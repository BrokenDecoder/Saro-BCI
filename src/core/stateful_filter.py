"""
Phase 2 – Core Processing Engine
=================================
Component 2: Real-Time Stateful EEG Filter
-------------------------------------------
Applies notch + bandpass filtering across multiple channels in real-time,
preserving filter state (zi) between chunks to avoid edge artifacts.

Key design choices:
- scipy.signal.sosfilt (second-order sections): numerically stable for
  high-order IIR filters on 24-bit EEG data. Never use lfilter.
- Persistent zi state (shape: [n_sections, n_channels, 2]): the filter
  "remembers" where it left off on each chunk. Omitting this produces
  large transient artifacts at every chunk boundary.
- Separate bandpass and notch filters chained in sequence.
- Thread-safe: filter state protected by a threading.Lock.
"""

from __future__ import annotations

import threading
import logging
import numpy as np
from typing import Optional, Tuple

try:
    from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class RealtimeEEGFilter:
    """Stateful, real-time, multi-channel bandpass + notch filter.

    Parameters
    ----------
    sfreq : float
        Sampling frequency in Hz (e.g. 160.0 for PhysioNet, 250.0 for OpenBCI).
    n_channels : int
        Number of EEG channels.
    lowcut : float
        Low-frequency cutoff in Hz (default 1.0 Hz removes DC drift).
    highcut : float
        High-frequency cutoff in Hz (default 50.0 Hz).
    notch_freq : float
        Power line frequency to notch out, Hz (50 or 60). Set to None to skip.
    order : int
        Butterworth filter order (4 is a good balance of roll-off vs. latency).

    Notes
    -----
    On the very first call to `process()` the filter state is warm-started
    using `sosfilt_zi` so there is no step response transient even on the
    first chunk.
    """

    def __init__(
        self,
        sfreq: float,
        n_channels: int,
        lowcut: float  = 1.0,
        highcut: float = 50.0,
        notch_freq: Optional[float] = 50.0,
        order: int = 4,
    ) -> None:
        if not SCIPY_AVAILABLE:
            raise ImportError(
                "scipy is required for RealtimeEEGFilter. "
                "Install with: pip install scipy"
            )

        self.sfreq      = sfreq
        self.n_channels = n_channels
        self.lowcut     = lowcut
        self.highcut    = highcut
        self.notch_freq = notch_freq
        self.order      = order

        # Design filters (second-order sections = numerically stable)
        self._sos_bp: np.ndarray = butter(
            order, [lowcut, highcut], btype="bandpass",
            fs=sfreq, output="sos"
        )
        logger.info(
            "Bandpass filter designed: %.1f–%.1f Hz, order=%d, fs=%.1f Hz",
            lowcut, highcut, order, sfreq
        )

        self._sos_notch: Optional[np.ndarray] = None
        if notch_freq is not None:
            # iirnotch gives a clinically narrow notch (Q=35 → ±0.15 Hz at 50 Hz)
            # compared to a 4th-order Butterworth bandstop (±1 Hz), preserving
            # the gamma band directly adjacent to the powerline frequency.
            b_notch, a_notch = iirnotch(notch_freq, Q=35, fs=sfreq)
            self._sos_notch = tf2sos(b_notch, a_notch)
            logger.info(
                "Notch filter designed: iirnotch %.1f Hz Q=35 (±%.2f Hz), fs=%.1f Hz",
                notch_freq, notch_freq / (2 * 35), sfreq
            )

        # Filter states — initialised to None, set on first process() call
        self._zi_bp:    Optional[np.ndarray] = None
        self._zi_notch: Optional[np.ndarray] = None

        self._lock = threading.Lock()
        self._initialised = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_state(self, first_chunk: np.ndarray) -> None:
        """Warm-start filter state from the DC level of the first chunk.

        Using zi=0 causes a step transient when the filter starts.
        Using sosfilt_zi scaled by the first sample's value avoids that.
        """
        # zi shape for sosfilt: (n_sections, n_channels, 2)
        # sosfilt_zi returns shape (n_sections, 2) for a single channel,
        # so we broadcast across channels.
        dc = first_chunk[:, :1]  # (n_channels, 1)

        zi_bp_1ch = sosfilt_zi(self._sos_bp)   # (n_sections, 2)
        # Scale zi by DC level of each channel → shape (n_sections, n_channels, 2)
        self._zi_bp = (
            zi_bp_1ch[:, np.newaxis, :] * dc[np.newaxis, :, :]
        ).transpose(0, 1, 2)  # (n_sections, n_channels, 2)

        if self._sos_notch is not None:
            zi_notch_1ch = sosfilt_zi(self._sos_notch)
            self._zi_notch = (
                zi_notch_1ch[:, np.newaxis, :] * dc[np.newaxis, :, :]
            ).transpose(0, 1, 2)

        self._initialised = True
        logger.debug("Filter state initialised (warm-start).")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """Filter a chunk of EEG data, updating state for continuity.

        Parameters
        ----------
        chunk : (n_channels, n_samples) float32 ndarray

        Returns
        -------
        filtered : (n_channels, n_samples) float32 ndarray — same shape as input.
        """
        if chunk.ndim != 2 or chunk.shape[0] != self.n_channels:
            raise ValueError(
                f"chunk must be ({self.n_channels}, n_samples), got {chunk.shape}."
            )

        chunk = np.ascontiguousarray(chunk, dtype=np.float64)  # sosfilt likes float64

        with self._lock:
            if not self._initialised:
                self._init_state(chunk)

            # --- Bandpass ---
            filtered, self._zi_bp = sosfilt(
                self._sos_bp, chunk, axis=1, zi=self._zi_bp
            )

            # --- Notch (chained after bandpass) ---
            if self._sos_notch is not None and self._zi_notch is not None:
                filtered, self._zi_notch = sosfilt(
                    self._sos_notch, filtered, axis=1, zi=self._zi_notch
                )

        return filtered.astype(np.float32)

    def reset(self) -> None:
        """Reset filter state (call between sessions / recording segments)."""
        with self._lock:
            self._zi_bp    = None
            self._zi_notch = None
            self._initialised = False
        logger.info("Filter state reset.")

    def get_filter_info(self) -> dict:
        """Return a summary dict for logging / dashboard status panel."""
        return {
            "sfreq_hz":      self.sfreq,
            "n_channels":    self.n_channels,
            "bandpass_hz":   [self.lowcut, self.highcut],
            "notch_hz":      self.notch_freq,
            "order":         self.order,
            "initialised":   self._initialised,
        }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    SFREQ      = 160.0   # PhysioNet EEG BCI sampling rate
    N_CH       = 64
    CHUNK_SIZE = 16      # 100 ms chunks at 160 Hz

    filt = RealtimeEEGFilter(sfreq=SFREQ, n_channels=N_CH)

    # Generate 5 s of synthetic 60 Hz + 10 Hz sinusoid (simulates noisy EEG)
    t = np.linspace(0, 5, int(5 * SFREQ))
    signal = (
        np.sin(2 * np.pi * 10 * t)   # 10 Hz alpha (target)
        + 2 * np.sin(2 * np.pi * 60 * t)  # 60 Hz power line (noise)
        + 0.1 * np.random.randn(len(t))
    )
    signal_mc = np.tile(signal, (N_CH, 1)).astype(np.float32)  # (64, 800)

    # Process in 100 ms chunks (simulating real-time)
    n_chunks = signal_mc.shape[1] // CHUNK_SIZE
    t0 = time.perf_counter()
    filtered = []
    for i in range(n_chunks):
        chunk = signal_mc[:, i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
        out   = filt.process(chunk)
        filtered.append(out)
    elapsed = time.perf_counter() - t0

    result = np.concatenate(filtered, axis=1)
    print(f"Filtered {n_chunks} chunks in {elapsed*1000:.2f} ms")
    print(f"Output shape: {result.shape}, dtype: {result.dtype}")
    print(f"Filter info: {filt.get_filter_info()}")
    print("stateful_filter.py self-test PASSED ✓")
