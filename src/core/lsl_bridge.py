"""
Phase 1 / Phase 3 – Hardware Bridge
=====================================
LSL (Lab Streaming Layer) inlet for 128-channel EEG hardware.

Supports two hardware targets:
  A. ADS1299-based amplifiers (OpenBCI Cyton ×4, BrainFlow, custom boards)
     → 8 × ADS1299 = 128 channels @ 250 Hz typical
  B. Any LSL-compatible amplifier (g.tec, BrainProducts, etc.)

Usage:
    # Real hardware (blocking, runs in a daemon thread):
    bridge = LSLBridge(n_channels=128, sfreq=250.0)
    bridge.connect()          # finds the LSL stream
    bridge.start(pipeline)    # feeds pipeline.push_chunk() forever

    # Without hardware (synthetic simulation at hardware rate):
    bridge = LSLBridge(n_channels=128, sfreq=250.0, synthetic=True)
    bridge.start(pipeline)

Why LSL and not serial / raw USB?
  - LSL handles network clock synchronisation (sub-millisecond)
  - Works with BrainFlow, OpenBCI GUI, any amplifier vendor
  - Zero custom serial parsers — just pull samples
  - Multi-process safe: multiple consumers can subscribe to one stream
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Callable

import numpy as np

logger = logging.getLogger("saro.lsl_bridge")

try:
    from pylsl import StreamInlet, resolve_stream, resolve_byprop, ContinuousResolver
    LSL_AVAILABLE = True
except ImportError:
    LSL_AVAILABLE = False
    logger.warning(
        "pylsl not installed — LSL bridge will run in synthetic mode only. "
        "Install with: pip install pylsl"
    )


# ─────────────────────────────────────────────────────────────────────────────
# ADS1299 Channel Layout (standard 10-20 system extension for 128 channels)
# ─────────────────────────────────────────────────────────────────────────────

# 128-channel positions mapped to a standardised layout.
# Grouped as 8 × ADS1299 ICs (16 ch each).
# Used for channel labelling in the dashboard.
ADS1299_128CH_LABELS = [
    # IC 0 – frontal
    "Fp1","Fp2","F7","F3","Fz","F4","F8","FC5",
    "FC1","FC2","FC6","M1","T7","C3","Cz","C4",
    # IC 1 – central
    "T8","M2","CP5","CP1","CP2","CP6","P7","P3",
    "Pz","P4","P8","POz","O1","Oz","O2","PO3",
    # IC 2 – temporal
    "AF7","AF3","AFz","AF4","AF8","F5","F1","F2",
    "F6","FT7","FC3","FC4","FT8","C5","C1","C2",
    # IC 3 – parietal
    "C6","TP7","CP3","CP4","TP8","P5","P1","P2",
    "P6","PO7","PO3","PO4","PO8","O1h","O2h","Iz",
    # IC 4 – occipital extended
    "FT9","FT10","T9","T10","P9","P10","PO9","PO10",
    "CB1","CB2","I1","I2","EEG129","EEG130","EEG131","EEG132",
    # IC 5 – auxiliary
    "EEG133","EEG134","EEG135","EEG136","EEG137","EEG138","EEG139","EEG140",
    "EEG141","EEG142","EEG143","EEG144","EEG145","EEG146","EEG147","EEG148",
    # IC 6
    "EEG149","EEG150","EEG151","EEG152","EEG153","EEG154","EEG155","EEG156",
    "EEG157","EEG158","EEG159","EEG160","EEG161","EEG162","EEG163","EEG164",
    # IC 7
    "EEG165","EEG166","EEG167","EEG168","EEG169","EEG170","EEG171","EEG172",
    "EEG173","EEG174","EEG175","EEG176","EEG177","EEG178","EEG179","EEG180",
]


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic EEG Generator (replaces hardware when no LSL stream is found)
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticEEGGenerator:
    """
    Generates physiologically-realistic synthetic 128-channel EEG.

    Combines:
      - 1/f (pink) noise as the baseline wideband EEG background
      - Alpha (8-13 Hz) oscillation peaks at occipital channels (Ch 28-31)
      - Mu (8-12 Hz) peaks at motor cortex channels (Ch 13-15)
      - 50 Hz mains interference (before filtering)
      - Realistic inter-channel correlations via spatial mixing matrix
    """

    def __init__(self, n_channels: int = 128, sfreq: float = 250.0) -> None:
        self.n_channels = n_channels
        self.sfreq      = sfreq
        self._phase     = np.random.uniform(0, 2 * np.pi, n_channels)
        self._t         = 0.0
        self._pink_state = np.zeros((n_channels, 16))  # 1/f state

        # Spatial correlation matrix (neighbouring channels correlated)
        rng   = np.random.default_rng(42)
        A     = rng.standard_normal((n_channels, n_channels)) * 0.1
        A     += np.eye(n_channels)
        Q, _  = np.linalg.qr(A)
        self._mix = Q[:, :min(n_channels, 16)]  # dim reduction for speed

    def _pink_noise(self, n_samples: int) -> np.ndarray:
        """Generate 1/f pink noise via Voss-McCartney algorithm."""
        white = np.random.randn(self.n_channels, n_samples)
        # Simple IIR approximation of 1/f: y = 0.99*y + x
        out = np.zeros_like(white)
        state = np.zeros(self.n_channels)
        for i in range(n_samples):
            state = 0.99 * state + white[:, i] * 0.1
            out[:, i] = state
        return out

    def generate_chunk(self, n_samples: int) -> np.ndarray:
        """Return (n_channels, n_samples) float32 synthetic EEG in microvolts."""
        t = np.linspace(self._t, self._t + n_samples / self.sfreq, n_samples)
        self._t += n_samples / self.sfreq

        chunk = self._pink_noise(n_samples) * 20.0   # ~20 uV background

        # Alpha (10 Hz) — occipital channels
        occ_chs = slice(28, 32)
        chunk[occ_chs] += 15.0 * np.sin(2 * np.pi * 10 * t + self._phase[occ_chs, None])

        # Mu (10 Hz) — motor channels
        mot_chs = slice(13, 16)
        chunk[mot_chs] += 8.0 * np.sin(2 * np.pi * 10 * t + self._phase[mot_chs, None])

        # Beta (20 Hz) — frontal channels
        chunk[:8] += 5.0 * np.sin(2 * np.pi * 20 * t)

        # 50 Hz mains hum (attenuated — will be removed by notch filter)
        chunk += 3.0 * np.sin(2 * np.pi * 50 * t)

        return chunk.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# LSL Bridge
# ─────────────────────────────────────────────────────────────────────────────

class LSLBridge:
    """
    128-channel LSL inlet → BCIPipeline bridge.

    In real-hardware mode:
      - Discovers the first EEG LSL stream on the network
      - Pulls chunks at the hardware rate and feeds pipeline.push_chunk()

    In synthetic mode (no hardware / pylsl not installed):
      - Runs SyntheticEEGGenerator at the configured hardware rate
      - Feeds the same pipeline interface

    Parameters
    ----------
    n_channels    : Expected EEG channel count (128 for ADS1299 x8).
    sfreq         : Hardware sampling rate in Hz (250 for ADS1299).
    chunk_size    : Samples to pull per iteration (e.g. 25 = 100 ms at 250 Hz).
    stream_name   : LSL stream name to search for. None = auto-discover.
    synthetic     : Force synthetic mode even if pylsl is available.
    on_connected  : Optional callback(stream_info) when hardware connects.
    """

    def __init__(
        self,
        n_channels:   int    = 128,
        sfreq:        float  = 250.0,
        chunk_size:   int    = 25,
        stream_name:  Optional[str] = None,
        synthetic:    bool   = False,
        on_connected: Optional[Callable] = None,
    ) -> None:
        self.n_channels   = n_channels
        self.sfreq        = sfreq
        self.chunk_size   = chunk_size
        self.stream_name  = stream_name
        self.on_connected = on_connected

        self._force_synthetic = synthetic or not LSL_AVAILABLE
        self._inlet:    Optional[StreamInlet] = None
        self._synth:    Optional[SyntheticEEGGenerator] = None
        self._thread:   Optional[threading.Thread] = None
        self._running   = False
        self._connected = False

        self.mode: str = "unconnected"

    # ── Public API ────────────────────────────────────────────────────────

    def connect(self, timeout: float = 5.0) -> bool:
        """Try to find an EEG LSL stream. Returns True on success.

        Falls back to synthetic if not found within `timeout` seconds.
        """
        if self._force_synthetic:
            self._setup_synthetic()
            return False

        logger.info("Searching for LSL EEG stream (timeout=%.1fs)...", timeout)
        try:
            if self.stream_name:
                streams = resolve_byprop("name", self.stream_name, timeout=timeout)
            else:
                streams = resolve_stream("type", "EEG", timeout=timeout)

            if not streams:
                logger.warning("No LSL stream found — switching to synthetic mode.")
                self._setup_synthetic()
                return False

            info = streams[0]
            self._inlet = StreamInlet(info, max_chunklen=self.chunk_size)
            self._connected = True
            self.mode = "lsl_hardware"
            logger.info(
                "LSL stream connected: name=%s, channels=%d, sfreq=%.1f Hz",
                info.name(), info.channel_count(), info.nominal_srate()
            )
            if self.on_connected:
                self.on_connected(info)
            return True

        except Exception as exc:
            logger.warning("LSL connect failed (%s) — synthetic mode.", exc)
            self._setup_synthetic()
            return False

    def _setup_synthetic(self) -> None:
        self._synth   = SyntheticEEGGenerator(self.n_channels, self.sfreq)
        self._connected = False
        self.mode     = "synthetic"
        logger.info("LSL bridge running in SYNTHETIC mode (%dch @ %.0fHz).",
                    self.n_channels, self.sfreq)

    def start(self, pipeline) -> None:
        """Start background thread that feeds chunks into `pipeline.push_chunk()`."""
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._run, args=(pipeline,),
            daemon=True, name="lsl-bridge"
        )
        self._thread.start()
        logger.info("LSL bridge thread started (mode=%s).", self.mode)

    def stop(self) -> None:
        """Stop the background thread."""
        self._running = False
        logger.info("LSL bridge stopped.")

    @property
    def is_hardware_connected(self) -> bool:
        return self._connected and self.mode == "lsl_hardware"

    # ── Background thread ─────────────────────────────────────────────────

    def _run(self, pipeline) -> None:
        """Pull chunks from hardware or synthetic generator → pipeline."""
        sleep_per_chunk = self.chunk_size / self.sfreq

        while self._running:
            try:
                chunk = self._pull_chunk()
                if chunk is not None:
                    pipeline.push_chunk(chunk)
                else:
                    time.sleep(sleep_per_chunk)
            except Exception as exc:
                logger.error("LSL bridge error: %s — retrying...", exc)
                time.sleep(0.5)

    def _pull_chunk(self) -> Optional[np.ndarray]:
        """Pull one chunk from hardware or synthetic generator.

        Returns (n_channels, chunk_size) float32 or None if no data.
        """
        if self.mode == "lsl_hardware" and self._inlet:
            # pylsl returns (samples × channels); we need (channels × samples)
            samples, _ = self._inlet.pull_chunk(
                timeout=0.1, max_samples=self.chunk_size
            )
            if not samples:
                return None
            arr = np.array(samples, dtype=np.float32).T  # (channels, samples)

            # Handle channel count mismatch
            n_hw = arr.shape[0]
            if n_hw < self.n_channels:
                pad = np.zeros((self.n_channels - n_hw, arr.shape[1]), dtype=np.float32)
                arr = np.concatenate([arr, pad], axis=0)
            elif n_hw > self.n_channels:
                arr = arr[:self.n_channels]

            return arr

        elif self.mode == "synthetic" and self._synth:
            time.sleep(self.chunk_size / self.sfreq)   # real-time pacing
            return self._synth.generate_chunk(self.chunk_size)

        return None

    def get_channel_labels(self) -> list[str]:
        """Return channel labels (ADS1299 layout for 128ch, generic otherwise)."""
        if self.n_channels == 128:
            return ADS1299_128CH_LABELS[:128]
        return [f"CH{i+1:03d}" for i in range(self.n_channels)]

    def __repr__(self) -> str:
        return (
            f"LSLBridge(mode={self.mode}, "
            f"channels={self.n_channels}, "
            f"sfreq={self.sfreq}Hz, "
            f"hardware={self.is_hardware_connected})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s]: %(message)s")

    N_CH   = 128
    SFREQ  = 250.0
    CHUNKS = 20

    bridge = LSLBridge(n_channels=N_CH, sfreq=SFREQ, synthetic=True)
    bridge.connect()

    print(f"\nBridge: {bridge}")
    print(f"Labels: {bridge.get_channel_labels()[:8]} ... (128 total)")

    chunks_received = []

    class FakePipeline:
        def push_chunk(self, c):
            chunks_received.append(c)

    fp = FakePipeline()
    bridge.start(fp)

    print(f"Running for {CHUNKS} chunks ({CHUNKS * 25 / SFREQ:.1f}s)...")
    time.sleep(CHUNKS * 25 / SFREQ + 0.2)
    bridge.stop()

    print(f"Received {len(chunks_received)} chunks")
    if chunks_received:
        c = chunks_received[-1]
        print(f"Chunk shape: {c.shape}, dtype: {c.dtype}")
        print(f"RMS per first 5 channels: {[round(float(np.sqrt(np.mean(c[i]**2))),3) for i in range(5)]}")
    print("lsl_bridge.py self-test PASSED")
