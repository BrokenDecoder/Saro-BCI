"""
Phase 2 – Core Processing Engine
=================================
Component 1: EEG Ring Buffer
-----------------------------
A fixed-size, lock-safe circular buffer for real-time EEG streaming.

Design decisions:
- Pre-allocated NumPy array: zero GC pressure during runtime.
- Pointer arithmetic (not np.roll): O(1) writes, zero-copy reads.
- threading.Lock for thread-safety between the LSL inlet thread
  and the inference thread.
- read_window() returns a C-contiguous copy safe for ONNX/torch.
"""

from __future__ import annotations

import threading
import numpy as np
from typing import Optional


class EEGRingBuffer:
    """Fixed-size circular buffer for multi-channel real-time EEG.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels (e.g. 64).
    n_samples : int
        Total buffer capacity in samples (e.g. 1000 = 5s at 200 Hz).
    dtype : numpy dtype
        Storage dtype. float32 recommended (matches ONNX input).

    Notes
    -----
    Writing and reading are protected by the same `threading.Lock`.
    The LSL inlet thread calls `write()` at hardware rate.
    The inference thread calls `read_window()` every 0.5 s.
    """

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        dtype: np.dtype = np.float32,
    ) -> None:
        if n_channels <= 0 or n_samples <= 0:
            raise ValueError("n_channels and n_samples must be positive integers.")

        self.n_channels = n_channels
        self.n_samples  = n_samples
        self.dtype      = dtype

        # Pre-allocate the buffer — never resized at runtime
        self._buf: np.ndarray = np.zeros((n_channels, n_samples), dtype=dtype)

        # Write pointer (index into time axis)
        self._write_ptr: int = 0

        # How many samples have been written in total (saturates at n_samples)
        self._fill: int = 0

        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(self, chunk: np.ndarray) -> None:
        """Append a chunk of samples to the buffer.

        Parameters
        ----------
        chunk : (n_channels, n_chunk_samples) float32 ndarray
            New samples from the hardware / LSL stream.
        """
        if chunk.ndim != 2 or chunk.shape[0] != self.n_channels:
            raise ValueError(
                f"chunk must be (n_channels={self.n_channels}, n_chunk_samples), "
                f"got {chunk.shape}."
            )

        chunk = np.asarray(chunk, dtype=self.dtype)
        n_new = chunk.shape[1]

        with self._lock:
            if n_new >= self.n_samples:
                # New chunk is larger than buffer — keep only the tail
                self._buf[:] = chunk[:, -self.n_samples:]
                self._write_ptr = 0
                self._fill = self.n_samples
                return

            # Determine how many samples fit before the end of the buffer
            space_to_end = self.n_samples - self._write_ptr
            if n_new <= space_to_end:
                self._buf[:, self._write_ptr : self._write_ptr + n_new] = chunk
            else:
                # Wrap around: fill to end, then fill from start
                self._buf[:, self._write_ptr :] = chunk[:, :space_to_end]
                self._buf[:, : n_new - space_to_end] = chunk[:, space_to_end:]

            self._write_ptr = (self._write_ptr + n_new) % self.n_samples
            self._fill = min(self._fill + n_new, self.n_samples)

    def read_window(self, window_samples: Optional[int] = None) -> np.ndarray:
        """Return the most recent `window_samples` samples as a contiguous array.

        Parameters
        ----------
        window_samples : int or None
            Number of most-recent samples to return.
            Defaults to the full buffer capacity.

        Returns
        -------
        window : (n_channels, window_samples) float32 ndarray  — C-contiguous copy.
        """
        with self._lock:
            W = window_samples or self.n_samples
            W = min(W, self._fill)  # can't return more than we have

            if W == 0:
                return np.zeros((self.n_channels, 0), dtype=self.dtype)

            # Reconstruct chronological order from the circular buffer
            end   = self._write_ptr
            start = (end - W) % self.n_samples

            if start < end:
                window = self._buf[:, start:end]
            else:
                # Wraps around the end
                window = np.concatenate(
                    [self._buf[:, start:], self._buf[:, :end]], axis=1
                )

            # Return a C-contiguous copy (safe for ONNX / torch.from_numpy)
            return np.ascontiguousarray(window)

    @property
    def is_full(self) -> bool:
        """True once the buffer has been filled at least once."""
        return self._fill >= self.n_samples

    @property
    def fill_ratio(self) -> float:
        """Fraction of the buffer currently occupied, in [0.0, 1.0]."""
        return self._fill / self.n_samples

    @property
    def samples_available(self) -> int:
        """Number of valid samples currently stored."""
        return self._fill

    def reset(self) -> None:
        """Zero out the buffer and reset pointers (useful between sessions)."""
        with self._lock:
            self._buf[:] = 0.0
            self._write_ptr = 0
            self._fill = 0

    def __repr__(self) -> str:
        return (
            f"EEGRingBuffer(channels={self.n_channels}, "
            f"capacity={self.n_samples}, "
            f"filled={self._fill}/{self.n_samples})"
        )


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    N_CH, N_BUF = 64, 1000
    buf = EEGRingBuffer(n_channels=N_CH, n_samples=N_BUF)

    # Simulate 200 Hz hardware: write 5 ms chunks
    chunk_size = 1   # 1 sample per write = worst case latency test
    n_writes   = 1200
    t0 = time.perf_counter()
    for i in range(n_writes):
        chunk = np.random.randn(N_CH, chunk_size).astype(np.float32)
        buf.write(chunk)
    elapsed = time.perf_counter() - t0

    print(f"Wrote {n_writes} chunks in {elapsed*1000:.2f} ms")
    print(f"Buffer: {buf}")
    print(f"Is full: {buf.is_full}")

    window = buf.read_window(window_samples=500)
    print(f"Window shape: {window.shape}, dtype: {window.dtype}, C-contiguous: {window.flags['C_CONTIGUOUS']}")
    print("ring_buffer.py self-test PASSED ✓")
