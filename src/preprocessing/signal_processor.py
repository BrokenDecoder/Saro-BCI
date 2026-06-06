import mne
import numpy as np
import pywt
import logging
from typing import Optional, Literal

# Sampling frequency used in mock / default configuration
_DEFAULT_SFREQ = 250
_DEFAULT_N_CHANNELS = 19
_VALID_MODES = frozenset({"testing", "realtime"})


class VersatileSignalProcessor:
    def __init__(
        self,
        mode: Literal["testing", "realtime"] = "testing",
        mock_data_path: Optional[str] = None,
        n_channels: int = _DEFAULT_N_CHANNELS,
        sfreq: int = _DEFAULT_SFREQ,
    ):
        """
        Parameters
        ----------
        mode : 'realtime' for live LSL streaming, 'testing' for simulated data.
        mock_data_path : Path to a pre-recorded EDF file (testing mode only).
        n_channels : Number of EEG channels (default 19).
        sfreq : Sampling frequency in Hz (default 250).
        """
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got '{mode}'.")
        self.mode = mode
        self.mock_data_path = mock_data_path
        self.n_channels = n_channels
        self.sfreq = sfreq
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Data acquisition
    # ------------------------------------------------------------------
    def fetch_epoch(self) -> np.ndarray:
        """Fetch the next chunk of EEG data as (n_channels, n_times)."""
        if self.mode == "testing":
            self.logger.info("Fetching simulated epoch (%d ch, %d samples).",
                             self.n_channels, self.sfreq)
            return np.random.randn(self.n_channels, self.sfreq).astype(np.float32)

        elif self.mode == "realtime":
            self.logger.info("Fetching epoch from live LSL stream.")
            # TODO: instantiate an LSL StreamInlet and call inlet.pull_chunk()
            raise NotImplementedError(
                "LSL real-time streaming is not yet wired up. "
                "Install pylsl and implement the StreamInlet pull here."
            )

    # ------------------------------------------------------------------
    # Signal processing
    # ------------------------------------------------------------------
    def apply_bandpass(self, data: np.ndarray,
                       l_freq: float = 1.0,
                       h_freq: float = 40.0) -> np.ndarray:
        """Apply a zero-phase bandpass FIR filter to each channel.

        Parameters
        ----------
        data : (n_channels, n_times)
        l_freq, h_freq : Lower / upper cut-off frequencies in Hz.

        Returns
        -------
        Filtered data, same shape as input.
        """
        from scipy.signal import firwin, filtfilt
        nyq = self.sfreq / 2.0
        numtaps = int(self.sfreq // l_freq) | 1  # nearest odd number
        b = firwin(numtaps, [l_freq / nyq, h_freq / nyq], pass_zero=False)
        return filtfilt(b, [1.0], data, axis=-1).astype(data.dtype)

    def apply_cwt(self, data: np.ndarray,
                  wavelet: str = "morl",
                  n_scales: int = 30) -> np.ndarray:
        """Apply Continuous Wavelet Transform **per channel**.

        Parameters
        ----------
        data : (n_channels, n_times) or (n_times,) for a single channel.
        wavelet : PyWavelets wavelet name (default 'morl').
        n_scales : Number of scale steps (1 .. n_scales inclusive).

        Returns
        -------
        coeffs : (n_channels, n_scales, n_times) — or (n_scales, n_times)
                  when a 1-D array is supplied.
        """
        widths = np.arange(1, n_scales + 1)
        if data.ndim == 1:
            coeffs, _ = pywt.cwt(data, widths, wavelet)
            return coeffs
        # Vectorise over channels
        results = [pywt.cwt(data[ch], widths, wavelet)[0] for ch in range(data.shape[0])]
        return np.stack(results, axis=0)  # (n_ch, n_scales, T)

    # ------------------------------------------------------------------
    # Artefact rejection
    # ------------------------------------------------------------------
    def fuse_multimodal(
        self,
        eeg: np.ndarray,
        ecg: Optional[np.ndarray] = None,
        emg: Optional[np.ndarray] = None,
        ecg_thresh_z: float = 3.0,
        emg_thresh_z: float = 3.0,
    ) -> np.ndarray:
        """Flag and zero-out EEG samples that coincide with ECG/EMG artefacts.

        Parameters
        ----------
        eeg : (n_channels, n_times)
        ecg : (n_times,) or None  — cardiac channel.
        emg : (n_times,) or None  — muscle/motion channel.
        ecg_thresh_z : Z-score threshold for cardiac spike detection.
        emg_thresh_z : Z-score threshold for EMG burst detection.

        Returns
        -------
        EEG array with artefact epochs zeroed out.
        """
        if ecg is None and emg is None:
            return eeg

        self.logger.info("Applying multimodal artefact rejection.")
        eeg_clean = eeg.copy()
        mask = np.zeros(eeg.shape[1], dtype=bool)

        for signal, thresh in [(ecg, ecg_thresh_z), (emg, emg_thresh_z)]:
            if signal is None:
                continue
            z = (signal - signal.mean()) / (signal.std() + 1e-12)
            mask |= np.abs(z) > thresh

        if mask.any():
            eeg_clean[:, mask] = 0.0
            self.logger.info("Zeroed %d / %d artefact samples.",
                             mask.sum(), mask.size)
        return eeg_clean
