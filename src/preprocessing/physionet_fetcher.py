"""
PhysioNet EEG Data Fetcher
==========================
Fetches real human brainwaves from the PhysioNet EEG Motor Movement/Imagery Dataset
to replace synthetic stubs and test the BCI pipeline with real data.

When target_channels > n_hardware_channels (e.g. 128 > 64), the data is
interleaved-repeated to simulate a 128-channel array for pipeline testing.
Real 128-ch hardware data from an ADS1299 x8 setup would not need this.
"""

import logging
from typing import Tuple

import mne
import numpy as np

logger = logging.getLogger(__name__)


def fetch_real_eeg_data(
    subject: int = 1,
    run: int = 3,
    tmin: float = 0.0,
    tmax: float = 60.0,
    target_channels: int = 128,
) -> Tuple[np.ndarray, int]:
    """
    Fetches real EEG data from the PhysioNet dataset and optionally pads
    it to `target_channels` for hardware-size pipeline testing.

    Parameters
    ----------
    subject : int
        The subject ID (1 to 109).
    run : int
        The run number (e.g., 3 is a motor imagery run).
    tmin : float
        Start time in seconds to crop.
    tmax : float
        End time in seconds to crop.
    target_channels : int
        Desired output channel count.  If > actual channels, data is
        interleave-padded (simulates a denser electrode array).

    Returns
    -------
    data : np.ndarray
        Shape (target_channels, n_times). float32.
    sfreq : int
        The sampling frequency.
    """
    logger.info("Fetching real EEG data for Subject %d, Run %d", subject, run)

    # Download or load from cache
    raw_fnames = mne.datasets.eegbci.load_data(subject, run, update_path=False)
    raw = mne.io.read_raw_edf(raw_fnames[0], preload=True, verbose=False)

    # NOTE: Do NOT apply any filter here.
    # The pipeline's RealtimeEEGFilter (stateful_filter.py) handles bandpass +
    # notch filtering in real-time to preserve filter state continuity.
    # Filtering here would cause double-filtering and phase distortion.

    # Crop
    max_time = raw.times[-1]
    raw.crop(tmin=tmin, tmax=min(tmax, max_time))

    sfreq  = int(raw.info['sfreq'])
    data   = raw.get_data().astype(np.float32)   # (n_actual_ch, n_times)
    n_real = data.shape[0]

    logger.info(
        "PhysioNet raw: shape=%s sfreq=%dHz -> padding to %d channels",
        data.shape, sfreq, target_channels
    )

    # Pad / trim to target_channels
    if n_real < target_channels:
        # Interleave-repeat: cycle through real channels
        repeats = int(np.ceil(target_channels / n_real))
        data = np.tile(data, (repeats, 1))[:target_channels, :]
        # Add per-channel micro-noise so channels aren't identical
        noise_scale = np.std(data) * 0.05
        data += (np.random.randn(*data.shape) * noise_scale).astype(np.float32)
    elif n_real > target_channels:
        data = data[:target_channels, :]

    logger.info("EEG data ready: shape=%s sfreq=%dHz", data.shape, sfreq)
    return data, sfreq


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data, sfreq = fetch_real_eeg_data(target_channels=128)
    print(f"Data shape: {data.shape}")
    print(f"Sampling Frequency: {sfreq}")
