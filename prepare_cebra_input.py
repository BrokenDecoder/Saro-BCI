# scripts/prepare_cebra_input.py
"""
Create NumPy inputs for CEBRA sanity-check grid
----------------------------------------------
Outputs: 12 .npy files in data/processed/<clean>/<scale>/
         shape = (n_channels_total, n_times)

Grid:
    cleaning  : zeropad_30  |  cut_60
    pairing   : spk9-lst10, lst9-spk10, stacked
    scaling   : raw         |  normalized
"""

from pathlib import Path
import logging
import numpy as np
import mne

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# 1.  Paths
# ---------------------------------------------------------------------
ROOT   = Path(__file__).resolve().parents[0]          # repo root
RAW    = ROOT / "data" / "raw"
PROC   = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# 2.  Helper functions
# ---------------------------------------------------------------------
def load_eeg(edf_path: Path, max_samples: int = 1000) -> np.ndarray:
    """Return EEG data as np.ndarray (channels, time).

    Parameters
    ----------
    edf_path   : Path to the EDF file.
    max_samples: Maximum number of time samples to load (default 1000).

    Returns
    -------
    data : (n_channels, max_samples) float32 array.

    Raises
    ------
    FileNotFoundError : If the EDF file does not exist.
    """
    if not edf_path.exists():
        raise FileNotFoundError(f"EDF not found: {edf_path}")
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    data = raw.get_data(picks="eeg").astype(np.float32)
    return data[:, :max_samples]


def generate_mock_ai_states(channels: int, time_steps: int) -> np.ndarray:
    """Generate synthetic AI internal states (random walk) for Twin Brain.

    Returns
    -------
    ai_state : (channels, time_steps) float32 array.
    """
    steps = np.random.randn(channels, time_steps) * 0.1
    return np.cumsum(steps, axis=1).astype(np.float32)


def align_lengths(a: np.ndarray, b: np.ndarray):
    """Trim the longer array so a and b share the same number of samples."""
    T = min(a.shape[1], b.shape[1])
    return a[:, :T], b[:, :T]


def minmax_per_channel(x: np.ndarray) -> np.ndarray:
    """Scale each channel to [0, 1] independently."""
    xmin = x.min(axis=1, keepdims=True)
    xmax = x.max(axis=1, keepdims=True)
    rng  = np.where((xmax - xmin) == 0, 1, xmax - xmin)
    return (x - xmin) / rng


def zscore_per_channel(x: np.ndarray) -> np.ndarray:
    """Z-score each channel independently (zero mean, unit variance)."""
    mu  = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (x - mu) / std


def save_npy(array: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, array)
    logger.info("✓ Saved %-45s  %s", out_path.name, array.shape)

# ---------------------------------------------------------------------
# 3.  Grid definition
# ---------------------------------------------------------------------
GRID = [
    # cleaning, pairing_name,  EDF paths (relative to RAW/<clean>/...)
    (
        "zeropad_30",
        "spk9-lst10",
        ("individual/nt9_speak_zeropad_30_components_preprocessed.edf",
         "individual/nt10_listen_zeropad_30_components_preprocessed.edf"),
    ),
    (
        "zeropad_30",
        "lst9-spk10",
        ("individual/nt9_listen_zeropad_30_components_preprocessed.edf",
         "individual/nt10_speak_zeropad_30_components_preprocessed.edf"),
    ),
    (
        "zeropad_30",
        "stacked",
        ("stacked/nt9_zeropad_speak_listen_stacked.edf",
         "stacked/nt10_zeropad_listen_speak_stacked.edf"),
    ),

    (
        "cut_60",
        "spk9-lst10",
        ("individual/nt9_speak_cut_60_components_preprocessed.edf",
         "individual/nt10_listen_cut_60_components_preprocessed.edf"),
    ),
    (
        "cut_60",
        "lst9-spk10",
        ("individual/nt9_listen_cut_60_components_preprocessed.edf",
         "individual/nt10_speak_cut_60_components_preprocessed.edf"),
    ),
    (
        "cut_60",
        "stacked",
        ("stacked/nt9_cut_speak_listen_stacked.edf",
         "stacked/nt10_cut_listen_speak_stacked.edf"),
    ),
    (
        "cut_60",
        "patient-ai",
        ("individual/nt9_speak_cut_60_components_preprocessed.edf",
         "AI_MOCK"),
    ),
]

# ---------------------------------------------------------------------
# 4.  Main loop
# ---------------------------------------------------------------------

def main() -> None:
    for clean, pairing, (edf_a_rel, edf_b_rel) in GRID:
        edf_a = RAW / clean / edf_a_rel
        edf_b = RAW / clean / edf_b_rel

        # ---------- load ----------
        try:
            A = load_eeg(edf_a)
        except FileNotFoundError as exc:
            logger.warning("Skipping %s/%s: %s", clean, pairing, exc)
            continue

        if str(edf_b_rel) == "AI_MOCK":
            B = generate_mock_ai_states(A.shape[0], A.shape[1])
        else:
            try:
                B = load_eeg(edf_b)
            except FileNotFoundError as exc:
                logger.warning("Skipping %s/%s: %s", clean, pairing, exc)
                continue

        # ---------- align ----------
        A, B = align_lengths(A, B)            # ensure equal length T
        combined = np.vstack([A, B])          # (ch_A+ch_B, T)

        # ---------- save raw ----------
        out_raw = PROC / clean / "raw" / f"{pairing}.npy"
        save_npy(combined, out_raw)

        # ---------- save normalized ----------
        combined_norm = minmax_per_channel(combined)
        out_norm = PROC / clean / "normalized" / f"{pairing}.npy"
        save_npy(combined_norm, out_norm)

        # ---------- save z-scored ----------
        combined_z = zscore_per_channel(combined)
        out_z = PROC / clean / "zscored" / f"{pairing}.npy"
        save_npy(combined_z, out_z)

    logger.info("\nAll NumPy files generated.")


if __name__ == "__main__":
    main()
