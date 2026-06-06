import mne
from mne.preprocessing import ICA
from mne.channels import read_custom_montage
from pyprep.find_noisy_channels import NoisyChannels
import logging
import os
import json
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

def convert_to_gsn_hydrocel_names(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """
    Rename EEG channels from 'EEG X' format to 'EX' (GSN-HydroCel).
    'EEG VREF' is renamed to 'Cz'.
    """
    mapping = {}
    for ch in raw.ch_names:
        if ch.upper() == 'EEG VREF':
            mapping[ch] = 'Cz'
        elif ch.startswith('EEG '):
            try:
                num = int(ch.split(' ')[1])
                mapping[ch] = f"E{num}"
            except ValueError:
                logger.warning("Skipping unrecognised channel: %s", ch)
    raw.rename_channels(mapping)
    return raw

def preprocess_eeg(
    path_to_set_file: str,
    montage_path: str = 'GSN-HydroCel-65_1.0.sfp',
    output_dir: str = 'preprocessed_eeg/',
    l_freq: float = 1.0,
    h_freq: float = 40.0,
    notch_freq: float = 50.0,
    n_ica_components: int = 15,
    auto_exclude_eog: bool = False,
) -> str:
    """
    Preprocess EEG data from a .set file and save as .edf after ICA.

    Parameters
    ----------
    path_to_set_file  : Path to the EEGLAB .set file.
    montage_path      : Path to the montage .sfp file.
    output_dir        : Directory where the output EDF file will be saved.
    l_freq            : High-pass filter cut-off in Hz (default 1.0).
    h_freq            : Low-pass filter cut-off in Hz (default 40.0).
    notch_freq        : Notch filter frequency for mains noise (default 50 Hz).
    n_ica_components  : Number of ICA components to fit (default 15).
    auto_exclude_eog  : If True, use MNE's EOG auto-detection instead of
                        prompting the user interactively (default False).

    Returns
    -------
    output_path : str – path to the saved pre-processed EDF file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize log dictionary
    processing_log = {
        'original_file': path_to_set_file,
        'timestamp': datetime.now().isoformat(),
        'processing_steps': []
    }

    # ------------------------------------------------------------------ #
    # Load data
    # ------------------------------------------------------------------ #
    raw = mne.io.read_raw_eeglab(path_to_set_file, preload=True)
    processing_log['processing_steps'].append({
        'step': 'load_data',
        'description': f'Loaded EEG data from {path_to_set_file}',
        'n_channels': len(raw.ch_names),
        'duration_seconds': raw.times[-1]
    })

    # ------------------------------------------------------------------ #
    # Rename channels and set montage
    # ------------------------------------------------------------------ #
    raw = convert_to_gsn_hydrocel_names(raw)
    montage = read_custom_montage(montage_path)
    raw.set_montage(montage)
    processing_log['processing_steps'].append({
        'step': 'set_montage',
        'description': f'Applied montage from {montage_path}',
        'montage_channels': montage.ch_names
    })

    # ------------------------------------------------------------------ #
    # Bandpass + notch filter
    # ------------------------------------------------------------------ #
    raw.filter(l_freq=l_freq, h_freq=h_freq, method='fir', fir_window='hamming')
    raw.notch_filter(freqs=notch_freq)
    processing_log['processing_steps'].append({
        'step': 'bandpass_notch_filter',
        'description': f'Applied {l_freq}–{h_freq} Hz bandpass + {notch_freq} Hz notch filter',
        'filter_settings': {'l_freq': l_freq, 'h_freq': h_freq, 'notch_freq': notch_freq}
    })

    # Automatic bad channel detection using pyprep
    nc = NoisyChannels(raw, random_state=1337)
    nc.find_all_bads()

    bad_channels = {
        'bad_by_nan': nc.bad_by_nan,
        'bad_by_flat': nc.bad_by_flat,
        'bad_by_deviation': nc.bad_by_deviation,
        'bad_by_hf_noise': nc.bad_by_hf_noise,
        'bad_by_correlation': nc.bad_by_correlation,
        'bad_by_ransac': nc.bad_by_ransac
    }

    all_bads = list(set(
        nc.bad_by_nan +
        nc.bad_by_flat +
        nc.bad_by_deviation +
        nc.bad_by_hf_noise +
        nc.bad_by_correlation +
        nc.bad_by_ransac
    ))

    raw.info['bads'] = all_bads
    processing_log['processing_steps'].append({
        'step': 'bad_channel_detection',
        'description': 'Identified bad channels using pyprep',
        'bad_channels': bad_channels,
        'all_bad_channels': all_bads,
        'n_bad_channels': len(all_bads)
    })

    # Interpolate bad channels
    raw.interpolate_bads(reset_bads=True)
    processing_log['processing_steps'].append({
        'step': 'interpolate_bads',
        'description': 'Interpolated bad channels',
        'interpolated_channels': all_bads
    })

    # ------------------------------------------------------------------ #
    # ICA
    # ------------------------------------------------------------------ #
    ica = ICA(n_components=n_ica_components, random_state=97, max_iter='auto')
    ica.fit(raw)
    processing_log['processing_steps'].append({
        'step': 'ica_fit',
        'description': 'Fitted ICA components',
        'ica_settings': {
            'n_components': n_ica_components,
            'random_state': 97,
            'max_iter': 'auto'
        }
    })

    # Component selection: auto (EOG) or interactive
    if auto_exclude_eog:
        eog_indices, _ = ica.find_bads_eog(raw)
        ica.exclude = eog_indices
        logger.info("Auto-excluded %d EOG components: %s", len(eog_indices), eog_indices)
        to_exclude = eog_indices
    else:
        logger.info("Plotting ICA components for visual inspection...")
        ica.plot_components(show=True)
        raw_input = input("Enter ICA component numbers to exclude (comma-separated, or blank to skip): ")
        to_exclude = [
            int(i.strip()) for i in raw_input.split(',') if i.strip().isdigit()
        ]
        ica.exclude = to_exclude

    processing_log['processing_steps'].append({
        'step': 'ica_component_selection',
        'description': 'EOG auto-detection' if auto_exclude_eog else 'User-selected ICA components',
        'excluded_components': to_exclude,
        'n_components_excluded': len(to_exclude)
    })

    # ------------------------------------------------------------------ #
    # Apply ICA
    # ------------------------------------------------------------------ #
    raw = ica.apply(raw.copy())
    processing_log['processing_steps'].append({
        'step': 'ica_apply',
        'description': 'Applied ICA cleaning',
        'n_components_removed': len(to_exclude)
    })

    # ------------------------------------------------------------------ #
    # Save output
    # ------------------------------------------------------------------ #
    base = Path(path_to_set_file).stem
    output_path = output_dir / f"{base}_preprocessed.edf"
    log_path    = output_dir / f"{base}_processing_log.json"

    mne.export.export_raw(str(output_path), raw, fmt='edf', overwrite=True)

    with open(log_path, 'w') as f:
        json.dump(processing_log, f, indent=4)

    logger.info("Preprocessed file saved to: %s", output_path)
    logger.info("Processing log saved to:    %s", log_path)
    return str(output_path)