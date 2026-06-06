"""
OP Upgrade 2: Multi-Modal AI Stacking
======================================
Takes human EEG and stacks it with multiple AI modalities
(EEGMamba state, LLM text embeddings, audio features) into a single
matrix suitable for CEBRA's contrastive learning.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Expected channel counts (document them here for validation)
_MODALITY_DIMS: dict[str, int] = {
    "eeg":           64,
    "mamba_state":   64,
    "llm_embedding": 1024,
    "audio_features": 128,
}


def _check_time_dim(arrays: dict[str, np.ndarray]) -> int:
    """Assert all arrays share the same time dimension and return T."""
    time_dims = {name: arr.shape[1] for name, arr in arrays.items()}
    unique = set(time_dims.values())
    if len(unique) > 1:
        raise ValueError(
            f"All modalities must have the same number of time steps, "
            f"got: {time_dims}"
        )
    return unique.pop()


def _validate_channels(name: str, arr: np.ndarray) -> None:
    """Warn if a modality has an unexpected channel count."""
    expected = _MODALITY_DIMS.get(name)
    if expected is not None and arr.shape[0] != expected:
        logger.warning(
            "Modality '%s': expected %d channels, got %d. "
            "Proceeding with actual shape.",
            name, expected, arr.shape[0],
        )


def stack_multimodal_data(
    eeg_data: np.ndarray,
    mamba_state: np.ndarray,
    llm_embedding: np.ndarray,
    audio_features: np.ndarray,
    *,
    normalize: bool = False,
) -> np.ndarray:
    """Stack all modalities into a single (n_features, T) matrix for CEBRA.

    Parameters
    ----------
    eeg_data      : (64, T)   – human EEG channels.
    mamba_state   : (64, T)   – EEGMamba AI twin latent state.
    llm_embedding : (1024, T) – LLM token or text-embedding features.
    audio_features: (128, T)  – audio/acoustic features.
    normalize     : If True, each row is z-scored before stacking.

    Returns
    -------
    stacked_matrix : (n_total_channels, T)  e.g. (1280, T).

    Raises
    ------
    ValueError  : If any two arrays have different time dimensions.
    """
    named: dict[str, np.ndarray] = {
        "eeg":            eeg_data,
        "mamba_state":    mamba_state,
        "llm_embedding":  llm_embedding,
        "audio_features": audio_features,
    }

    for name, arr in named.items():
        if arr.ndim != 2:
            raise ValueError(
                f"Modality '{name}' must be 2-D (channels, time), "
                f"got shape {arr.shape}."
            )
        _validate_channels(name, arr)

    T = _check_time_dim(named)

    if normalize:
        normed = {}
        for name, arr in named.items():
            std = arr.std(axis=1, keepdims=True)
            std = np.where(std == 0, 1.0, std)
            normed[name] = (arr - arr.mean(axis=1, keepdims=True)) / std
        arrays_to_stack = list(normed.values())
    else:
        arrays_to_stack = [eeg_data, mamba_state, llm_embedding, audio_features]

    stacked_matrix = np.vstack(arrays_to_stack)
    logger.info(
        "Multi-modal matrix: shape=%s, T=%d, normalize=%s",
        stacked_matrix.shape, T, normalize,
    )
    return stacked_matrix


def trim_to_shortest(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Trim all (C, T) arrays to the shortest common T.

    Useful as a preprocessing step before ``stack_multimodal_data`` when
    different modalities have slightly different recording lengths.
    """
    T_min = min(a.shape[1] for a in arrays)
    return tuple(a[:, :T_min] for a in arrays)
