"""
OP Upgrade 3: Generative Pre-training with EEGMamba
====================================================
Uses the EEGMamba SSM architecture to generate synthetic human brainwaves
from the AI twin's internal state.  These synthetic EEG epochs are used
to pre-train the CEBRA joint-embedding model before real patient data is
available.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_N_CHANNELS = 64
_DEFAULT_SFREQ      = 250


# ---------------------------------------------------------------------------
# Synthetic EEG generation
# ---------------------------------------------------------------------------

def generate_synthetic_eeg(
    mamba_model: Optional[object],
    ai_state: torch.Tensor,
    *,
    duration_seconds: float = 60.0,
    sfreq: int = _DEFAULT_SFREQ,
    n_channels: int = _DEFAULT_N_CHANNELS,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Generate synthetic human EEG from the AI twin's internal state.

    Parameters
    ----------
    mamba_model     : Fitted EEGMamba model with a ``generate(state, length)``
                      method, or *None* to fall back to structured noise.
    ai_state        : Tensor containing the AI twin's current latent state.
    duration_seconds: Duration of the synthetic clip in seconds (default 60).
    sfreq           : Sampling frequency in Hz (default 250).
    n_channels      : Number of EEG channels to generate (default 64).
    seed            : Optional RNG seed for the noise fallback path.

    Returns
    -------
    synthetic_eeg : (n_channels, n_timesteps) float32 tensor.
    """
    n_timesteps = int(duration_seconds * sfreq)

    if mamba_model is not None:
        logger.info(
            "Generating %d samples via EEGMamba (%.1f s @ %d Hz).",
            n_timesteps, duration_seconds, sfreq,
        )
        try:
            if hasattr(mamba_model, "generate"):
                synthetic_eeg = mamba_model.generate(ai_state, length=n_timesteps)
            elif isinstance(mamba_model, torch.nn.Module):
                device = next(mamba_model.parameters()).device
                noise = torch.randn(1, n_timesteps, n_channels, device=device)
                with torch.no_grad():
                    out, _ = mamba_model(noise)
                synthetic_eeg = out.squeeze(0).T.cpu()
            else:
                raise ValueError("Unsupported mamba_model type")

            if not isinstance(synthetic_eeg, torch.Tensor):
                synthetic_eeg = torch.as_tensor(synthetic_eeg, dtype=torch.float32)
            return synthetic_eeg
        except Exception as exc:
            logger.warning(
                "EEGMamba generation failed (%s); falling back to noise.", exc
            )

    # -----------------------------------------------------------------------
    # Structured-noise fallback: band-limited pink-ish noise
    # -----------------------------------------------------------------------
    logger.info(
        "Generating %d samples of structured noise fallback (%d ch).",
        n_timesteps, n_channels,
    )
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)

    # White noise coloured by a 1/f envelope in frequency space
    white = torch.randn(n_channels, n_timesteps, generator=rng)
    freqs = torch.fft.rfftfreq(n_timesteps)
    # Avoid division by zero at DC
    freqs[0] = 1.0
    pink_filter = (1.0 / freqs.sqrt()).unsqueeze(0)     # (1, F)
    spectrum = torch.fft.rfft(white) * pink_filter
    synthetic_eeg = torch.fft.irfft(spectrum, n=n_timesteps).float()

    return synthetic_eeg


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_pretraining_dataset(
    mamba_model: Optional[object],
    ai_state: torch.Tensor,
    n_clips: int = 10,
    *,
    duration_seconds: float = 60.0,
    sfreq: int = _DEFAULT_SFREQ,
    n_channels: int = _DEFAULT_N_CHANNELS,
    seed: int = 0,
) -> torch.Tensor:
    """Generate multiple synthetic EEG clips and concatenate along time.

    Returns
    -------
    dataset : (n_channels, n_clips * n_timesteps) float32 tensor,
              ready to be fed to CEBRA.
    """
    clips = [
        generate_synthetic_eeg(
            mamba_model, ai_state,
            duration_seconds=duration_seconds,
            sfreq=sfreq,
            n_channels=n_channels,
            seed=seed + i,
        )
        for i in range(n_clips)
    ]
    dataset = torch.cat(clips, dim=1)
    logger.info(
        "Pre-training dataset built: %d clips → shape %s.",
        n_clips, tuple(dataset.shape),
    )
    return dataset
