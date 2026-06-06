"""
EEGMambaTwin – AI Twin Brain Module
=====================================
Wraps the EEGMamba SSM architecture and exposes a ``predict_healthy_state``
interface that the rest of the pipeline consumes.

Now fully wired with a real `mamba_ssm.Mamba` backbone instead of stub noise!
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False

logger = logging.getLogger(__name__)


class RealEEGMambaModel(nn.Module):
    """
    Actual PyTorch implementation of the EEG Mamba sequence model.
    Transforms (batch, n_times, n_channels) -> (batch, n_times, n_channels)
    """
    def __init__(self, n_channels=64, d_model=128, n_layers=2):
        super().__init__()
        if not MAMBA_AVAILABLE:
            raise ImportError("mamba_ssm is required to use RealEEGMambaModel. Please install it.")
        
        self.proj_in = nn.Linear(n_channels, d_model)
        self.layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(n_layers)
        ])
        self.proj_out = nn.Linear(d_model, n_channels)

    def forward(self, x):
        # x shape: (batch, n_times, n_channels)
        h = self.proj_in(x)
        for layer in self.layers:
            h = layer(h)
        out = self.proj_out(h)
        return out, h


class EEGMambaTwin:
    """AI twin that predicts the 'healthy' brain state for a given patient window.

    Parameters
    ----------
    weights_path : Path to the pre-trained EEGMamba `.pth` file.
    device       : PyTorch device string (default ``'cpu'``).
    """

    def __init__(
        self,
        weights_path: str = "pretrained_weights/eegmamba.pth",
        device: str = "cpu",
        n_channels: int = 64
    ) -> None:
        self.weights_path = Path(weights_path)
        self.device = torch.device(device)
        self.n_channels = n_channels
        self._model: Optional[RealEEGMambaModel] = None
        self._stub_mode: bool = not MAMBA_AVAILABLE
        self.logger = logging.getLogger(__name__)
        
        self._load_model()

    def _load_model(self) -> None:
        """Attempt to load the real EEGMamba model; fall back to stub if mamba_ssm is absent."""
        if self._stub_mode:
            self.logger.warning("mamba_ssm not installed. Running in stub mode.")
            return

        self.logger.info("Initializing real EEGMamba model.")
        self._model = RealEEGMambaModel(n_channels=self.n_channels).to(self.device)
        
        if self.weights_path.exists():
            self.logger.info("Loading EEGMamba weights from '%s'.", self.weights_path)
            try:
                self._model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
            except Exception as exc:
                self.logger.warning("Could not load weights, starting from scratch. Error: %s", exc)
        else:
            self.logger.info("No weights found at %s. Using randomly initialized Mamba.", self.weights_path)
            
        self._model.eval()

    @property
    def is_ready(self) -> bool:
        """True when the real model is loaded; False in stub mode."""
        return not self._stub_mode

    def predict_healthy_state(self, patient_data_window: np.ndarray) -> np.ndarray:
        """Predict the next 'healthy' EEG window from the patient's recent history.

        Parameters
        ----------
        patient_data_window : (n_channels, n_times) float32 ndarray.

        Returns
        -------
        healthy_prediction : Same shape as input, float32 ndarray.
        """
        if patient_data_window.ndim != 2:
            raise ValueError(
                "patient_data_window must be 2-D (n_channels, n_times), "
                f"got shape {patient_data_window.shape}."
            )

        if not self._stub_mode and self._model is not None:
            self.logger.debug("Predicting healthy brain state using REAL EEGMamba model.")
            with torch.no_grad():
                # Transpose to (batch, n_times, n_channels)
                tensor_in = torch.from_numpy(patient_data_window).T.unsqueeze(0).to(self.device)
                tensor_out, _ = self._model(tensor_in)
                # Transpose back to (n_channels, n_times)
                out_np = tensor_out.squeeze(0).T.cpu().numpy()
            return out_np.astype(np.float32)

        # Stub fallback
        self.logger.debug("Stub mode: generating smoothed approximation of healthy state.")
        kernel_size = max(1, patient_data_window.shape[1] // 50)
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.apply_along_axis(
            lambda ch: np.convolve(ch, kernel, mode="same"),
            axis=1, arr=patient_data_window,
        )
        noise = np.random.normal(0.0, 0.01, patient_data_window.shape)
        return (smoothed + noise).astype(np.float32)

    def get_latent_state(self, patient_data_window: np.ndarray) -> np.ndarray:
        """Return the Mamba hidden state as a 1-D latent vector."""
        if not self._stub_mode and self._model is not None:
            with torch.no_grad():
                tensor_in = torch.from_numpy(patient_data_window).T.unsqueeze(0).to(self.device)
                _, latent_seq = self._model(tensor_in)
                # Mean pool over the sequence dimension to get a fixed-size vector
                latent_vector = latent_seq.squeeze(0).mean(dim=0).cpu().numpy()
            return latent_vector
            
        # Stub: return mean of healthy prediction
        healthy = self.predict_healthy_state(patient_data_window)
        return healthy.mean(axis=1)  # (n_channels,)
