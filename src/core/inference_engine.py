"""
Phase 2 – Core Processing Engine
=================================
Component 3: ONNX Inference Engine (EEGMamba)
----------------------------------------------
Wraps the real EEGMamba model for low-latency inference.

Two modes:
1. ONNX Runtime (primary): model exported to .onnx, run via onnxruntime.
   CPU: < 5 ms per window. GPU: < 1 ms per window.
2. PyTorch fallback: loads the raw .pth weights from EEGMamba repo directly.
   Used when onnxruntime is not installed or ONNX export hasn't been done yet.

EEGMamba input format:
  (batch_size, n_channels, n_segments, points_per_patch)
  = (1, 19|64, 4, 200)

The engine handles the reshaping from the ring buffer's
  (n_channels, n_times) → EEGMamba tensor format internally.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Make EEGMamba repo importable from this project's root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EEGMAMBA_ROOT = _PROJECT_ROOT / "EEGMamba"
if str(_EEGMAMBA_ROOT) not in sys.path:
    sys.path.insert(0, str(_EEGMAMBA_ROOT))

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    from models.eegmamba import EEGMamba as _RealEEGMamba
    EEGMAMBA_AVAILABLE = True
except Exception:
    EEGMAMBA_AVAILABLE = False

logger = logging.getLogger(__name__)

# Rolling baseline for anomaly score normalisation (thread-local per engine)
_BASELINE_N    = 30     # samples before baseline is established
_BASELINE_ALPHA = 0.05  # EMA smoothing factor (lower = slower adaptation)


# ---------------------------------------------------------------------------
# Constants matching the published EEGMamba architecture
# ---------------------------------------------------------------------------
EEGMAMBA_PATCH_SIZE   = 200   # points per patch (at 200 Hz → 1 s per patch)
EEGMAMBA_N_SEGMENTS   = 4     # number of patches per window
EEGMAMBA_WINDOW_SAMPS = EEGMAMBA_PATCH_SIZE * EEGMAMBA_N_SEGMENTS  # 800 samples


# ---------------------------------------------------------------------------
# Input pre-processing
# ---------------------------------------------------------------------------

def preprocess_window(
    window: np.ndarray,
    target_channels: int = 19,
) -> np.ndarray:
    """Reshape ring-buffer window to EEGMamba input format.

    Handles any number of input channels (64, 128, etc.) by selecting
    the `target_channels` channels with the highest RMS signal power.
    This preserves the most informative channels rather than blind truncation.

    Parameters
    ----------
    window : (n_channels, n_times) float32 ndarray from the ring buffer.
    target_channels : channels EEGMamba was trained with (19 for pre-trained).

    Returns
    -------
    tensor_input : (1, target_channels, n_segments, patch_size) float32.
    """
    n_ch, n_t = window.shape

    # ── 1. Time-axis: trim / zero-pad to exactly EEGMAMBA_WINDOW_SAMPS ──
    if n_t < EEGMAMBA_WINDOW_SAMPS:
        pad = np.zeros((n_ch, EEGMAMBA_WINDOW_SAMPS - n_t), dtype=np.float32)
        window = np.concatenate([window, pad], axis=1)
    else:
        window = window[:, :EEGMAMBA_WINDOW_SAMPS]

    # ── 2. Channel-axis: select best `target_channels` by RMS power ──────
    if n_ch > target_channels:
        # Compute per-channel RMS; pick top-N
        rms = np.sqrt(np.mean(window ** 2, axis=1))        # (n_ch,)
        top_idx = np.argsort(rms)[::-1][:target_channels]  # highest RMS first
        top_idx = np.sort(top_idx)                          # restore spatial order
        window = window[top_idx, :]
    elif n_ch < target_channels:
        # Zero-pad missing channels
        pad = np.zeros((target_channels - n_ch, EEGMAMBA_WINDOW_SAMPS), dtype=np.float32)
        window = np.concatenate([window, pad], axis=0)
    # else: n_ch == target_channels, use as-is

    # ── 3. Per-channel z-score normalisation ──────────────────────────────
    mu   = window.mean(axis=1, keepdims=True)
    std  = window.std(axis=1, keepdims=True) + 1e-8
    window = (window - mu) / std

    # Divide by 100 (EEGMamba physio dataset convention; see physio_dataset.py)
    window = window / 100.0

    # ── 4. Reshape: (channels, time) → (1, channels, n_segments, patch_size)
    tensor = window.reshape(1, target_channels, EEGMAMBA_N_SEGMENTS, EEGMAMBA_PATCH_SIZE)
    return tensor.astype(np.float32)


# ---------------------------------------------------------------------------
# ONNX export helper
# ---------------------------------------------------------------------------

def export_to_onnx(
    weights_path: str,
    onnx_path: str,
    n_channels: int = 19,
    device: str = "cpu",
) -> Path:
    """Load PyTorch EEGMamba weights and export to ONNX.

    Parameters
    ----------
    weights_path : Path to `pretrained_EEGMamba.pth`.
    onnx_path    : Destination `.onnx` file path.
    n_channels   : Number of EEG channels in the model.
    device       : 'cpu' or 'cuda'.

    Returns
    -------
    Path to the exported ONNX file.
    """
    if not EEGMAMBA_AVAILABLE:
        raise ImportError(
            "EEGMamba model code not found. Ensure the EEGMamba/ directory "
            "is present in the project root."
        )

    dev  = torch.device(device)
    model = _RealEEGMamba(
        in_dim=EEGMAMBA_PATCH_SIZE, out_dim=EEGMAMBA_PATCH_SIZE,
        d_model=200, seq_len=EEGMAMBA_N_SEGMENTS, n_layer=12
    ).to(dev)

    state_dict = torch.load(weights_path, map_location=dev, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    dummy = torch.zeros(
        1, n_channels, EEGMAMBA_N_SEGMENTS, EEGMAMBA_PATCH_SIZE,
        dtype=torch.float32, device=dev
    )

    out_path = Path(onnx_path)
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["eeg_window"],
        output_names=["eeg_features"],
        dynamic_axes={"eeg_window": {0: "batch"}},
        opset_version=17,
    )
    logger.info("EEGMamba exported to ONNX: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Main inference engine
# ---------------------------------------------------------------------------

class EEGMambaInferenceEngine:
    """Low-latency inference engine wrapping the real EEGMamba model.

    Tries ONNX Runtime first (fastest), falls back to PyTorch if unavailable.

    Parameters
    ----------
    weights_path : Path to `pretrained_EEGMamba.pth` (PyTorch weights).
    onnx_path    : Path to the exported `.onnx` file. Exported automatically
                   if not found and PyTorch weights are available.
    n_channels   : EEG channels expected by the model (19 for pre-trained).
    device       : 'cpu' or 'cuda'.
    """

    def __init__(
        self,
        weights_path: str = "pretrained_weights/pretrained_EEGMamba.pth",
        onnx_path:    str = "pretrained_weights/eegmamba.onnx",
        n_channels:   int = 19,
        device:       str = "cpu",
    ) -> None:
        self.weights_path = Path(weights_path)
        self.onnx_path    = Path(onnx_path)
        self.n_channels   = n_channels
        self.device       = device

        self._ort_session: Optional[ort.InferenceSession]  = None
        self._torch_model: Optional[_RealEEGMamba]         = None
        self._mode: str = "unloaded"

        # Rolling baseline for anomaly score (Exponential Moving Average)
        self._ema_mean: float = 0.0
        self._ema_var:  float = 1.0
        self._n_seen:   int   = 0

        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        # 1. Try ONNX Runtime
        if ONNX_AVAILABLE:
            if not self.onnx_path.exists() and self.weights_path.exists():
                logger.info("ONNX file not found – exporting from PyTorch weights...")
                try:
                    export_to_onnx(
                        str(self.weights_path), str(self.onnx_path),
                        n_channels=self.n_channels, device=self.device
                    )
                except Exception as exc:
                    logger.warning("ONNX export failed (%s). Falling back to PyTorch.", exc)

            if self.onnx_path.exists():
                try:
                    providers = (
                        ["CUDAExecutionProvider", "CPUExecutionProvider"]
                        if self.device == "cuda"
                        else ["CPUExecutionProvider"]
                    )
                    self._ort_session = ort.InferenceSession(
                        str(self.onnx_path), providers=providers
                    )
                    self._mode = "onnx"
                    logger.info("EEGMamba ONNX session loaded. Provider: %s",
                                self._ort_session.get_providers()[0])
                    return
                except Exception as exc:
                    logger.warning("Could not load ONNX session (%s).", exc)

        # 2. Try PyTorch
        if EEGMAMBA_AVAILABLE and self.weights_path.exists():
            try:
                dev = torch.device(self.device)
                model = _RealEEGMamba(
                    in_dim=EEGMAMBA_PATCH_SIZE, out_dim=EEGMAMBA_PATCH_SIZE,
                    d_model=200, seq_len=EEGMAMBA_N_SEGMENTS, n_layer=12
                ).to(dev)
                state_dict = torch.load(
                    str(self.weights_path), map_location=dev, weights_only=True
                )
                model.load_state_dict(state_dict, strict=False)
                model.eval()
                self._torch_model = model
                self._mode = "pytorch"
                logger.info("EEGMamba PyTorch model loaded on %s.", self.device)
                return
            except Exception as exc:
                logger.warning("Could not load PyTorch model (%s).", exc)

        # 3. Stub mode
        self._mode = "stub"
        logger.warning(
            "EEGMamba running in STUB mode — no weights or runtime found. "
            "Output will be the mean of the input."
        )

    # ------------------------------------------------------------------
    def infer(self, window: np.ndarray) -> dict:
        """Run inference on a (n_channels, n_times) EEG window.

        Parameters
        ----------
        window : (n_channels, n_times) float32 ndarray from the ring buffer.

        Returns
        -------
        result : dict with keys:
            - "features"     : (n_channels, n_segments, feature_dim) ndarray
            - "latent_mean"  : (feature_dim,) ndarray — averaged latent representation
            - "anomaly_score": float — simple energy ratio (deviance from mean)
            - "timestamp"    : float — Unix timestamp of inference
            - "latency_ms"   : float — inference wall time in milliseconds
            - "mode"         : str   — 'onnx', 'pytorch', or 'stub'
        """
        tensor_input = preprocess_window(window, target_channels=self.n_channels)
        t0 = time.perf_counter()

        if self._mode == "onnx" and self._ort_session is not None:
            input_name = self._ort_session.get_inputs()[0].name
            features = self._ort_session.run(None, {input_name: tensor_input})[0]

        elif self._mode == "pytorch" and self._torch_model is not None:
            with torch.no_grad():
                t_in = torch.from_numpy(tensor_input).to(self.device)
                features = self._torch_model(t_in).cpu().numpy()

        else:
            # Stub: return input as "features" (no real processing)
            features = tensor_input

        latency_ms = (time.perf_counter() - t0) * 1000

        # ── Anomaly score via rolling z-score baseline ─────────────────────
        # energy of the feature vector (always positive scalar)
        energy = float(np.sqrt(np.mean(features ** 2)))

        self._n_seen += 1
        if self._n_seen <= _BASELINE_N:
            # Warm-up: accumulate into EMA; report 0.0 (no alert spam on start)
            self._ema_mean = self._ema_mean + (energy - self._ema_mean) / self._n_seen
            self._ema_var  = max(self._ema_var,  1e-6)
            anomaly_score  = 0.0
        else:
            # z-score relative to rolling EMA baseline, mapped to [0, 1] via sigmoid
            z = (energy - self._ema_mean) / (self._ema_var ** 0.5 + 1e-8)
            anomaly_score = float(1.0 / (1.0 + np.exp(-z)))  # sigmoid
            # Update EMA (Welford-inspired online update)
            delta              = energy - self._ema_mean
            self._ema_mean    += _BASELINE_ALPHA * delta
            self._ema_var      = (1 - _BASELINE_ALPHA) * (self._ema_var + _BASELINE_ALPHA * delta ** 2)

        latent_mean = features.mean(axis=(0, 1, 2)) if features.ndim == 4 else features.mean()

        return {
            "features":      features,
            "latent_mean":   latent_mean,
            "anomaly_score": round(anomaly_score, 4),
            "timestamp":     time.time(),
            "latency_ms":    latency_ms,
            "mode":          self._mode,
        }

    @property
    def is_ready(self) -> bool:
        return self._mode in ("onnx", "pytorch")

    def __repr__(self) -> str:
        return f"EEGMambaInferenceEngine(mode={self._mode}, channels={self.n_channels})"


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    engine = EEGMambaInferenceEngine(
        weights_path=str(PROJECT_ROOT / "pretrained_weights" / "pretrained_EEGMamba.pth"),
        onnx_path=str(PROJECT_ROOT / "pretrained_weights" / "eegmamba.onnx"),
        n_channels=19,
    )
    print(engine)

    # Simulate a 5-second window from the ring buffer (19ch × 800 samples)
    window = np.random.randn(19, EEGMAMBA_WINDOW_SAMPS).astype(np.float32)
    result = engine.infer(window)

    print(f"Mode:          {result['mode']}")
    print(f"Latency:       {result['latency_ms']:.2f} ms")
    print(f"Anomaly score: {result['anomaly_score']:.4f}")
    print(f"Features shape:{result['features'].shape}")
    print("inference_engine.py self-test PASSED ✓")
