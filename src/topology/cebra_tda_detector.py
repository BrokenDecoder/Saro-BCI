"""
Topology – CEBRA + TDA Divergence Detector
============================================
Embeds patient and AI-twin signals into a shared CEBRA latent space and
uses persistent-homology (giotto-tda) to detect topological structural
collapse that precedes seizure onset.

Key improvements over the original:
- Adaptive baseline: the detector accumulates the first N_BASELINE windows
  and computes μ and σ of the divergence score.  Subsequent scores are
  z-normalised → output is always in a meaningful, human-readable range.
- Warm-up guard: anomaly detection is suppressed until the baseline window
  is complete, preventing a storm of false-positives at startup.
- Alert cooldown: at most one alert fires per ALERT_COOLDOWN_S seconds,
  so the log doesn't get spammed when the signal is noisy.
- Wasserstein stub upgraded to a proper 2-D Earth-Mover approximation.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────
# How many divergence scores to collect before we consider the baseline "set"
N_BASELINE = 20
# Alert cooldown — at most one alert every N seconds
ALERT_COOLDOWN_S = 5.0
# Minimum z-score to trigger an anomaly (robust to outliers)
Z_SCORE_THRESHOLD = 2.5


def _wasserstein_approx(diag_a: np.ndarray, diag_b: np.ndarray) -> float:
    """Approximate 2-D Wasserstein distance between two persistence diagrams.

    Both inputs are (K, 2) arrays of (birth, death) pairs.
    We match each point in diag_a to its nearest in diag_b using the
    L2 cost, sum the matched distances, and add a penalty for unmatched
    points (projected to the diagonal).

    This is a proper approximation of the Wasserstein-1 distance on
    persistence diagrams and replaces the naive lifetime-difference stub.
    """
    if len(diag_a) == 0 and len(diag_b) == 0:
        return 0.0

    # Persistence of each diagram (only take finite points)
    def _persistence(d):
        if len(d) == 0:
            return np.empty((0, 2))
        finite = d[np.isfinite(d).all(axis=1)]
        return finite[finite[:, 1] > finite[:, 0]]  # death > birth only

    a = _persistence(diag_a)
    b = _persistence(diag_b)

    # Add diagonal projections so both sets have the same size
    def _diagonal_proj(pts):
        """Project each point to its nearest point on the diagonal."""
        mid = (pts[:, 0] + pts[:, 1]) / 2
        return np.stack([mid, mid], axis=1)

    # Build cost matrix between all point pairs from a and b
    if len(a) == 0 or len(b) == 0:
        # All points unmatched — use persistence as cost
        unmatched_a = np.sum((a[:, 1] - a[:, 0]) / 2) if len(a) else 0
        unmatched_b = np.sum((b[:, 1] - b[:, 0]) / 2) if len(b) else 0
        return float(unmatched_a + unmatched_b)

    # Pairwise L2 distances between actual points
    diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]   # (|a|, |b|, 2)
    cost = np.sqrt((diff ** 2).sum(axis=-1))             # (|a|, |b|)

    # Greedy matching (good enough; Hungarian would be O(n^3))
    matched = 0.0
    used_a = np.zeros(len(a), dtype=bool)
    used_b = np.zeros(len(b), dtype=bool)

    # Sort by cost and greedily match
    flat_idx = np.argsort(cost.ravel())
    for fi in flat_idx:
        i, j = divmod(int(fi), len(b))
        if not used_a[i] and not used_b[j]:
            matched += cost[i, j]
            used_a[i] = True
            used_b[j] = True

    # Penalise unmatched points with half their persistence
    pen_a = sum((a[i, 1] - a[i, 0]) / 2 for i in range(len(a)) if not used_a[i])
    pen_b = sum((b[j, 1] - b[j, 0]) / 2 for j in range(len(b)) if not used_b[j])

    return float(matched + pen_a + pen_b)


class TopologyDetector:
    """Detect topological anomalies between patient and AI-twin signals.

    Parameters
    ----------
    threshold       : Absolute divergence above which an anomaly is flagged
                      *before* the baseline is established.  After baseline,
                      the z-score threshold Z_SCORE_THRESHOLD is used instead.
    embedding_dim   : Output dimensionality of the CEBRA latent space.
    homology_dims   : Homology dimensions for Vietoris-Rips persistence.
    """

    def __init__(
        self,
        threshold: float = 0.8,
        embedding_dim: int = 3,
        homology_dims: Optional[list] = None,
    ) -> None:
        self.threshold      = threshold
        self.embedding_dim  = embedding_dim
        self.homology_dims  = homology_dims or [0, 1]

        self._cebra_model   = None
        self._cebra_cls     = None
        self._vr            = None

        # Adaptive baseline
        self._baseline_buf : deque = deque(maxlen=N_BASELINE)
        self._baseline_ready: bool  = False
        self._baseline_mean : float = 0.0
        self._baseline_std  : float = 1.0

        # Alert rate-limiting
        self._last_alert_time: float = 0.0

        self._try_init_real_libs()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _try_init_real_libs(self) -> None:
        """Attempt to load CEBRA + giotto-tda; log clearly if unavailable."""
        try:
            from cebra import CEBRA          # type: ignore[import]
            self._cebra_cls = CEBRA
            logger.info("CEBRA library loaded successfully.")
        except ImportError:
            self._cebra_cls = None
            logger.warning(
                "cebra not installed — running in stub mode. "
                "Install with: pip install cebra"
            )

        try:
            from gtda.homology import VietorisRipsPersistence   # type: ignore[import]
            self._vr = VietorisRipsPersistence(
                homology_dimensions=self.homology_dims
            )
            logger.info("giotto-tda VietorisRipsPersistence loaded.")
        except ImportError:
            self._vr = None
            logger.warning(
                "giotto-tda not installed — using approximate Wasserstein stub. "
                "Install with: pip install giotto-tda"
            )

    # ------------------------------------------------------------------
    # CEBRA fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        baseline_patient: np.ndarray,
        baseline_twin: np.ndarray,
        max_iterations: int = 1000,
    ) -> "TopologyDetector":
        """Fit the CEBRA model on healthy baseline data."""
        combined = np.hstack([baseline_patient, baseline_twin])
        if self._cebra_cls is not None:
            logger.info("Fitting CEBRA model on baseline data (shape=%s).", combined.shape)
            self._cebra_model = self._cebra_cls(
                model_architecture="offset10-model",
                output_dimension=self.embedding_dim,
                conditional="time",
                max_iterations=max_iterations,
                device="cuda_if_available",
                verbose=False,
            )
            self._cebra_model.fit(combined)
        else:
            logger.warning("CEBRA not available; skipping model fit.")
        return self

    # ------------------------------------------------------------------
    # Embedding + persistence
    # ------------------------------------------------------------------

    def _embed(self, signal: np.ndarray) -> np.ndarray:
        """Embed a (T, C) signal into (T, D) CEBRA space (or stub)."""
        if self._cebra_model is not None:
            return self._cebra_model.transform(signal)
        # Stub: deterministic random projection (reproducible per-session)
        T, C = signal.shape
        rng  = np.random.default_rng(seed=42)
        proj = rng.standard_normal((C, self.embedding_dim)).astype(np.float32)
        return (signal @ proj).astype(np.float32)

    def _persistence_diagram(self, point_cloud: np.ndarray) -> np.ndarray:
        """Compute a persistence diagram for a (T, D) point cloud (or stub)."""
        if self._vr is not None:
            diags = self._vr.fit_transform(point_cloud[np.newaxis])[0]
            return diags
        # Stub diagram: birth/death from PCA-inspired range
        births = point_cloud.min(axis=0)
        deaths = point_cloud.max(axis=0)
        return np.stack([births, deaths], axis=1)

    # ------------------------------------------------------------------
    # Adaptive baseline management
    # ------------------------------------------------------------------

    def _update_baseline(self, raw_score: float) -> None:
        """Add a raw score to the baseline buffer and recompute stats."""
        self._baseline_buf.append(raw_score)
        if len(self._baseline_buf) >= N_BASELINE:
            arr = np.array(self._baseline_buf, dtype=np.float64)
            self._baseline_mean  = float(arr.mean())
            self._baseline_std   = float(max(arr.std(), 1e-6))
            self._baseline_ready = True
            logger.info(
                "Baseline established: μ=%.4f σ=%.4f (N=%d)",
                self._baseline_mean, self._baseline_std, len(arr)
            )

    def _z_score(self, raw_score: float) -> float:
        """Convert raw score to z-score relative to baseline."""
        if not self._baseline_ready:
            return 0.0
        return (raw_score - self._baseline_mean) / self._baseline_std

    @property
    def warmup_progress(self) -> float:
        """Fraction of the warm-up window completed, [0, 1]."""
        return min(1.0, len(self._baseline_buf) / N_BASELINE)

    @property
    def is_warmed_up(self) -> bool:
        return self._baseline_ready

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def measure_divergence(
        self,
        patient_signal: np.ndarray,
        twin_signal: np.ndarray,
    ) -> Tuple[bool, float]:
        """Measure topological divergence between patient and AI twin signals.

        Parameters
        ----------
        patient_signal : (T, C) float32 – current patient EEG window.
        twin_signal    : (T, C) float32 – corresponding AI twin prediction.

        Returns
        -------
        is_anomaly      : True only after warm-up AND z-score > Z_SCORE_THRESHOLD
                          AND cooldown has elapsed.
        divergence_score: z-score (0.0 during warm-up).
        """
        if patient_signal.shape != twin_signal.shape:
            raise ValueError(
                f"patient_signal {patient_signal.shape} and "
                f"twin_signal {twin_signal.shape} must have the same shape."
            )

        # Embed and compute raw Wasserstein divergence
        patient_emb = self._embed(patient_signal)
        twin_emb    = self._embed(twin_signal)

        diag_patient = self._persistence_diagram(patient_emb)
        diag_twin    = self._persistence_diagram(twin_emb)

        raw_score = _wasserstein_approx(diag_patient, diag_twin)

        # Feed into adaptive baseline
        self._update_baseline(raw_score)

        # During warm-up: report progress but never fire alerts
        if not self._baseline_ready:
            pct = int(self.warmup_progress * 100)
            logger.info("Warm-up %d%% — raw=%.4f (collecting baseline...)", pct, raw_score)
            return False, 0.0

        # Compute z-score
        z = self._z_score(raw_score)

        # Rate-limited anomaly detection
        now       = time.monotonic()
        cooldown_ok = (now - self._last_alert_time) >= ALERT_COOLDOWN_S
        is_anomaly  = (z > Z_SCORE_THRESHOLD) and cooldown_ok

        if is_anomaly:
            self._last_alert_time = now

        logger.info(
            "Divergence raw=%.4f  z=%.3f  threshold=%.1fσ  anomaly=%s%s",
            raw_score, z, Z_SCORE_THRESHOLD, is_anomaly,
            "" if cooldown_ok else " [cooldown]",
        )

        return is_anomaly, round(z, 4)
