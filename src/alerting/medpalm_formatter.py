"""
Alerting – MedPaLM Alert Formatter
====================================
Formats topological anomaly data (from the CEBRA/TDA pipeline) into a
structured clinical alert payload and a ready-to-send LLM prompt.

Improvements over the original
-------------------------------
* Severity bucketing (low / medium / high / critical) so the caller does
  not need to interpret the raw float score.
* ``generate_alert`` validates inputs and raises on out-of-range scores.
* ``AlertPayload`` dataclass makes the structured data reusable and
  serialisable without re-parsing the prompt string.
* Severity-specific prompt templates for better LLM calibration.
* Type hints and NumPy-style docstrings throughout.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Severity classification
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"

    @classmethod
    def from_score(cls, score: float) -> "Severity":
        """Map a [0, ∞) divergence score to a severity level.

        Thresholds are illustrative; calibrate against your clinical dataset.
        """
        if score < 0.5:
            return cls.LOW
        elif score < 1.0:
            return cls.MEDIUM
        elif score < 2.0:
            return cls.HIGH
        else:
            return cls.CRITICAL


# ---------------------------------------------------------------------------
# Alert data container
# ---------------------------------------------------------------------------

@dataclass
class AlertPayload:
    event: str
    severity_score: float
    severity_level: str          # Severity enum value
    seizure_onset_zone: str
    clinical_context: str
    recommendation_requested: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Formatter class
# ---------------------------------------------------------------------------

class MedPalmAlertFormatter:
    """Format manifold-divergence alerts into LLM-ready clinical prompts.

    Parameters
    ----------
    model_name : Identifier of the target LLM (for prompt tailoring).
    max_score  : Maximum expected divergence score; used for sanity checks.
    """

    _CONTEXT = (
        "Divergence between real-time EEG and the healthy AI Twin "
        "state-space prediction exceeds threshold.  "
        "Topological structural collapse identified in the CEBRA latent manifold."
    )

    def __init__(
        self,
        model_name: str = "Med-PaLM-2",
        max_score: float = 10.0,
    ) -> None:
        self.model_name = model_name
        self.max_score = max_score
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    def _build_payload(
        self,
        divergence_score: float,
        soz_location: str,
    ) -> AlertPayload:
        if not (0.0 <= divergence_score <= self.max_score):
            raise ValueError(
                f"divergence_score must be in [0, {self.max_score}], "
                f"got {divergence_score}."
            )
        severity = Severity.from_score(divergence_score)
        return AlertPayload(
            event="High-Precision Manifold Divergence Detected",
            severity_score=round(divergence_score, 4),
            severity_level=severity.value,
            seizure_onset_zone=soz_location,
            clinical_context=self._CONTEXT,
        )

    # ------------------------------------------------------------------
    def generate_alert(
        self,
        divergence_score: float,
        soz_location: str = "Left Temporal Lobe",
    ) -> str:
        """Build a clinical LLM prompt from a divergence event.

        Parameters
        ----------
        divergence_score : Scalar output of ``TopologyDetector.measure_divergence``.
        soz_location     : Anatomical label for the seizure onset zone.

        Returns
        -------
        prompt : str – ready to POST to the Med-PaLM endpoint.
        """
        payload = self._build_payload(divergence_score, soz_location)
        severity = Severity(payload.severity_level)

        # Severity-specific instruction prefix
        urgency_map = {
            Severity.LOW:      "Please review the following low-priority neural divergence event.",
            Severity.MEDIUM:   "A moderate neural divergence event requires clinical attention.",
            Severity.HIGH:     "URGENT: High neural divergence event detected. Immediate review required.",
            Severity.CRITICAL: "CRITICAL ALERT: Seizure onset likely imminent. Emergency protocol advised.",
        }
        urgency_line = urgency_map[severity]

        prompt = (
            f"{urgency_line}\n\n"
            f"Event         : {payload.event}\n"
            f"Severity Score: {payload.severity_score:.4f} "
            f"({payload.severity_level.upper()})\n"
            f"Focal Region  : {payload.seizure_onset_zone}\n"
            f"Clinical Note : {payload.clinical_context}\n\n"
            "Based on the above, please generate a concise clinical "
            "recommendation including: (1) immediate next steps, "
            "(2) medication adjustments if applicable, "
            "(3) referral urgency."
        )

        self.logger.info(
            "Alert generated: score=%.4f severity=%s zone=%s model=%s",
            divergence_score, payload.severity_level, soz_location, self.model_name,
        )
        return prompt

    # ------------------------------------------------------------------
    def generate_alert_json(
        self,
        divergence_score: float,
        soz_location: str = "Left Temporal Lobe",
    ) -> str:
        """Return the structured alert as a JSON string (no LLM prompt)."""
        payload = self._build_payload(divergence_score, soz_location)
        return payload.to_json()
