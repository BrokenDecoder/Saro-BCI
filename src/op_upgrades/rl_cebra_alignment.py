"""
OP Upgrade 4: Reinforcement Learning (RL) Alignment
====================================================
Uses the CEBRA synchrony score as a reward signal to train an RL agent
that adapts the AI twin's behaviour to maximise alignment with the patient.

Design notes
------------
* ``calculate_reward`` accepts pre-computed embeddings **or** raw signals
  together with a CEBRA model, so it can be used in both offline and
  online settings.
* ``RLAlignmentAgent`` is a thin wrapper that keeps the policy separate
  from environment concerns; swap ``_select_action`` / ``_update_policy``
  for any RL library (stable-baselines3, RLlib, etc.).
* All random-number seeds are explicit for reproducibility.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity in [−1, 1] between two flat vectors."""
    a, b = a.flatten(), b.flatten()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def calculate_reward(
    patient_embedding: np.ndarray,
    ai_embedding: np.ndarray,
    *,
    metric: str = "cosine",
) -> float:
    """Compute a scalar reward in [0, 1] from paired CEBRA embeddings.

    Parameters
    ----------
    patient_embedding : (T, D) or (D,) – patient latent representation.
    ai_embedding      : (T, D) or (D,) – AI twin latent representation.
    metric            : 'cosine' (default) or 'euclidean'.

    Returns
    -------
    reward : float in [0, 1].  Higher means better alignment.
    """
    if metric == "cosine":
        sim = cosine_similarity(patient_embedding.mean(axis=0),
                                ai_embedding.mean(axis=0))
        # Map [−1, 1] → [0, 1]
        reward = (sim + 1.0) / 2.0
    elif metric == "euclidean":
        distance = np.linalg.norm(
            patient_embedding.mean(axis=0) - ai_embedding.mean(axis=0)
        )
        reward = 1.0 / (1.0 + distance)
    else:
        raise ValueError(f"Unknown metric '{metric}'. Choose 'cosine' or 'euclidean'.")

    logger.debug("Alignment reward (%s): %.4f", metric, reward)
    return reward


# ---------------------------------------------------------------------------
# Simple RL agent skeleton
# ---------------------------------------------------------------------------

@dataclass
class RLAlignmentAgent:
    """Minimal RL agent that maximises the CEBRA alignment reward.

    Attributes
    ----------
    action_dim   : Dimensionality of the action space (AI parameter tweaks).
    learning_rate: Step size for policy gradient updates.
    gamma        : Discount factor.
    rng          : Seeded random state for reproducibility.
    """

    action_dim: int = 8
    learning_rate: float = 1e-3
    gamma: float = 0.99
    seed: int = 42
    _policy: np.ndarray = field(init=False)
    _episode_rewards: list[float] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        # Simple linear policy: action = policy @ state
        self._policy = self.rng.standard_normal(self.action_dim).astype(np.float32)
        logger.info("RLAlignmentAgent initialised (action_dim=%d).", self.action_dim)

    # ------------------------------------------------------------------
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Sample an action given the current environment state.

        Replace this stub with a neural network forward pass for
        production use.
        """
        noise = self.rng.standard_normal(self.action_dim).astype(np.float32)
        action = np.tanh(self._policy + noise * 0.1)   # bounded in (−1, 1)
        logger.debug("Selected action: %s", action)
        return action

    # ------------------------------------------------------------------
    def update_policy(self, reward: float, state: np.ndarray) -> None:
        """REINFORCE-style policy gradient step (stub).

        Replace with a proper RL update (PPO, SAC, etc.) for production.
        """
        self._episode_rewards.append(reward)
        # Simple gradient estimate: push policy toward high-reward states
        gradient = reward * state.flatten()[: self.action_dim]
        self._policy += self.learning_rate * gradient
        logger.debug("Policy updated. Cumulative rewards so far: %d steps.",
                     len(self._episode_rewards))

    # ------------------------------------------------------------------
    @property
    def mean_reward(self) -> float:
        """Running mean reward across all steps seen so far."""
        if not self._episode_rewards:
            return 0.0
        return float(np.mean(self._episode_rewards))


# ---------------------------------------------------------------------------
# Training loop helper
# ---------------------------------------------------------------------------

def rl_training_step(
    agent: RLAlignmentAgent,
    environment: Any,
    *,
    cebra_model: Any,
    metric: str = "cosine",
) -> float:
    """Execute one RL step: select action → step env → compute reward → update.

    Parameters
    ----------
    agent       : RLAlignmentAgent instance.
    environment : Object with ``state`` property and ``step(action)`` method
                  returning ``(new_state, patient_eeg)``.
    cebra_model : Fitted CEBRA model with a ``transform(X)`` method.
    metric      : Reward metric passed to ``calculate_reward``.

    Returns
    -------
    reward : float – reward obtained at this step.
    """
    state = environment.state
    action = agent.select_action(state)

    new_state, patient_eeg = environment.step(action)

    patient_emb = cebra_model.transform(patient_eeg)
    ai_emb      = cebra_model.transform(new_state)
    reward = calculate_reward(patient_emb, ai_emb, metric=metric)

    agent.update_policy(reward, state)
    logger.info("RL step complete – reward=%.4f, mean_reward=%.4f",
                reward, agent.mean_reward)
    return reward
