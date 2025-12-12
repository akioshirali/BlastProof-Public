 """
buffer.py — PPO rollout buffer

This module defines:
    • PPOBuffer — for single-environment PPO rollouts
    • PPOBatch  — optional container for vectorized PPO (v2)
"""

from __future__ import annotations
from typing import List, Any


class PPOBuffer:
    """
    Minimal buffer for standard PPO (single-environment rollout).

    Stores:
        states:      a list of PyG Data or HeteroData graphs
        actions:     list[int]
        rewards:     list[float]
        values:      list[float]
        logps:       list[float]
        dones:       list[bool]

    This buffer is emptied after each PPO update.
    """

    def __init__(self):
        self.states: List[Any] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.logps: List[float] = []
        self.dones: List[bool] = []

    # -------------------------------------------------------------

    def store(self, state, action, reward, value, logp, done):
        """Append one timestep of rollout data."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.logps.append(logp)
        self.dones.append(done)

    # -------------------------------------------------------------

    def clear(self):
        """Reset the buffer."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.logps.clear()
        self.dones.clear()


# ============================================================================
# OPTIONAL CONTAINER — used by PPO v2 (batched / vectorized rollout)
# ============================================================================

class PPOBatch:
    """
    Structured container returned by vectorized PPO rollout.

    Attributes:
        graphs[t][n]  → graph for env n at timestep t
        actions[t][n]
        rewards[t][n]
        values[t][n]
        logps[t][n]
        dones[t][n]
        last_values[n] → bootstrap values for final state
    """

    def __init__(
        self,
        graphs,
        actions,
        rewards,
        values,
        logps,
        dones,
        last_values
    ):
        self.graphs = graphs
        self.actions = actions
        self.rewards = rewards
        self.values = values
        self.logps = logps
        self.dones = dones
        self.last_values = last_values
