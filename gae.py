 
"""
gae.py — Generalized Advantage Estimation (GAE)

Provides:
    • compute_gae()      — single-environment GAE
    • compute_gae_batch() — multi-environment (vectorized) GAE

Both follow the standard GAE formula:

    δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)

    A_t = δ_t + γλ * (1 - done_t) * A_{t+1}

Returns:
    advantages, returns
"""

from __future__ import annotations
import numpy as np
import torch


# ============================================================================
# SINGLE-ENVIRONMENT GAE
# ============================================================================

def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    lam: float = 0.95
):
    """
    Compute GAE for a single rollout.

    Args:
        rewards: shape (T,)
        values:  shape (T+1,) — bootstrap value at end
        dones:   shape (T,), 1 if terminal else 0

    Returns:
        advantages: shape (T,)
        returns:    shape (T,)
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0

    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae

    returns = advantages + values[:-1]
    return advantages, returns


# ============================================================================
# MULTI-ENV (VECTORIZED) GAE
# ============================================================================

def compute_gae_batch(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    lam: float = 0.95
):
    """
    Vectorized GAE across multiple parallel environments (PPO v2).

    Args:
        rewards: shape (T, N)
        values:  shape (T+1, N)
        dones:   shape (T, N)
        gamma, lam: scalars

    Returns:
        advantages: shape (T, N)
        returns:    shape (T, N)
    """

    T, N = rewards.shape
    advantages = np.zeros((T, N), dtype=np.float32)
    gae = np.zeros(N, dtype=np.float32)

    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae

    returns = advantages + values[:-1]
    return advantages, returns

