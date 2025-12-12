
"""
Board Generation Utilities for Minesweeper

This module handles the creation of minefields according to
different strategies:

    • Uniform random density-based placement
    • Sampling boards by difficulty weight (3BV)
    • Constrained sampling for dataset creation

The functions here do *not* compute clue boards or modify
the MinesweeperGame environment; they only generate mine layouts.
"""

from __future__ import annotations
import numpy as np
import random
from typing import Tuple, Callable, Any


# ============================================================
# BASIC DENSITY-BASED BOARD GENERATION
# ============================================================

def generate_mines_board(rows: int, cols: int, density: float) -> np.ndarray:
    """
    Generate a minefield with mines placed randomly according to a density.

    Args:
        rows: Number of board rows.
        cols: Number of board columns.
        density: A float in [0,1] giving the fraction of cells that contain mines.

    Returns:
        mines: ndarray (rows, cols) where 1 = mine, 0 = safe.
    """
    total = rows * cols
    num_mines = int(total * density)

    arr = np.zeros(total, dtype=int)
    arr[:num_mines] = 1
    np.random.shuffle(arr)

    return arr.reshape(rows, cols)


# ============================================================
# DIFFICULTY-BASED SAMPLING (OPTIONAL)
# ============================================================

def sample_board_by_difficulty(
    env_class: Callable,
    rows: int,
    cols: int,
    density: float,
    difficulty_weight: float = 2.0,
    num_candidates: int = 6
) -> Tuple[Any, dict]:
    """
    Produce an environment by sampling multiple candidate boards and selecting
    one with probability proportional to (3BV + 1) ** difficulty_weight.

    This allows training to emphasize boards that are more difficult.

    Args:
        env_class: A callable (typically MinesweeperGame) that takes (rows, cols, density).
        rows: Number of rows.
        cols: Number of columns.
        density: Mine density.
        difficulty_weight: Exponent controlling difficulty emphasis.
        num_candidates: Number of boards sampled per draw.

    Returns:
        env: A MinesweeperGame instance.
        obs: Initial observation dict (from env.reset()).
    """
    candidates = []

    for _ in range(num_candidates):
        env = env_class(rows, cols, density)
        obs = env.reset()
        threebv = obs["threebv"]
        difficulty_score = (threebv + 1) ** difficulty_weight
        candidates.append((difficulty_score, env, obs))

    # Weighted random choice
    weights = np.array([c[0] for c in candidates], dtype=float)
    weights /= weights.sum()

    idx = np.random.choice(len(candidates), p=weights)
    _, env, obs = candidates[idx]

    return env, obs


# ============================================================
# 3BV-CONSTRAINED GENERATION
# ============================================================

def generate_board_with_3bv(
    env_class: Callable,
    rows: int,
    cols: int,
    density: float,
    min_3bv: int,
    max_3bv: int,
    max_attempts: int = 200
) -> Tuple[Any, dict, int]:
    """
    Generate a board whose 3BV is within [min_3bv, max_3bv].
    Repeats generation until a matching board is found or max_attempts is reached.

    Args:
        env_class: Constructor for MinesweeperGame.
        rows: Number of board rows.
        cols: Number of board columns.
        density: Mine density.
        min_3bv: Minimum acceptable 3BV.
        max_3bv: Maximum acceptable 3BV.
        max_attempts: Safety cap.

    Returns:
        env: The MinesweeperGame instance.
        obs: Its initial observation.
        threebv: The final 3BV obtained.
    """
    best_env = None
    best_obs = None
    best_tbv = None

    for _ in range(max_attempts):
        env = env_class(rows, cols, density)
        obs = env.reset()
        tbv = obs["threebv"]

        if min_3bv <= tbv <= max_3bv:
            return env, obs, tbv

        best_env, best_obs, best_tbv = env, obs, tbv

    # Fallback to closest match if none found in range
    return best_env, best_obs, best_tbv
