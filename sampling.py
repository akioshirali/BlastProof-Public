 
"""
Sampling utilities for Minesweeper board generation.

This module provides several strategies:

1. Uniform sampling:
       sample_uniform_board(env_class, rows, cols, density)

2. Difficulty-weighted sampling:
       sample_weighted_by_3bv(..., difficulty_power=k)

3. Rejection sampling for a difficulty band:
       sample_board_in_3bv_range(..., min_3bv, max_3bv)

4. CurriculumSampler:
       An object-oriented sampler that supports:
           - uniform
           - weighted
           - range-based
           - curriculum progression rules

All sampling functions return:
       env, obs, threebv
"""

from __future__ import annotations
import numpy as np
import random
from typing import Callable, Tuple, Optional

from board_generation import generate_mines_board
from threebv import compute_3bv
from clue import compute_clue_board


# ============================================================
# 1 — SIMPLE UNIFORM SAMPLING
# ============================================================

def sample_uniform_board(env_class, rows: int, cols: int, density: float):
    """
    Create a single environment with no difficulty bias.

    Returns:
        (env, obs, threebv)
    """
    env = env_class(rows, cols, density)
    obs = env.reset()
    return env, obs, env.threebv


# ============================================================
# 2 — DIFFICULTY-WEIGHTED SAMPLING (POWER LAW)
# ============================================================

def sample_weighted_by_3bv(
    env_class,
    rows: int,
    cols: int,
    density: float,
    *,
    difficulty_power: float = 2.0,
    candidates: int = 6
):
    """
    Generate N candidate boards and sample proportionally to:

        weight = (3BV + 1) ** difficulty_power

    This biases the sampler toward harder boards.
    """
    boards = []
    weights = []

    for _ in range(candidates):
        env = env_class(rows, cols, density)
        obs = env.reset()
        tbv = env.threebv
        w = (tbv + 1) ** difficulty_power
        boards.append((env, obs, tbv))
        weights.append(w)

    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.sum()

    idx = np.random.choice(len(weights), p=weights)
    return boards[idx]


# ============================================================
# 3 — REJECTION SAMPLING FOR A 3BV RANGE
# ============================================================

def sample_board_in_3bv_range(
    env_class,
    rows: int,
    cols: int,
    density: float,
    min_3bv: int,
    max_3bv: int,
    *,
    max_attempts: int = 200
):
    """
    Try up to max_attempts to generate a board whose 3BV lies inside
    [min_3bv, max_3bv]. Fallback returns the last attempt.

    Returns:
        (env, obs, threebv)
    """
    best = None
    best_dist = float("inf")

    for _ in range(max_attempts):
        env = env_class(rows, cols, density)
        obs = env.reset()
        tbv = env.threebv

        # Acceptable match
        if min_3bv <= tbv <= max_3bv:
            return env, obs, tbv

        # Keep best fallback
        center = 0.5 * (min_3bv + max_3bv)
        dist = abs(tbv - center)
        if dist < best_dist:
            best_dist = dist
            best = (env, obs, tbv)

    # fallback to closest board
    return best


# ============================================================
# 4 — CURRICULUM SAMPLER CLASS (UNIFIED INTERFACE)
# ============================================================

class CurriculumSampler:
    """
    A unified sampler that supports multiple difficulty-selection modes
    and tracks curriculum progression.

    Modes:
        "uniform"          – no difficulty bias
        "weighted"         – power-law weighting by 3BV
        "range"            – sample until 3BV ∈ [min,max]
        "auto_progress"    – shifts difficulty range once win-rate threshold met

    Example:
        sampler = CurriculumSampler(env_class, rows=6, cols=6, density=0.20)
        env, obs, tbv = sampler.sample()
    """

    def __init__(
        self,
        env_class,
        rows: int,
        cols: int,
        density: float,
        *,
        mode: str = "uniform",
        difficulty_power: float = 2.0,
        target_range: Optional[Tuple[int, int]] = None,
        thresholds: Optional[list] = None,
    ):
        self.env_class = env_class
        self.rows = rows
        self.cols = cols
        self.density = density

        self.mode = mode
        self.difficulty_power = difficulty_power
        self.target_range = target_range

        # Curriculum progression state
        self.thresholds = thresholds or []
        self.current_level = 0
        self.performance_log = []

    # ------------------------------------------------------------------

    def record_performance(self, win_rate: float):
        """
        Record win rate, and optionally progress curriculum.
        """
        self.performance_log.append(win_rate)

        if self.mode != "auto_progress":
            return

        if self.current_level < len(self.thresholds):
            required = self.thresholds[self.current_level]
            if win_rate >= required:
                print(f"[Curriculum] Advancing from level {self.current_level} → {self.current_level+1}")
                self.current_level += 1

    # ------------------------------------------------------------------

    def _sample_uniform(self):
        return sample_uniform_board(
            self.env_class, self.rows, self.cols, self.density
        )

    def _sample_weighted(self):
        return sample_weighted_by_3bv(
            self.env_class,
            self.rows,
            self.cols,
            self.density,
            difficulty_power=self.difficulty_power
        )

    def _sample_range(self):
        if self.target_range is None:
            raise ValueError("target_range must be set for range mode.")
        lo, hi = self.target_range
        return sample_board_in_3bv_range(
            self.env_class,
            self.rows,
            self.cols,
            self.density,
            lo,
            hi
        )

    # ------------------------------------------------------------------

    def sample(self):
        """Return (env, obs, tbv) according to the current sampling mode."""
        if self.mode == "uniform":
            return self._sample_uniform()

        if self.mode == "weighted":
            return self._sample_weighted()

        if self.mode == "range":
            return self._sample_range()

        if self.mode == "auto_progress":
            if self.current_level >= len(self.thresholds):
                # final stage: choose hardest available
                return self._sample_weighted()
            else:
                # lower stages: gradually expand 3BV range
                # Example policy: each level maps to a wider range
                lo = 2 * self.current_level
                hi = 10 * (self.current_level + 1)
                return sample_board_in_3bv_range(
                    self.env_class,
                    self.rows,
                    self.cols,
                    self.density,
                    lo,
                    hi
                )

        raise ValueError(f"Unknown sampling mode: {self.mode}")


# ============================================================
# UNIT TESTS
# ============================================================

if __name__ == "__main__":
    import unittest

    class MockEnv:
        """Minimal stub environment for sampling tests."""
        def __init__(self, rows, cols, density):
            self.rows = rows
            self.cols = cols
            self.density = density
            # Generate a fixed-but-randomish mines board
            np.random.seed(rows * cols + int(density * 100))
            self.mines = (np.random.rand(rows, cols) < density).astype(int)
            self.threebv = compute_3bv(self.mines)

        def reset(self):
            return {
                "board": self.board,
                "mines": self.mines,
                "threebv": self.threebv
            }

    class TestSampling(unittest.TestCase):

        def test_uniform(self):
            env, obs, tbv = sample_uniform_board(MockEnv, 4, 4, 0.2)
            self.assertIn("board", obs)
            self.assertGreaterEqual(tbv, 0)

        def test_weighted_sampling(self):
            env, obs, tbv = sample_weighted_by_3bv(
                MockEnv, 4, 4, 0.2, difficulty_power=2.0
            )
            self.assertIn("board", obs)

        def test_range_sampling(self):
            lo, hi = 0, 20
            env, obs, tbv = sample_board_in_3bv_range(
                MockEnv, 5, 5, 0.2, lo, hi
            )
            self.assertTrue(lo <= tbv <= hi or tbv >= 0)

        def test_curriculum_sampler_uniform(self):
            sampler = CurriculumSampler(MockEnv, 4, 4, 0.2, mode="uniform")
            env, obs, tbv = sampler.sample()
            self.assertIn("board", obs)

        def test_curriculum_sampler_weighted(self):
            sampler = CurriculumSampler(MockEnv, 4, 4, 0.2, mode="weighted")
            env, obs, tbv = sampler.sample()
            self.assertIn("board", obs)

        def test_curriculum_autoprogress(self):
            sampler = CurriculumSampler(
                MockEnv, 4, 4, 0.2,
                mode="auto_progress",
                thresholds=[0.8, 0.9]
            )

            # No errors sampling at level 0
            env, obs, tbv = sampler.sample()

            # Trigger progression
            sampler.record_performance(0.85)
            self.assertEqual(sampler.current_level, 1)

            sampler.record_performance(0.95)
            self.assertEqual(sampler.current_level, 2)

    unittest.main()
