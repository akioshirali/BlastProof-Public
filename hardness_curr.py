 
"""
hardness_curriculum.py — Curriculum driven by Minesweeper hardness

This module uses the hardness metrics (3BV, density, openings, composite score)
from hardness.py to create boards of increasing difficulty bands:

    easy → medium → hard → expert

The curriculum progresses when:
    • agent win rate ≥ win_threshold
or  • episodes_per_stage reached (failsafe)

    curriculum = HardnessCurriculum()
    config = curriculum.next()     # gives rows, cols, mines, target_band
    curriculum.record_result(win)  # trainer reports win/loss
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Dict

from .hardness import compute_hardness, hardness_band


class HardnessCurriculum:
    """
    Difficulty progression based on actual hardness of generated boards.

    Args:
        sizes: list of board sizes to sample from
        density_range: min/max density for sampling
        win_threshold: advancement threshold
        episodes_per_stage: forced advancement
        seed: RNG seed
    """

    def __init__(
        self,
        sizes=None,
        density_range=(0.08, 0.22),
        win_threshold=0.55,
        episodes_per_stage=300,
        seed=0
    ):
        self.rng = np.random.default_rng(seed)

        # Possible board sizes (sampled randomly)
        self.sizes = sizes or [
            (8, 8),
            (12, 12),
            (16, 16),
            (20, 20),
            (24, 24),
        ]

        self.density_min, self.density_max = density_range

        self.bands = ["easy", "medium", "hard", "expert"]
        self.i_band = 0  # start at easiest

        self.win_threshold = win_threshold
        self.episodes_per_stage = episodes_per_stage

        self.recent_results = []
        self.episode_counter = 0

    # ------------------------------------------------------------------
    # Recording results
    # ------------------------------------------------------------------

    def record_result(self, win: bool):
        self.recent_results.append(1 if win else 0)
        if len(self.recent_results) > 200:
            self.recent_results.pop(0)
        self.episode_counter += 1

    # ------------------------------------------------------------------

    def should_advance(self):
        if len(self.recent_results) == 0:
            return False

        win_rate = np.mean(self.recent_results)

        if win_rate >= self.win_threshold:
            return True

        if self.episode_counter >= self.episodes_per_stage:
            return True

        return False

    # ------------------------------------------------------------------

    def advance(self):
        self.episode_counter = 0
        self.recent_results.clear()

        if self.i_band + 1 < len(self.bands):
            self.i_band += 1

    # ------------------------------------------------------------------
    # Generating hardness-targeted boards
    # ------------------------------------------------------------------

    def sample_board_config(self) -> Dict:
        """
        Returns a board config (rows, cols, mines) sampled so that
        expected hardness falls within the current difficulty band.
        """

        target_band = self.bands[self.i_band]

        # Try several samples until we find one of correct hardness.
        # Worst case: return last sample.
        best_cfg = None

        for _ in range(50):
            rows, cols = self.sizes[self.rng.integers(len(self.sizes))]
            density = float(self.rng.uniform(self.density_min, self.density_max))
            mines = max(1, int(rows * cols * density))

            # Create a synthetic empty board for hardness estimation.
            # Full board generation happens in the environment,
            # but hardness estimation requires true clue layout.
            # Here we *approximate* target hardness using mines + density.
            est_board = np.zeros((rows, cols), dtype=int)
            mine_mask = np.zeros((rows, cols), dtype=bool)
            mine_positions = self.rng.choice(rows * cols, mines, replace=False)
            mine_mask.flat[mine_positions] = True

            # Compute clue numbers
            for r in range(rows):
                for c in range(cols):
                    if mine_mask[r, c]:
                        est_board[r, c] = -1
                    else:
                        count = 0
                        for dr in (-1, 0, 1):
                            for dc in (-1, 0, 1):
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < rows and 0 <= nc < cols:
                                    if mine_mask[nr, nc]:
                                        count += 1
                        est_board[r, c] = count

            hardness = compute_hardness(est_board, mine_mask)
            if hardness["band"] == target_band:
                return {
                    "rows": rows,
                    "cols": cols,
                    "mines": mines,
                    "band": target_band,
                    "hardness": hardness,
                }

            best_cfg = {
                "rows": rows,
                "cols": cols,
                "mines": mines,
                "band": hardness["band"],
                "hardness": hardness,
            }

        # If no exact band match after N tries, fallback to closest found
        return best_cfg

    # ------------------------------------------------------------------

    def next(self):
        """Called at the start of each episode."""
        if self.should_advance():
            self.advance()
        return self.sample_board_config()
