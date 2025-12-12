#@title size_Density.py
"""
size_density.py — Curriculum scheduler for Minesweeper

This module implements adaptive difficulty progression based on:

    - board size
    - mine density
    - optional 3BV-based difficulty banding

The trainer calls:
    cfg = curriculum.next()

Which returns:
    {
        "rows": rows,
        "cols": cols,
        "mines": mines,
        "density": mines / (rows * cols),
        "level": current_stage
    }

The curriculum increases difficulty when:
    • average win rate ≥ win_threshold
or  • episodes_per_stage exceeded (failsafe)
"""

from __future__ import annotations
import numpy as np


class SizeDensityCurriculum:
    """
    Curriculum that schedules Minesweeper difficulty using:
        • Board size schedule
        • Mine density schedule
        • Automatic level-ups

    Args:
        size_list:    list of (rows, cols)
        density_list: list of floats, 0–1 (mines / area)
        win_threshold: minimum win rate to advance
        episodes_per_stage: max episodes before forced advance
    """

    def __init__(
        self,
        size_list=None,
        density_list=None,
        win_threshold=0.55,
        episodes_per_stage=300,
        seed=0
    ):
        self.rng = np.random.default_rng(seed)

        self.size_list = size_list or [
            (5, 5),
            (8, 8),
            (12, 12),
            (16, 16),
            (24, 24),
        ]

        self.density_list = density_list or [
            0.08,  # very sparse
            0.12,
            0.15,
            0.18,
            0.21,  # hard
        ]

        # curriculum coordinates
        self.i_size = 0
        self.i_density = 0

        # progression rules
        self.win_threshold = win_threshold
        self.episodes_per_stage = episodes_per_stage

        # statistics
        self.episode_counter = 0
        self.recent_results = []

    # ------------------------------------------------------------------

    def record_result(self, win: bool):
        """
        Called by trainer after each episode.
        """
        self.recent_results.append(1 if win else 0)
        if len(self.recent_results) > 200:
            self.recent_results.pop(0)

        self.episode_counter += 1

    # ------------------------------------------------------------------

    def should_advance(self) -> bool:
        """
        Determine whether curriculum should move to next difficulty.
        """
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
        """
        Move to the next curriculum cell (size,density).
        """
        self.episode_counter = 0
        self.recent_results.clear()

        # Try increasing density first
        if self.i_density + 1 < len(self.density_list):
            self.i_density += 1
            return

        # Otherwise increase board size and reset density
        if self.i_size + 1 < len(self.size_list):
            self.i_size += 1
            self.i_density = 0
            return

        # Cap at max difficulty (do nothing)
        return

    # ------------------------------------------------------------------

    def current(self):
        """
        Returns the current curriculum configuration dict.
        """
        rows, cols = self.size_list[self.i_size]
        density = self.density_list[self.i_density]

        mines = int(rows * cols * density)
        mines = max(1, mines)

        return {
            "rows": rows,
            "cols": cols,
            "mines": mines,
            "density": density,
            "level": (self.i_size, self.i_density),
        }

    # ------------------------------------------------------------------

    def next(self):
        """
        Called by trainer each new episode.
        Decides whether to advance schedule.
        Returns config for new episode.
        """
        if self.should_advance():
            self.advance()
        return self.current()
