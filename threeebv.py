

"""
3BV Computation for Minesweeper

3BV ("Bechtel's Board Benchmark Value") is a classical difficulty metric:
    3BV = (# zero-clue connected regions) + (# isolated non-zero clue cells)

Definitions:
    • A zero region is a connected component of cells whose clue is 0.
    • An isolated clue is a cell with clue > 0 that does NOT touch any zero cell.

This module provides a single function:

    compute_3bv(mines: ndarray) -> int

and keeps all logic pure & independent from RL and environment code.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, List
from clue import compute_clue_board, get_adjacent, extract_zero_regions


# ============================================================
# ISOLATED CLUE DETECTION
# ============================================================

def count_isolated_clues(clue_board: np.ndarray) -> int:
    """
    Count clue cells (value > 0) that do NOT have any adjacent zero cells.

    Args:
        clue_board: ndarray with -1 for mines, >=0 for safe cells.

    Returns:
        int: number of isolated clues.
    """
    rows, cols = clue_board.shape
    isolated = 0

    for r in range(rows):
        for c in range(cols):
            if clue_board[r, c] <= 0:
                continue

            touching_zero = False
            for nr, nc in get_adjacent(r, c, rows, cols):
                if clue_board[nr, nc] == 0:
                    touching_zero = True
                    break

            if not touching_zero:
                isolated += 1

    return isolated


# ============================================================
# 3BV COMPUTATION
# ============================================================

def compute_3bv(mines: np.ndarray) -> int:
    """
    Compute the Minesweeper 3BV difficulty metric.

    Args:
        mines: (rows, cols) binary array where 1 = mine, 0 = safe.

    Returns:
        int: 3BV value.

    Procedure:
        1. Compute clue board.
        2. Count zero regions.
        3. Count isolated clues.
        4. Return sum.
    """
    clue_board = compute_clue_board(mines)
    zero_regions = extract_zero_regions(clue_board)
    isolated = count_isolated_clues(clue_board)
    return len(zero_regions) + isolated


# ============================================================
# UNIT TESTS
# ============================================================

if __name__ == "__main__":
    import unittest

    class TestThreeBV(unittest.TestCase):

        def test_no_mines(self):
            """
            All blank board → one giant zero region → 3BV = 1
            """
            mines = np.zeros((3, 3), dtype=int)
            tbv = compute_3bv(mines)
            self.assertEqual(tbv, 1)

        def test_single_mine_center(self):
            """
            Board:
                0 0 0
                0 M 0
                0 0 0

            Clue board:
                1 1 1
                1 -1 1
                1 1 1

            Zero regions = 0
            Isolated clues = 8 (all clue cells have no zeros adjacent)
            3BV = 8
            """
            mines = np.array([
                [0,0,0],
                [0,1,0],
                [0,0,0]
            ])
            tbv = compute_3bv(mines)
            self.assertEqual(tbv, 8)

        def test_two_zero_regions(self):
            """
            Intentionally build a board with two zero components.

            Mines layout:
                0 0 1 0
                0 1 1 0
                1 1 0 0
                0 0 0 0

            Expected:
                zero regions = 2
                isolated clues = cells >0 with no zero-neighbor

            Ensure correct count.
            """
            mines = np.array([
                [0,0,1,0],
                [0,1,1,0],
                [1,1,0,0],
                [0,0,0,0]
            ])

            tbv = compute_3bv(mines)

            # Manually derive:
            #   Zero regions = 2 (matches clue.extract_zero_regions test)
            #   Isolated clues = count manually:
            clue = compute_clue_board(mines)
            rows, cols = clue.shape

            isolated_manual = 0
            for r in range(rows):
                for c in range(cols):
                    if clue[r, c] > 0:
                        touching_zero = False
                        for nr, nc in get_adjacent(r, c, rows, cols):
                            if clue[nr, nc] == 0:
                                touching_zero = True
                                break
                        if not touching_zero:
                            isolated_manual += 1

            expected = 2 + isolated_manual
            self.assertEqual(tbv, expected)

        def test_all_mines(self):
            """
            Degenerate board: everything is a mine → 3BV = 0.
            """
            mines = np.ones((4, 4), dtype=int)
            tbv = compute_3bv(mines)
            self.assertEqual(tbv, 0)

        def test_isolated_clue_simple(self):
            """
            Manually create an isolated clue:
                M 0
                0 0

            Compute:
                Clues = [-1,1]
                        [1,1]

            Zero region = 0
            All clues are touching zeros → isolated = 0
            3BV = 0
            """
            mines = np.array([
                [1,0],
                [0,0]
            ])
            tbv = compute_3bv(mines)
            self.assertEqual(tbv, 0)

    unittest.main()
