"""
Clue Computation and Adjacency Utilities for Minesweeper

This module provides:
    • compute_clue_board():    compute numeric adjacency clues (-1 for mines)
    • get_adjacent():          list of valid neighbor coordinates
    • flood_zero_region():     zero-region expansion
    • extract_zero_regions():  return all connected zero-clue components
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Iterable


# ============================================================
# ADJACENCY UTILITIES
# ============================================================

def get_adjacent(r: int, c: int, rows: int, cols: int) -> Iterable[Tuple[int, int]]:
    """Yield all 8-directionally adjacent neighbors of (r,c)."""
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                yield nr, nc


# ============================================================
# CLUE BOARD
# ============================================================

def compute_clue_board(mines: np.ndarray) -> np.ndarray:
    """
    Compute the board of adjacency clues given a binary mine matrix.

    Args:
        mines: ndarray of shape (rows, cols), 1 = mine, 0 = safe.

    Returns:
        clue_board: ndarray with:
            -1  for mines
             k  for number of adjacent mines for safe cells
    """
    rows, cols = mines.shape
    clue = np.zeros_like(mines, dtype=int)

    for r in range(rows):
        for c in range(cols):
            if mines[r, c] == 1:
                clue[r, c] = -1
                continue

            count = 0
            for nr, nc in get_adjacent(r, c, rows, cols):
                count += mines[nr, nc]
            clue[r, c] = count

    return clue


# ============================================================
# ZERO-REGION DETECTION
# ============================================================

def flood_zero_region(
    board: np.ndarray,
    start_r: int,
    start_c: int,
    visited: np.ndarray
) -> List[Tuple[int, int]]:
    """
    Expand a connected zero-clue region beginning at (start_r, start_c).

    Args:
        board: clue board (with 0, 1, 2, ..., -1 for mines)
        start_r, start_c: starting coordinates with clue 0
        visited: boolean mask updated in-place

    Returns:
        region: list of (r,c) cells belonging to the zero-region
    """
    rows, cols = board.shape
    stack = [(start_r, start_c)]
    visited[start_r, start_c] = True
    region = [(start_r, start_c)]

    while stack:
        r, c = stack.pop()
        for nr, nc in get_adjacent(r, c, rows, cols):
            if board[nr, nc] == 0 and not visited[nr, nc]:
                visited[nr, nc] = True
                region.append((nr, nc))
                stack.append((nr, nc))

    return region


def extract_zero_regions(board: np.ndarray) -> List[List[Tuple[int, int]]]:
    """
    Identify all connected zero-clue regions on a clue board.

    Args:
        board: clue board (computed by compute_clue_board)

    Returns:
        A list of regions, each a list of (r,c) coordinates.
    """
    rows, cols = board.shape
    visited = np.zeros((rows, cols), dtype=bool)
    regions: List[List[Tuple[int, int]]] = []

    for r in range(rows):
        for c in range(cols):
            if board[r, c] == 0 and not visited[r, c]:
                region = flood_zero_region(board, r, c, visited)
                regions.append(region)

    return regions


# ============================================================
# UNIT TESTS
# ============================================================

if __name__ == "__main__":
    import unittest

    class TestClueUtilities(unittest.TestCase):

        def test_get_adjacent_center(self):
            adj = list(get_adjacent(1, 1, 3, 3))
            expected = {
                (0,0),(0,1),(0,2),
                (1,0),       (1,2),
                (2,0),(2,1),(2,2)
            }
            self.assertEqual(set(adj), expected)

        def test_get_adjacent_corner(self):
            adj = list(get_adjacent(0, 0, 3, 3))
            expected = {(0,1), (1,0), (1,1)}
            self.assertEqual(set(adj), expected)

        def test_compute_clue_board_basic(self):
            # 3x3 board, center mine
            mines = np.array([
                [0,0,0],
                [0,1,0],
                [0,0,0]
            ])
            clue = compute_clue_board(mines)
            expected = np.array([
                [1,1,1],
                [1,-1,1],
                [1,1,1]
            ])
            np.testing.assert_array_equal(clue, expected)

        def test_compute_clue_board_edges(self):
            mines = np.array([
                [1,0,0],
                [0,0,0],
                [0,0,1]
            ])
            clue = compute_clue_board(mines)

            # manually compute expected clue values
            expected = np.array([
                [-1,1,0],
                [1,2,1],
                [0,1,-1]
            ])

            np.testing.assert_array_equal(clue, expected)

        def test_flood_zero_region(self):
            # Board with two connected zeros in top-left
            board = np.array([
                [0,0,1],
                [0,1,1],
                [1,1,1]
            ])
            visited = np.zeros_like(board, dtype=bool)
            region = flood_zero_region(board, 0, 0, visited)

            region_set = set(region)
            expected = {(0,0),(0,1),(1,0)}  # zero-cells reachable
            self.assertEqual(region_set, expected)

        def test_extract_zero_regions_multiple(self):
            board = np.array([
                [0,0,1,0],
                [0,1,1,0],
                [1,1,0,0],
                [0,0,0,0]
            ])
            regions = extract_zero_regions(board)

            # Sort regions for stable comparison
            regions_sorted = [set(region) for region in regions]

            # Expected regions:
            # Region 1: connected zero cluster in top-left
            r1 = {(0,0),(0,1),(1,0)}

            # Region 2: isolated zero at (2,2),(2,3),(3,2),(3,3),(3,1),(3,0) cluster
            # Actually these form a single large cluster:
            r2 = {(0,3),(1,3),(2,2),(2,3),(3,0),(3,1),(3,2),(3,3)}

            expected_sets = [r1, r2]
            self.assertEqual(len(regions_sorted), 2)
            self.assertEqual({frozenset(s) for s in regions_sorted},
                             {frozenset(s) for s in expected_sets})

    unittest.main()
