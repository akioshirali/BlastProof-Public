#@title hardness.py
"""
hardness.py — Minesweeper Hardness Metrics

This module computes several standard and advanced difficulty metrics:

    • 3BV (Bechtel's Board Value)
    • Opening count (zero-valued regions)
    • Mine density
    • Clue density
    • Composite hardness score

Used for:
    • Curriculum learning
    • Difficulty labeling
    • Automatic level scheduling

Main API:
    hardness = compute_hardness(board, mines)

Returns a dictionary:
    {
        "3bv": int,
        "openings": int,
        "mine_density": float,
        "clue_density": float,
        "score": float,
        "band": str,
    }
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple


# ============================================================================
# Helper: 8-neighborhood
# ============================================================================

def neighbors(r: int, c: int, rows: int, cols: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                yield nr, nc


# ============================================================================
# Zero-region identification for 3BV
# ============================================================================

def flood_zero(board: np.ndarray, visited: np.ndarray, r: int, c: int):
    """
    Flood fill starting from a zero cell to find an "opening".
    """
    stack = [(r, c)]
    visited[r, c] = True
    rows, cols = board.shape

    while stack:
        cr, cc = stack.pop()
        for nr, nc in neighbors(cr, cc, rows, cols):
            # Zero neighbors included in flood-fill
            if board[nr, nc] == 0 and not visited[nr, nc]:
                visited[nr, nc] = True
                stack.append((nr, nc))


def count_openings(board: np.ndarray) -> int:
    """
    Count the number of zero-valued connected regions (openings).
    """
    rows, cols = board.shape
    visited = np.zeros((rows, cols), dtype=bool)

    openings = 0

    for r in range(rows):
        for c in range(cols):
            if board[r, c] == 0 and not visited[r, c]:
                openings += 1
                flood_zero(board, visited, r, c)

    return openings


# ============================================================================
# 3BV Calculation
# ============================================================================

def compute_3bv(board: np.ndarray) -> int:
    """
    Compute Bechtel's 3BV (Board Benchmark Value):

        3BV = (# zero-openings) + (# non-zero unrevealed clue cells)

    Mines must be encoded as -1 or another negative marker.
    """
    rows, cols = board.shape

    # Identify zero-openings
    openings = count_openings(board)

    # Count all positive clue cells (1–8)
    clue_count = int(np.sum((board > 0)))

    return openings + clue_count


# ============================================================================
# Composite Hardness Score
# ============================================================================

def composite_score(
    bv: int,
    openings: int,
    mine_density: float,
    clue_density: float,
    rows: int,
    cols: int
) -> float:
    """
    Heuristic composite hardness score.

    Higher score = harder board.

    Weighted contributions:
        • 3BV:              structural difficulty
        • mine_density:     randomness difficulty
        • clue_density:     informational density
        • openings:         inversely affects difficulty (more openings = easier)
    """

    area = rows * cols

    score = (
        0.6 * (bv / area)
        + 0.4 * mine_density
        + 0.3 * clue_density
        - 0.2 * (openings / area)
    )

    return float(score)


# ============================================================================
# Difficulty Banding
# ============================================================================

def hardness_band(score: float) -> str:
    """
    Assign board to a difficulty band.
    Thresholds tuned for typical Minesweeper scales.
    """
    if score < 0.05:
        return "easy"
    if score < 0.09:
        return "medium"
    if score < 0.14:
        return "hard"
    return "expert"


# ============================================================================
# Main API
# ============================================================================

def compute_hardness(board: np.ndarray, mines: np.ndarray) -> Dict:
    """
    Compute all hardness metrics for a given Minesweeper board.

    Args:
        board: 2D array with values:
                   -1 for mines
                   0–8 for clues
        mines: boolean or int array of same shape

    Returns dict containing:
        • 3bv
        • openings
        • mine_density
        • clue_density
        • score
        • band
    """
    rows, cols = board.shape
    area = rows * cols

    mine_density = float(mines.sum() / area)
    clue_density = float(np.sum(board > 0) / area)

    openings = count_openings(board)
    bv = compute_3bv(board)

    score = composite_score(
        bv=bv,
        openings=openings,
        mine_density=mine_density,
        clue_density=clue_density,
        rows=rows,
        cols=cols
    )

    band = hardness_band(score)

    return {
        "3bv": int(bv),
        "openings": int(openings),
        "mine_density": mine_density,
        "clue_density": clue_density,
        "score": score,
        "band": band,
    }
