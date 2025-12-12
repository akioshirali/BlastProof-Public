 
"""
Minesweeper Environment (Core Game Engine)

This module provides:
    • Minefield generation by density
    • Clue (adjacency count) computation
    • 3BV difficulty metric
    • MinesweeperGame class:
          - reveal logic with flood fill
          - rollback on mine hit
          - reward shaping hooks
          - observation dictionary used by downstream GNN builders

No RL logic lives here. No graph builders live here.
This is a self-contained game engine.
"""

from __future__ import annotations
import numpy as np
import random
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple



# ============================================================
# BOARD GENERATION
# ============================================================

def generate_mines_board(rows: int, cols: int, density: float) -> np.ndarray:
    """
    Generate a board with mines randomly placed according to a density ∈ [0,1].

    Returns:
        mines: ndarray shape (rows, cols), 1 = mine, 0 = safe.
    """
    total = rows * cols
    num_mines = int(total * density)

    arr = np.zeros(total, dtype=int)
    arr[:num_mines] = 1
    np.random.shuffle(arr)

    return arr.reshape(rows, cols)


# ============================================================
# CLUE CALCULATION
# ============================================================

def compute_clue_board(mines: np.ndarray) -> np.ndarray:
    """
    Computes the clue board:
        clue[r,c] = number of adjacent mines
        clue[r,c] = -1 for mine cells
    """
    rows, cols = mines.shape
    board = np.zeros_like(mines, dtype=int)

    for r in range(rows):
        for c in range(cols):
            if mines[r, c] == 1:
                board[r, c] = -1
                continue
            count = 0
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        count += mines[nr, nc]
            board[r, c] = count

    return board


# ============================================================
# 3BV — Bechtel’s Board Benchmark Value
# ============================================================

def compute_3bv(mines: np.ndarray) -> int:
    """
    Computes 3BV as (# zero regions) + (# isolated non-zero clues).

    Zero regions:
        Connected components of cells where clue = 0.

    Isolated clues:
        Clue > 0 with no adjacent zero cells.
    """
    rows, cols = mines.shape
    board = compute_clue_board(mines)
    visited = np.zeros_like(board, dtype=bool)

    # ---------------------
    # Count zero regions
    # ---------------------
    zero_regions = 0
    for r in range(rows):
        for c in range(cols):
            if board[r, c] == 0 and not visited[r, c]:
                zero_regions += 1
                stack = [(r, c)]
                visited[r, c] = True

                while stack:
                    cr, cc = stack.pop()
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < rows and 0 <= nc < cols:
                                if board[nr, nc] == 0 and not visited[nr, nc]:
                                    visited[nr, nc] = True
                                    stack.append((nr, nc))

    # ---------------------
    # Count isolated clues
    # ---------------------
    isolated = 0
    for r in range(rows):
        for c in range(cols):
            if board[r, c] > 0:
                touching_zero = False
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if board[nr, nc] == 0:
                                touching_zero = True
                                break
                if not touching_zero:
                    isolated += 1

    return int(zero_regions + isolated)


# ============================================================
# MINESWEEPER ENVIRONMENT (WITH ROLLBACK)
# ============================================================

@dataclass
class MinesweeperGame:
    """
    A minimal but complete Minesweeper environment suitable for RL or analysis.

    Mechanics:
        • Random safe-cell starting position.
        • Reveal logic supports flood fill for zero clues.
        • Rollback: if a mine is clicked, revert to the last safe snapshot.
        • Step returns:
              obs, reward, done, info
        • Observation is a dict of full board state.
    """

    rows: int
    cols: int
    density: float
    max_steps: int = 500

    # Dynamic fields (created in reset)
    mines: np.ndarray = field(init=False)
    board: np.ndarray = field(init=False)
    revealed: np.ndarray = field(init=False)
    flags: np.ndarray = field(init=False)
    step_count: int = 0
    done: bool = False
    last_safe_state: Dict[str, Any] = None
    threebv: int = 0

    # ----------------------------------------------------------
    # RESET
    # ----------------------------------------------------------
    def reset(self) -> Dict[str, Any]:
        """Generate a new board and initialize state."""

        self.mines = generate_mines_board(self.rows, self.cols, self.density)
        self.threebv = compute_3bv(self.mines)

        self.revealed = np.zeros_like(self.mines, dtype=bool)
        self.flags = np.zeros_like(self.mines, dtype=bool)
        self.step_count = 0
        self.done = False

        # Guaranteed safe starting position
        safe_positions = np.argwhere(self.mines == 0)
        r0, c0 = safe_positions[random.randrange(len(safe_positions))]
        self._reveal(r0, c0)

        # Store rollback snapshot
        self.last_safe_state = self.snapshot()

        return self.get_obs()

    # ----------------------------------------------------------
    # SNAPSHOT / RESTORE
    # ----------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        """Deep copy the visible state (mines are static)."""
        return {
            "revealed": self.revealed.copy(),
            "flags": self.flags.copy(),
            "step_count": self.step_count,
            "done": self.done,
        }
    def load_snapshot(self, snap: Dict[str, Any]):
        """Restore only visible state; do NOT restore step_count or done."""
        self.revealed = snap["revealed"].copy()
        self.flags = snap["flags"].copy()

    # ----------------------------------------------------------
    # REVEAL LOGIC
    # ----------------------------------------------------------
   def _reveal(self, r: int, c: int):
    """Reveal a cell and flood-fill zero regions including border clues."""

    # If already revealed, do nothing
    if self.revealed[r, c]:
        return

    # Reveal the clicked cell
    self.revealed[r, c] = True

    # If clue > 0 → stop
    if self.board[r, c] != 0:
        return

    # Zero region → flood fill and reveal bordering clues
    stack = [(r, c)]
    while stack:
        cr, cc = stack.pop()

        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:

                    if not self.revealed[nr, nc] and self.mines[nr, nc] == 0:
                        self.revealed[nr, nc] = True

                        # If it's also zero, expand the region
                        if self.board[nr, nc] == 0:
                            stack.append((nr, nc))



    # ----------------------------------------------------------
    # LEGAL ACTIONS
    # ----------------------------------------------------------
    def list_legal_actions(self) -> np.ndarray:
    actions = []
    for r in range(self.rows):
        for c in range(self.cols):
            if not self.revealed[r, c] and not self.flags[r, c]:
                actions.append(r * self.cols + c)
    return np.array(actions, dtype=int)

    # ----------------------------------------------------------
    # STEP
    # ----------------------------------------------------------
    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Apply an action:
            • If mine → rollback, reward = -1
            • If safe → reveal, reward = 0.1 + shape term
            • Track 3BV change for optional shaping
            • Terminal when all non-mines revealed
        """
        if self.done:
            return self.get_obs(), 0.0, True, {}

        r = action // self.cols
        c = action % self.cols
        self.step_count += 1

        prev_3bv = self.threebv

        # Hit mine → rollback
        if self.mines[r, c] == 1:
            self.load_snapshot(self.last_safe_state)
            return self.get_obs(), -1.0, False, {"rollback": True}

        # Safe reveal
        self._reveal(r, c)
        self.last_safe_state = self.snapshot()

        # Base reward for safe reveal
        reward = 0.1


        # Win condition
        if np.all((self.revealed == True) | (self.mines == 1)):
            self.done = True
            reward += 5.0
            return self.get_obs(), reward, True, {"win": True}

        # Step limit
        if self.step_count >= self.max_steps:
            self.done = True
            reward -= 0.5
            return self.get_obs(), reward, True, {"timeout": True}

        return self.get_obs(), reward, False, {}

    # ----------------------------------------------------------
    # OBSERVATION
    # ----------------------------------------------------------
    def get_obs(self) -> Dict[str, Any]:
        """
        Observation is a dictionary consumed by graph builders.

        Returns:
            {
                "board": ndarray,
                "revealed": bool ndarray,
                "flags": bool ndarray,
                "mines": int ndarray,
                "threebv": int
            }
        """
        return {
            "board": self.board,
            "revealed": self.revealed,
            "flags": self.flags,
            "mines": self.mines,
            "threebv": self.threebv,
        }
