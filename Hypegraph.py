 
"""
hypergraph.py

Factor-graph hypergraph builder for Minesweeper.

Nodes:
    • "var"   – variable nodes (one per grid cell)
    • "con"   – constraint nodes (one per clue cell, i.e. board[r,c] >= 0)

Edges:
    • ("var", "to_con", "con")
    • ("con", "to_var", "var")

Meaning:
    A "con" node for cell (r,c) connects to variable nodes for EVERY
    adjacent cell (including diagonals). This captures the minesweeper
    constraints as a hyperedge represented by a factor node.

Caching:
    struct = builder.build_structure(obs)
    data   = builder.update_features(struct, obs)

Unit tests included below.
"""

from __future__ import annotations
import numpy as np
import torch
from torch_geometric.data import HeteroData
from typing import Dict, Tuple




from __future__ import annotations
import numpy as np
import torch
from torch_geometric.data import HeteroData
from typing import Dict, Tuple


# ======================================================================
# Low-level helpers
# ======================================================================

def cell_index(r: int, c: int, cols: int) -> int:
    return r * cols + c


def get_adjacent_8(r: int, c: int, rows: int, cols: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                yield nr, nc


# ======================================================================
# Hypergraph Builder
# ======================================================================

class HypergraphBuilder:

    # -------------------------------------------------------------
    def build_structure(self, obs: Dict) -> Dict:
        """
        Build static hypergraph structure:
            - constraint node map (true clues only)
            - adjacency edges:
                 var → con
                 con → var
        """

        board = obs["board"]
        mines = obs["mines"]
        rows, cols = board.shape

        # --------------------------------------------------
        # Correct constraint-node definition:
        #     mines[r,c] == 0 and board[r,c] > 0
        # --------------------------------------------------
        fact_map = {}
        fid = 0
        for r in range(rows):
            for c in range(cols):
                if mines[r, c] == 0 and board[r, c] > 0:
                    fact_map[(r, c)] = fid
                    fid += 1

        # --------------------------------------------------
        # Hyperedge construction
        # Factor node connects to:
        #   • Its own clue cell
        #   • All 8 neighbors
        # --------------------------------------------------
        var_to_con_src = []
        var_to_con_dst = []

        con_to_var_src = []
        con_to_var_dst = []

        for (r, c), fid in fact_map.items():

            # connect clue cell itself
            vid_self = cell_index(r, c, cols)

            var_to_con_src.append(vid_self)
            var_to_con_dst.append(fid)

            con_to_var_src.append(fid)
            con_to_var_dst.append(vid_self)

            # connect neighbors
            for nr, nc in get_adjacent_8(r, c, rows, cols):
                vid = cell_index(nr, nc, cols)

                var_to_con_src.append(vid)
                var_to_con_dst.append(fid)

                con_to_var_src.append(fid)
                con_to_var_dst.append(vid)

        return {
            "rows": rows,
            "cols": cols,
            "fact_map": fact_map,

            "var_to_con_src": torch.tensor(var_to_con_src, dtype=torch.long),
            "var_to_con_dst": torch.tensor(var_to_con_dst, dtype=torch.long),

            "con_to_var_src": torch.tensor(con_to_var_src, dtype=torch.long),
            "con_to_var_dst": torch.tensor(con_to_var_dst, dtype=torch.long),
        }

    # -------------------------------------------------------------
    def update_features(self, struct: Dict, obs: Dict) -> HeteroData:

        board = obs["board"]
        revealed = obs["revealed"]
        mines = obs["mines"]
        rows, cols = board.shape

        data = HeteroData()

        # ---------------------------------------------------------
        # VAR NODES
        # ---------------------------------------------------------
        var_features = []
        for r in range(rows):
            for c in range(cols):
                rev = 1 if revealed[r, c] else 0
                clue = board[r, c] if rev else -1
                mine_label = int(mines[r, c])
                idx = cell_index(r, c, cols)
                var_features.append([rev, clue, mine_label, idx])

        data["var"].x = torch.tensor(var_features, dtype=torch.float32)

        # ---------------------------------------------------------
        # CONSTRAINT NODES
        # ---------------------------------------------------------
        fact_map = struct["fact_map"]
        con_features = []

        for (r, c), fid in sorted(fact_map.items(), key=lambda x: x[1]):
            con_features.append([board[r, c]])  # clue value

        if len(con_features) == 0:
            con_features = [[0]]

        data["con"].x = torch.tensor(con_features, dtype=torch.float32)

        # ---------------------------------------------------------
        # EDGES
        # ---------------------------------------------------------
        data["var","to_con","con"].edge_index = torch.stack(
            [struct["var_to_con_src"], struct["var_to_con_dst"]]
        )

        data["con","to_var","var"].edge_index = torch.stack(
            [struct["con_to_var_src"], struct["con_to_var_dst"]]
        )

        return data

    # -------------------------------------------------------------
    def build_graph(self, obs: Dict) -> HeteroData:
        struct = self.build_structure(obs)
        return self.update_features(struct, obs)


# ======================================================================
# UNIT TESTS
# ======================================================================

if __name__ == "__main__":
    import unittest

    class FakeObs:
        def __init__(self, board, revealed, mines):
            self.board = board
            self.revealed = revealed
            self.mines = mines

        def to_dict(self):
            return {
                "board": self.board,
                "revealed": self.revealed,
                "mines": self.mines
            }

    class TestHypergraphBuilder(unittest.TestCase):

        def test_basic_hypergraph(self):
            board = np.array([
                [0, 1],
                [1, 0]
            ])
            revealed = np.array([
                [True, False],
                [False, True]
            ])
            mines = np.array([
                [0, 0],
                [0, 0]
            ])

            obs = FakeObs(board, revealed, mines).to_dict()
            builder = HypergraphBuilder()
            g = builder.build_graph(obs)

            # Node types exist
            self.assertIn("var", g.node_types)
            self.assertIn("con", g.node_types)

            # Each var node has 4 features
            self.assertEqual(g["var"].x.shape[1], 4)

            # At least one con node
            self.assertGreater(g["con"].x.shape[0], 0)

            # Edges exist
            self.assertIn(("var", "to_con", "con"), g.edge_types)
            self.assertIn(("con", "to_var", "var"), g.edge_types)

        def test_caching(self):
            board = np.zeros((3, 3))
            revealed = np.zeros((3, 3), dtype=bool)
            mines = np.zeros((3, 3))

            obs = FakeObs(board, revealed, mines).to_dict()

            builder = HypergraphBuilder()
            struct = builder.build_structure(obs)

            # change board & revealed
            obs2 = FakeObs(
                board=np.ones((3,3)),
                revealed=np.ones((3,3), dtype=bool),
                mines=mines
            ).to_dict()

            g2 = builder.update_features(struct, obs2)
            self.assertTrue((g2["var"].x[:,1] == 1).all())  # all clues=1

    unittest.main()
