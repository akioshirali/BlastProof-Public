  
"""
hetero.py — FINAL FIXED VERSION
Includes:
- Correct clue-node definition
- Correct touches edges
- Correct region grouping including bordering clues
- No dummy regions
- Dynamic region extraction
"""

from __future__ import annotations
import numpy as np
import torch
from torch_geometric.data import HeteroData
from typing import Dict, Tuple, List


# ======================================================================
# Helper functions
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


def get_adjacent_4(r: int, c: int, rows: int, cols: int):
    for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc


# ======================================================================
# Zero-region extraction
# ======================================================================

def extract_zero_regions(clue_board: np.ndarray, revealed: np.ndarray):
    rows, cols = clue_board.shape
    visited = np.zeros_like(clue_board, dtype=bool)
    region_ids = np.full((rows, cols), -1, dtype=int)
    region_list = []

    rid = 0
    for r in range(rows):
        for c in range(cols):
            if revealed[r,c] and clue_board[r,c] == 0 and not visited[r,c]:
                stack = [(r,c)]
                visited[r,c] = True
                cells = [(r,c)]

                while stack:
                    cr, cc = stack.pop()
                    for nr, nc in get_adjacent_8(cr, cc, rows, cols):
                        if revealed[nr,nc] and clue_board[nr,nc] == 0 and not visited[nr,nc]:
                            visited[nr,nc] = True
                            stack.append((nr,nc))
                            cells.append((nr,nc))

                for rr, cc in cells:
                    region_ids[rr,cc] = rid

                region_list.append(cells)
                rid += 1

    return region_ids, region_list


# ======================================================================
# Heterogeneous Graph Builder
# ======================================================================

class HeteroGraphBuilder:

    # -------------------------------------------------------------
    def build_structure(self, obs: Dict) -> Dict:
        board = obs["board"]
        mines = obs["mines"]
        rows, cols = board.shape

        # VAR adjacency
        var_src = []
        var_dst = []
        for r in range(rows):
            for c in range(cols):
                v = cell_index(r, c, cols)
                for nr, nc in get_adjacent_4(r, c, rows, cols):
                    var_src.append(v)
                    var_dst.append(cell_index(nr, nc, cols))

        # CONSTRAINT NODES: true clue cells only (board > 0)
        fact_map = {}
        fid = 0
        for r in range(rows):
            for c in range(cols):
                if mines[r,c] == 0 and board[r,c] > 0:   # FIXED
                    fact_map[(r,c)] = fid
                    fid += 1

        return {
            "rows": rows,
            "cols": cols,
            "var_edges": (
                torch.tensor(var_src, dtype=torch.long),
                torch.tensor(var_dst, dtype=torch.long)
            ),
            "fact_map": fact_map,
        }


    # -------------------------------------------------------------
    def update_features(self, struct: Dict, obs: Dict) -> HeteroData:

        board = obs["board"]
        revealed = obs["revealed"]
        mines = obs["mines"]
        rows, cols = board.shape

        # DYNAMIC REGION EXTRACTION
        region_ids, region_list = extract_zero_regions(board, revealed)

        data = HeteroData()

        # VAR NODES
        var_features = []
        for r in range(rows):
            for c in range(cols):
                rev = 1 if revealed[r,c] else 0
                clue = board[r,c] if revealed[r,c] else -1
                mine_label = int(mines[r,c])
                idx = cell_index(r,c,cols)
                var_features.append([rev, clue, mine_label, idx])

        data["var"].x = torch.tensor(var_features, dtype=torch.float32)

        # CONSTRAINT NODES
        fact_map = struct["fact_map"]
        con_features = [[board[r,c]] for (r,c), _ in sorted(fact_map.items(), key=lambda x: x[1])]
        data["con"].x = torch.tensor(con_features, dtype=torch.float32)

        # REGION NODES (size, region_id)
        if len(region_list) > 0:
            data["region"].x = torch.tensor(
                [[len(cells), i] for i, cells in enumerate(region_list)],
                dtype=torch.float32
            )
        else:
            data["region"].x = torch.zeros((0,2), dtype=torch.float32)

        # -------------------------------------------
        # EDGES
        # -------------------------------------------

        # var <-> var adjacency
        var_src, var_dst = struct["var_edges"]
        data["var","adj","var"].edge_index = torch.stack([var_src, var_dst])

        # var -> con (touches)
        touch_src = []
        touch_dst = []

        for (r,c), fid in fact_map.items():
            # Connect clue cell itself
            vid = cell_index(r,c,cols)
            touch_src.append(vid)
            touch_dst.append(fid)

            # And neighbors
            for nr, nc in get_adjacent_8(r,c,rows,cols):
                vid = cell_index(nr,nc,cols)
                touch_src.append(vid)
                touch_dst.append(fid)

        data["var","touches","con"].edge_index = torch.tensor(
            [touch_src, touch_dst], dtype=torch.long
        )

        # region -> var (zero cells + bordering clues)
        rg_src = []
        rg_dst = []

        for rid, cells in enumerate(region_list):

            # zero cells
            for (rr,cc) in cells:
                rg_src.append(rid)
                rg_dst.append(cell_index(rr,cc,cols))

            # bordering clues
            for (rr,cc) in cells:
                for nr,nc in get_adjacent_8(rr,cc,rows,cols):
                    if board[nr,nc] > 0:  # clue
                        rg_src.append(rid)
                        rg_dst.append(cell_index(nr,nc,cols))

        if len(rg_src) > 0:
            data["region","groups","var"].edge_index = torch.tensor(
                [rg_src, rg_dst], dtype=torch.long
            )
        else:
            data["region","groups","var"].edge_index = torch.zeros((2,0), dtype=torch.long)

        return data


    def build_graph(self, obs: Dict) -> HeteroData:
        return self.update_features(self.build_structure(obs), obs)
