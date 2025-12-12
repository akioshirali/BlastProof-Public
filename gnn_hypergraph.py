 
"""


Stabilized Hypergraph GNN for Minesweeper.

Graph specification (from hypergraph.py):
    Node types: "var", "con"
    Edge types:
        ("var", "belongs", "con")      # var -> con
        ("con", "influences", "var")   # con -> var

This module implements:
    • Degree-normalized sum aggregation
    • VarToCon and ConToVar message blocks
    • LayerNorm-stabilized updates
    • Learnable gated residual mixing
    • Multi-round message passing

Interface:
    var_embeddings = model.forward_graph(graph)

Used directly by policy_network.py.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from typing import Tuple


# ============================================================================
# Utility: degree-normalized aggregation
# ============================================================================

def normalized_aggregate(msg: torch.Tensor,
                         dst_index: torch.Tensor,
                         dst_count: torch.Tensor) -> torch.Tensor:
    """
    Perform degree-normalized scatter-add:

        out[v] = sum_{u in N(v)} msg[u] / deg(v)

    Args:
        msg: (E, H_k) sender → receiver messages
        dst_index: (E,) indices of receivers
        dst_count: (#receivers,) number of incoming edges per receiver

    Returns:
        out: (#receivers, H_k)
    """
    out = torch.zeros(dst_count.shape[0], msg.size(1), device=msg.device)
    out = out.index_add(0, dst_index, msg)
    return out / (dst_count.unsqueeze(-1) + 1e-6)


# ============================================================================
# Var → Con Message Block
# ============================================================================

class VarToCon(nn.Module):
    """
    Message block for var → con.

    con_new = LN( ReLU( Linear( [agg(var_msg), clue_value] ) ) )
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.msg = nn.Linear(hidden_dim, hidden_dim)
        self.update = nn.Linear(hidden_dim + 1, hidden_dim)  # + clue scalar
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x_var: torch.Tensor,
        edge_index: Tuple[torch.Tensor, torch.Tensor],
        clue_values: torch.Tensor,
        num_con: int
    ) -> torch.Tensor:

        src, dst = edge_index
        var_messages = self.msg(x_var[src])

        dst_count = torch.bincount(dst, minlength=num_con)
        agg = normalized_aggregate(var_messages, dst, dst_count)

        # Concatenate clue values
        agg = torch.cat([agg, clue_values.unsqueeze(-1)], dim=1)

        out = F.relu(self.update(agg))
        return self.norm(out)


# ============================================================================
# Con → Var Message Block
# ============================================================================

class ConToVar(nn.Module):
    """
    Message block for con → var.

    var_new = LN( ReLU( Linear( agg(con_msg) ) ) )
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.msg = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x_con: torch.Tensor,
        edge_index: Tuple[torch.Tensor, torch.Tensor],
        num_var: int,
    ) -> torch.Tensor:

        src, dst = edge_index
        con_messages = self.msg(x_con[src])

        dst_count = torch.bincount(dst, minlength=num_var)
        agg = normalized_aggregate(con_messages, dst, dst_count)

        return self.norm(F.relu(agg))


# ============================================================================
# Hypergraph GNN Encoder
# ============================================================================

class HypergraphGNN(nn.Module):
    """
    Stabilized degree-normalized hypergraph neural network.

    Args:
        var_dim: input dimension of var nodes
        con_dim: input dimension of con nodes (typically 1 = clue)
        hidden_dim: message-passing hidden size
        rounds: number of var→con→var rounds (default 2)

    Returns:
        var_node_embeddings: (#var_nodes, hidden_dim)
    """

    def __init__(
        self,
        var_dim: int = 4,
        con_dim: int = 1,
        hidden_dim: int = 128,
        rounds: int = 2
    ):
        super().__init__()
        self.rounds = rounds
        self.hidden_dim = hidden_dim

        # Initial projections
        self.var_proj = nn.Linear(var_dim, hidden_dim)
        self.con_proj = nn.Linear(con_dim, hidden_dim)

        # Message blocks
        self.v2c = VarToCon(hidden_dim)
        self.c2v = ConToVar(hidden_dim)

        # Gated residuals
        self.var_gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.con_gate = nn.Linear(2 * hidden_dim, hidden_dim)

        # Node embedding layer norms
        self.var_norm = nn.LayerNorm(hidden_dim)
        self.con_norm = nn.LayerNorm(hidden_dim)

    # ----------------------------------------------------------------------

    def gated_residual(self, old: torch.Tensor, new: torch.Tensor, gate_layer):
        g = torch.sigmoid(gate_layer(torch.cat([old, new], dim=1)))
        return g * new + (1 - g) * old

    # ----------------------------------------------------------------------

    def forward_graph(self, data: HeteroData) -> torch.Tensor:
        """
        Run the HGNN on a graph and return only var-node embeddings.
        """

        # Initial embeddings
        x_var = self.var_norm(F.relu(self.var_proj(data["var"].x)))
        x_con = self.con_norm(F.relu(self.con_proj(data["con"].x)))

        clue_values = data["con"].x.squeeze(-1)
        num_var = x_var.size(0)
        num_con = x_con.size(0)

        e_var_con = data["var", "belongs", "con"].edge_index
        e_con_var = data["con", "influences", "var"].edge_index

        # Multi-round message passing
        for _ in range(self.rounds):
            # --- var → con update ---
            con_new = self.v2c(x_var, e_var_con, clue_values, num_con)
            x_con = self.gated_residual(x_con, con_new, self.con_gate)

            # --- con → var update ---
            var_new = self.c2v(x_con, e_con_var, num_var)
            x_var = self.gated_residual(x_var, var_new, self.var_gate)

        return x_var  # policy network uses only var embeddings


# ============================================================================
# UNIT TESTS
# ============================================================================

if __name__ == "__main__":
    import unittest

    class TestHypergraphGNN(unittest.TestCase):

        def build_test_graph(self):
            """
            Create minimal graph:
                Var nodes: 3
                Con nodes: 2
            Factor graph edges linking them.
            """
            data = HeteroData()

            data["var"].x = torch.randn(3, 4)      # var_dim = 4
            data["con"].x = torch.randn(2, 1)      # con_dim = 1

            # var→con (belongs)
            data["var", "belongs", "con"].edge_index = torch.tensor([
                [0, 1, 2],
                [0, 1, 1],
            ])

            # con→var (influences)
            data["con", "influences", "var"].edge_index = torch.tensor([
                [0, 1, 1],
                [0, 1, 2],
            ])

            return data

        def test_output_shape(self):
            model = HypergraphGNN(var_dim=4, con_dim=1, hidden_dim=32, rounds=2)
            graph = self.build_test_graph()
            out = model.forward_graph(graph)
            self.assertEqual(out.shape, (3, 32))

    unittest.main()
