 
"""
gnn_hetero.py

Heterogeneous GNN encoder for Minesweeper structured graphs.

Graph structure (from hetero.py):
    Node types:  "var", "con", "region"
    Edge types:
        ("var", "adj", "var")
        ("var", "touches", "con")
        ("region", "groups", "var")

Forward API:
    var_embeddings = model.forward_graph(hetero_graph)

Used by PolicyNetwork (policy_network.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    HeteroConv,
    GraphConv,
    GraphNorm,
)

from typing import Dict, Literal


# =============================================================================
# Hetero Layer Block
# =============================================================================

class HeteroLayer(nn.Module):
    """
    One heterogeneous convolutional layer:
        heterogeneous conv  → norm  → activation

    Args:
        in_dims: dict[node_type] = input_dim
        out_dim: hidden dimension for all node types
    """

    def __init__(self, in_dims: Dict[str, int], out_dim: int):
        super().__init__()

        # --- Message passing definition ---
        self.conv = HeteroConv(
            {
                ("var", "adj", "var"): GraphConv(in_dims["var"], out_dim),
                ("var", "touches", "con"): GraphConv(in_dims["var"], out_dim),
                ("region", "groups", "var"): GraphConv(in_dims["region"], out_dim),
            },
            aggr="sum",
        )

        # --- GraphNorm per node type ---
        self.norms = nn.ModuleDict({
            "var": GraphNorm(out_dim),
            "con": GraphNorm(out_dim),
            "region": GraphNorm(out_dim),
        })

    # ------------------------------------------------------------------

    def forward(self, x_dict, edge_index_dict):
        """
        Args:
            x_dict: dict[node_type -> tensor(N_t, D)]
            edge_index_dict: edge structure

        Returns:
            updated node embeddings in a dict
        """
        out_dict = self.conv(x_dict, edge_index_dict)

        for nt, x in out_dict.items():
            out_dict[nt] = F.relu(self.norms[nt](x))

        return out_dict


# =============================================================================
# HETEROGN N-GNN ENCODER
# =============================================================================

class HeteroGNN(nn.Module):
    """
    Heterogeneous GNN used with heterograph topology.

    Args:
        in_dims: dict with keys {"var", "con", "region"} mapping to input dims
        hidden_dim: hidden feature dimension for all node types
        num_layers: number of hetero layers

    Interface:
        var_embeddings = forward_graph(graph)
            → tensor(#var_nodes, hidden_dim)
    """

    def __init__(
        self,
        in_dims: Dict[str, int] = None,
        hidden_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()

        if in_dims is None:
            in_dims = {"var": 4, "con": 1, "region": 1}

        self.input_dims = in_dims
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Initial projection per node type
        self.proj = nn.ModuleDict({
            "var": nn.Linear(in_dims["var"], hidden_dim),
            "con": nn.Linear(in_dims["con"], hidden_dim),
            "region": nn.Linear(in_dims["region"], hidden_dim),
        })

        # Build hetero layers
        dims = {"var": hidden_dim, "con": hidden_dim, "region": hidden_dim}
        self.layers = nn.ModuleList([
            HeteroLayer(dims, hidden_dim) for _ in range(num_layers)
        ])

    # ------------------------------------------------------------------

    def forward_graph(self, data):
        """
        Args:
            data: HeteroData with node types: var, con, region

        Returns:
            var_embeddings: torch.Tensor (#var_nodes, hidden_dim)
        """

        # Initial node embedding dict
        x_dict = {
            "var": F.relu(self.proj["var"](data["var"].x)),
            "con": F.relu(self.proj["con"](data["con"].x)),
            "region": F.relu(self.proj["region"](data["region"].x)),
        }

        # Message passing
        for layer in self.layers:
            x_dict = layer(x_dict, data.edge_index_dict)

        return x_dict["var"]  # policy network consumes var nodes only


# =============================================================================
# UNIT TESTS
# =============================================================================

if __name__ == "__main__":
    import unittest
    from torch_geometric.data import HeteroData

    class TestHeteroGNN(unittest.TestCase):

        def make_graph(self):
            """Creates a tiny 2-var, 1-con, 1-region heterograph."""

            data = HeteroData()

            # Two var nodes
            data["var"].x = torch.randn(2, 4)

            # One con node (clue = 1)
            data["con"].x = torch.randn(1, 1)

            # One region node
            data["region"].x = torch.randn(1, 1)

            # Edges
            data["var", "adj", "var"].edge_index = torch.tensor([[0, 1],
                                                                 [1, 0]])

            data["var", "touches", "con"].edge_index = torch.tensor([[0, 1],
                                                                     [0, 0]])

            data["region", "groups", "var"].edge_index = torch.tensor([[0, 0],
                                                                       [0, 1]])

            return data

        # -------------------------------------------------------------

        def test_forward_shape(self):
            model = HeteroGNN(
                in_dims={"var": 4, "con": 1, "region": 1},
                hidden_dim=32,
                num_layers=2,
            )

            graph = self.make_graph()
            out = model.forward_graph(graph)

            self.assertEqual(out.shape, (2, 32))

    unittest.main()
