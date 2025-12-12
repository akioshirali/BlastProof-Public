 
"""
Conventional GNN encoder for the Minesweeper homogeneous graph.

Supports:
    - GCN
    - GraphSAGE
    - GAT

Outputs:
    node_embeddings = model.forward_graph(graph)

Graph assumptions (from ConventionalGraphBuilder):
    graph.x:          (N, F)
    graph.edge_index: (2, E)

Used by PolicyNetwork.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GCNConv,
    SAGEConv,
    GATConv,
    GraphNorm
)

from typing import Literal


# =============================================================================
# Graph Encoder Block
# =============================================================================

class GNNLayer(nn.Module):
    """
    One GNN layer with:
        conv → norm → activation
    """

    def __init__(self, conv: nn.Module, hidden_dim: int):
        super().__init__()
        self.conv = conv
        self.norm = GraphNorm(hidden_dim)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.norm(x)
        x = F.relu(x)
        return x


# =============================================================================
# CONVENTIONAL GNN ENCODER
# =============================================================================

class ConventionalGNN(nn.Module):
    """
    Args:
        in_dim:        Input feature dimension
        hidden_dim:    Hidden layer dimension
        num_layers:    Number of graph layers (>= 1)
        gnn_type:      "gcn", "sage", or "gat"

    Interface:
        forward_graph(graph) -> node_embeddings (N × hidden_dim)
    """

    def __init__(
        self,
        in_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 3,
        gnn_type: Literal["gcn", "sage", "gat"] = "gcn"
    ):
        super().__init__()

        assert num_layers >= 1, "num_layers must be >= 1"
        self.gnn_type = gnn_type
        self.layers = nn.ModuleList()

        # ---- Build layers ----
        dims = [in_dim] + [hidden_dim] * num_layers

        for i in range(num_layers):
            self.layers.append(
                GNNLayer(
                    conv=self._make_conv(gnn_type, dims[i], dims[i + 1]),
                    hidden_dim=dims[i + 1]
                )
            )

    # ----------------------------------------------------------------------

    def _make_conv(self, gnn_type: str, in_dim: int, out_dim: int):
        """Factory method that builds the selected GNN convolution."""
        if gnn_type == "gcn":
            return GCNConv(in_dim, out_dim, add_self_loops=True)

        elif gnn_type == "sage":
            return SAGEConv(in_dim, out_dim)

        elif gnn_type == "gat":
            # 2 heads × (out_dim // 2) = out_dim
            heads = 2
            assert out_dim % heads == 0
            return GATConv(
                in_dim,
                out_dim // heads,
                heads=heads,
                add_self_loops=True
            )

        else:
            raise ValueError(f"Unknown gnn_type: {gnn_type}")

    # ----------------------------------------------------------------------

    def forward_graph(self, graph):
        """
        Args:
            graph: PyG Data with fields:
                   - x: (N, F)
                   - edge_index: (2, E)

        Returns:
            node_embeddings: (N, hidden_dim)
        """
        x = graph.x
        edge_index = graph.edge_index

        for layer in self.layers:
            x = layer(x, edge_index)

        return x  # node embeddings


# =============================================================================
# UNIT TESTS
# =============================================================================

if __name__ == "__main__":
    import unittest
    from torch_geometric.data import Data

    class TestConventionalGNN(unittest.TestCase):

        def test_gcn_forward(self):
            x = torch.randn(6, 4)
            edge_index = torch.tensor([[0,1,2,3,4,5],
                                       [1,2,3,4,5,0]])

            graph = Data(x=x, edge_index=edge_index)

            model = ConventionalGNN(in_dim=4, hidden_dim=64, num_layers=2, gnn_type="gcn")
            out = model.forward_graph(graph)

            self.assertEqual(out.shape, (6, 64))

        def test_sage_forward(self):
            x = torch.randn(10, 4)
            edge_index = torch.tensor([[0,1,2,3,4,5,6,7,8,9],
                                       [1,2,3,4,5,6,7,8,9,0]])

            graph = Data(x=x, edge_index=edge_index)

            model = ConventionalGNN(in_dim=4, hidden_dim=32, num_layers=3, gnn_type="sage")
            out = model.forward_graph(graph)

            self.assertEqual(out.shape, (10, 32))

        def test_gat_forward(self):
            x = torch.randn(5, 4)
            edge_index = torch.tensor([[0,1,2,3,4],
                                       [1,2,3,4,0]])

            graph = Data(x=x, edge_index=edge_index)

            model = ConventionalGNN(in_dim=4, hidden_dim=64, num_layers=2, gnn_type="gat")
            out = model.forward_graph(graph)

            self.assertEqual(out.shape, (5, 64))

    unittest.main()
