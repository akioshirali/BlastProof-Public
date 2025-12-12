  
"""
policy_network.py

Unified Policy/Value Network wrapper for Minesweeper RL.

Supports three backends:
    - Conventional GNN
    - Heterogeneous GNN
    - Hypergraph GNN

Each backend must implement:
    forward_graph(graph) -> node_embeddings (N × D)

This module provides:
    class PolicyNetwork(nn.Module)

Features:
    - Actor head outputs logits for all variable nodes
    - Action masking (illegal moves get -1e9)
    - Critic head produces scalar value
    - Clean interface compatible with PPO-style training

Unit tests included below.
"""
 
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ============================================================================
# SIMPLE MULTILAYER PERCEPTRON HEAD
# ============================================================================

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


# ============================================================================
# POLICY NETWORK (WRAPPER)
# ============================================================================

class PolicyNetwork(nn.Module):
    """
    Wraps a graph encoder and produces:
        - action logits (masked)
        - state value (scalar)

    Required encoder interface:
        embeddings = encoder.forward_graph(graph)
        -> returns tensor of shape (num_var_nodes, embedding_dim)
    """

    def __init__(
        self,
        encoder: nn.Module,
        embedding_dim: int,
        hidden_dim: int = 128
    ):
        super().__init__()
        self.encoder = encoder
        self.embedding_dim = embedding_dim

        # Actor: produces 1 logit per variable node
        self.actor = MLP(embedding_dim, 1, hidden_dim)

        # Critic: receives pooled graph embedding (1 × D)
        self.critic = MLP(embedding_dim, 1, hidden_dim)

    # ----------------------------------------------------------------------
    def forward(
        self,
        graph,
        legal_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # ----------------------------------------------------------
        # Encode the graph → var node embeddings
        # ----------------------------------------------------------
        embeddings = self.encoder.forward_graph(graph)  # (N_var, D)

        # ----------------------------------------------------------
        # Critic head (graph value)
        # Use mean pooling over variable nodes
        # ----------------------------------------------------------
        pooled = embeddings.mean(dim=0, keepdim=True)  # (1, D)
        value = self.critic(pooled).squeeze(-1)        # scalar

        # ----------------------------------------------------------
        # Actor head: one logit per variable cell
        # ----------------------------------------------------------
        logits = self.actor(embeddings).squeeze(-1)    # (N_var,)

        # ----------------------------------------------------------
        # Mask illegal actions
        # ----------------------------------------------------------
        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask, -1e9)

        # Second output kept for compatibility with PPO code
        dist_logits = logits.clone()

        return logits, dist_logits, value


# ============================================================================
# EXAMPLE ENCODERS (STUBS FIXED TO RETURN ONLY VAR EMBEDDINGS)
# ============================================================================

class ConventionalGNN(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=128):
        super().__init__()
        self.fc = nn.Linear(in_dim, hidden_dim)

    def forward_graph(self, graph):
        # graph.x = (N_var, in_dim)
        return F.relu(self.fc(graph.x))


class HeteroGNN(nn.Module):
    def __init__(self, var_in=4, con_in=1, hidden_dim=128):
        super().__init__()
        self.var_fc = nn.Linear(var_in, hidden_dim)
        self.con_fc = nn.Linear(con_in, hidden_dim)

    def forward_graph(self, graph):
        # Return variable embeddings only
        var_emb = F.relu(self.var_fc(graph["var"].x))
        return var_emb


class HypergraphGNN(nn.Module):
    def __init__(self, var_in=4, con_in=1, hidden_dim=128):
        super().__init__()
        self.var_fc = nn.Linear(var_in, hidden_dim)
        self.con_fc = nn.Linear(con_in, hidden_dim)

    def forward_graph(self, graph):
        # Hypergraph encoder returns variable node embeddings only
        var_emb = F.relu(self.var_fc(graph["var"].x))
        return var_emb


# ============================================================================
# UNIT TESTS
# ============================================================================

if __name__ == "__main__":
    import unittest
    from torch_geometric.data import Data, HeteroData

    class TestPolicyNetwork(unittest.TestCase):

        def test_conventional_policy(self):
            x = torch.randn(5, 4)
            graph = Data(x=x)

            enc = ConventionalGNN()
            net = PolicyNetwork(enc, embedding_dim=128)

            legal = torch.tensor([True, True, False, True, False])
            logits, dist, value = net(graph, legal)

            self.assertEqual(logits.shape[0], 5)
            self.assertTrue((logits[~legal] < -1e8).all())  # masked
            self.assertTrue(value.ndim == 0)

        def test_hetero_policy(self):
            g = HeteroData()
            g["var"].x = torch.randn(6, 4)
            g["con"].x = torch.randn(3, 1)

            enc = HeteroGNN()
            net = PolicyNetwork(enc, embedding_dim=128)

            legal = torch.ones(6, dtype=torch.bool)
            logits, dist, value = net(g, legal)
            self.assertEqual(logits.shape[0], 6)

        def test_hypergraph_policy(self):
            g = HeteroData()
            g["var"].x = torch.randn(10, 4)
            g["con"].x = torch.randn(5, 1)

            enc = HypergraphGNN()
            net = PolicyNetwork(enc, embedding_dim=128)

            legal = torch.rand(10) > 0.2
            logits, dist, value = net(g, legal)
            self.assertEqual(logits.shape[0], 10)

    unittest.main()
