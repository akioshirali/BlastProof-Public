## Overview

BlastProof is a fully graph-native Minesweeper solver that uses reinforcement learning, graph neural networks, and curriculum training to perform multi-step relational inference on Minesweeper boards of increasing complexity. The project explores three graph representations conventional GNNs, heterogeneous GNNs, and factor-graph hypergraphs—and evaluates their ability to solve Minesweeper efficiently and generalize across board sizes.


# Directory Structure  
<pre>
'''
├── Board_Generation.py        # Utilities for mine layouts and board configurations.
├── Clue_wUTs.py               # Clue computation
├── Evals.py                   # Eval metrics and comparisons.
├── Finetune.py                # Fine-tuning for PPO w/reward shaping
├── Hetro.py                   # heterogeneous graph builder (var, con, region nodes + edges).
├── Hypergraph.py              # hypergraph builder like constraint-GNNs.
├── Minesweeper.py             # Core Minesweeper environment with reveal logic and rollback.
├── density.py                 # Size/density difficulty curriculum
├── gae.py                     # Generalized Advantage Estimation (GAE).
├── gnn_conventional.py        # GCN/SAGE/GAT variants
├── gnn_hypergraph.py          # Stabilized hypergraph encoder
├── hardness.py                # Hardness metrics: 3BV, openings, densities, composite score.
├── hardness_curr.py           # Hardness curriculum generator (easy to expert progression).
├── hetro.py                   # Old heterogeneous graph builder — DO NOT USE ~!!
├── policy_network.py          # Unified actor–critic network
├── ppo_buffer.py              # Rollout buffers for PPO
├── ppo_v1.py                  # PPO v1 trainer (single-environment, per-episode updates).
├── ppo_v2.py                  # PPO v2 trainer with vectorized rollouts and parallel environments.
├── sampling.py                # Uniform, weighted, 3BV-range sampling + curriculum utilities.
├── supid_hard_curr.py         # “Stupid Hard” multi-stage 3BV curriculum with adaptive episode counts.
├── threeebv.py                # Original 3BV computation module + isolated clue detection.
├── train_v1.py                # Training script for PPO v1 with logging and checkpoints.
└── utils.py                   # Misc PPO utilities
'''
</pre>

# Graph Builders
Three graph representations are supported:
### 1. Conventional Graph
<img width="4671" height="1466" alt="one_panel_grayscale_ Conventional " src="https://github.com/user-attachments/assets/a82222ee-25f9-4ea2-97ab-4e1021580ccc" />
The simplest representation treats every tile as a node and connects it to the eight tiles around it. All tiles—hidden cells, revealed clues, zeros, and mines—are represented with the same node type. This graph captures only basic spatial adjacency, and the model can learn shallow local patterns such as simple 1-2-1 or corner configurations. However, it cannot express how numerical clues impose multi-variable constraints. Because each clue must influence all of its surrounding tiles jointly, forcing everything into pairwise edges strips away most of the logical structure of Minesweeper. 
This baseline is fast, intuitive, and easy to generate, but it fails to provide the relational detail needed for deeper inference.


### 2. Heterogeneous Graph
<img width="1662" height="690" alt="hetrero graph" src="https://github.com/user-attachments/assets/facf44e3-657f-4ca3-9bc8-625f87c2174b" />

To capture more of the structure of Minesweeper, the heterogeneous graph introduces different types of nodes and edges. Variable nodes represent tiles, constraint nodes represent revealed clues, and region nodes represent connected zero-valued areas. Edges also differ depending on their meaning: spatial adjacency, clue–tile “touch” relationships, and region–tile grouping connections each use their own edge type. This allows the model to treat different interactions differently and propagate information in a semantically appropriate way. It captures more global structure than the grid graph, especially around zero-propagation and boundary formation, but it still represents clue constraints as a collection of pairwise links rather than a single joint constraint.


### 3. Hypergraph (Factor Graph)

<img width="1662" height="690" alt="hyper graph" src="https://github.com/user-attachments/assets/1948af95-8a85-45f9-bbb5-294384e591f3" />

The most expressive representation models Minesweeper exactly as a constraint satisfaction problem. Each clue becomes a factor node, and all the tiles in its 8-neighborhood form a hyperedge with it. Instead of decomposing the clue into eight separate pairwise edges, the hypergraph treats the clue as one constraint that simultaneously governs all adjacent variables. This allows joint reasoning about all tiles involved in a constraint, matches the structure of SAT-like formulations, and enables long-range inference through repeated rounds of variable-to-constraint and constraint-to-variable message passing. Empirically, this representation produces the fastest curriculum progression, the highest win rates, and the most human-like efficiency, because it mirrors how Minesweeper logic actually works.
