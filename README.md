## Overview

BlastProof is a complete reinforcement learning framework for Minesweeper, enabling researchers to explore representation learning, reasoning, and constraint-satisfaction algorithms using:
- Multi-graph representations of game state
- GNN-based policies
- PPO with advanced shaping and curricula
- Difficulty-targeted training (hardness, 3BV, density ranges)

## Project Features

### Full Minesweeper environment implementation
- Reveal logic with zero-region flood fill
- Clue board computation
- 3BV difficulty metric
- Rollback on mine hits

### Graph-based state representations
- Conventional GNN (GCN/SAGE/GAT)
- Heterogeneous graph with region + constraint nodes
- Hypergraph factor-graph representation

### Reinforcement Learning
- PPO V1 (single environment)
- PPO V2 (vectorized / batched rollout)
- GAE, value clipping, entropy regularization

### Curricula
- Size/density progression
- Hardness-driven (easy → expert)
- 3BV rejection sampling
- “Stupid Hard” multi-level curriculum

### Evaluation Tools
- Win rate
- Step stats
- 3BV efficiency
- Difficulty buckets


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
- Each cell is a node
- 4- or 8-neighborhood adjacency
- Features: revealed flag, clue value, mine label, index
### 2. Heterogeneous Graph
**Node types:**
- var → variable cells  
- con → clue constraints  
- region → zero regions  
**Edge types:**
- var ↔ var adjacency  
- var → con (touches)  
- region → var (group memberships)  
### 3. Hypergraph (Factor Graph)
Constraint nodes represent clues:
- var → con (variable participates in constraint)  
- con → var (constraint influences variable)  

Used for robust relational reasoning.


