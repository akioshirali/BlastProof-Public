 
"""
analysis_tools.py — Diagnostic utilities for Minesweeper RL system

NOTE:
    # --------------------------------------------------------------
    # We wrote this during finals week.
    # We had no time to properly run or validate all components here.
    # This module was implemented as a complete system
    # mainly because we were confused about
    # eval on the 251104 run wanted a unified place
    # for all future diagnostics.
    # We will probably use and refine this in the next few iterations.
    # --------------------------------------------------------------

This module includes:
    • Board-level analysis
    • 3BV / hardness summary statistics
    • Action legality verification
    • Policy rollout inspection
    • PPO diagnostic metrics (entropy, value drift, etc.)
    • Graph statistic extraction (node counts, edge counts, hyperedge structure)
    • Logging helpers for debugging / research

All functions are designed to be safe to import even in production runs.
No plotting is performed.
"""

from __future__ import annotations
import numpy as np
import torch
from typing import Dict, List, Any


# ======================================================================
#  BOARD & HARDNESS ANALYSIS
# ======================================================================

def count_mines(board: np.ndarray) -> int:
    """Return number of mines on board."""
    return int(np.sum(board < 0))


def count_clues(board: np.ndarray) -> int:
    """Return number of clue cells (1–8)."""
    return int(np.sum(board > 0))


def count_zeros(board: np.ndarray) -> int:
    """Return number of zeros on board."""
    return int(np.sum(board == 0))


def hardness_summary(board: np.ndarray, threebv: int) -> Dict[str, Any]:
    """
    Return basic board characteristics + 3BV summary.
    No hardness bands; those exist in hardness.py.
    """
    rows, cols = board.shape
    return {
        "rows": rows,
        "cols": cols,
        "area": rows * cols,
        "num_mines": count_mines(board),
        "num_clues": count_clues(board),
        "num_zeros": count_zeros(board),
        "threebv": int(threebv),
        "mine_density": float(count_mines(board) / (rows * cols)),
        "clue_density": float(count_clues(board) / (rows * cols)),
    }


# ======================================================================
#  DISTRIBUTION ANALYSIS
# ======================================================================

def distribution_stats(values: List[float]) -> Dict[str, float]:
    """
    Generic distribution summarizer for lists.
    No plotting; returns only numeric summaries.
    """
    if len(values) == 0:
        return {}

    arr = np.array(values, dtype=float)
    return {
        "count": len(arr),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
    }


def threebv_distribution(threebv_list: List[int]) -> Dict[str, float]:
    """Convenience wrapper for 3BV distribution."""
    return distribution_stats(threebv_list)


def step_distribution(step_list: List[int]) -> Dict[str, float]:
    """Stats for episode step counts."""
    return distribution_stats(step_list)


def reward_distribution(reward_list: List[float]) -> Dict[str, float]:
    """Stats for reward sequences."""
    return distribution_stats(reward_list)


# ======================================================================
#  ACTION LEGALITY VERIFICATION
# ======================================================================

def illegal_actions(actions: List[int], legal_mask_list: List[np.ndarray]) -> List[int]:
    """
    Detect illegal actions chosen by the policy.
    Returns list of indices where illegal moves occurred.
    """
    bad = []
    for i, (act, mask) in enumerate(zip(actions, legal_mask_list)):
        if act < 0 or act >= len(mask) or not mask[act]:
            bad.append(i)
    return bad


def explain_illegal_action(action: int, mask: np.ndarray) -> str:
    """
    Explain why an action was illegal, useful for debugging PPO updates.
    """
    if action < 0 or action >= len(mask):
        return f"Action {action} out of bounds [0, {len(mask)-1}]"
    if not mask[action]:
        return f"Action {action} corresponds to a revealed or invalid cell."
    return "Action is legal."


# ======================================================================
#  POLICY ROLLOUT INSPECTION
# ======================================================================

def inspect_policy_rollout(trainer, env) -> Dict[str, Any]:
    """
    Run one rollout, collect all intermediate info:
        - actions
        - rewards
        - values
        - legality
        - final outcome
    """
    obs = env.reset()
    done = False

    actions = []
    rewards = []
    values = []
    legal_masks = []

    while not done:
        action, logp, value, g = trainer.choose_action(obs)
        node_x = g.x if hasattr(g, "x") else g["var"].x
        legal_mask = (node_x[:, 0] == 0).cpu().numpy()

        next_obs, reward, done, info = env.step(action)

        actions.append(int(action))
        rewards.append(float(reward))
        values.append(float(value))
        legal_masks.append(legal_mask)

        obs = next_obs

    illegal_idx = illegal_actions(actions, legal_masks)

    return {
        "actions": actions,
        "rewards": rewards,
        "values": values,
        "illegal_action_indices": illegal_idx,
        "win": info.get("win", False),
        "steps": len(actions),
        "final_threebv": env.threebv,
    }


# ======================================================================
#  PPO DIAGNOSTICS
# ======================================================================

def entropy_from_logits(logits: torch.Tensor) -> float:
    """Return entropy of policy distribution."""
    logp = torch.log_softmax(logits, dim=0)
    p = logp.exp()
    ent = -(p * logp).sum()
    return float(ent.item())


def value_drift(values: List[float]) -> float:
    """
    Detect runaway value estimates (instability).
    Returns mean absolute change between consecutive values.
    """
    if len(values) < 2:
        return 0.0
    diffs = np.abs(np.diff(values))
    return float(diffs.mean())


def advantage_scale(advantages: np.ndarray) -> float:
    """
    Useful to detect insufficient or excessive advantage signal.
    """
    return float(np.abs(advantages).mean())


# ======================================================================
#  GRAPH STATISTICS (FOR GNN DEBUGGING)
# ======================================================================

def graph_stats(g) -> Dict[str, Any]:
    """
    Return node/edge statistics for PyG Data or HeteroData objects.
    """
    if hasattr(g, "edge_index"):
        # Homogeneous graph
        num_nodes = g.num_nodes
        num_edges = g.edge_index.shape[1]
        return {
            "type": "homogeneous",
            "num_nodes": int(num_nodes),
            "num_edges": int(num_edges),
            "node_feat_dim": g.x.shape[1] if hasattr(g, "x") else None,
        }

    # Heterogeneous graph
    stats = {"type": "hetero", "node_types": {}, "edge_types": {}}

    for nt, nt_data in g.node_items():
        stats["node_types"][nt] = {
            "num_nodes": nt_data.x.shape[0],
            "feat_dim": nt_data.x.shape[1],
        }

    for et, et_data in g.edge_items():
        stats["edge_types"][et] = {
            "num_edges": et_data.edge_index.shape[1],
        }

    return stats


# ======================================================================
#  LOGGING HELPERS
# ======================================================================

def summarize_rollout_info(info_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarize a list of rollout dictionaries returned by inspect_policy_rollout().
    Useful for batch diagnostics.
    """
    wins = [info["win"] for info in info_list]
    steps = [info["steps"] for info in info_list]
    threebv = [info["final_threebv"] for info in info_list]
    illegal_total = sum(len(info["illegal_action_indices"]) for info in info_list)

    return {
        "rollouts": len(info_list),
        "win_rate": float(np.mean(wins)),
        "avg_steps": float(np.mean(steps)),
        "avg_threebv": float(np.mean(threebv)),
        "total_illegal_actions": illegal_total,
    }
