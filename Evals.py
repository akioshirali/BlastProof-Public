 
"""
eval_all.py — Consolidated Minesweeper evaluation utilities
(Non-plotting version)

Includes:
    • evaluate_model()               — full evaluation loop
    • evaluate_on_density()          — simple density-based evaluation
    • evaluate_single_game()         — step-by-step env execution
    • difficulty_buckets()           — human Minesweeper difficulty tiers
    • summarize_eval()               — produce summary metrics
    • compare_models()               — text-only model comparison
"""

from __future__ import annotations
import numpy as np
from tqdm import tqdm


# ======================================================================
# Difficulty buckets (human Minesweeper categories)
# ======================================================================

DIFFICULTY_BUCKETS = [
    (0, 5, "Trivial"),
    (5, 10, "Easy"),
    (10, 20, "Medium"),
    (20, 40, "Hard"),
    (40, 80, "Expert"),
    (80, 999, "Extreme"),
]


# ======================================================================
# Evaluate a single complete game
# ======================================================================

def evaluate_single_game(model, trainer, env):
    """
    Runs a single Minesweeper game to completion.
    Returns:
        win (bool)
        steps (int)
        final_3bv (int)
        mine_hits (int)
    """
    obs = env.reset()
    done = False
    steps = 0
    mine_hits = 0
    win = False

    while not done:
        action, logp, value, g = trainer.choose_action(obs)
        obs, reward, done, info = env.step(action)
        steps += 1

        if info.get("rollback", False):
            mine_hits += 1

    if info.get("win", False):
        win = True

    return win, steps, env.threebv, mine_hits


# ======================================================================
# Evaluate the model over N episodes at a fixed board size + density
# ======================================================================

def evaluate_model(
    model,
    trainer_class,
    env_class,
    graph_builder,
    board_size: int,
    density: float,
    episodes: int = 200,
    device: str = "cpu",
):
    """
    Full evaluation:
        - win rate
        - avg steps
        - avg 3BV
        - weighted accuracy
        - mine hit rate
        - efficiency (steps per 3BV)
        - per-episode logs

    trainer_class: class, not instance — created fresh for evaluation.
    """

    # Build trainer for evaluation only
    trainer = trainer_class(model, graph_builder, device=device)

    wins = 0
    mine_hits = 0
    all_steps = []
    all_3bv = []

    for _ in tqdm(range(episodes), desc=f"Eval {board_size}x{board_size}"):
        env = env_class(board_size, board_size, density)
        win, steps, tbv, hits = evaluate_single_game(model, trainer, env)

        wins += int(win)
        mine_hits += hits
        all_steps.append(steps)
        all_3bv.append(tbv)

    # Convert to numpy
    all_steps = np.array(all_steps, dtype=np.float32)
    all_3bv = np.array(all_3bv, dtype=np.float32)

    # Steps per 3BV (difficulty-normalized efficiency)
    efficiency = np.mean(all_steps / (all_3bv + 1e-6))

    # Difficulty-weighted accuracy:
    # success = steps < threshold proportional to 3BV
    weighted_accuracy = np.mean((all_steps < all_3bv * 3).astype(np.float32))

    return {
        "wins": int(wins),
        "episodes": episodes,
        "win_rate": wins / episodes,
        "avg_steps": float(all_steps.mean()),
        "avg_3bv": float(all_3bv.mean()),
        "mine_hit_rate": mine_hits / episodes,
        "efficiency_steps_per_3bv": float(efficiency),
        "weighted_accuracy": float(weighted_accuracy),
        "steps": all_steps.tolist(),
        "three_bv": all_3bv.tolist(),
    }


# ======================================================================
# Simple evaluation on density (win rate only)
# ======================================================================

def evaluate_on_density(
    trainer,
    env_class,
    board_size: int,
    density: float,
    episodes: int = 40,
):
    """
    Fast evaluation:
        - Only win rate + minimal stats
        - Used inside curricula
    """
    wins = 0
    steps_list = []
    bv_list = []

    for _ in range(episodes):
        env = env_class(board_size, board_size, density)
        obs = env.reset()
        done = False
        steps = 0

        while not done:
            action, logp, value, g = trainer.choose_action(obs)
            obs, reward, done, info = env.step(action)
            steps += 1

        steps_list.append(steps)
        bv_list.append(env.threebv)
        if info.get("win", False):
            wins += 1

    return wins / episodes, {
        "steps": steps_list,
        "3bv": bv_list,
        "win": wins / episodes,
    }


# ======================================================================
# Difficulty bucket analysis
# ======================================================================

def difficulty_buckets(eval_data: dict):
    """
    Splits episodes into human difficulty categories based on 3BV.
    Returns a list of bucket summaries.
    """
    steps = np.array(eval_data["steps"])
    bv = np.array(eval_data["three_bv"])

    results = []

    for lo, hi, label in DIFFICULTY_BUCKETS:
        mask = (bv >= lo) & (bv < hi)
        count = int(mask.sum())
        if count == 0:
            continue

        avg_steps = float(steps[mask].mean())

        results.append({
            "difficulty": label,
            "count": count,
            "avg_steps": avg_steps,
        })

    return results


# ======================================================================
# Summaries and comparisons (non-plotting)
# ======================================================================

def summarize_eval(name: str, results: dict) -> str:
    """
    Produce a readable summary string for a model evaluation.
    """
    return (
        f"\n===== {name} Evaluation Summary =====\n"
        f"Win Rate:                {results['win_rate']:.3f}\n"
        f"Average Steps:           {results['avg_steps']:.2f}\n"
        f"Average 3BV:             {results['avg_3bv']:.2f}\n"
        f"Mine Hit Rate:           {results['mine_hit_rate']:.3f}\n"
        f"Steps per 3BV:           {results['efficiency_steps_per_3bv']:.3f}\n"
        f"Weighted Accuracy:       {results['weighted_accuracy']:.3f}\n"
    )


def compare_models(result_dict: dict):
    """
    Print a readable table comparing models on standard metrics.
    Input:
        result_dict = {
            "hypergraph": {...},
            "conventional": {...},
            "hetero": {...},
        }
    """

    print("\n==================== MODEL COMPARISON ====================")
    print(f"{'Model':<15} {'WinRate':<10} {'AvgSteps':<10} {'MineHit':<10} {'Eff':<10}")

    for name, res in result_dict.items():
        print(
            f"{name:<15} "
            f"{res['win_rate']:<10.3f} "
            f"{res['avg_steps']:<10.2f} "
            f"{res['mine_hit_rate']:<10.3f} "
            f"{res['efficiency_steps_per_3bv']:<10.3f}"
        )
