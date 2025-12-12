 
"""
finetune.py — Optional fine-tuning routines

This module provides:
    • fine_tune_3bv()          – emphasize reducing remaining board difficulty
    • fine_tune_standard()     – generic PPO fine-tuning
    • evaluate_after_finetune()– wrapper around eval_all.evaluate_model

Fine-tuning is typically run after the main curriculum or size/density training.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from tqdm import tqdm


# =============================================================================
#  Standard Fine-Tuning Loop
# =============================================================================

def fine_tune_standard(
    trainer,
    env_class,
    graph_builder,
    episodes: int = 2000,
    board_size: int = 6,
    density: float = 0.20,
):
    """
    A simple, general-purpose PPO fine-tuning loop.
    Does not include additional reward shaping.
    """
    print("\n==========================================")
    print("        STARTING STANDARD FINE-TUNE")
    print("==========================================\n")

    logs = {
        "episode_win": [],
        "episode_steps": [],
        "episode_3bv": [],
    }

    for ep in tqdm(range(episodes), desc="Fine-tune"):
        env = env_class(board_size, board_size, density)
        obs = env.reset()
        done = False
        buffer = trainer.new_buffer()
        steps = 0

        while not done:
            action, logp, value, g = trainer.choose_action(obs)
            next_obs, reward, done, info = env.step(action)

            buffer.store(
                g.cpu(),
                action,
                float(reward),
                float(value),
                logp.cpu(),
                float(done),
            )

            obs = next_obs
            steps += 1

        trainer.update(buffer)

        logs["episode_win"].append(info.get("win", False))
        logs["episode_steps"].append(steps)
        logs["episode_3bv"].append(env.threebv)

    print("\nFine-tuning complete.\n")
    return logs


# =============================================================================
#  Fine-Tuning That Emphasizes *Reducing 3BV*
# =============================================================================

def fine_tune_3bv(
    trainer,
    env_class,
    graph_builder,
    episodes: int = 2000,
    board_size: int = 6,
    density: float = 0.20,
    reward_scale: float = 0.03,
):
    """
    Fine-tuning with special emphasis on REDUCING REMAINING 3BV.

    We do NOT modify the environment. Instead, we simply allow the agent
    to receive the internal reward shaping that already exists inside the
    MinesweeperGame (delta_3BV * reward_scale).

    This loop just focuses heavily and repeatedly on boards at a fixed size
    and density so the agent improves its reasoning and "zero-expansion"
    behavior.

    Args:
        trainer: PPO_v1 or PPO_v2 trainer (must have choose_action + update)
        env_class: MinesweeperGame or compatible environment
        graph_builder: builder used by trainer
        episodes: total fine-tuning episodes
        reward_scale: if the environment supports changing shaping intensity
    """

    print("\n==========================================")
    print("        STARTING 3BV REDUCTION TUNE")
    print("==========================================\n")

    logs = {
        "episode_win": [],
        "episode_steps": [],
        "episode_3bv": [],
    }

    for ep in tqdm(range(episodes), desc="Fine-tune-3BV"):
        env = env_class(board_size, board_size, density)
        # If environment supports dynamic reward shaping:
        if hasattr(env, "set_reward_scale"):
            env.set_reward_scale(reward_scale)

        obs = env.reset()
        done = False
        buffer = trainer.new_buffer()
        steps = 0

        while not done:
            action, logp, value, g = trainer.choose_action(obs)
            next_obs, reward, done, info = env.step(action)

            buffer.store(
                g.cpu(),
                action,
                float(reward),
                float(value),
                logp.cpu(),
                float(done),
            )

            obs = next_obs
            steps += 1

        trainer.update(buffer)

        logs["episode_win"].append(info.get("win", False))
        logs["episode_steps"].append(steps)
        logs["episode_3bv"].append(env.threebv)

    print("\n3BV Fine-tuning Complete.\n")
    return logs


# =============================================================================
#  Post-Training Evaluation
# =============================================================================

def evaluate_after_finetune(
    model,
    trainer_class,
    env_class,
    graph_builder,
    episodes: int = 200,
    board_size: int = 6,
    density: float = 0.20,
    device: str = "cpu",
):
    """
    A convenience wrapper around evaluation/eval_all.py.

    Returns a standard evaluation result dict.
    """

    from evaluation.eval_all import evaluate_model

    print("\n==========================================")
    print("        EVALUATING FINE-TUNED MODEL")
    print("==========================================\n")

    results = evaluate_model(
        model=model,
        trainer_class=trainer_class,
        env_class=env_class,
        graph_builder=graph_builder,
        board_size=board_size,
        density=density,
        episodes=episodes,
        device=device,
    )

    print("\n===== FINE-TUNE EVALUATION RESULTS =====")
    print(f"Win Rate:          {results['win_rate']:.3f}")
    print(f"Avg Steps:         {results['avg_steps']:.2f}")
    print(f"Avg 3BV:           {results['avg_3bv']:.2f}")
    print(f"Mine Hit Rate:     {results['mine_hit_rate']:.3f}")
    print(f"Efficiency:        {results['efficiency_steps_per_3bv']:.3f}")
    print(f"Weighted Accuracy: {results['weighted_accuracy']:.3f}")
    print("========================================\n")

    return results
