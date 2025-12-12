 
"""
stupid_hard.py — Extremely aggressive 3BV-driven curriculum

This curriculum forces the agent to master progressively harder 3BV bands.
It repeatedly generates random Minesweeper boards until the 3BV falls
inside the desired hardness interval.

Stages (default):
    Level 1:  3BV ∈ [0,  5]     (trivial)
    Level 2:  3BV ∈ [5, 10]
    Level 3:  3BV ∈ [10, 20]
    Level 4:  3BV ∈ [20, 40]
    Level 5:  3BV ∈ [40, 80]
    Level 6:  3BV ∈ [80, 999]

The agent advances when:
    - win_rate >= required_winrate[level]
    OR
    - forced episode limit reached (adaptive)
"""

from __future__ import annotations
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any, Optional
from datetime import datetime
import os
import json


# ================================================================
# HARDNESS LEVELS (3BV RANGES)
# ================================================================
HARDNESS_LEVELS = [
    (0, 5),      # Level 1 – trivial
    (5, 10),     # Level 2 – easy
    (10, 20),    # Level 3 – medium
    (20, 40),    # Level 4 – hard
    (40, 80),    # Level 5 – expert
    (80, 999),   # Level 6 – extreme
]

# Required win rate per level before advancement
HARDNESS_REQUIRED_WINRATE = [
    0.95,
    0.90,
    0.85,
    0.75,
    0.65,
    0.55,
]


# ================================================================
# Helper — regenerate until board meets target hardness
# ================================================================
def generate_board_with_3bv(env_class, rows, cols, density, min_3bv, max_3bv):
    """
    Keep creating environments until 3BV is inside the desired range.
    Returns: env, obs, threebv
    """
    for _ in range(300):  # safety limit
        env = env_class(rows, cols, density)
        obs = env.reset()
        tbv = env.threebv
        if min_3bv <= tbv <= max_3bv:
            return env, obs, tbv

    # If not found (rare), return last sample
    return env, obs, tbv


# ================================================================
# Checkpoint utilities
# ================================================================
def create_checkpoint_folder(model_type: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder = os.path.join("checkpoints", model_type, f"stupid_hard_{ts}")
    os.makedirs(folder, exist_ok=True)
    return folder


def save_checkpoint(folder: str, model, optimizer, log: dict, step_info: dict):
    path = os.path.join(folder, "last_checkpoint.pt")
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "curriculum_log": log,
        "step_info": step_info,
    }
    os.makedirs(folder, exist_ok=True)
    torch_save(path, payload)

    # Also save readable CSV
    try:
        import pandas as pd
        pd.DataFrame(log).to_csv(os.path.join(folder, "curriculum_log.csv"), index=False)
    except ImportError:
        pass

    print(f"[CHECKPOINT] Saved → {path}")


def torch_save(path, payload):
    import torch
    torch.save(payload, path)


# ================================================================
# Adaptive Episode Count
# ================================================================
def adaptive_episode_count(win_hist: List[float], base: int,
                           min_ep=20, max_ep=400) -> int:
    """
    A simple but effective adaptive scheduling rule:
        • win↑↑↑ → fewer episodes
        • win stagnates → more episodes
        • win drops → increase episodes aggressively
    """
    if len(win_hist) < 3:
        return base

    w1, w2, w3 = win_hist[-3:]

    # Strong convergence
    if w1 > 0.9 and w2 > 0.9 and w3 > 0.9:
        return max(min_ep, int(base * 0.6))

    # Improving steadily
    if w3 > w2 > w1:
        return max(min_ep, int(base * 0.8))

    # Stagnation
    if abs(w3 - w2) < 0.05 and abs(w2 - w1) < 0.05:
        return min(max_ep, int(base * 1.4))

    # Regression
    if w3 < w2 < w1:
        return min(max_ep, int(base * 1.8))

    return base


# ================================================================
# Evaluation for a hardness level
# ================================================================
def evaluate_hardness_level(
    trainer,
    env_class,
    min_3bv: int,
    max_3bv: int,
    rows=8,
    cols=8,
    density=0.20,
    episodes=40,
) -> float:
    """
    Run evaluation episodes where all boards fall inside the target
    3BV hardness band.
    """
    wins = 0

    for _ in range(episodes):
        env, obs, tbv = generate_board_with_3bv(
            env_class,
            rows=rows,
            cols=cols,
            density=density,
            min_3bv=min_3bv,
            max_3bv=max_3bv,
        )

        done = False
        while not done:
            action, logp, value, g = trainer.choose_action(obs)
            obs, reward, done, info = env.step(action)

        if info.get("win", False):
            wins += 1

    return wins / episodes


# ================================================================
# MAIN CURRICULUM
# ================================================================
@dataclass
class StupidHardCurriculum:
    """
    The most aggressive curriculum we offer.

    This will:
        • Train exclusively inside 3BV bands
        • Not advance until win rate requirement is met
        • Adaptively scale number of episodes
    """

    base_episodes: int = 120
    rows: int = 8
    cols: int = 8
    density: float = 0.20
    device: str = "cpu"

    curriculum_log: dict = field(default_factory=lambda: {
        "level": [],
        "min_3bv": [],
        "max_3bv": [],
        "win_rate": [],
        "episodes_used": [],
    })

    def run(self, model_type: str, trainer, env_class):
        """
        Full multi-stage hardness curriculum.
        """

        folder = create_checkpoint_folder(model_type)
        optimizer = trainer.optimizer

        start_time = datetime.now()

        # ------------------------------------------------------------
        # Iterate over hardness tiers
        # ------------------------------------------------------------
        for level_idx, (min_3bv, max_3bv) in enumerate(HARDNESS_LEVELS):

            required = HARDNESS_REQUIRED_WINRATE[level_idx]
            print("")
            print("=" * 60)
            print(f" LEVEL {level_idx+1} — Target 3BV [{min_3bv}, {max_3bv}]")
            print(f" Required win rate: {required * 100:.0f}%")
            print("=" * 60)

            win_hist = []
            converged = False

            # ------------------------------
            # Repeat until win rate achieved
            # ------------------------------
            while not converged:

                # Decide how many episodes to run this round
                episodes_now = adaptive_episode_count(win_hist, self.base_episodes)
                print(f"\nTraining {episodes_now} episodes for hardness level {level_idx+1}")

                # --------------------------
                # TRAINING
                # --------------------------
                for _ in tqdm(range(episodes_now), desc="Training"):
                    env, obs, tbv = generate_board_with_3bv(
                        env_class,
                        rows=self.rows,
                        cols=self.cols,
                        density=self.density,
                        min_3bv=min_3bv,
                        max_3bv=max_3bv,
                    )

                    # standard on-policy episode
                    buffer = trainer.new_buffer()
                    done = False
                    while not done:
                        action, logp, value, g = trainer.choose_action(obs)
                        next_obs, reward, done, info = env.step(action)
                        buffer.store(g.cpu(), action, float(reward),
                                     float(value), logp.cpu(), float(done))
                        obs = next_obs

                    trainer.update(buffer)

                # --------------------------
                # VALIDATION
                # --------------------------
                win_rate = evaluate_hardness_level(
                    trainer=trainer,
                    env_class=env_class,
                    min_3bv=min_3bv,
                    max_3bv=max_3bv,
                    rows=self.rows,
                    cols=self.cols,
                    density=self.density,
                    episodes=40,
                )

                win_hist.append(win_rate)

                print(f" Win Rate @ Level {level_idx+1}: {win_rate:.3f}")

                # Store results
                self.curriculum_log["level"].append(level_idx)
                self.curriculum_log["min_3bv"].append(min_3bv)
                self.curriculum_log["max_3bv"].append(max_3bv)
                self.curriculum_log["win_rate"].append(win_rate)
                self.curriculum_log["episodes_used"].append(episodes_now)

                # Checkpoint
                save_checkpoint(
                    folder,
                    trainer.model,
                    optimizer,
                    self.curriculum_log,
                    step_info={
                        "level": level_idx,
                        "win_rate": win_rate,
                        "episodes": episodes_now,
                        "timestamp": str(datetime.now()),
                    }
                )

                if win_rate >= required:
                    print(f"\n*** LEVEL {level_idx+1} MASTERED! ***\n")
                    converged = True

        # End of all hardness levels
        total_time = datetime.now() - start_time
        print("=" * 60)
        print(" STUPID HARD CURRICULUM COMPLETE ")
        print(f" TOTAL TIME: {total_time}")
        print(f" CHECKPOINTS: {folder}")
        print("=" * 60)

        # Save training-time JSON
        with open(os.path.join(folder, "training_time.json"), "w") as f:
            json.dump({"total_time_seconds": total_time.total_seconds(),
                       "total_time_str": str(total_time)}, f, indent=2)

        return self.curriculum_log, folder
