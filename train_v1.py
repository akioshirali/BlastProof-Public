 
"""
 Training script for PPO V1 (single environment)

This script is for :
    • Standard training
    • Optional curriculum scheduling
    • Logging
    • Checkpointing
    • Final evaluation

"""

from __future__ import annotations
import os
import json
import torch
from datetime import datetime
from tqdm import tqdm

from envs.minesweeper_env import MinesweeperGame
from graph_builders.graph_builder import GraphBuilder
from ppo.ppo_v1 import PPOTrainer
from evaluation.eval_all import evaluate_model


# ================================================================
# CHECKPOINT UTILITIES
# ================================================================

def create_checkpoint_folder(model_name: str):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder = os.path.join("checkpoints", f"{model_name}_v1", ts)
    os.makedirs(folder, exist_ok=True)
    return folder


def save_checkpoint(folder: str, trainer, episode: int, extra=None):
    path = os.path.join(folder, f"checkpoint_{episode}.pt")
    payload = {
        "model": trainer.model.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "episode": episode,
        "extra": extra or {},
    }
    torch.save(payload, path)
    print(f"[CHECKPOINT] Saved → {path}")


# ================================================================
# TRAINING LOOP — PPO V1
# ================================================================

def train_v1(
    model,
    graph_builder_cls=GraphBuilder,
    env_cls=MinesweeperGame,
    episodes: int = 50000,
    rows: int = 6,
    cols: int = 6,
    density: float = 0.18,
    save_every: int = 2000,
    device: str = "cpu",
    curriculum=None,
    model_name: str = "ppo_v1_model",
):
    """
    Primary training function for PPO V1.

    Args:
        model: your GNN (hypergraph, hetero, conventional, etc.)
        curriculum: optional scheduler with .next() and .record_result()
        env_cls: must support Minesweeper-like env interface
        graph_builder_cls: produces PyG graphs from observations
    """

    print("\n=====================================")
    print("        TRAINING PPO V1")
    print("=====================================\n")

    # --------------------------------------------
    # Instantiate builder + trainer
    # --------------------------------------------
    graph_builder = graph_builder_cls()
    trainer = PPOTrainer(model, graph_builder, device=device)

    # --------------------------------------------
    # Prepare checkpoint folder
    # --------------------------------------------
    folder = create_checkpoint_folder(model_name)

    # --------------------------------------------
    # Logging
    # --------------------------------------------
    logs = {
        "episode": [],
        "win": [],
        "steps": [],
        "3bv": [],
    }

    # --------------------------------------------
    # Main Training Loop
    # --------------------------------------------
    for episode in tqdm(range(1, episodes + 1), desc="Training PPO V1"):

        # Curriculum override:
        if curriculum is not None:
            cfg = curriculum.next()
            r = cfg["rows"]
            c = cfg["cols"]
            m = cfg["mines"]
            env = env_cls(r, c, mine_density=None, mine_count=m)
        else:
            env = env_cls(rows, cols, density)

        obs = env.reset()
        done = False

        buffer = trainer.new_buffer()
        steps = 0

        # --------- Rollout Episode ---------
        while not done:
            action, logp, value, g = trainer.choose_action(obs)
            next_obs, reward, done, info = env.step(action)

            buffer.store(
                g.cpu(),
                int(action),
                float(reward),
                float(value),
                logp.cpu(),
                float(done),
            )

            obs = next_obs
            steps += 1

        trainer.update(buffer)

        # --------- Logging ---------
        win = info.get("win", False)
        logs["episode"].append(episode)
        logs["win"].append(int(win))
        logs["steps"].append(steps)
        logs["3bv"].append(env.threebv)

        if curriculum is not None:
            curriculum.record_result(win)

        # --------- Periodic Checkpoints ---------
        if episode % save_every == 0:
            save_checkpoint(folder, trainer, episode, extra={"logs": logs})

    # =======================================
    # Final save
    # =======================================
    final_path = os.path.join(folder, "final_model.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\n[SAVED] Final model → {final_path}")

    # Save logs
    with open(os.path.join(folder, "train_logs.json"), "w") as f:
        json.dump(logs, f, indent=2)

    return logs, folder, trainer


# ================================================================
#  OPTIONAL: STANDALONE EXECUTION ENTRY
# ================================================================

if __name__ == "__main__":
    print("This script is meant to be imported and called from a launcher.")
