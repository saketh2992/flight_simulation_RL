"""Train PPO in rounds and capture a GIF after each round to show improvement."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor

import flight_sim  # noqa: F401
from flight_sim.env import FlightDodgeEnv

STEPS_PER_ROUND = 100_000
ROUNDS = 7          # 700k more steps (total ~1M)
GIF_DIR = Path("gifs")
MODEL_PATH = Path("models/ppo_flight_dodge")


def capture_episode(model, seed: int, max_steps: int = 500) -> list[Image.Image]:
    env = FlightDodgeEnv(render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    frames = []
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        frame = env.render()
        frames.append(Image.fromarray(frame))
        if term or trunc:
            break
    env.close()
    return frames


def save_gif(frames: list[Image.Image], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=50,
        loop=0,
    )


def main():
    GIF_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    vec_env = VecMonitor(make_vec_env("FlightDodge-v0", n_envs=8, seed=42))

    # Resume from saved model
    model = PPO.load(str(MODEL_PATH), env=vec_env)
    steps_already = 300_000
    print(f"Resuming from {MODEL_PATH}.zip ({steps_already:,} steps already done)")

    for round_idx in range(ROUNDS):
        steps_done = steps_already + round_idx * STEPS_PER_ROUND
        print(f"\n--- Round {round_idx + 1}/{ROUNDS}  ({steps_done:,} steps so far) ---")
        model.learn(total_timesteps=STEPS_PER_ROUND, reset_num_timesteps=False)

        total_steps = steps_done + STEPS_PER_ROUND
        gif_path = GIF_DIR / f"round_{total_steps // 1000}k_steps.gif"
        print(f"Capturing eval episode → {gif_path}")
        frames = capture_episode(model, seed=round_idx)
        save_gif(frames, gif_path)
        print(f"  survived {len(frames)} frames")

    model.save(str(MODEL_PATH))
    print(f"\nDone. Model saved to {MODEL_PATH}.zip  (total ~1M steps)")


if __name__ == "__main__":
    main()
