"""Train aircraft vs scripted turret for a long run, saving a GIF every 500k steps."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor

import flight_sim  # noqa: F401
from flight_sim.env import FlightDodgeEnv

MODEL_PATH   = Path("models/ppo_flight_dodge")
GIF_DIR      = Path("gifs/long_run")
STEPS_RESUME = 1_000_000   # already trained
STEPS_TOTAL  = 10_000_000  # train 9M more


def capture_gif(model, seed: int, max_steps: int = 600) -> tuple[list[Image.Image], int]:
    env = FlightDodgeEnv(render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    frames = []
    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        frames.append(Image.fromarray(env.render()))
        if term or trunc:
            break
    env.close()
    return frames, len(frames)


def save_gif(frames: list[Image.Image], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=50, loop=0)


class GifCallback(BaseCallback):
    def __init__(self, save_every: int, steps_offset: int):
        super().__init__()
        self.save_every   = save_every
        self.steps_offset = steps_offset
        self._next_save   = save_every

    def _on_step(self) -> bool:
        total = self.num_timesteps + self.steps_offset
        if self.num_timesteps >= self._next_save:
            label = f"{total // 1000}k"
            print(f"\n[{label}] capturing eval gif...", flush=True)
            frames, survived = capture_gif(self.model, seed=total)
            path = GIF_DIR / f"{label}_steps_{survived}frames.gif"
            save_gif(frames, path)
            print(f"  survived {survived} frames → {path.name}", flush=True)
            self._next_save += self.save_every
        return True


def main():
    GIF_DIR.mkdir(parents=True, exist_ok=True)

    vec_env = VecMonitor(make_vec_env("FlightDodge-v0", n_envs=8, seed=0))

    print(f"Loading {MODEL_PATH}.zip  (resuming from ~{STEPS_RESUME:,} steps)")
    model = PPO.load(str(MODEL_PATH), env=vec_env)

    remaining = STEPS_TOTAL - STEPS_RESUME
    print(f"Training {remaining:,} more steps (target {STEPS_TOTAL:,} total)\n")

    callback = GifCallback(save_every=500_000, steps_offset=STEPS_RESUME)
    model.learn(total_timesteps=remaining, callback=callback,
                reset_num_timesteps=False, progress_bar=True)

    model.save(str(MODEL_PATH))
    print(f"\nDone. Model saved to {MODEL_PATH}.zip")


if __name__ == "__main__":
    main()
