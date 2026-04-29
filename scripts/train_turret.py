"""Freeze the best aircraft policy and train the turret against it."""

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
from flight_sim.multi_env import MultiFlightDodgeEnv
from flight_sim.wrappers import SingleAgentWrapper, policy_from_sb3

AIRCRAFT_MODEL = Path("models/ppo_flight_dodge")
TURRET_MODEL   = Path("models/ppo_turret")
GIF_DIR        = Path("gifs/turret_training")
STEPS_TOTAL    = 5_000_000


def make_aircraft_policy():
    model = PPO.load(str(AIRCRAFT_MODEL))
    return policy_from_sb3(model)


def capture_matchup(aircraft_policy, turret_model, seed: int, max_steps=600):
    turret_policy = policy_from_sb3(turret_model)
    env = MultiFlightDodgeEnv(render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    frames = []
    for _ in range(max_steps):
        actions = {
            "aircraft": aircraft_policy(obs["aircraft"]),
            "turret":   turret_policy(obs["turret"]),
        }
        obs, _, terms, truncs, _ = env.step(actions)
        frames.append(Image.fromarray(env.render()))
        if terms["aircraft"] or truncs["aircraft"]:
            break
    env.close()
    return frames


def save_gif(frames, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=50, loop=0)


class GifCallback(BaseCallback):
    def __init__(self, save_every: int, aircraft_policy, gif_dir: Path):
        super().__init__()
        self.save_every      = save_every
        self.aircraft_policy = aircraft_policy
        self.gif_dir         = gif_dir
        self._next_save      = save_every

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_save:
            label = f"{self.num_timesteps // 1000}k"
            print(f"\n[{label}] capturing matchup gif...", flush=True)
            frames = capture_matchup(self.aircraft_policy, self.model, seed=self.num_timesteps)
            survived = len(frames)
            path = self.gif_dir / f"{label}_aircraft_survived_{survived}frames.gif"
            save_gif(frames, path)
            print(f"  aircraft survived {survived} frames → {path.name}", flush=True)
            self._next_save += self.save_every
        return True


def main():
    GIF_DIR.mkdir(parents=True, exist_ok=True)
    TURRET_MODEL.parent.mkdir(parents=True, exist_ok=True)

    aircraft_policy = make_aircraft_policy()
    print(f"Loaded aircraft policy from {AIRCRAFT_MODEL}.zip (frozen)")

    def env_factory():
        return SingleAgentWrapper(controlled="turret", opponent_policy=aircraft_policy)

    vec_env = VecMonitor(make_vec_env(env_factory, n_envs=8, seed=0))

    turret_model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=3e-4, n_steps=1024, batch_size=256,
        gae_lambda=0.95, gamma=0.99, ent_coef=0.005, clip_range=0.2,
        verbose=0, seed=0,
    )

    print(f"Training turret for {STEPS_TOTAL:,} steps...\n")
    callback = GifCallback(save_every=500_000, aircraft_policy=aircraft_policy, gif_dir=GIF_DIR)
    turret_model.learn(total_timesteps=STEPS_TOTAL, callback=callback,
                       reset_num_timesteps=True, progress_bar=True)

    turret_model.save(str(TURRET_MODEL))
    print(f"\nDone. Turret model saved to {TURRET_MODEL}.zip")


if __name__ == "__main__":
    main()
