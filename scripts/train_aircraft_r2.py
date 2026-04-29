"""Resume aircraft training against the frozen RL turret (self-play round 2)."""

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
GIF_DIR        = Path("gifs/aircraft_r2")
STEPS_TOTAL    = 5_000_000


def make_turret_policy():
    model = PPO.load(str(TURRET_MODEL))
    return policy_from_sb3(model)


def capture_matchup(aircraft_model, turret_policy, seed: int, max_steps=600):
    aircraft_policy = policy_from_sb3(aircraft_model)
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
    def __init__(self, save_every: int, turret_policy, gif_dir: Path):
        super().__init__()
        self.save_every    = save_every
        self.turret_policy = turret_policy
        self.gif_dir       = gif_dir
        self._next_save    = save_every

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_save:
            label = f"{self.num_timesteps // 1000}k"
            print(f"\n[{label}] capturing matchup gif...", flush=True)
            frames = capture_matchup(self.model, self.turret_policy, seed=self.num_timesteps)
            survived = len(frames)
            path = self.gif_dir / f"{label}_survived_{survived}frames.gif"
            save_gif(frames, path)
            print(f"  aircraft survived {survived} frames → {path.name}", flush=True)
            self._next_save += self.save_every
        return True


def main():
    GIF_DIR.mkdir(parents=True, exist_ok=True)

    turret_policy = make_turret_policy()
    print(f"Loaded turret policy from {TURRET_MODEL}.zip (frozen)")

    def env_factory():
        return SingleAgentWrapper(controlled="aircraft", opponent_policy=turret_policy)

    vec_env = VecMonitor(make_vec_env(env_factory, n_envs=8, seed=1))

    print(f"Resuming aircraft from {AIRCRAFT_MODEL}.zip")
    aircraft_model = PPO.load(str(AIRCRAFT_MODEL), env=vec_env)

    print(f"Training aircraft for {STEPS_TOTAL:,} steps against frozen RL turret...\n")
    callback = GifCallback(save_every=500_000, turret_policy=turret_policy, gif_dir=GIF_DIR)
    aircraft_model.learn(total_timesteps=STEPS_TOTAL, callback=callback,
                         reset_num_timesteps=True, progress_bar=True)

    aircraft_model.save(str(AIRCRAFT_MODEL))
    print(f"\nDone. Aircraft model saved to {AIRCRAFT_MODEL}.zip")


if __name__ == "__main__":
    main()
