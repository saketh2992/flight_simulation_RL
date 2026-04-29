"""Alternating self-play: freeze one side, train the other for 1M steps. 3 full cycles."""

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
from flight_sim.multi_env import MultiFlightDodgeEnv
from flight_sim.wrappers import SingleAgentWrapper, policy_from_sb3

AIRCRAFT_MODEL = Path("models/ppo_flight_dodge")
TURRET_MODEL   = Path("models/ppo_turret")
CKPT_DIR       = Path("models/selfplay_v2")
GIF_DIR        = Path("gifs/selfplay_v2")
STEPS_PER_ROUND = 1_000_000
CYCLES          = 3          # 3 × (turret + aircraft) = 6 rounds total


def make_policy(path: Path):
    model = PPO.load(str(path))
    return policy_from_sb3(model)


def capture_gif(aircraft_path: Path, turret_path: Path, seed: int, max_steps=600):
    ac_pi = make_policy(aircraft_path)
    tu_pi = make_policy(turret_path)
    env = MultiFlightDodgeEnv(render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    frames = []
    for _ in range(max_steps):
        actions = {"aircraft": ac_pi(obs["aircraft"]), "turret": tu_pi(obs["turret"])}
        obs, _, terms, truncs, _ = env.step(actions)
        frames.append(Image.fromarray(env.render()))
        if terms["aircraft"] or truncs["aircraft"]:
            break
    env.close()
    return frames


def save_gif(frames, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=50, loop=0)


def train(controlled: str, active_path: Path, frozen_path: Path, save_path: Path,
          cycle: int, round_label: str):
    frozen_policy = make_policy(frozen_path)

    def env_factory():
        return SingleAgentWrapper(controlled=controlled, opponent_policy=frozen_policy)

    vec_env = VecMonitor(make_vec_env(env_factory, n_envs=8, seed=cycle))
    model = PPO.load(str(active_path), env=vec_env)
    model.learn(total_timesteps=STEPS_PER_ROUND, reset_num_timesteps=True, progress_bar=True)
    model.save(str(save_path))
    vec_env.close()
    print(f"  Saved → {save_path}")


def main():
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    GIF_DIR.mkdir(parents=True, exist_ok=True)

    aircraft_path = AIRCRAFT_MODEL
    turret_path   = TURRET_MODEL

    for cycle in range(1, CYCLES + 1):
        # ── Train turret ────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  CYCLE {cycle}/{CYCLES}  —  training TURRET (aircraft frozen)")
        print(f"{'='*60}")
        turret_ckpt = CKPT_DIR / f"cycle{cycle:02d}_turret.zip"
        train("turret", turret_path, aircraft_path, turret_ckpt, cycle, f"c{cycle}_turret")
        turret_path = turret_ckpt

        frames = capture_gif(aircraft_path, turret_path, seed=cycle * 100)
        gif_path = GIF_DIR / f"cycle{cycle:02d}_after_turret_{len(frames)}frames.gif"
        save_gif(frames, gif_path)
        print(f"  GIF: aircraft survived {len(frames)} frames → {gif_path.name}")

        # ── Train aircraft ───────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  CYCLE {cycle}/{CYCLES}  —  training AIRCRAFT (turret frozen)")
        print(f"{'='*60}")
        aircraft_ckpt = CKPT_DIR / f"cycle{cycle:02d}_aircraft.zip"
        train("aircraft", aircraft_path, turret_path, aircraft_ckpt, cycle + 100, f"c{cycle}_aircraft")
        aircraft_path = aircraft_ckpt

        frames = capture_gif(aircraft_path, turret_path, seed=cycle * 100 + 1)
        gif_path = GIF_DIR / f"cycle{cycle:02d}_after_aircraft_{len(frames)}frames.gif"
        save_gif(frames, gif_path)
        print(f"  GIF: aircraft survived {len(frames)} frames → {gif_path.name}")

    # Update the main model files with the latest
    import shutil
    shutil.copy(aircraft_path, AIRCRAFT_MODEL.with_stem("ppo_flight_dodge_selfplay"))
    shutil.copy(turret_path,   TURRET_MODEL.with_stem("ppo_turret_selfplay"))
    print(f"\nDone. Final models saved as *_selfplay.zip")
    print(f"All checkpoints in {CKPT_DIR}/")
    print(f"All GIFs in {GIF_DIR}/")


if __name__ == "__main__":
    main()
