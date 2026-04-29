"""Alternating self-play: 20 cycles of (turret 1M → aircraft 1M), starting from selfplay_v2 cycle 3."""

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

# Start from the best checkpoints produced by selfplay_v2
AIRCRAFT_START = Path("models/selfplay_v2/cycle03_aircraft.zip")
TURRET_START   = Path("models/selfplay_v2/cycle03_turret.zip")
CKPT_DIR       = Path("models/selfplay_v3")
GIF_DIR        = Path("gifs/selfplay_v3")
STEPS_PER_ROUND = 1_000_000
CYCLES          = 20


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


def train(controlled: str, active_path: Path, frozen_path: Path, save_path: Path, seed: int):
    frozen_policy = make_policy(frozen_path)

    def env_factory():
        return SingleAgentWrapper(controlled=controlled, opponent_policy=frozen_policy)

    vec_env = VecMonitor(make_vec_env(env_factory, n_envs=8, seed=seed))
    model = PPO.load(str(active_path), env=vec_env)
    model.learn(total_timesteps=STEPS_PER_ROUND, reset_num_timesteps=True, progress_bar=True)
    model.save(str(save_path))
    vec_env.close()
    print(f"  saved → {save_path.name}", flush=True)


def main():
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    GIF_DIR.mkdir(parents=True, exist_ok=True)

    aircraft_path = AIRCRAFT_START
    turret_path   = TURRET_START

    results = []

    for cycle in range(1, CYCLES + 1):
        # ── Train turret ────────────────────────────────────────────────
        print(f"\n{'='*60}", flush=True)
        print(f"  CYCLE {cycle:02d}/{CYCLES}  —  TURRET trains  (aircraft frozen)", flush=True)
        print(f"{'='*60}", flush=True)
        turret_ckpt = CKPT_DIR / f"cycle{cycle:02d}_turret.zip"
        train("turret", turret_path, aircraft_path, turret_ckpt, seed=cycle)
        turret_path = turret_ckpt

        frames = capture_gif(aircraft_path, turret_path, seed=cycle * 100)
        gif_path = GIF_DIR / f"cycle{cycle:02d}_after_turret_{len(frames)}fr.gif"
        save_gif(frames, gif_path)
        results.append((f"C{cycle:02d} turret", len(frames)))
        print(f"  → aircraft survived {len(frames)} frames", flush=True)

        # ── Train aircraft ───────────────────────────────────────────────
        print(f"\n{'='*60}", flush=True)
        print(f"  CYCLE {cycle:02d}/{CYCLES}  —  AIRCRAFT trains  (turret frozen)", flush=True)
        print(f"{'='*60}", flush=True)
        aircraft_ckpt = CKPT_DIR / f"cycle{cycle:02d}_aircraft.zip"
        train("aircraft", aircraft_path, turret_path, aircraft_ckpt, seed=cycle + 1000)
        aircraft_path = aircraft_ckpt

        frames = capture_gif(aircraft_path, turret_path, seed=cycle * 100 + 1)
        gif_path = GIF_DIR / f"cycle{cycle:02d}_after_aircraft_{len(frames)}fr.gif"
        save_gif(frames, gif_path)
        results.append((f"C{cycle:02d} aircraft", len(frames)))
        print(f"  → aircraft survived {len(frames)} frames", flush=True)

        # Running summary every 5 cycles
        if cycle % 5 == 0:
            print(f"\n--- Progress so far ---", flush=True)
            for label, s in results[-10:]:
                print(f"  {label}: {s} frames", flush=True)

    print(f"\n{'='*60}", flush=True)
    print("ALL DONE — final results:", flush=True)
    for label, s in results:
        print(f"  {label}: {s} frames", flush=True)
    print(f"\nCheckpoints: {CKPT_DIR}/", flush=True)
    print(f"GIFs:        {GIF_DIR}/", flush=True)


if __name__ == "__main__":
    main()
