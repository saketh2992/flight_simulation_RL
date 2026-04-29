"""Run alternating self-play and capture a GIF after every round."""

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
from flight_sim.multi_env import MultiFlightDodgeEnv, heuristic_turret_policy, heuristic_aircraft_policy
from flight_sim.wrappers import SingleAgentWrapper, policy_from_sb3

STEPS_PER_ROUND = 150_000
ROUNDS = 6
GIF_DIR = Path("gifs/selfplay")
MODEL_DIR = Path("models/selfplay")


# ── helpers ────────────────────────────────────────────────────────────────

def make_sb3_policy(path: Path):
    model = PPO.load(str(path))
    def policy(obs):
        action, _ = model.predict(np.asarray(obs), deterministic=True)
        return np.asarray(action, dtype=np.float32)
    return policy


def capture_matchup(aircraft_policy, turret_policy, seed: int, max_steps: int = 500) -> list[Image.Image]:
    env = MultiFlightDodgeEnv(render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    frames = []
    for _ in range(max_steps):
        actions = {
            "aircraft": aircraft_policy(obs["aircraft"]),
            "turret":   turret_policy(obs["turret"]),
        }
        obs, _, terms, truncs, _ = env.step(actions)
        frame = env.render()
        frames.append(Image.fromarray(frame))
        if terms["aircraft"] or truncs["aircraft"]:
            break
    env.close()
    return frames


def save_gif(frames: list[Image.Image], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=50, loop=0)


def make_env_factory(controlled: str, opponent_policy):
    def _factory():
        return SingleAgentWrapper(controlled=controlled, opponent_policy=opponent_policy)
    return _factory


def train_round(controlled, opponent_policy, warm_start, seed):
    env = VecMonitor(make_vec_env(make_env_factory(controlled, opponent_policy), n_envs=8, seed=seed))
    if warm_start is not None and warm_start.exists():
        model = PPO.load(str(warm_start), env=env)
    else:
        model = PPO("MlpPolicy", env=env,
                    learning_rate=3e-4, n_steps=1024, batch_size=256,
                    gae_lambda=0.95, gamma=0.99, ent_coef=0.005,
                    clip_range=0.2, verbose=0, seed=seed)
    model.learn(total_timesteps=STEPS_PER_ROUND, reset_num_timesteps=(warm_start is None))
    env.close()
    return model


# ── main ───────────────────────────────────────────────────────────────────

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    GIF_DIR.mkdir(parents=True, exist_ok=True)

    aircraft_path: Path | None = None
    turret_path:   Path | None = None

    for r in range(ROUNDS):
        if r % 2 == 0:
            controlled = "aircraft"
            opp_policy = make_sb3_policy(turret_path) if turret_path else heuristic_turret_policy
            warm = aircraft_path
        else:
            controlled = "turret"
            opp_policy = make_sb3_policy(aircraft_path)
            warm = turret_path

        print(f"\n=== Round {r} — training {controlled} ({STEPS_PER_ROUND:,} steps) ===")
        model = train_round(controlled, opp_policy, warm, seed=42 + r)

        save_path = MODEL_DIR / f"iter_{r:02d}_{controlled}.zip"
        model.save(str(save_path))
        if controlled == "aircraft":
            aircraft_path = save_path
        else:
            turret_path = save_path

        # Capture matchup with best available policies
        ac_pi = make_sb3_policy(aircraft_path) if aircraft_path else heuristic_aircraft_policy
        tu_pi = make_sb3_policy(turret_path)   if turret_path   else heuristic_turret_policy
        frames = capture_matchup(ac_pi, tu_pi, seed=r)
        label = "ac" if controlled == "aircraft" else "tu"
        gif_path = GIF_DIR / f"round_{r:02d}_{controlled}_{len(frames)}frames.gif"
        save_gif(frames, gif_path)
        print(f"  Aircraft survived {len(frames)} frames → {gif_path.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
