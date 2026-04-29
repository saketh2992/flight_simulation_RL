"""Train a PPO policy to dodge ground-fired projectiles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402
from stable_baselines3.common.vec_env import VecMonitor  # noqa: E402

import flight_sim  # noqa: E402,F401  (registers the env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("models/ppo_flight_dodge"))
    parser.add_argument("--tb", type=Path, default=Path("tensorboard/"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.tb.mkdir(parents=True, exist_ok=True)

    env = VecMonitor(make_vec_env("FlightDodge-v0", n_envs=args.n_envs, seed=args.seed))

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        gae_lambda=0.95,
        gamma=0.99,
        ent_coef=0.005,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=str(args.tb),
        seed=args.seed,
    )

    model.learn(total_timesteps=args.steps, progress_bar=True)
    model.save(str(args.out))
    print(f"Saved policy to {args.out}.zip")


if __name__ == "__main__":
    main()
