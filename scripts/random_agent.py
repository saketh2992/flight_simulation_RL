"""Sanity-check the env with a random-action agent (no RL deps required)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import flight_sim  # noqa: E402,F401
from flight_sim.env import FlightDodgeEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    env = FlightDodgeEnv(render_mode="human" if args.render else None)
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        total = 0.0
        steps = 0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, term, trunc, _ = env.step(action)
            total += reward
            steps += 1
            if args.render:
                env.render()
            done = term or trunc
        print(f"Episode {ep + 1}: steps={steps} reward={total:.2f}")
    env.close()


if __name__ == "__main__":
    main()
