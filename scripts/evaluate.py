"""Watch a trained PPO policy fly in the simulator."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pygame  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from flight_sim.env import FlightDodgeEnv  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/ppo_flight_dodge.zip"))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    model = PPO.load(str(args.model))
    env = FlightDodgeEnv(render_mode="human")

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        total = 0.0
        steps = 0
        done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, term, trunc, _ = env.step(action)
            total += reward
            steps += 1
            env.render()
            if env._renderer is not None:
                env._renderer.draw(
                    aircraft=env._aircraft,
                    turret=env._turret,
                    projectiles=[p for p in env._projectiles if p.alive],
                    info_lines=[
                        f"ep {ep + 1}/{args.episodes}",
                        f"steps: {steps}",
                        f"reward: {total:+.2f}",
                    ],
                )
            done = term or trunc
        print(f"Episode {ep + 1}: steps={steps} reward={total:.2f}")
        time.sleep(0.5)
    env.close()


if __name__ == "__main__":
    main()
