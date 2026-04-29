"""Manually fly the aircraft with the arrow keys to test the simulator."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pygame  # noqa: E402

from flight_sim.env import FlightDodgeEnv  # noqa: E402


def main():
    env = FlightDodgeEnv(render_mode="human")
    obs, _ = env.reset(seed=0)
    done = False
    total_reward = 0.0
    steps = 0

    while True:
        # Read keyboard.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                sys.exit(0)
        keys = pygame.key.get_pressed()
        ax = (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT])
        ay = (keys[pygame.K_UP] - keys[pygame.K_DOWN])
        action = np.array([ax, ay], dtype=np.float32)

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1

        # Hook a HUD into the renderer.
        env.render()
        if env._renderer is not None:
            env._renderer.draw(
                aircraft=env._aircraft,
                turret=env._turret,
                projectiles=[p for p in env._projectiles if p.alive],
                info_lines=[
                    f"steps: {steps}",
                    f"reward: {total_reward:+.2f}",
                    "arrows = thrust   esc = quit",
                ],
            )

        if keys[pygame.K_ESCAPE]:
            break
        if terminated or truncated:
            print(f"Episode ended — steps={steps} reward={total_reward:.2f}")
            obs, _ = env.reset()
            total_reward = 0.0
            steps = 0

    env.close()


if __name__ == "__main__":
    main()
