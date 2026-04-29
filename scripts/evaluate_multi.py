"""Watch any pair of policies play against each other.

Either side can be a saved PPO checkpoint or the built-in heuristic. Pass
`heuristic` (or omit) for the scripted policy.

Examples:
    # Scripted turret vs scripted aircraft (no models needed):
    python scripts/evaluate_multi.py

    # Trained aircraft (round 0) vs scripted turret:
    python scripts/evaluate_multi.py --aircraft models/selfplay/iter_00_aircraft.zip

    # Final co-evolved matchup:
    python scripts/evaluate_multi.py \\
        --aircraft models/selfplay/iter_02_aircraft.zip \\
        --turret   models/selfplay/iter_03_turret.zip
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from flight_sim.multi_env import (  # noqa: E402
    MultiFlightDodgeEnv,
    heuristic_aircraft_policy,
    heuristic_turret_policy,
)


def load_policy(name_or_path: str, agent: str):
    if name_or_path in ("heuristic", "scripted", ""):
        return heuristic_aircraft_policy if agent == "aircraft" else heuristic_turret_policy
    from stable_baselines3 import PPO

    model = PPO.load(name_or_path)
    deterministic = False

    def policy(obs):
        action, _ = model.predict(obs, deterministic=deterministic)
        return np.asarray(action, dtype=np.float32)

    return policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aircraft", default="heuristic")
    parser.add_argument("--turret", default="heuristic")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    aircraft_pi = load_policy(args.aircraft, "aircraft")
    turret_pi = load_policy(args.turret, "turret")

    pygame = None
    if not args.no_render:
        import pygame as _pygame  # noqa: F401

        pygame = _pygame

    env = MultiFlightDodgeEnv(render_mode=None if args.no_render else "human")
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        ac_total = tu_total = 0.0
        steps = 0
        done = False
        while not done:
            if pygame is not None:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return
            actions = {
                "aircraft": aircraft_pi(obs["aircraft"]),
                "turret": turret_pi(obs["turret"]),
            }
            obs, rewards, terms, truncs, _ = env.step(actions)
            ac_total += rewards["aircraft"]
            tu_total += rewards["turret"]
            steps += 1
            if not args.no_render:
                env.render()
            done = terms["aircraft"] or truncs["aircraft"]
        print(
            f"Episode {ep + 1}: steps={steps:4d}  "
            f"aircraft={ac_total:+.2f}  turret={tu_total:+.2f}"
        )
        if not args.no_render:
            time.sleep(0.4)
    env.close()


if __name__ == "__main__":
    main()
