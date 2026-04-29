"""Co-evolve aircraft & turret policies via alternating self-play.

Round 0: train aircraft against the scripted (heuristic) turret.
Round 1: train turret  against the frozen aircraft from round 0.
Round 2: train aircraft against the frozen turret  from round 1.
... and so on.

Both policies are saved each round to `models/selfplay/iter_<i>_<agent>.zip`,
so you can replay any matchup with `scripts/evaluate_multi.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402
from stable_baselines3.common.vec_env import VecMonitor  # noqa: E402

from flight_sim.wrappers import SingleAgentWrapper, policy_from_sb3  # noqa: E402


def make_env_factory(controlled: str, opponent_policy):
    def _factory():
        return SingleAgentWrapper(controlled=controlled, opponent_policy=opponent_policy)

    return _factory


def train_round(
    controlled: str,
    opponent_policy,
    steps: int,
    n_envs: int,
    save_path: Path,
    tb_log: Path,
    seed: int,
    warm_start: Path | None,
) -> Path:
    env = VecMonitor(make_vec_env(make_env_factory(controlled, opponent_policy), n_envs=n_envs, seed=seed))

    if warm_start is not None and warm_start.exists():
        print(f"  Warm-starting {controlled} from {warm_start}")
        model = PPO.load(str(warm_start), env=env)
    else:
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
            verbose=0,
            tensorboard_log=str(tb_log),
            seed=seed,
        )

    model.learn(total_timesteps=steps, progress_bar=True)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    env.close()
    return save_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--steps-per-round", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=Path("models/selfplay"))
    parser.add_argument("--tb", type=Path, default=Path("tensorboard/selfplay"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warm-start", action="store_true",
                        help="Each round, warm-start from the previous policy of the same agent.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.tb.mkdir(parents=True, exist_ok=True)

    aircraft_path: Path | None = None
    turret_path: Path | None = None

    for r in range(args.rounds):
        # Even rounds train the aircraft; odd rounds train the turret.
        if r % 2 == 0:
            controlled = "aircraft"
            opponent_policy = None  # default heuristic
            if turret_path is not None:
                turret_model = PPO.load(str(turret_path))
                opponent_policy = policy_from_sb3(turret_model)
            warm = aircraft_path if args.warm_start else None
        else:
            controlled = "turret"
            aircraft_model = PPO.load(str(aircraft_path))
            opponent_policy = policy_from_sb3(aircraft_model)
            warm = turret_path if args.warm_start else None

        save_path = args.out_dir / f"iter_{r:02d}_{controlled}.zip"
        print(f"\n=== Round {r}: training {controlled} for {args.steps_per_round} steps ===")
        train_round(
            controlled=controlled,
            opponent_policy=opponent_policy,
            steps=args.steps_per_round,
            n_envs=args.n_envs,
            save_path=save_path,
            tb_log=args.tb,
            seed=args.seed + r,
            warm_start=warm,
        )
        if controlled == "aircraft":
            aircraft_path = save_path
        else:
            turret_path = save_path
        print(f"  Saved → {save_path}")

    print("\nDone. Latest checkpoints:")
    print(f"  aircraft : {aircraft_path}")
    print(f"  turret   : {turret_path}")


if __name__ == "__main__":
    main()
