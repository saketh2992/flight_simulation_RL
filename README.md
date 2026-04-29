# Flight Simulation RL

A small open-source 2D flight simulator built as a Gymnasium environment, plus
PPO training scripts. The setup: a stationary turret on the ground tracks an
aircraft and fires projectiles. The aircraft is controlled by a reinforcement
learning policy that has to dodge them for as long as possible.

This is intentionally minimal — pygame for rendering, NumPy for physics,
Gymnasium for the env API, and Stable-Baselines3 (PPO) for the agent. No
proprietary game engine required.

## Layout

```
flight_sim/
  env.py        # Gymnasium env: physics, turret AI, reward
  render.py     # pygame renderer
scripts/
  play_human.py    # Fly manually with arrow keys
  random_agent.py  # Smoke-test the env
  train.py         # Train PPO
  evaluate.py      # Watch a trained policy fly
tests/
  test_env.py      # Reset/step shape & termination tests
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

Smoke-test (no RL libs required if you skip `train.py` / `evaluate.py`):

```bash
python -m tests.test_env
python scripts/random_agent.py --episodes 3
```

Fly it yourself with the arrow keys:

```bash
python scripts/play_human.py
```

Train PPO and then watch it dodge:

```bash
python scripts/train.py --steps 500000
python scripts/evaluate.py --model models/ppo_flight_dodge.zip
```

Tensorboard logs land in `tensorboard/` by default:

```bash
tensorboard --logdir tensorboard/
```

## Environment details

| Field          | Value                                                 |
| -------------- | ----------------------------------------------------- |
| Action space   | `Box(-1, 1, shape=(2,))` — `[ax, ay]` thrust          |
| Observation    | `Box(shape=(5 + 4 * MAX_PROJECTILES,))`               |
| World          | 200 × 100 (units), turret on the ground (`y = 0`)     |
| Turret AI      | Leads the target, fires every 0.6 s with aim noise    |
| Reward         | `+0.01/step + 0.001*altitude − 10` on hit, `−1` OOB   |
| Episode length | 2000 steps max (≈ 200 simulated seconds at `dt=0.1`)  |

The observation packs the aircraft state, the relative turret position, and the
`MAX_PROJECTILES` nearest live projectiles (relative position + velocity each).

## Roadmap

- 2D dodging baseline (this repo) ← you are here
- Multiple turrets / moving turrets
- Continuous fuel + altitude penalties for more interesting flight dynamics
- 3D extension once the 2D baseline is solid
