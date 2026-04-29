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
  env.py         # Single-agent Gymnasium env (scripted turret)
  multi_env.py   # Two-agent env: aircraft AND turret are trainable
  wrappers.py    # SingleAgentWrapper for self-play training with SB3
  render.py      # pygame renderer
scripts/
  play_human.py        # Fly manually with arrow keys
  random_agent.py      # Smoke-test the single-agent env
  train.py             # Train PPO aircraft vs scripted turret
  evaluate.py          # Watch a trained aircraft policy fly
  train_selfplay.py    # Co-evolve aircraft & turret via alternating self-play
  evaluate_multi.py    # Watch any aircraft policy vs any turret policy
tests/
  test_env.py          # Single-agent reset/step/termination tests
  test_multi_env.py    # Multi-agent + wrapper smoke tests
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

## Multi-agent self-play (turret as a second RL agent)

In the original env the turret is hard-coded — it computes a lead-aim angle
analytically and fires every 0.6 s. `multi_env.py` makes the turret a real RL
agent with its own action and observation:

| Field          | Aircraft                       | Turret                                   |
| -------------- | ------------------------------ | ---------------------------------------- |
| Action         | `Box(2,)` thrust `[ax, ay]`    | `Box(2,)` `[aim, fire]`                  |
| Aim            | —                              | `aim ∈ [-1, 1]` → barrel angle in `[0.05π, 0.95π]`, slewed at 4 rad/s |
| Fire           | —                              | `fire > 0` requests a shot (cooldown 0.5 s) |
| Observation    | aircraft state + 6 nearest projectiles | turret pose + relative aircraft + aim + cooldown |
| Reward         | per-step survival + altitude − proximity − hit penalty | **negative** of aircraft's reward, minus a small per-fire cost |

Training is alternating self-play: freeze one side, train the other with PPO,
swap, repeat.

```bash
python scripts/train_selfplay.py --rounds 4 --steps-per-round 200000
```

This produces `models/selfplay/iter_00_aircraft.zip`,
`iter_01_turret.zip`, `iter_02_aircraft.zip`, … one per round.

Replay any matchup:

```bash
# Two scripted policies (no models needed)
python scripts/evaluate_multi.py

# Trained aircraft from round 0 vs scripted turret
python scripts/evaluate_multi.py --aircraft models/selfplay/iter_00_aircraft.zip

# Final co-evolved matchup
python scripts/evaluate_multi.py \
  --aircraft models/selfplay/iter_02_aircraft.zip \
  --turret   models/selfplay/iter_03_turret.zip
```

## Roadmap

- 2D dodging baseline ✅
- Trainable turret + alternating self-play ✅ ← you are here
- Population-based self-play (sample opponent from a pool of past policies)
- Multiple turrets / moving turrets
- Continuous fuel + altitude penalties for richer flight dynamics
- 3D extension once the 2D baseline is solid
