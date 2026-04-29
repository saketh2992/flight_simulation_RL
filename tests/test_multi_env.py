"""Smoke tests for the multi-agent env and the SingleAgentWrapper."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from flight_sim.multi_env import (  # noqa: E402
    AIRCRAFT_OBS_DIM,
    MultiFlightDodgeEnv,
    TURRET_OBS_DIM,
    heuristic_aircraft_policy,
    heuristic_turret_policy,
)
from flight_sim.wrappers import SingleAgentWrapper  # noqa: E402


def test_multi_reset_and_step_shapes():
    env = MultiFlightDodgeEnv()
    obs, _ = env.reset(seed=0)
    assert obs["aircraft"].shape == (AIRCRAFT_OBS_DIM,)
    assert obs["turret"].shape == (TURRET_OBS_DIM,)

    actions = {
        "aircraft": np.zeros(2, dtype=np.float32),
        "turret": np.array([0.0, 1.0], dtype=np.float32),
    }
    obs, rewards, terms, truncs, infos = env.step(actions)
    assert obs["aircraft"].shape == (AIRCRAFT_OBS_DIM,)
    assert obs["turret"].shape == (TURRET_OBS_DIM,)
    assert "aircraft" in rewards and "turret" in rewards
    assert isinstance(terms["aircraft"], bool)


def test_zero_sum_rewards_when_no_fire():
    env = MultiFlightDodgeEnv()
    env.reset(seed=0)
    actions = {
        "aircraft": np.zeros(2, dtype=np.float32),
        "turret": np.array([0.0, -1.0], dtype=np.float32),  # never fire
    }
    _, rewards, _, _, infos = env.step(actions)
    assert infos["aircraft"]["fired"] is False
    # Without firing the rewards must be exactly opposite.
    assert abs(rewards["aircraft"] + rewards["turret"]) < 1e-6


def test_heuristic_vs_heuristic_terminates():
    env = MultiFlightDodgeEnv()
    obs, _ = env.reset(seed=2)
    for _ in range(2000):
        actions = {
            "aircraft": heuristic_aircraft_policy(obs["aircraft"]),
            "turret": heuristic_turret_policy(obs["turret"]),
        }
        obs, _, terms, truncs, _ = env.step(actions)
        if terms["aircraft"] or truncs["aircraft"]:
            return
    raise AssertionError("Heuristic-vs-heuristic episode never ended")


def test_single_agent_wrapper_aircraft():
    env = SingleAgentWrapper(controlled="aircraft")
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    obs, reward, term, trunc, _ = env.step(np.zeros(2, dtype=np.float32))
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(term, bool)


def test_single_agent_wrapper_turret():
    env = SingleAgentWrapper(controlled="turret")
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    # Always fire upward; should run for several steps before the heuristic
    # aircraft survives or dies.
    action = np.array([0.0, 1.0], dtype=np.float32)
    for _ in range(50):
        obs, reward, term, trunc, _ = env.step(action)
        assert obs.shape == env.observation_space.shape
        if term or trunc:
            return


if __name__ == "__main__":
    test_multi_reset_and_step_shapes()
    test_zero_sum_rewards_when_no_fire()
    test_heuristic_vs_heuristic_terminates()
    test_single_agent_wrapper_aircraft()
    test_single_agent_wrapper_turret()
    print("All multi-agent smoke tests passed.")
