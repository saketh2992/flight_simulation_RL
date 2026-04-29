"""Lightweight smoke tests for the FlightDodge environment."""

from __future__ import annotations

import numpy as np

from flight_sim.env import FlightDodgeEnv, MAX_PROJECTILES


def test_reset_returns_correct_shape():
    env = FlightDodgeEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (5 + 4 * MAX_PROJECTILES,)
    assert obs.dtype == np.float32
    assert isinstance(info, dict)


def test_step_returns_5_tuple():
    env = FlightDodgeEnv()
    env.reset(seed=0)
    out = env.step(np.zeros(2, dtype=np.float32))
    assert len(out) == 5
    obs, reward, term, trunc, info = out
    assert obs.shape == (5 + 4 * MAX_PROJECTILES,)
    assert isinstance(reward, float)
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)


def test_episode_eventually_terminates():
    env = FlightDodgeEnv()
    env.reset(seed=1)
    # Hold full down-thrust — should eventually crash into the floor or get hit.
    action = np.array([0.0, -1.0], dtype=np.float32)
    for _ in range(2000):
        _, _, term, trunc, _ = env.step(action)
        if term or trunc:
            return
    raise AssertionError("Episode never ended after 2000 steps with downward thrust")


if __name__ == "__main__":
    test_reset_returns_correct_shape()
    test_step_returns_5_tuple()
    test_episode_eventually_terminates()
    print("All smoke tests passed.")
