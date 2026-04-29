"""Single-agent gym.Env wrapper around MultiFlightDodgeEnv.

Use this when you want to train one side at a time with Stable-Baselines3:
the opponent's actions come from a frozen callable (e.g. a saved PPO policy
or the built-in heuristic).
"""

from __future__ import annotations

from typing import Callable, Optional

import gymnasium as gym
import numpy as np

from flight_sim.multi_env import (
    MultiFlightDodgeEnv,
    heuristic_aircraft_policy,
    heuristic_turret_policy,
)


PolicyFn = Callable[[np.ndarray], np.ndarray]


def _default_opponent(agent: str) -> PolicyFn:
    if agent == "turret":
        return heuristic_turret_policy
    return heuristic_aircraft_policy


class SingleAgentWrapper(gym.Env):
    """Expose one of the two agents as a standard single-agent gym env."""

    metadata = MultiFlightDodgeEnv.metadata

    def __init__(
        self,
        controlled: str = "aircraft",
        opponent_policy: Optional[PolicyFn] = None,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        if controlled not in ("aircraft", "turret"):
            raise ValueError(f"controlled must be 'aircraft' or 'turret', got {controlled!r}")
        super().__init__()
        self.multi = MultiFlightDodgeEnv(render_mode=render_mode, seed=seed)
        self.controlled = controlled
        self.opp = "turret" if controlled == "aircraft" else "aircraft"
        self.opp_policy = opponent_policy or _default_opponent(self.opp)
        self.observation_space = self.multi.observation_spaces[controlled]
        self.action_space = self.multi.action_spaces[controlled]
        self._last_opp_obs: Optional[np.ndarray] = None

    def reset(self, *, seed: Optional[int] = None, options=None):
        obs, info = self.multi.reset(seed=seed)
        self._last_opp_obs = obs[self.opp]
        return obs[self.controlled], info.get(self.controlled, {}) if isinstance(info, dict) else {}

    def step(self, action):
        opp_action = self.opp_policy(self._last_opp_obs)
        actions = {self.controlled: np.asarray(action), self.opp: np.asarray(opp_action)}
        obs, rewards, terms, truncs, infos = self.multi.step(actions)
        self._last_opp_obs = obs[self.opp]
        terminated = bool(terms[self.controlled] or terms[self.opp])
        truncated = bool(truncs[self.controlled] or truncs[self.opp])
        return (
            obs[self.controlled],
            float(rewards[self.controlled]),
            terminated,
            truncated,
            infos.get(self.controlled, {}),
        )

    def render(self):
        return self.multi.render()

    def close(self):
        self.multi.close()


def policy_from_sb3(model) -> PolicyFn:
    """Wrap a Stable-Baselines3 model as an opponent policy callable."""

    def policy(obs: np.ndarray) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=False)
        return np.asarray(action, dtype=np.float32)

    return policy
