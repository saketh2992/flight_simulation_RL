"""Two-agent FlightDodge: both the aircraft AND the ground turret act.

This is the *non-Gym* multi-agent core. It exposes:

  observation_spaces["aircraft"], observation_spaces["turret"]
  action_spaces["aircraft"],      action_spaces["turret"]
  reset()                         -> (obs_dict, info_dict)
  step(actions_dict)              -> (obs, rewards, terms, truncs, infos) dicts

For training with Stable-Baselines3 you wrap this with `SingleAgentWrapper`
in flight_sim.wrappers, which fixes the opponent to a frozen policy and
exposes a normal gym.Env.

Action spaces
-------------
aircraft : Box(2,)  [ax, ay]   — thrust, same as FlightDodgeEnv
turret   : Box(2,)  [aim, fire]
    aim  ∈ [-1, 1] → desired barrel angle, mapped linearly to
                     [TURRET_AIM_MIN, TURRET_AIM_MAX] (mostly upward arc).
    fire ∈ [-1, 1] → fire request (>0). Shot only launches when the
                     reload cooldown has expired.

Reward scheme (zero-sum + small efficiency cost)
------------------------------------------------
aircraft_reward  : same dense reward used by FlightDodgeEnv.
turret_reward    : -aircraft_reward - TURRET_FIRE_COST when it fired.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from gymnasium import spaces

from flight_sim.env import (
    AIRCRAFT_ACCEL,
    AIRCRAFT_DRAG,
    AIRCRAFT_MAX_SPEED,
    AIRCRAFT_RADIUS,
    Aircraft,
    CEIL_Y,
    DANGER_RADIUS,
    DT,
    FLOOR_Y,
    MAX_PROJECTILES,
    PROJECTILE_LIFETIME,
    PROJECTILE_RADIUS,
    PROJECTILE_SPEED,
    Projectile,
    Turret,
    WORLD_H,
    WORLD_W,
)


TURRET_AIM_MIN = 0.05 * np.pi  # ≈ 9°  (just above the horizon, right side)
TURRET_AIM_MAX = 0.95 * np.pi  # ≈ 171° (just above the horizon, left side)
TURRET_AIM_RATE = 4.0  # rad / s — how fast the barrel can slew
TURRET_FIRE_COOLDOWN = 0.5  # s minimum reload after a shot
TURRET_FIRE_COST = 0.02  # per-fire reward penalty (efficiency)

AIRCRAFT_OBS_DIM = 5 + 4 * MAX_PROJECTILES
TURRET_OBS_DIM = 8


def _aim_action_to_angle(a: float) -> float:
    """Map a [-1, 1] aim action to an absolute aim angle in radians."""
    a = float(np.clip(a, -1.0, 1.0))
    return TURRET_AIM_MIN + (a + 1.0) * 0.5 * (TURRET_AIM_MAX - TURRET_AIM_MIN)


def _angle_to_aim_action(angle: float) -> float:
    """Inverse mapping (used by the heuristic opponent)."""
    angle = float(np.clip(angle, TURRET_AIM_MIN, TURRET_AIM_MAX))
    return 2.0 * (angle - TURRET_AIM_MIN) / (TURRET_AIM_MAX - TURRET_AIM_MIN) - 1.0


class MultiFlightDodgeEnv:
    """Two-agent flight-vs-turret environment."""

    AGENTS = ("aircraft", "turret")
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: Optional[str] = None, seed: Optional[int] = None):
        self.render_mode = render_mode

        self.observation_spaces = {
            "aircraft": spaces.Box(-2.0, 2.0, (AIRCRAFT_OBS_DIM,), np.float32),
            "turret": spaces.Box(-2.0, 2.0, (TURRET_OBS_DIM,), np.float32),
        }
        self.action_spaces = {
            "aircraft": spaces.Box(-1.0, 1.0, (2,), np.float32),
            "turret": spaces.Box(-1.0, 1.0, (2,), np.float32),
        }

        self._np_random = np.random.default_rng(seed)
        self._aircraft = Aircraft(0.0, 0.0)
        self._turret = Turret(0.0)
        self._projectiles: list[Projectile] = []
        self._steps = 0
        self._renderer = None

        if seed is not None:
            self.reset(seed=seed)

    # ------------------------------------------------------------------ core
    def reset(self, *, seed: Optional[int] = None, options=None):
        self._np_random = np.random.default_rng(seed)
        self._aircraft = Aircraft(
            x=float(self._np_random.uniform(-WORLD_W * 0.4, WORLD_W * 0.4)),
            y=float(self._np_random.uniform(WORLD_H * 0.5, WORLD_H * 0.85)),
        )
        self._turret = Turret(
            x=float(self._np_random.uniform(-WORLD_W * 0.4, WORLD_W * 0.4)),
            cooldown=TURRET_FIRE_COOLDOWN,
            aim=float(np.pi / 2),
        )
        self._projectiles = []
        self._steps = 0
        return self._observations(), {}

    def step(self, actions):
        ac_action = np.asarray(actions["aircraft"], dtype=np.float32).clip(-1.0, 1.0)
        tu_action = np.asarray(actions["turret"], dtype=np.float32).clip(-1.0, 1.0)

        self._step_aircraft(ac_action)
        fired = self._step_turret(tu_action)
        self._step_projectiles()

        ac_reward, terminated = self._compute_aircraft_reward()

        # Zero-sum opponent reward + small efficiency cost on firing.
        tu_reward = -ac_reward
        if fired:
            tu_reward -= TURRET_FIRE_COST

        self._steps += 1
        rewards = {"aircraft": float(ac_reward), "turret": float(tu_reward)}
        terms = {"aircraft": terminated, "turret": terminated}
        truncs = {"aircraft": False, "turret": False}
        infos = {"aircraft": {"fired": fired}, "turret": {"fired": fired}}
        return self._observations(), rewards, terms, truncs, infos

    # ------------------------------------------------------------------ physics
    def _step_aircraft(self, action: np.ndarray):
        ax = float(action[0]) * AIRCRAFT_ACCEL
        ay = float(action[1]) * AIRCRAFT_ACCEL
        ac = self._aircraft
        ac.vx += ax * DT - AIRCRAFT_DRAG * ac.vx * DT
        ac.vy += ay * DT - AIRCRAFT_DRAG * ac.vy * DT
        speed = float(np.hypot(ac.vx, ac.vy))
        if speed > AIRCRAFT_MAX_SPEED:
            scale = AIRCRAFT_MAX_SPEED / speed
            ac.vx *= scale
            ac.vy *= scale
        ac.x += ac.vx * DT
        ac.y += ac.vy * DT

    def _step_turret(self, action: np.ndarray) -> bool:
        t = self._turret
        target_angle = _aim_action_to_angle(action[0])

        # Slew the barrel toward the requested angle.
        delta = target_angle - t.aim
        max_step = TURRET_AIM_RATE * DT
        delta = float(np.clip(delta, -max_step, max_step))
        t.aim = float(np.clip(t.aim + delta, TURRET_AIM_MIN, TURRET_AIM_MAX))

        t.cooldown -= DT
        fired = False
        if (
            float(action[1]) > 0.0
            and t.cooldown <= 0.0
            and len(self._projectiles) < MAX_PROJECTILES
        ):
            vx = PROJECTILE_SPEED * float(np.cos(t.aim))
            vy = PROJECTILE_SPEED * float(np.sin(t.aim))
            self._projectiles.append(Projectile(x=t.x, y=t.y + 1.0, vx=vx, vy=vy))
            t.cooldown = TURRET_FIRE_COOLDOWN
            fired = True
        return fired

    def _step_projectiles(self):
        for p in self._projectiles:
            if not p.alive:
                continue
            p.x += p.vx * DT
            p.y += p.vy * DT
            p.age += DT
            if (
                p.age > PROJECTILE_LIFETIME
                or p.x < -WORLD_W / 2
                or p.x > WORLD_W / 2
                or p.y < FLOOR_Y - 2.0
                or p.y > CEIL_Y + 2.0
            ):
                p.alive = False
        self._projectiles = [p for p in self._projectiles if p.alive]

    def _compute_aircraft_reward(self) -> tuple[float, bool]:
        ac = self._aircraft
        terminated = False
        reward = 0.01
        reward += 0.001 * max(0.0, ac.y - FLOOR_Y) / WORLD_H

        if not (-WORLD_W / 2 <= ac.x <= WORLD_W / 2):
            reward -= 1.0
            terminated = True
        if ac.y < FLOOR_Y or ac.y > CEIL_Y:
            reward -= 1.0
            terminated = True

        if not terminated:
            for p in self._projectiles:
                d = float(np.hypot(p.x - ac.x, p.y - ac.y))
                if d <= AIRCRAFT_RADIUS + PROJECTILE_RADIUS:
                    reward -= 10.0
                    terminated = True
                    break
                if d < DANGER_RADIUS:
                    reward -= 0.05 * (1.0 - d / DANGER_RADIUS)
        return reward, terminated

    # ------------------------------------------------------------------ obs
    def _observations(self) -> dict:
        return {
            "aircraft": self._aircraft_observation(),
            "turret": self._turret_observation(),
        }

    def _aircraft_observation(self) -> np.ndarray:
        ac = self._aircraft
        t = self._turret
        obs = np.zeros(AIRCRAFT_OBS_DIM, dtype=np.float32)
        obs[0] = ac.x / (WORLD_W / 2)
        obs[1] = ac.y / WORLD_H
        obs[2] = ac.vx / AIRCRAFT_MAX_SPEED
        obs[3] = ac.vy / AIRCRAFT_MAX_SPEED
        obs[4] = (t.x - ac.x) / (WORLD_W / 2)
        live = sorted(
            (p for p in self._projectiles if p.alive),
            key=lambda p: (p.x - ac.x) ** 2 + (p.y - ac.y) ** 2,
        )
        for i, p in enumerate(live[:MAX_PROJECTILES]):
            base = 5 + 4 * i
            obs[base + 0] = (p.x - ac.x) / (WORLD_W / 2)
            obs[base + 1] = (p.y - ac.y) / WORLD_H
            obs[base + 2] = p.vx / PROJECTILE_SPEED
            obs[base + 3] = p.vy / PROJECTILE_SPEED
        return obs

    def _turret_observation(self) -> np.ndarray:
        ac = self._aircraft
        t = self._turret
        obs = np.zeros(TURRET_OBS_DIM, dtype=np.float32)
        obs[0] = t.x / (WORLD_W / 2)
        obs[1] = (ac.x - t.x) / (WORLD_W / 2)
        obs[2] = ac.y / WORLD_H
        obs[3] = ac.vx / AIRCRAFT_MAX_SPEED
        obs[4] = ac.vy / AIRCRAFT_MAX_SPEED
        obs[5] = float(np.cos(t.aim))
        obs[6] = float(np.sin(t.aim))
        obs[7] = float(np.clip(t.cooldown / TURRET_FIRE_COOLDOWN, 0.0, 1.0))
        return obs

    # ------------------------------------------------------------------ render
    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from flight_sim.render import PygameRenderer

            self._renderer = PygameRenderer(WORLD_W, WORLD_H, self.render_mode)
        return self._renderer.draw(
            aircraft=self._aircraft,
            turret=self._turret,
            projectiles=[p for p in self._projectiles if p.alive],
        )

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# ----------------------------------------------------------------- heuristics
def heuristic_turret_policy(obs: np.ndarray) -> np.ndarray:
    """Lead-aim policy used as the default turret opponent.

    Reads the turret's own observation (relative aircraft position + velocity),
    predicts where the aircraft will be in `range / projectile_speed` seconds,
    and returns the corresponding aim action plus an always-on fire request.
    """
    rel_x = float(obs[1]) * (WORLD_W / 2)
    ac_y = float(obs[2]) * WORLD_H
    rel_y = ac_y - FLOOR_Y
    vx = float(obs[3]) * AIRCRAFT_MAX_SPEED
    vy = float(obs[4]) * AIRCRAFT_MAX_SPEED

    rng = np.hypot(rel_x, rel_y)
    lead_t = max(0.1, rng / PROJECTILE_SPEED)
    target_x = rel_x + vx * lead_t
    target_y = rel_y + vy * lead_t
    angle = float(np.arctan2(target_y, target_x))
    return np.array([_angle_to_aim_action(angle), 1.0], dtype=np.float32)


def heuristic_aircraft_policy(obs: np.ndarray) -> np.ndarray:
    """Very simple evader: dodge perpendicular to the nearest projectile."""
    if MAX_PROJECTILES == 0 or len(obs) < 9:
        return np.zeros(2, dtype=np.float32)
    pdx = float(obs[5])
    pdy = float(obs[6])
    if pdx == 0.0 and pdy == 0.0:
        return np.array([0.0, 0.3], dtype=np.float32)  # gentle climb
    # Move perpendicular to the projectile's relative bearing.
    away = np.array([-pdy, pdx], dtype=np.float32)
    nrm = float(np.linalg.norm(away))
    if nrm > 0:
        away /= nrm
    return np.clip(away, -1.0, 1.0)
