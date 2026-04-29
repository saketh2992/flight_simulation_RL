"""2D flight-vs-ground-shooter Gymnasium environment.

A single aircraft flies horizontally across a 2D world. A stationary turret on
the ground tracks the aircraft and periodically launches projectiles. The agent
controls the aircraft's vertical and horizontal acceleration and must dodge
projectiles for as long as possible.

Observation (Box, float32, shape=(5 + 4 * MAX_PROJECTILES,)):
    [px, py, vx, vy, turret_x_rel,
     proj_dx_0, proj_dy_0, proj_vx_0, proj_vy_0,
     ...]
    All positions and velocities are normalised to [-1, 1] / [-2, 2] roughly.
    Inactive projectile slots are zero-filled.

Action (Box, float32, shape=(2,), range [-1, 1]):
    [ax, ay] — horizontal and vertical acceleration commands.

Reward:
    +0.01 per step survived
    +0.001 * height_above_floor (encourages staying airborne)
    -0.05 per projectile that comes very close (within DANGER_RADIUS)
    -10.0 on hit (terminal)
    -1.0 on going out of bounds (terminal)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces


WORLD_W = 200.0
WORLD_H = 100.0
FLOOR_Y = 0.0
CEIL_Y = WORLD_H

DT = 0.1
MAX_PROJECTILES = 6

AIRCRAFT_MAX_SPEED = 25.0
AIRCRAFT_ACCEL = 20.0
AIRCRAFT_DRAG = 0.15
AIRCRAFT_RADIUS = 2.0

PROJECTILE_SPEED = 30.0
PROJECTILE_RADIUS = 1.0
PROJECTILE_LIFETIME = 8.0  # seconds
TURRET_FIRE_INTERVAL = 0.6  # seconds between shots
TURRET_AIM_NOISE = 0.08  # radians std-dev

DANGER_RADIUS = 6.0


@dataclass
class Aircraft:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0


@dataclass
class Projectile:
    x: float
    y: float
    vx: float
    vy: float
    age: float = 0.0
    alive: bool = True


@dataclass
class Turret:
    x: float
    y: float = FLOOR_Y
    cooldown: float = 0.0
    aim: float = float(np.pi / 2)  # current barrel angle, radians (π/2 = straight up)


class FlightDodgeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: Optional[str] = None, seed: Optional[int] = None):
        super().__init__()
        self.render_mode = render_mode

        obs_dim = 5 + 4 * MAX_PROJECTILES
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self._np_random: np.random.Generator
        self._aircraft: Aircraft = Aircraft(0.0, 0.0)
        self._turret: Turret = Turret(0.0)
        self._projectiles: list[Projectile] = []
        self._steps: int = 0

        # Lazy-loaded renderer (pygame).
        self._renderer = None

        if seed is not None:
            self.reset(seed=seed)

    # ------------------------------------------------------------------ core
    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self._np_random = np.random.default_rng(seed)

        self._aircraft = Aircraft(
            x=float(self._np_random.uniform(-WORLD_W * 0.4, WORLD_W * 0.4)),
            y=float(self._np_random.uniform(WORLD_H * 0.5, WORLD_H * 0.85)),
            vx=0.0,
            vy=0.0,
        )
        self._turret = Turret(
            x=float(self._np_random.uniform(-WORLD_W * 0.4, WORLD_W * 0.4)),
            cooldown=TURRET_FIRE_INTERVAL,
        )
        self._projectiles = []
        self._steps = 0

        return self._observation(), {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
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

        self._update_turret()
        self._update_projectiles()

        terminated = False
        truncated = False
        reward = 0.01

        # Stay-airborne shaping (height above the floor, normalised).
        reward += 0.001 * max(0.0, ac.y - FLOOR_Y) / WORLD_H

        # Out-of-bounds termination.
        if not (-WORLD_W / 2 <= ac.x <= WORLD_W / 2):
            reward -= 1.0
            terminated = True
        if ac.y < FLOOR_Y or ac.y > CEIL_Y:
            reward -= 1.0
            terminated = True

        # Collision / proximity penalty.
        if not terminated:
            for p in self._projectiles:
                if not p.alive:
                    continue
                d = float(np.hypot(p.x - ac.x, p.y - ac.y))
                if d <= AIRCRAFT_RADIUS + PROJECTILE_RADIUS:
                    reward -= 10.0
                    terminated = True
                    break
                if d < DANGER_RADIUS:
                    reward -= 0.05 * (1.0 - d / DANGER_RADIUS)

        self._steps += 1
        return self._observation(), float(reward), terminated, truncated, {}

    # ------------------------------------------------------------------ helpers
    def _update_turret(self):
        t = self._turret
        t.cooldown -= DT
        if t.cooldown <= 0.0:
            t.cooldown = TURRET_FIRE_INTERVAL
            if sum(1 for p in self._projectiles if p.alive) < MAX_PROJECTILES:
                self._fire_projectile()

    def _fire_projectile(self):
        ac = self._aircraft
        t = self._turret
        # Lead the target a little based on its current velocity.
        lead_t = max(0.1, np.hypot(ac.x - t.x, ac.y - t.y) / PROJECTILE_SPEED)
        target_x = ac.x + ac.vx * lead_t
        target_y = ac.y + ac.vy * lead_t
        dx = target_x - t.x
        dy = target_y - t.y
        angle = float(np.arctan2(dy, dx))
        angle += float(self._np_random.normal(0.0, TURRET_AIM_NOISE))
        vx = PROJECTILE_SPEED * float(np.cos(angle))
        vy = PROJECTILE_SPEED * float(np.sin(angle))
        self._projectiles.append(Projectile(x=t.x, y=t.y + 1.0, vx=vx, vy=vy))

    def _update_projectiles(self):
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

    def _observation(self) -> np.ndarray:
        ac = self._aircraft
        t = self._turret
        obs = np.zeros(5 + 4 * MAX_PROJECTILES, dtype=np.float32)
        obs[0] = ac.x / (WORLD_W / 2)
        obs[1] = ac.y / WORLD_H
        obs[2] = ac.vx / AIRCRAFT_MAX_SPEED
        obs[3] = ac.vy / AIRCRAFT_MAX_SPEED
        obs[4] = (t.x - ac.x) / (WORLD_W / 2)

        # Sort projectiles by distance so the nearest threats appear first.
        live = [p for p in self._projectiles if p.alive]
        live.sort(key=lambda p: (p.x - ac.x) ** 2 + (p.y - ac.y) ** 2)
        for i, p in enumerate(live[:MAX_PROJECTILES]):
            base = 5 + 4 * i
            obs[base + 0] = (p.x - ac.x) / (WORLD_W / 2)
            obs[base + 1] = (p.y - ac.y) / WORLD_H
            obs[base + 2] = p.vx / PROJECTILE_SPEED
            obs[base + 3] = p.vy / PROJECTILE_SPEED
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
