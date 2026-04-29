"""Minimal pygame renderer for the FlightDodge environment."""

from __future__ import annotations

import os
from typing import Iterable, Optional

import numpy as np


class PygameRenderer:
    SCREEN_W = 900
    SCREEN_H = 500

    def __init__(self, world_w: float, world_h: float, render_mode: str):
        if render_mode == "rgb_array":
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

        import pygame  # imported lazily so headless training doesn't need it

        pygame.init()
        if render_mode == "human":
            self.screen = pygame.display.set_mode((self.SCREEN_W, self.SCREEN_H))
            pygame.display.set_caption("FlightDodge — RL Simulator")
        else:
            self.screen = pygame.Surface((self.SCREEN_W, self.SCREEN_H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 16)
        self.world_w = world_w
        self.world_h = world_h
        self.render_mode = render_mode
        self._pg = pygame

    # ------------------------------------------------------------------ utils
    def _to_screen(self, x: float, y: float) -> tuple[int, int]:
        sx = int((x + self.world_w / 2) / self.world_w * self.SCREEN_W)
        sy = int(self.SCREEN_H - (y / self.world_h) * self.SCREEN_H)
        return sx, sy

    # ------------------------------------------------------------------ draw
    def draw(
        self,
        *,
        aircraft,
        turret,
        projectiles: Iterable,
        info_lines: Optional[list[str]] = None,
    ):
        pg = self._pg
        # Pump events so the OS doesn't think the window is unresponsive.
        if self.render_mode == "human":
            for _ in pg.event.get():
                pass

        self.screen.fill((10, 18, 38))  # night-sky blue

        # Ground
        floor_y = self._to_screen(0, 0)[1]
        pg.draw.rect(
            self.screen,
            (40, 60, 30),
            pg.Rect(0, floor_y, self.SCREEN_W, self.SCREEN_H - floor_y),
        )

        # Turret
        tx, ty = self._to_screen(turret.x, turret.y)
        pg.draw.rect(self.screen, (180, 180, 180), pg.Rect(tx - 8, ty - 12, 16, 12))
        pg.draw.circle(self.screen, (220, 80, 80), (tx, ty - 14), 4)

        # Projectiles
        for p in projectiles:
            px, py = self._to_screen(p.x, p.y)
            pg.draw.circle(self.screen, (255, 200, 80), (px, py), 4)
            tail_x, tail_y = self._to_screen(p.x - p.vx * 0.05, p.y - p.vy * 0.05)
            pg.draw.line(self.screen, (255, 140, 40), (tail_x, tail_y), (px, py), 2)

        # Aircraft (simple triangle pointing in velocity direction)
        ax, ay = self._to_screen(aircraft.x, aircraft.y)
        heading = float(np.arctan2(aircraft.vy, aircraft.vx)) if (aircraft.vx or aircraft.vy) else 0.0
        size = 12
        nose = (ax + size * np.cos(heading), ay - size * np.sin(heading))
        left = (
            ax + size * np.cos(heading + 2.6),
            ay - size * np.sin(heading + 2.6),
        )
        right = (
            ax + size * np.cos(heading - 2.6),
            ay - size * np.sin(heading - 2.6),
        )
        pg.draw.polygon(self.screen, (200, 230, 255), [nose, left, right])

        # HUD
        for i, line in enumerate(info_lines or []):
            surf = self.font.render(line, True, (220, 220, 220))
            self.screen.blit(surf, (8, 8 + i * 18))

        if self.render_mode == "human":
            pg.display.flip()
            self.clock.tick(30)
            return None

        # rgb_array
        arr = pg.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def close(self):
        self._pg.quit()
