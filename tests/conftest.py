"""Shared helpers for the physics test-suite."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from aircraft_sim.physics import (
    Constraints,
    Phase,
    Simulator,
    State,
    G,
    KNOTS_TO_MS,
    MS_TO_KNOTS,
    wrap_pi,
)

AREA = 20_000.0
DT = 0.02


@dataclass
class Flight:
    """A completed simulation run, with numpy arrays for analysis."""

    states: List[State]
    poi: Tuple[float, float]
    cfg: Constraints
    entry: Tuple[float, float]
    heading_deg: float
    speed_kt: float
    divert_index: Optional[int]
    loiter_index: Optional[int]

    # --- raw arrays --------------------------------------------------- #
    @property
    def t(self) -> np.ndarray:
        return np.array([s.t for s in self.states])

    @property
    def x(self) -> np.ndarray:
        return np.array([s.x for s in self.states])

    @property
    def y(self) -> np.ndarray:
        return np.array([s.y for s in self.states])

    @property
    def z(self) -> np.ndarray:
        return np.array([s.z for s in self.states])

    @property
    def v(self) -> np.ndarray:
        return np.array([s.v for s in self.states])

    @property
    def phi(self) -> np.ndarray:
        return np.array([s.phi for s in self.states])

    @property
    def psi(self) -> np.ndarray:
        return np.array([s.psi for s in self.states])

    # --- derived kinematics (numerical) ------------------------------- #
    def velocity(self):
        return np.gradient(self.x, self.t), np.gradient(self.y, self.t)

    def acceleration(self):
        vx, vy = self.velocity()
        return np.gradient(vx, self.t), np.gradient(vy, self.t)

    def jerk(self):
        ax, ay = self.acceleration()
        return np.gradient(ax, self.t), np.gradient(ay, self.t)

    def dist_to_poi(self) -> np.ndarray:
        return np.hypot(self.x - self.poi[0], self.y - self.poi[1])


def make_poi_behind(entry, heading_deg, back=3000.0, side=2500.0):
    """Return a POI that is *behind* the entry heading (turn > 90 deg needed)."""
    h = math.radians(heading_deg)
    hx, hy = math.cos(h), math.sin(h)
    # -back along heading (behind) and +side perpendicular (left).
    px = entry[0] - hx * back - hy * side
    py = entry[1] - hy * back + hx * side
    return (px, py)


def run_flight(
    speed_kt: float = 80.0,
    heading_deg: float = 0.0,
    entry: Tuple[float, float] = (0.0, 10_000.0),
    poi: Optional[Tuple[float, float]] = None,
    divert_at: float = 8.0,
    land: bool = False,
    cfg: Optional[Constraints] = None,
    max_time: Optional[float] = None,
) -> Flight:
    """Fly a full scenario and capture the trajectory."""
    cfg = cfg or Constraints()
    if poi is None:
        poi = make_poi_behind(entry, heading_deg)
    sim = Simulator(entry=entry, heading_deg=heading_deg, speed_knots=speed_kt,
                    area_size=AREA, constraints=cfg)

    if max_time is None:
        max_time = 2000.0 if land else 400.0

    states = [sim.state]
    divert_index = loiter_index = None
    diverted = landed_cmd = False
    n = int(max_time / DT)
    for i in range(n):
        if not diverted and sim.t >= divert_at:
            sim.command_divert(poi)
            diverted = True
            divert_index = len(states)
        if (land and not landed_cmd and sim.phase == Phase.LOITER
                and sim.t > divert_at + 40.0):
            sim.command_land()
            landed_cmd = True
        sim.step(DT)
        if loiter_index is None and sim.phase == Phase.LOITER:
            loiter_index = len(states)
        states.append(sim.state)
        if sim.phase in (Phase.LANDED, Phase.STOPPED):
            break

    return Flight(states=states, poi=poi, cfg=cfg, entry=entry,
                  heading_deg=heading_deg, speed_kt=speed_kt,
                  divert_index=divert_index, loiter_index=loiter_index)
