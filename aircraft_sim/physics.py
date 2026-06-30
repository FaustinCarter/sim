"""Physics / guidance engine for the aircraft simulator.

The engine models the aircraft as a point mass with a *heading* (spin) degree of
freedom.  It is deliberately GUI-free so that it can be exercised in isolation by
the test-suite.

Design goals
------------
The trajectory the engine produces must be **continuous, differentiable and
smooth** in position, velocity, altitude and acceleration.  This is achieved by
never commanding a discontinuous control:

* The horizontal speed ``v(t)`` and the altitude ``z(t)`` follow analytic
  *smootherstep* profiles (``6u^5 - 15u^4 + 10u^3``) whose first and second time
  derivatives vanish at both ends.  Speed and altitude are therefore C2.
* Turning is performed through a coordinated bank.  The guidance law produces a
  *continuous* commanded turn-rate; the bank angle slews towards the required
  value through a first-order lag, so the actual bank ``phi(t)`` is continuous,
  the heading-rate ``psi_dot = g*tan(phi)/v`` is continuous and hence the lateral
  acceleration is continuous.
* The bank command is *softly* saturated so it can never exceed the 30 deg limit
  and never develops a corner.
* Discrete pilot events (divert / land) are blended in with smootherstep ramps so
  that no control jumps at the moment of the event.

Coordinate system
-----------------
``x`` (east) and ``y`` (north) are ground-plane metres, ``z`` is altitude in
metres (ground = 0).  ``psi`` is the heading measured CCW from the +x axis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Physical constants
# --------------------------------------------------------------------------- #
G = 9.80665  # gravitational acceleration, m/s^2
KNOTS_TO_MS = 0.514444  # 1 knot in m/s
MS_TO_KNOTS = 1.0 / KNOTS_TO_MS


# --------------------------------------------------------------------------- #
# Small math helpers
# --------------------------------------------------------------------------- #
def wrap_pi(angle: float) -> float:
    """Wrap an angle to the range (-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def smootherstep(u: float) -> float:
    """Ken Perlin's smootherstep.  S(0)=0, S(1)=1, S'=S''=0 at both ends."""
    if u <= 0.0:
        return 0.0
    if u >= 1.0:
        return 1.0
    return u * u * u * (u * (u * 6.0 - 15.0) + 10.0)


def smootherstep_deriv(u: float, span: float) -> float:
    """d/dt of ``smootherstep(t/span)`` evaluated at ``u = t/span``."""
    if u <= 0.0 or u >= 1.0 or span <= 0.0:
        return 0.0
    return (30.0 * u * u * (u - 1.0) * (u - 1.0)) / span


def soft_clip(value: float, limit: float) -> float:
    """Smoothly limit ``value`` to ``(-limit, limit)`` using ``tanh``.

    For small inputs this is almost the identity; it never reaches +/-limit and
    has no corner, which keeps the commanded bank infinitely differentiable.
    """
    if limit <= 0.0:
        return 0.0
    return limit * math.tanh(value / limit)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Constraints:
    """Physical limits and tuning gains for the aircraft."""

    # Hard physical limits ------------------------------------------------- #
    max_bank: float = math.radians(30.0)          # rad, hard bank limit
    loiter_bank: float = math.radians(25.0)        # rad, steady holding bank
    max_speed: float = 100.0 * KNOTS_TO_MS         # m/s
    max_linear_accel: float = 2.0                  # m/s^2 (~0.2 g)
    max_roll_rate: float = math.radians(20.0)      # rad/s, |d(phi)/dt| limit
    max_descent_rate: float = 30.0                 # m/s, vertical speed limit

    # Guidance gains ------------------------------------------------------- #
    k_orbit: float = 2.0        # how sharply the approach steepens with range
    k_heading: float = 1.5      # proportional course-tracking gain, 1/s
    roll_tau: float = 0.6       # bank first-order lag time-constant, s

    # Event blend windows -------------------------------------------------- #
    divert_ramp: float = 4.0    # s, ramp-in of the turn after a divert command

    def min_turn_radius(self, speed: float) -> float:
        """Tightest turn radius achievable at ``speed`` (m/s) at max bank."""
        if speed <= 0.0:
            return 0.0
        return speed * speed / (G * math.tan(self.max_bank))

    def loiter_radius(self, speed: float) -> float:
        """Holding-circle radius for ``speed`` (m/s) -- set purely by speed."""
        if speed <= 0.0:
            return 0.0
        return speed * speed / (G * math.tan(self.loiter_bank))


class Phase(Enum):
    """High level state of the flight."""

    CRUISE = "cruise"        # straight-and-level entry leg
    DIVERT = "divert"        # turning to capture the holding circle
    LOITER = "loiter"        # established in the holding pattern
    LANDING = "landing"      # spiralling down onto the point of interest
    LANDED = "landed"        # on the ground, stopped
    STOPPED = "stopped"      # simulation halted by the user


@dataclass
class State:
    """Immutable snapshot of the aircraft at one instant."""

    t: float
    x: float
    y: float
    z: float          # altitude, m
    psi: float        # heading, rad
    v: float          # horizontal speed, m/s
    phi: float        # bank angle, rad
    phase: Phase

    # First derivatives (filled in by the simulator) ----------------------- #
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    psi_dot: float = 0.0

    @property
    def speed_knots(self) -> float:
        return self.v * MS_TO_KNOTS

    @property
    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)


# --------------------------------------------------------------------------- #
# Simulator
# --------------------------------------------------------------------------- #
class Simulator:
    """Integrates the aircraft state forward in time.

    Typical usage::

        sim = Simulator(entry=(0, 5000), heading_deg=0, speed_knots=80,
                        area_size=20000)
        while sim.state.phase not in (Phase.LANDED, Phase.STOPPED):
            sim.step(0.02)
    """

    V_EPS = 1e-3  # m/s, below this the aircraft is treated as stopped

    def __init__(
        self,
        entry: Tuple[float, float],
        heading_deg: float,
        speed_knots: float,
        area_size: float = 20_000.0,
        start_altitude: float = 10_000.0,
        constraints: Optional[Constraints] = None,
    ) -> None:
        self.cfg = constraints or Constraints()
        self.area_size = float(area_size)
        self.start_altitude = float(start_altitude)

        speed = max(0.0, min(speed_knots, self.cfg.max_speed * MS_TO_KNOTS)) * KNOTS_TO_MS
        self._cruise_speed = speed

        self.t = 0.0
        self._x = float(entry[0])
        self._y = float(entry[1])
        self._psi = math.radians(heading_deg)
        self._phi = 0.0  # wings level on entry

        self.phase = Phase.CRUISE

        # Point of interest / holding centre.
        self._poi: Optional[Tuple[float, float]] = None
        self._turn_dir = 1.0  # +1 CCW, -1 CW; chosen at divert time
        self._divert_t: Optional[float] = None

        # Landing profile parameters (filled in by command_land).
        self._land_t: Optional[float] = None
        self._land_T: float = 0.0
        self._land_v0: float = 0.0
        self._land_z0: float = 0.0
        self._land_radius_c: float = 0.0  # R = c * v during the spiral-in

        # Cache of the most recent derivatives for reporting.
        self._last_psi_dot = 0.0

    # ------------------------------------------------------------------ #
    # Pilot commands
    # ------------------------------------------------------------------ #
    def command_divert(self, poi: Tuple[float, float]) -> None:
        """Begin turning to capture a holding pattern around ``poi``."""
        if self.phase in (Phase.LANDING, Phase.LANDED, Phase.STOPPED):
            return
        self._poi = (float(poi[0]), float(poi[1]))
        self._divert_t = self.t
        # Pick the turn direction that initially rotates the nose toward the POI.
        bearing = math.atan2(self._poi[1] - self._y, self._poi[0] - self._x)
        self._turn_dir = 1.0 if wrap_pi(bearing - self._psi) >= 0.0 else -1.0
        self.phase = Phase.DIVERT

    def command_land(self) -> None:
        """Spiral down onto the point of interest and stop at zero altitude."""
        if self._poi is None or self.phase in (Phase.LANDED, Phase.STOPPED):
            return
        self._land_t = self.t
        self._land_v0 = self._speed_at(self.t)
        self._land_z0 = self._altitude_at(self.t)
        # Duration is paced by the vertical-speed limit (smootherstep peaks at
        # 1.875x the mean rate) and, secondarily, the deceleration limit.
        t_descent = self._land_z0 * 1.875 / max(self.cfg.max_descent_rate, 1e-6)
        t_decel = self._land_v0 * 1.875 / max(self.cfg.max_linear_accel, 1e-6)
        self._land_T = max(t_descent, t_decel, 1.0)
        # Radius shrinks linearly with speed so the turn-rate (and therefore the
        # number of loops) stays bounded as v -> 0, and the bank -> 0 on stop.
        loiter_r = self.cfg.loiter_radius(self._land_v0)
        self._land_radius_c = loiter_r / self._land_v0 if self._land_v0 > 0 else 0.0
        self.phase = Phase.LANDING

    def command_stop(self) -> None:
        """Halt the simulation in place."""
        if self.phase != Phase.LANDED:
            self.phase = Phase.STOPPED

    # ------------------------------------------------------------------ #
    # Analytic speed / altitude profiles (C2 in time)
    # ------------------------------------------------------------------ #
    def _speed_at(self, t: float) -> float:
        if self._land_t is None or t <= self._land_t:
            return self._cruise_speed
        u = (t - self._land_t) / self._land_T
        return self._land_v0 * (1.0 - smootherstep(u))

    def _speed_deriv_at(self, t: float) -> float:
        if self._land_t is None or t <= self._land_t or t >= self._land_t + self._land_T:
            return 0.0
        u = (t - self._land_t) / self._land_T
        return -self._land_v0 * smootherstep_deriv(u, self._land_T)

    def _altitude_at(self, t: float) -> float:
        if self._land_t is None or t <= self._land_t:
            return self.start_altitude
        u = (t - self._land_t) / self._land_T
        return self._land_z0 * (1.0 - smootherstep(u))

    def _altitude_deriv_at(self, t: float) -> float:
        if self._land_t is None or t <= self._land_t or t >= self._land_t + self._land_T:
            return 0.0
        u = (t - self._land_t) / self._land_T
        return -self._land_z0 * smootherstep_deriv(u, self._land_T)

    # ------------------------------------------------------------------ #
    # Guidance: commanded turn-rate as a continuous function of state
    # ------------------------------------------------------------------ #
    def _target_radius(self, speed: float) -> float:
        if self.phase == Phase.LANDING:
            return self._land_radius_c * speed
        return self.cfg.loiter_radius(speed)

    def _commanded_turn_rate(self, x: float, y: float, psi: float, speed: float, t: float) -> float:
        """Vector-field orbit guidance -> desired heading rate (rad/s).

        Returns a value that is a continuous function of the state, smoothly
        ramped after a divert command and soft-limited so that the resulting
        bank can never exceed ``max_bank``.
        """
        if self._poi is None or speed <= self.V_EPS:
            return 0.0

        cx, cy = self._poi
        dx, dy = x - cx, y - cy
        dist = math.hypot(dx, dy)
        r_t = self._target_radius(speed)

        if dist < 1e-6 or r_t < 1e-6:
            return 0.0

        gamma = math.atan2(dy, dx)  # bearing from centre to aircraft
        # Desired course: spiral that points inward when outside the circle and
        # becomes tangent (circulation) on the circle.
        approach = math.atan(self.cfg.k_orbit * (dist - r_t) / r_t)
        course_d = gamma + self._turn_dir * (math.pi / 2.0 + approach)

        heading_err = wrap_pi(course_d - psi)
        # Feed-forward circulation rate, weighted to ~1 near the circle.
        band = (dist - r_t) / r_t
        weight = math.exp(-band * band)
        feedforward = self._turn_dir * (speed / r_t) * weight

        rate = self.cfg.k_heading * heading_err + feedforward

        # Soft-limit so |phi| < max_bank, with margin handled by tanh.
        rate_max = G * math.tan(self.cfg.max_bank) / speed
        rate = soft_clip(rate, rate_max)

        # Blend the turn in after a divert so nothing jumps at the event.
        if self._divert_t is not None and self.cfg.divert_ramp > 0.0:
            ramp = smootherstep((t - self._divert_t) / self.cfg.divert_ramp)
            rate *= ramp
        return rate

    def _commanded_bank(self, x: float, y: float, psi: float, speed: float, t: float) -> float:
        """Coordinated-flight bank for the commanded turn-rate."""
        rate = self._commanded_turn_rate(x, y, psi, speed, t)
        phi = math.atan(speed * rate / G)
        # Hard guarantee against numerical overshoot of the limit.
        return max(-self.cfg.max_bank, min(self.cfg.max_bank, phi))

    # ------------------------------------------------------------------ #
    # Equations of motion
    # ------------------------------------------------------------------ #
    def _derivs(self, x: float, y: float, psi: float, phi: float, t: float):
        """Return (x_dot, y_dot, psi_dot, phi_dot) at the given state/time."""
        speed = self._speed_at(t)
        x_dot = speed * math.cos(psi)
        y_dot = speed * math.sin(psi)

        if speed > self.V_EPS:
            psi_dot = G * math.tan(phi) / speed
        else:
            psi_dot = 0.0

        phi_cmd = self._commanded_bank(x, y, psi, speed, t)
        # First-order lag, rate-limited -> phi is continuous (and C1 while the
        # command is continuous).
        phi_dot = (phi_cmd - phi) / self.cfg.roll_tau
        rr = self.cfg.max_roll_rate
        phi_dot = max(-rr, min(rr, phi_dot))
        return x_dot, y_dot, psi_dot, phi_dot

    def step(self, dt: float) -> State:
        """Advance the simulation by ``dt`` seconds (RK4) and return the state."""
        if self.phase in (Phase.LANDED, Phase.STOPPED):
            return self.state

        x, y, psi, phi, t = self._x, self._y, self._psi, self._phi, self.t

        k1 = self._derivs(x, y, psi, phi, t)
        k2 = self._derivs(
            x + 0.5 * dt * k1[0], y + 0.5 * dt * k1[1],
            psi + 0.5 * dt * k1[2], phi + 0.5 * dt * k1[3], t + 0.5 * dt)
        k3 = self._derivs(
            x + 0.5 * dt * k2[0], y + 0.5 * dt * k2[1],
            psi + 0.5 * dt * k2[2], phi + 0.5 * dt * k2[3], t + 0.5 * dt)
        k4 = self._derivs(
            x + dt * k3[0], y + dt * k3[1],
            psi + dt * k3[2], phi + dt * k3[3], t + dt)

        self._x = x + dt / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        self._y = y + dt / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        self._psi = wrap_pi(psi + dt / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]))
        self._phi = phi + dt / 6.0 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
        self.t = t + dt
        self._last_psi_dot = k1[2]

        self._update_phase()
        return self.state

    # ------------------------------------------------------------------ #
    # Phase bookkeeping
    # ------------------------------------------------------------------ #
    def _update_phase(self) -> None:
        if self.phase == Phase.LANDING:
            if self.t >= (self._land_t or 0.0) + self._land_T:
                # Snap the residual numerical dust to an exact touchdown.
                self.phase = Phase.LANDED
                self._phi = 0.0
            return
        if self.phase == Phase.DIVERT and self._poi is not None:
            speed = self._speed_at(self.t)
            r_t = self._target_radius(speed)
            dist = math.hypot(self._x - self._poi[0], self._y - self._poi[1])
            # Established once close to the circle and flying tangentially.
            radial_err = abs(dist - r_t) / max(r_t, 1.0)
            bearing = math.atan2(self._y - self._poi[1], self._x - self._poi[0])
            tangent_err = abs(wrap_pi(self._psi - (bearing + self._turn_dir * math.pi / 2.0)))
            if radial_err < 0.03 and tangent_err < math.radians(5.0):
                self.phase = Phase.LOITER

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    @property
    def state(self) -> State:
        speed = self._speed_at(self.t)
        if self.phase in (Phase.LANDED,):
            speed = 0.0
        vx = speed * math.cos(self._psi)
        vy = speed * math.sin(self._psi)
        vz = self._altitude_deriv_at(self.t)
        psi_dot = (G * math.tan(self._phi) / speed) if speed > self.V_EPS else 0.0
        return State(
            t=self.t, x=self._x, y=self._y, z=self._altitude_at(self.t),
            psi=self._psi, v=speed, phi=self._phi, phase=self.phase,
            vx=vx, vy=vy, vz=vz, psi_dot=psi_dot,
        )

    def run(self, duration: float, dt: float = 0.02) -> List[State]:
        """Integrate for ``duration`` seconds and return every sampled state."""
        n = int(round(duration / dt))
        out = [self.state]
        for _ in range(n):
            out.append(self.step(dt))
            if self.phase in (Phase.LANDED, Phase.STOPPED):
                break
        return out
