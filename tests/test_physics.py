"""Behavioural tests for the aircraft physics engine.

The trajectory must be continuous, differentiable and smooth in position,
velocity, altitude and acceleration; the aircraft must capture the holding
circle tangentially with a radius set by the speed; and a commanded landing must
finish at exactly zero altitude and zero velocity.

Numerical derivatives are taken with ``numpy.gradient``; its one-sided endpoint
stencils are noisy, so smoothness assertions look only at the *interior* of each
run (``EDGE`` samples trimmed from each end).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from aircraft_sim.physics import (
    Constraints,
    Phase,
    Simulator,
    G,
    KNOTS_TO_MS,
    MS_TO_KNOTS,
    wrap_pi,
)
from conftest import DT, AREA, run_flight, make_poi_behind


EDGE = 8  # samples trimmed from each end before smoothness checks

# A spread of initial conditions used across many tests.
SPEEDS = [30.0, 50.0, 80.0, 100.0]
HEADINGS = [0.0, 35.0, 90.0, 200.0, -60.0]


def _interior(arr):
    return arr[EDGE:-EDGE]


# --------------------------------------------------------------------------- #
# Continuity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("speed", SPEEDS)
@pytest.mark.parametrize("heading", HEADINGS)
def test_position_continuous(speed, heading):
    """No teleports: each step moves at most ~v_max * dt."""
    f = run_flight(speed_kt=speed, heading_deg=heading)
    step = np.hypot(np.diff(f.x), np.diff(f.y))
    max_step = f.cfg.max_speed * DT
    assert step.max() <= max_step * 1.05


@pytest.mark.parametrize("speed", SPEEDS)
def test_velocity_continuous(speed):
    """Velocity has no jumps -> acceleration is bounded by the limits."""
    f = run_flight(speed_kt=speed, land=True)
    vx, vy = f.velocity()
    dvx = _interior(np.diff(vx))
    dvy = _interior(np.diff(vy))
    # Largest credible accel: tangential + centripetal at max bank.
    a_cap = f.cfg.max_linear_accel + G * math.tan(f.cfg.max_bank)
    assert np.abs(dvx).max() / DT <= a_cap * 1.2
    assert np.abs(dvy).max() / DT <= a_cap * 1.2


@pytest.mark.parametrize("speed", SPEEDS)
def test_altitude_continuous_and_bounded_descent(speed):
    f = run_flight(speed_kt=speed, land=True)
    vz = np.gradient(f.z, f.t)
    assert np.abs(_interior(vz)).max() <= f.cfg.max_descent_rate * 1.05
    # altitude never goes below ground or above the start.
    assert f.z.min() >= -1e-6
    assert f.z.max() <= f.states[0].z + 1e-6


# --------------------------------------------------------------------------- #
# Differentiability / smoothness
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("speed", SPEEDS)
@pytest.mark.parametrize("heading", HEADINGS)
def test_acceleration_continuous(speed, heading):
    """Acceleration is itself continuous (bounded jerk, no spikes)."""
    f = run_flight(speed_kt=speed, heading_deg=heading)
    jx, jy = f.jerk()
    jmag = np.hypot(_interior(jx), _interior(jy))
    # Observed interior jerk is ~2-3 m/s^3; allow a wide margin.
    assert jmag.max() < 15.0


@pytest.mark.parametrize("speed", SPEEDS)
def test_bank_angle_smooth(speed):
    """Bank rate stays within the roll-rate limit (phi is C1)."""
    f = run_flight(speed_kt=speed, land=True)
    dphi = _interior(np.diff(f.phi)) / DT
    assert np.abs(dphi).max() <= f.cfg.max_roll_rate * 1.05


# --------------------------------------------------------------------------- #
# Hard physical constraints
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("speed", SPEEDS)
@pytest.mark.parametrize("heading", HEADINGS)
def test_bank_never_exceeds_limit(speed, heading):
    f = run_flight(speed_kt=speed, heading_deg=heading, land=True)
    assert np.degrees(np.abs(f.phi)).max() <= 30.0 + 1e-6


@pytest.mark.parametrize("speed", SPEEDS)
def test_speed_never_exceeds_max(speed):
    f = run_flight(speed_kt=speed, land=True)
    assert f.v.max() <= f.cfg.max_speed + 1e-9


@pytest.mark.parametrize("speed", SPEEDS)
def test_turn_radius_respects_minimum(speed):
    """Instantaneous radius of curvature never beats the min-turn-radius."""
    f = run_flight(speed_kt=speed, heading_deg=20.0)
    # radius = v / psi_dot ; compare against the speed-dependent minimum.
    psi_dot = np.array([s.psi_dot for s in f.states])
    v = f.v
    moving = v > 1.0
    radius = np.where(np.abs(psi_dot) > 1e-9, v / np.maximum(np.abs(psi_dot), 1e-9), np.inf)
    r_min = np.array([f.cfg.min_turn_radius(vv) for vv in v])
    # allow a hair of numerical slack
    assert np.all(radius[moving] >= r_min[moving] - 1.0)


# --------------------------------------------------------------------------- #
# Cruise leg
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("heading", HEADINGS)
def test_cruise_is_straight_and_level(heading):
    """Before diverting, the aircraft flies a straight, constant-speed line."""
    f = run_flight(speed_kt=80.0, heading_deg=heading, divert_at=1e9, max_time=20.0)
    # constant speed
    assert np.ptp(f.v) < 1e-9
    # constant altitude
    assert np.ptp(f.z) < 1e-9
    # wings level
    assert np.abs(f.phi).max() < 1e-9
    # collinear: cross-product of displacement with heading ~ 0
    h = math.radians(heading)
    cross = (f.x - f.x[0]) * math.sin(h) - (f.y - f.y[0]) * math.cos(h)
    assert np.abs(cross).max() < 1e-6


# --------------------------------------------------------------------------- #
# Holding-pattern capture
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("speed", SPEEDS)
@pytest.mark.parametrize("heading", HEADINGS)
def test_captures_holding_pattern(speed, heading):
    f = run_flight(speed_kt=speed, heading_deg=heading)
    assert f.loiter_index is not None, "never established in the holding pattern"


@pytest.mark.parametrize("speed", SPEEDS)
def test_loiter_radius_set_by_speed(speed):
    """Established holding radius matches v^2 / (g tan(loiter_bank))."""
    f = run_flight(speed_kt=speed)
    assert f.loiter_index is not None
    seg = slice(f.loiter_index + 500, f.loiter_index + 2500)
    d = f.dist_to_poi()[seg]
    expected = f.cfg.loiter_radius(speed * KNOTS_TO_MS)
    assert d.mean() == pytest.approx(expected, rel=0.05)
    # and it is steady (a real circle, not a slow drift)
    assert np.ptp(d) < 0.05 * expected


@pytest.mark.parametrize("speed", SPEEDS)
@pytest.mark.parametrize("heading", HEADINGS)
def test_enters_on_tangent(speed, heading):
    """In the hold, velocity is perpendicular to the radius vector."""
    f = run_flight(speed_kt=speed, heading_deg=heading)
    assert f.loiter_index is not None
    seg = slice(f.loiter_index + 500, f.loiter_index + 2000)
    bx = f.x[seg] - f.poi[0]
    by = f.y[seg] - f.poi[1]
    radial = np.arctan2(by, bx)
    err = np.array([abs(wrap_pi(p - (r + math.pi / 2))) for p, r in zip(f.psi[seg], radial)])
    err = np.minimum(err, math.pi - err)  # accept either circulation sense
    assert np.degrees(err).max() < 2.0


@pytest.mark.parametrize("heading", HEADINGS)
def test_turn_exceeds_90_degrees(heading):
    """A POI placed behind the entry requires a > 90 deg course change."""
    entry = (0.0, 10_000.0)
    poi = make_poi_behind(entry, heading)
    f = run_flight(speed_kt=80.0, heading_deg=heading, entry=entry, poi=poi)
    assert f.divert_index is not None and f.loiter_index is not None
    # POI really is behind: bearing-to-POI vs heading is an obtuse angle.
    h = math.radians(heading)
    bx, by = poi[0] - entry[0], poi[1] - entry[1]
    dot = bx * math.cos(h) + by * math.sin(h)
    assert dot < 0.0
    # cumulative heading change during capture exceeds 90 deg.
    psi = f.psi[f.divert_index:f.loiter_index + 1]
    total = np.sum(np.abs([wrap_pi(b - a) for a, b in zip(psi[:-1], psi[1:])]))
    assert math.degrees(total) > 90.0


# --------------------------------------------------------------------------- #
# Landing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("speed", SPEEDS)
@pytest.mark.parametrize("heading", [0.0, 90.0, 200.0])
def test_landing_ends_at_zero_altitude_and_speed(speed, heading):
    f = run_flight(speed_kt=speed, heading_deg=heading, land=True)
    assert f.states[-1].phase == Phase.LANDED
    final = f.states[-1]
    assert abs(final.z) < 1e-3, f"final altitude {final.z}"
    assert final.v * MS_TO_KNOTS < 1e-3, f"final speed {final.v * MS_TO_KNOTS} kt"
    # vertical speed has also arrested (soft touchdown).
    vz = np.gradient(f.z, f.t)
    assert abs(vz[-2]) < 0.5


@pytest.mark.parametrize("speed", SPEEDS)
def test_landing_touches_down_on_poi(speed):
    f = run_flight(speed_kt=speed, land=True)
    final = f.states[-1]
    miss = math.hypot(final.x - f.poi[0], final.y - f.poi[1])
    r = f.cfg.loiter_radius(speed * KNOTS_TO_MS)
    assert miss < max(5.0, 0.05 * r), f"missed POI by {miss:.1f} m"


def test_landing_altitude_monotonic():
    """Altitude only ever decreases during the descent (no porpoising)."""
    f = run_flight(speed_kt=80.0, land=True)
    li = next(i for i, s in enumerate(f.states) if s.phase == Phase.LANDING)
    z = f.z[li:]
    assert np.all(np.diff(z) <= 1e-6)


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_simulation_is_deterministic():
    a = run_flight(speed_kt=70.0, heading_deg=33.0, land=True)
    b = run_flight(speed_kt=70.0, heading_deg=33.0, land=True)
    assert a.x[-1] == b.x[-1]
    assert a.z[-1] == b.z[-1]
    assert len(a.states) == len(b.states)
