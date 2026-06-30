"""Run nine aircraft trajectories with varied initial conditions and plot them.

Produces a 3x3 grid of ground tracks (coloured by flight phase) plus, for the
landing cases, a second figure of time-series that demonstrates the trajectory
is smooth (speed, altitude, bank and lateral acceleration are continuous).

Usage::

    python scripts/plot_trajectories.py [output_dir]
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from aircraft_sim.physics import (
    Constraints,
    Phase,
    Simulator,
    KNOTS_TO_MS,
    MS_TO_KNOTS,
)

AREA = 20_000.0
DT = 0.05

PHASE_COLOR = {
    Phase.CRUISE: "#5ac8ff",
    Phase.DIVERT: "#ffaa3c",
    Phase.LOITER: "#7be88c",
    Phase.LANDING: "#ff5a78",
    Phase.LANDED: "#ff5a78",
    Phase.STOPPED: "#9096aa",
}


@dataclass
class Scenario:
    name: str
    speed_kt: float
    heading_deg: float
    entry: Tuple[float, float]
    land: bool = False
    back: float = 3500.0
    side: float = 2800.0


def poi_behind(entry, heading_deg, back, side):
    h = math.radians(heading_deg)
    hx, hy = math.cos(h), math.sin(h)
    return (entry[0] - hx * back - hy * side, entry[1] - hy * back + hx * side)


def run(sc: Scenario):
    cfg = Constraints()
    poi = poi_behind(sc.entry, sc.heading_deg, sc.back, sc.side)
    sim = Simulator(entry=sc.entry, heading_deg=sc.heading_deg,
                    speed_knots=sc.speed_kt, area_size=AREA, constraints=cfg)
    states = [sim.state]
    diverted = landed_cmd = False
    max_t = 1600.0 if sc.land else 360.0
    for _ in range(int(max_t / DT)):
        if not diverted and sim.t >= 8.0:
            sim.command_divert(poi)
            diverted = True
        if (sc.land and not landed_cmd and sim.phase == Phase.LOITER
                and sim.t > 8.0 + 35.0):
            sim.command_land()
            landed_cmd = True
        sim.step(DT)
        states.append(sim.state)
        if sim.phase in (Phase.LANDED, Phase.STOPPED):
            break
    return states, poi, cfg


def coloured_track(ax, states):
    x = np.array([s.x for s in states]) / 1000.0  # km
    y = np.array([s.y for s in states]) / 1000.0
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    cols = [PHASE_COLOR[states[i].phase] for i in range(len(states) - 1)]
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=1.8))


def plot_grid(scenarios, results, path):
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle("Nine aircraft trajectories  (colour = flight phase)",
                 fontsize=16, y=0.995)
    for ax, sc, (states, poi, cfg) in zip(axes.flat, scenarios, results):
        x = np.array([s.x for s in states]) / 1000.0
        y = np.array([s.y for s in states]) / 1000.0
        px, py = poi[0] / 1000.0, poi[1] / 1000.0
        # auto-zoom to the action (track + POI) with a margin, kept square.
        xs = np.append(x, px)
        ys = np.append(y, py)
        cx, cy = 0.5 * (xs.min() + xs.max()), 0.5 * (ys.min() + ys.max())
        half = 0.5 * max(np.ptp(xs), np.ptp(ys)) * 1.18 + 0.1
        rng = 2 * half

        coloured_track(ax, states)
        # entry + heading arrow (scaled to the view)
        ex, ey = sc.entry[0] / 1000.0, sc.entry[1] / 1000.0
        h = math.radians(sc.heading_deg)
        ax.plot(ex, ey, "o", color="#5ac8ff", ms=6, zorder=5)
        ax.arrow(ex, ey, math.cos(h) * 0.10 * rng, math.sin(h) * 0.10 * rng,
                 head_width=0.025 * rng, head_length=0.03 * rng,
                 fc="#5ac8ff", ec="#5ac8ff", length_includes_head=True, zorder=5)
        # POI + holding circle
        ax.plot(px, py, "*", color="#ff5a78", ms=15, zorder=6)
        r = cfg.loiter_radius(sc.speed_kt * KNOTS_TO_MS) / 1000.0
        ax.add_patch(plt.Circle((px, py), r, fill=False, ls="--",
                                ec="#888", lw=1.0, alpha=0.7))

        final = states[-1]
        max_bank = max(abs(math.degrees(s.phi)) for s in states)
        outcome = (f"landed  alt={final.z:.2f} m  v={final.speed_knots:.2f} kt"
                   if final.phase == Phase.LANDED
                   else f"loiter R={r * 1000:.0f} m")
        ax.set_title(f"{sc.name}\n{sc.speed_kt:.0f} kt, hdg {sc.heading_deg:.0f}°"
                     f"  |  max bank {max_bank:.1f}°\n{outcome}", fontsize=10)
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_aspect("equal")
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        ax.grid(alpha=0.2)

    legend = [Line2D([0], [0], color=c, lw=3, label=p.value)
              for p, c in [(Phase.CRUISE, PHASE_COLOR[Phase.CRUISE]),
                           (Phase.DIVERT, PHASE_COLOR[Phase.DIVERT]),
                           (Phase.LOITER, PHASE_COLOR[Phase.LOITER]),
                           (Phase.LANDING, PHASE_COLOR[Phase.LANDING])]]
    legend += [Line2D([0], [0], marker="*", color="w", markerfacecolor="#ff5a78",
                      markersize=13, label="point of interest")]
    fig.legend(handles=legend, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    fig.savefig(path, dpi=110)
    print("wrote", path)


def plot_smoothness(scenarios, results, path):
    """Time-series for the landing cases proving continuity/smoothness."""
    landing = [(sc, r) for sc, r in zip(scenarios, results) if sc.land]
    fig, axes = plt.subplots(len(landing), 4, figsize=(18, 4 * len(landing)))
    if len(landing) == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle("Landing cases: speed, altitude, bank and lateral acceleration "
                 "are continuous and smooth", fontsize=14)
    for row, (sc, (states, poi, cfg)) in zip(axes, landing):
        t = np.array([s.t for s in states])
        v = np.array([s.speed_knots for s in states])
        z = np.array([s.z for s in states]) / 1000.0
        phi = np.degrees(np.array([s.phi for s in states]))
        x = np.array([s.x for s in states])
        y = np.array([s.y for s in states])
        vx, vy = np.gradient(x, t), np.gradient(y, t)
        ax_, ay_ = np.gradient(vx, t), np.gradient(vy, t)
        a = np.hypot(ax_, ay_)
        sl = slice(3, -3)  # trim gradient edge artefacts
        for ax, data, lab, col in [
            (row[0], v[sl], "speed (kt)", "#5ac8ff"),
            (row[1], z[sl], "altitude (km)", "#7be88c"),
            (row[2], phi[sl], "bank (°)", "#ffaa3c"),
            (row[3], a[sl], "lateral accel (m/s²)", "#ff5a78"),
        ]:
            ax.plot(t[sl], data, color=col, lw=1.3)
            ax.set_xlabel("t (s)")
            ax.set_ylabel(lab)
            ax.grid(alpha=0.25)
        row[0].set_title(f"{sc.name}  ({sc.speed_kt:.0f} kt)", loc="left",
                         fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=110)
    print("wrote", path)


SCENARIOS = [
    Scenario("1. slow entry, east", 40, 0, (0, 7000)),
    Scenario("2. cruise, east", 70, 0, (0, 12000)),
    Scenario("3. fast, east", 100, 0, (0, 9000)),
    Scenario("4. diagonal NE", 80, 45, (1000, 1000)),
    Scenario("5. northbound", 55, 90, (10000, 0)),
    Scenario("6. SW-bound", 90, 200, (18000, 16000)),
    Scenario("7. land (slow)", 50, -30, (2000, 18000), land=True),
    Scenario("8. land (mid)", 75, 135, (17000, 3000), land=True),
    Scenario("9. land (fast)", 100, 250, (19000, 14000), land=True),
]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    results = [run(sc) for sc in SCENARIOS]
    for sc, (states, _, _) in zip(SCENARIOS, results):
        f = states[-1]
        print(f"{sc.name:22s} phase={f.phase.value:8s} "
              f"t={f.t:7.1f}s alt={f.z:8.1f} v={f.speed_knots:6.2f}kt")
    plot_grid(SCENARIOS, results, f"{out}/trajectories.png")
    plot_smoothness(SCENARIOS, results, f"{out}/landing_smoothness.png")


if __name__ == "__main__":
    main()
