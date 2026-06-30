# Aircraft Holding-Pattern Simulator

A small native Python app that simulates a fixed-wing aircraft entering a square
area on a constant-velocity, constant-altitude vector, then diverting into a
**holding pattern around a point of interest that is behind it** (requiring a
turn of more than 90°), and optionally **spiralling down to land** on that point.

The project is split in two:

| Module | Responsibility |
| --- | --- |
| `aircraft_sim/physics.py` | GUI-free physics / guidance engine (what the tests exercise) |
| `aircraft_sim/app.py` | pygame front-end (2-D top-down view + altitude indicator) |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# run the app
python -m aircraft_sim.app

# run the physics test-suite
pytest
```

## Using the app

The flight is set up with three mouse clicks, exactly as in the brief:

1. **Click a point on the edge** of the square → entry coordinate (the click is
   snapped to the nearest edge).
2. **Move the mouse** to aim the entry heading — a live arrow and angle read-out
   track the mouse — and **click** to lock it.
3. **Slide the mouse back and forth along that heading** to pick the speed
   (0–100 kt, shown live) and **click** to launch.

While flying:

* **Click anywhere** to drop a point of interest and divert into a holding turn
  around it.
* Press **L** to spiral down and land on the point of interest.
* Press **S** to stop, **R** to reset, **Esc** to quit.

At 100 kt the aircraft crosses the square edge-to-edge in 5 seconds; slower
speeds are proportionally slower. The aircraft glyph is an asymmetric arrowhead
so its heading is always obvious, and it tints toward red as it banks.

## Units and scale

* Speed is in **knots**, altitude in **metres**, ground is at altitude 0, and
  every flight starts at **10 000 m**.
* The square represents a **20 km** physical area. This keeps the maneuvering
  realistic: a 100-kt aircraft limited to a 30° bank has a ~580 m holding
  radius, which comfortably fits inside the area. The on-screen pacing
  ("5 s edge-to-edge at 100 kt") is a display-time compression layered on top of
  the real-time physics, so the physics stays honest while the animation stays
  watchable.

## How the physics stays smooth

The engine guarantees the trajectory is continuous, differentiable and smooth in
position, velocity, altitude and acceleration:

* **Speed and altitude** follow analytic *smootherstep* profiles whose first and
  second derivatives vanish at the ends — so they are C².
* **Turning** uses a coordinated bank. A vector-field guidance law produces a
  *continuous* commanded turn-rate; the bank slews to it through a rate-limited
  first-order lag, so the bank — and therefore the heading-rate and the lateral
  acceleration — are continuous. The bank command is softly saturated and can
  never exceed **30°** (`bank = atan(v² / (R·g))`).
* **Holding radius is set purely by speed**: `R = v² / (g·tan 25°)`, and the
  aircraft converges onto the circle **tangentially**.
* **Landing** keeps the turn-rate bounded by shrinking the target radius
  linearly with speed, so as `v → 0` the bank → 0, the spiral converges onto the
  point of interest, and the aircraft touches down at **exactly zero altitude
  and zero velocity** with an arrested descent rate.
* Discrete pilot events (divert / land) are blended in with smootherstep ramps
  so nothing jumps at the moment of the command.

## Tests

`tests/test_physics.py` runs a matrix of initial conditions (speeds, headings,
entry points, with and without landing) and asserts:

* position, velocity, altitude and acceleration are continuous and smooth
  (bounded jerk, no spikes);
* the bank never exceeds 30°, speed never exceeds the maximum, and the turn
  radius never beats the speed-dependent minimum;
* the cruise leg is straight and level;
* the aircraft captures the holding pattern, on a tangent, at the
  speed-determined radius, after a turn exceeding 90°;
* a commanded landing ends at zero altitude and zero velocity, on the point of
  interest, with a monotonic descent.
```bash
pytest -q
```
