"""A small fixed-wing aircraft simulator.

The package is split into a GUI-free physics engine (:mod:`aircraft_sim.physics`)
and a pygame front-end (:mod:`aircraft_sim.app`).  The physics engine is fully
self contained and is what the test-suite exercises.
"""

from .physics import (
    Constraints,
    Phase,
    Simulator,
    State,
    G,
    KNOTS_TO_MS,
    MS_TO_KNOTS,
)

__all__ = [
    "Constraints",
    "Phase",
    "Simulator",
    "State",
    "G",
    "KNOTS_TO_MS",
    "MS_TO_KNOTS",
]
