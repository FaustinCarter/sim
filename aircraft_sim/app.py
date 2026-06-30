"""pygame front-end for the aircraft simulator.

Run with::

    python -m aircraft_sim.app

Interaction flow
----------------
1. Click a point on the **edge** of the square to set the entry coordinate.
2. Move the mouse to aim the entry heading; **click** to lock it in.
3. Move the mouse back/forth along that heading to choose speed (0-100 kt);
   **click** to lock it and start the flight.
4. During the flight, **click anywhere** to divert into a holding pattern around
   that point of interest (which is typically behind the aircraft).
5. Once established, press **L** to spiral down and land, or **S** to stop.

At 100 kt the aircraft crosses the square edge-to-edge in 5 seconds; slower
speeds are proportionally slower.  Altitude starts at 10 000 m.
"""

from __future__ import annotations

import math
import sys
from enum import Enum, auto
from typing import List, Optional, Tuple

import pygame

from .physics import (
    Constraints,
    Phase,
    Simulator,
    KNOTS_TO_MS,
    MS_TO_KNOTS,
)

# --------------------------------------------------------------------------- #
# Layout / tuning
# --------------------------------------------------------------------------- #
AREA_SIZE = 20_000.0        # metres, physical size of the square
START_ALT = 10_000.0        # metres
MAX_SPEED_KT = 500.0        # selectable maximum speed
PACING_SPEED_KT = 100.0     # the speed whose edge-to-edge time is fixed
EDGE_TO_EDGE_AT_PACING = 5.0  # seconds to cross at PACING_SPEED_KT (display pacing)

CANVAS = 760                # px, the square flight area
MARGIN = 24
PANEL_W = 300
WIN_W = MARGIN * 2 + CANVAS + PANEL_W
WIN_H = MARGIN * 2 + CANVAS
FPS = 60

SPEED_DRAG_PX = 320.0       # mouse travel that maps to full speed

# Colours
BG = (12, 16, 24)
AREA_BG = (20, 28, 42)
GRID = (32, 44, 62)
EDGE = (70, 96, 130)
ACCENT = (90, 200, 255)
WARN = (255, 170, 60)
GOOD = (120, 230, 140)
POI_COL = (255, 90, 120)
TRAIL = (90, 130, 180)
TEXT = (210, 222, 235)
MUTED = (130, 150, 170)
GLYPH = (240, 246, 255)

# 3D ancillary view
V3D_BG = (16, 22, 33)
V3D_GROUND = (40, 70, 60)
V3D_GRID = (34, 50, 64)
V3D_STALK = (90, 110, 135)
V3D_SHADOW = (40, 52, 68)
V3D_W = 280
V3D_H = 244


class UI(Enum):
    PICK_ENTRY = auto()
    PICK_ANGLE = auto()
    PICK_SPEED = auto()
    FLYING = auto()
    DONE = auto()


# Time-compression so that PACING_SPEED_KT crosses the square in the pacing time.
def _time_scale() -> float:
    real_cross = AREA_SIZE / (PACING_SPEED_KT * KNOTS_TO_MS)
    return real_cross / EDGE_TO_EDGE_AT_PACING


class App:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Aircraft Holding-Pattern Simulator")
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas,menlo,monospace", 16)
        self.big = pygame.font.SysFont("consolas,menlo,monospace", 22, bold=True)
        self.scale = CANVAS / AREA_SIZE
        self.time_scale = _time_scale()
        # ancillary 3D camera (orbit angles); rotated by dragging in its panel
        self.cam_az = math.radians(-55.0)
        self.cam_el = math.radians(24.0)
        self.drag3d = False
        self.last_drag: Optional[Tuple[int, int]] = None
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.mode = UI.PICK_ENTRY
        self.entry: Optional[Tuple[float, float]] = None
        self.heading_deg: float = 0.0
        self.speed_kt: float = 0.0
        self.sim: Optional[Simulator] = None
        self.trail: List[Tuple[float, float, float]] = []  # (x, y, altitude)
        self.poi: Optional[Tuple[float, float]] = None
        self.mouse = (0, 0)

    # ------------------------------------------------------------------ #
    # Coordinate transforms (world metres <-> screen pixels)
    # ------------------------------------------------------------------ #
    def w2s(self, wx: float, wy: float) -> Tuple[float, float]:
        sx = MARGIN + wx * self.scale
        sy = MARGIN + (AREA_SIZE - wy) * self.scale
        return (sx, sy)

    def s2w(self, sx: float, sy: float) -> Tuple[float, float]:
        wx = (sx - MARGIN) / self.scale
        wy = AREA_SIZE - (sy - MARGIN) / self.scale
        return (wx, wy)

    def in_area(self, sx: float, sy: float) -> bool:
        return MARGIN <= sx <= MARGIN + CANVAS and MARGIN <= sy <= MARGIN + CANVAS

    def snap_to_edge(self, sx: float, sy: float) -> Tuple[float, float]:
        """Snap a screen point to the nearest edge of the flight square."""
        left, right = MARGIN, MARGIN + CANVAS
        top, bottom = MARGIN, MARGIN + CANVAS
        sx = min(max(sx, left), right)
        sy = min(max(sy, top), bottom)
        d = {"l": sx - left, "r": right - sx, "t": sy - top, "b": bottom - sy}
        side = min(d, key=d.get)
        if side == "l":
            sx = left
        elif side == "r":
            sx = right
        elif side == "t":
            sy = top
        else:
            sy = bottom
        return self.s2w(sx, sy)

    # ------------------------------------------------------------------ #
    # Event handling
    # ------------------------------------------------------------------ #
    def handle_click(self, pos) -> None:
        sx, sy = pos
        if self.mode == UI.PICK_ENTRY:
            if self.in_area(sx, sy):
                self.entry = self.snap_to_edge(sx, sy)
                self.mode = UI.PICK_ANGLE
        elif self.mode == UI.PICK_ANGLE:
            self.heading_deg = self._heading_from_mouse(pos)
            self.mode = UI.PICK_SPEED
        elif self.mode == UI.PICK_SPEED:
            self.speed_kt = self._speed_from_mouse(pos)
            self._start_flight()
        elif self.mode == UI.FLYING:
            if self.in_area(sx, sy) and self.sim is not None:
                if self.sim.phase in (Phase.CRUISE, Phase.DIVERT, Phase.LOITER):
                    self.poi = self.s2w(sx, sy)
                    self.sim.command_divert(self.poi)

    def _heading_from_mouse(self, pos) -> float:
        ex, ey = self.w2s(*self.entry)
        return math.degrees(math.atan2(-(pos[1] - ey), pos[0] - ex))

    def _speed_from_mouse(self, pos) -> float:
        ex, ey = self.w2s(*self.entry)
        h = math.radians(self.heading_deg)
        # heading unit vector in *screen* space (y is down)
        hx, hy = math.cos(h), -math.sin(h)
        proj = (pos[0] - ex) * hx + (pos[1] - ey) * hy
        frac = max(0.0, min(1.0, proj / SPEED_DRAG_PX))
        return frac * MAX_SPEED_KT

    def _start_flight(self) -> None:
        self.sim = Simulator(
            entry=self.entry, heading_deg=self.heading_deg,
            speed_knots=self.speed_kt, area_size=AREA_SIZE,
            start_altitude=START_ALT,
        )
        self.trail = [(self.entry[0], self.entry[1], START_ALT)]
        self.mode = UI.FLYING

    def handle_key(self, key) -> None:
        if key == pygame.K_r:
            self.reset()
        elif key == pygame.K_l and self.sim is not None:
            self.sim.command_land()
        elif key == pygame.K_s and self.sim is not None:
            self.sim.command_stop()

    # ------------------------------------------------------------------ #
    # Update
    # ------------------------------------------------------------------ #
    def update(self, wall_dt: float) -> None:
        if self.mode != UI.FLYING or self.sim is None:
            return
        sim_dt = wall_dt * self.time_scale
        # sub-step for integration accuracy/smoothness
        n = max(1, int(sim_dt / 0.05) + 1)
        h = sim_dt / n
        for _ in range(n):
            self.sim.step(h)
            if self.sim.phase in (Phase.LANDED, Phase.STOPPED):
                break
        st = self.sim.state
        if not self.trail or math.hypot(st.x - self.trail[-1][0], st.y - self.trail[-1][1]) > 30:
            self.trail.append((st.x, st.y, st.z))
        if len(self.trail) > 4000:
            self.trail = self.trail[-4000:]

    # ------------------------------------------------------------------ #
    # Drawing
    # ------------------------------------------------------------------ #
    def draw(self) -> None:
        self.screen.fill(BG)
        self._draw_area()
        if self.mode == UI.PICK_ANGLE:
            self._draw_aim()
        elif self.mode == UI.PICK_SPEED:
            self._draw_aim()
            self._draw_speed_preview()
        elif self.mode in (UI.FLYING, UI.DONE):
            self._draw_flight()
        self._draw_panel()
        self._draw_view3d()
        pygame.display.flip()

    def _draw_area(self) -> None:
        pygame.draw.rect(self.screen, AREA_BG, (MARGIN, MARGIN, CANVAS, CANVAS))
        for i in range(1, 10):
            p = MARGIN + CANVAS * i / 10
            pygame.draw.line(self.screen, GRID, (p, MARGIN), (p, MARGIN + CANVAS))
            pygame.draw.line(self.screen, GRID, (MARGIN, p), (MARGIN + CANVAS, p))
        pygame.draw.rect(self.screen, EDGE, (MARGIN, MARGIN, CANVAS, CANVAS), 2)
        if self.entry is not None:
            ex, ey = self.w2s(*self.entry)
            pygame.draw.circle(self.screen, ACCENT, (int(ex), int(ey)), 6, 2)

    def _draw_aim(self) -> None:
        ex, ey = self.w2s(*self.entry)
        h = math.radians(self.heading_deg if self.mode == UI.PICK_SPEED
                         else self._heading_from_mouse(self.mouse))
        tip = (ex + math.cos(h) * 120, ey - math.sin(h) * 120)
        pygame.draw.line(self.screen, ACCENT, (ex, ey), tip, 2)
        self._arrowhead((ex, ey), tip, ACCENT)
        deg = self.heading_deg if self.mode == UI.PICK_SPEED else self._heading_from_mouse(self.mouse)
        self._label(f"{deg:6.1f} deg", (ex + 8, ey - 24), ACCENT)

    def _draw_speed_preview(self) -> None:
        ex, ey = self.w2s(*self.entry)
        kt = self._speed_from_mouse(self.mouse)
        h = math.radians(self.heading_deg)
        length = 40 + (kt / MAX_SPEED_KT) * 200
        tip = (ex + math.cos(h) * length, ey - math.sin(h) * length)
        pygame.draw.line(self.screen, GOOD, (ex, ey), tip, 4)
        self._arrowhead((ex, ey), tip, GOOD)
        self._label(f"{kt:5.1f} kt", (tip[0] + 6, tip[1] - 8), GOOD)

    def _draw_flight(self) -> None:
        st = self.sim.state
        # trail
        if len(self.trail) > 1:
            pts = [self.w2s(p[0], p[1]) for p in self.trail]
            pygame.draw.lines(self.screen, TRAIL, False, pts, 2)
        # POI + holding circle
        if self.poi is not None:
            px, py = self.w2s(*self.poi)
            pygame.draw.circle(self.screen, POI_COL, (int(px), int(py)), 6)
            pygame.draw.line(self.screen, POI_COL, (px - 10, py), (px + 10, py), 1)
            pygame.draw.line(self.screen, POI_COL, (px, py - 10), (px, py + 10), 1)
            r = self.sim.cfg.loiter_radius(st.v) * self.scale
            if r > 2:
                pygame.draw.circle(self.screen, (70, 50, 70), (int(px), int(py)), int(r), 1)
        self._draw_glyph(st)

    def _draw_glyph(self, st) -> None:
        cx, cy = self.w2s(st.x, st.y)
        # non-symmetric arrowhead: long nose, swept tail
        local = [(16, 0), (-10, 9), (-5, 0), (-10, -9)]
        ca, sa = math.cos(st.psi), math.sin(st.psi)
        pts = []
        for lx, ly in local:
            rx = lx * ca - ly * sa
            ry = lx * sa + ly * ca
            pts.append((cx + rx, cy - ry))  # y flip for screen
        # bank tint: redder as bank increases
        bank_frac = min(1.0, abs(st.phi) / self.sim.cfg.max_bank)
        col = (240, int(246 - 120 * bank_frac), int(255 - 160 * bank_frac))
        pygame.draw.polygon(self.screen, col, pts)
        pygame.draw.polygon(self.screen, (20, 30, 45), pts, 1)

    # ------------------------------------------------------------------ #
    def _draw_panel(self) -> None:
        x0 = MARGIN + CANVAS + 20
        y = MARGIN
        self._label("AIRCRAFT SIM", (x0, y), ACCENT, big=True)
        y += 34
        for line in self._instructions():
            self._label(line, (x0, y), MUTED)
            y += 20
        y += 8
        if self.sim is not None:
            st = self.sim.state
            rows = [
                ("phase", st.phase.value.upper()),
                ("speed", f"{st.speed_knots:6.1f} kt"),
                ("altitude", f"{st.z:8.0f} m"),
                ("heading", f"{(math.degrees(st.psi)) % 360:6.1f} deg"),
                ("bank", f"{math.degrees(st.phi):6.1f} deg"),
                ("v.speed", f"{st.vz:6.1f} m/s"),
            ]
            for k, v in rows:
                self._label(f"{k:>9}: {v}", (x0, y), TEXT)
                y += 22
            y += 6
            self._draw_alt_bar(x0, y, st)

    def _draw_alt_bar(self, x0: int, y0: int, st) -> None:
        h = 220
        w = 26
        pygame.draw.rect(self.screen, (30, 40, 56), (x0, y0, w, h))
        pygame.draw.rect(self.screen, EDGE, (x0, y0, w, h), 1)
        frac = max(0.0, min(1.0, st.z / START_ALT))
        fh = int(h * frac)
        col = GOOD if frac > 0.05 else WARN
        pygame.draw.rect(self.screen, col, (x0, y0 + (h - fh), w, fh))
        self._label("ALT", (x0, y0 + h + 4), MUTED)
        self._label(f"{st.z:6.0f} m", (x0 + w + 8, y0 + h - 14), TEXT)
        self._label(f"{START_ALT:6.0f} m", (x0 + w + 8, y0), MUTED)

    # ------------------------------------------------------------------ #
    # Ancillary 3D perspective view (orbit by dragging inside its panel)
    # ------------------------------------------------------------------ #
    def _view3d_rect(self) -> "pygame.Rect":
        x = MARGIN + CANVAS + 20
        y = WIN_H - MARGIN - V3D_H
        return pygame.Rect(x, y, V3D_W, V3D_H)

    def _rotate_cam(self, pos) -> None:
        dx = pos[0] - self.last_drag[0]
        dy = pos[1] - self.last_drag[1]
        self.cam_az -= dx * 0.01
        self.cam_el = max(math.radians(-10.0),
                          min(math.radians(85.0), self.cam_el + dy * 0.01))
        self.last_drag = pos

    def _project3d(self, X, Y, Z, rect):
        """Perspective-project a world point into the 3D viewport (or None)."""
        cx = cy = AREA_SIZE * 0.5
        cz = START_ALT * 0.5
        dx, dy, dz = X - cx, Y - cy, Z - cz
        ca, sa = math.cos(self.cam_az), math.sin(self.cam_az)
        x1 = dx * ca - dy * sa
        y1 = dx * sa + dy * ca
        ce, se = math.cos(self.cam_el), math.sin(self.cam_el)
        y2 = y1 * ce - dz * se
        z2 = y1 * se + dz * ce
        depth = y2 + AREA_SIZE * 2.4
        if depth < 1e-3:
            return None
        f = V3D_W * 1.25
        return (rect.centerx + f * x1 / depth, rect.centery - f * z2 / depth, depth)

    def _seg3d(self, p1, p2, rect, col, w=1) -> None:
        a = self._project3d(*p1, rect)
        b = self._project3d(*p2, rect)
        if a and b:
            pygame.draw.line(self.screen, col, a[:2], b[:2], w)

    def _draw_view3d(self) -> None:
        rect = self._view3d_rect()
        self._label("3D VIEW  (drag to rotate)", (rect.x, rect.y - 20), MUTED)
        pygame.draw.rect(self.screen, V3D_BG, rect)
        prev = self.screen.get_clip()
        self.screen.set_clip(rect)
        try:
            self._draw_ground3d(rect)
            if self.sim is not None:
                self._draw_aircraft3d(rect)
        finally:
            self.screen.set_clip(prev)
        pygame.draw.rect(self.screen, EDGE, rect, 1)

    def _draw_ground3d(self, rect) -> None:
        n = 4
        step = AREA_SIZE / n
        for i in range(n + 1):
            self._seg3d((i * step, 0, 0), (i * step, AREA_SIZE, 0), rect, V3D_GRID)
            self._seg3d((0, i * step, 0), (AREA_SIZE, i * step, 0), rect, V3D_GRID)
        corners = [(0, 0, 0), (AREA_SIZE, 0, 0),
                   (AREA_SIZE, AREA_SIZE, 0), (0, AREA_SIZE, 0)]
        for i in range(4):
            self._seg3d(corners[i], corners[(i + 1) % 4], rect, V3D_GROUND, 2)
        if self.entry is not None:
            e = self._project3d(self.entry[0], self.entry[1], 0, rect)
            if e:
                pygame.draw.circle(self.screen, ACCENT, (int(e[0]), int(e[1])), 3, 1)

    def _draw_aircraft3d(self, rect) -> None:
        st = self.sim.state
        # POI on the ground and the holding circle at the current altitude
        if self.poi is not None:
            pg = self._project3d(self.poi[0], self.poi[1], 0, rect)
            if pg:
                pygame.draw.circle(self.screen, POI_COL, (int(pg[0]), int(pg[1])), 4)
            r = self.sim.cfg.loiter_radius(st.v)
            if r > 1 and st.phase in (Phase.DIVERT, Phase.LOITER, Phase.LANDING):
                ring = []
                for k in range(33):
                    a = 2 * math.pi * k / 32
                    p = self._project3d(self.poi[0] + r * math.cos(a),
                                        self.poi[1] + r * math.sin(a), st.z, rect)
                    if p:
                        ring.append(p[:2])
                if len(ring) > 1:
                    pygame.draw.lines(self.screen, (120, 80, 95), True, ring, 1)
        # 3D trail
        if len(self.trail) > 1:
            pts = [p[:2] for p in (self._project3d(x, y, z, rect)
                                   for x, y, z in self.trail) if p]
            if len(pts) > 1:
                pygame.draw.lines(self.screen, TRAIL, False, pts, 1)
        # altitude stalk + ground shadow
        base = self._project3d(st.x, st.y, 0, rect)
        top = self._project3d(st.x, st.y, st.z, rect)
        if base and top:
            pygame.draw.line(self.screen, V3D_STALK, base[:2], top[:2], 1)
            pygame.draw.circle(self.screen, V3D_SHADOW, (int(base[0]), int(base[1])), 3)
        # heading arrow + body
        nose = self._project3d(st.x + math.cos(st.psi) * 1600.0,
                               st.y + math.sin(st.psi) * 1600.0, st.z, rect)
        if top and nose:
            pygame.draw.line(self.screen, GLYPH, top[:2], nose[:2], 2)
            self._arrowhead(top[:2], nose[:2], GLYPH)
        if top:
            pygame.draw.circle(self.screen, ACCENT, (int(top[0]), int(top[1])), 4)

    def _instructions(self) -> List[str]:
        if self.mode == UI.PICK_ENTRY:
            return ["1. Click a point on the", "   square's edge (entry)."]
        if self.mode == UI.PICK_ANGLE:
            return ["2. Move mouse to aim the", "   heading; click to lock."]
        if self.mode == UI.PICK_SPEED:
            return ["3. Slide along the heading", "   for speed (0-100 kt);",
                    "   click to launch."]
        if self.mode == UI.FLYING and self.sim is not None:
            p = self.sim.phase
            if p == Phase.CRUISE:
                return ["Click anywhere to set a", "point of interest and", "divert to a holding turn."]
            if p in (Phase.DIVERT, Phase.LOITER):
                return ["[L] spiral down & land", "[S] stop simulation", "[R] reset"]
            if p == Phase.LANDING:
                return ["Landing...", "[R] reset"]
            return ["Simulation finished.", "[R] reset"]
        return ["[R] reset"]

    # ------------------------------------------------------------------ #
    # small drawing helpers
    # ------------------------------------------------------------------ #
    def _arrowhead(self, a, b, col) -> None:
        ang = math.atan2(b[1] - a[1], b[0] - a[0])
        for s in (-1, 1):
            d = ang + s * math.radians(150)
            pygame.draw.line(self.screen, col, b,
                             (b[0] + math.cos(d) * 12, b[1] + math.sin(d) * 12), 2)

    def _label(self, text, pos, col, big=False) -> None:
        font = self.big if big else self.font
        self.screen.blit(font.render(text, True, col), pos)

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        running = True
        while running:
            wall_dt = self.clock.tick(FPS) / 1000.0
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.MOUSEMOTION:
                    self.mouse = ev.pos
                    if self.drag3d and self.last_drag is not None:
                        self._rotate_cam(ev.pos)
                elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    if self._view3d_rect().collidepoint(ev.pos):
                        self.drag3d = True       # rotate the ancillary 3D view
                        self.last_drag = ev.pos
                    else:
                        self.handle_click(ev.pos)
                elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                    self.drag3d = False
                    self.last_drag = None
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        running = False
                    else:
                        self.handle_key(ev.key)
            self.update(wall_dt)
            self.draw()
        pygame.quit()


def main() -> None:
    App().run()


if __name__ == "__main__":
    main()
