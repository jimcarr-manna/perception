"""
Lightweight visualisers for the sector-proximity message.

Two renderers are provided, both with the same ``update(msg)`` / ``close()``
interface so they are interchangeable:

* :class:`SectorFanViz` — a Tkinter half-disc fan of wedges. Tkinter draws
  vector primitives (polygons / arcs), so over a remote X11 link
  (e.g. xQuartz + SSH) only the draw commands travel — not a raster frame —
  keeping per-frame payload tiny compared to ``cv2.imshow``. Wedges are created
  once and only recoloured / resized each frame via ``itemconfig`` + ``coords``.

* :class:`SectorAsciiViz` — a pure terminal renderer (one coloured bar per
  sector, refreshed in place with ANSI escapes). No X server at all: a few KB
  of text per frame travels over the plain SSH channel, so it is the lightest
  and most robust option for a remote session.

Usage::

    viz = SectorFanViz(num_sectors=36, max_range=10.0)   # or SectorAsciiViz(...)
    ...
    msg = generate_sector_proximity(lidar_xy, feet_bev)
    viz.update(msg)   # Tk viz must be called from its creating thread
"""
from __future__ import annotations

import math
from typing import Any

# 0 = CLEAR, 1 = LIDAR_STATIC, 2 = HUMAN_DANGEROUS
_FILL = {
    0: "#1f2a30",   # faint slate — sector is clear
    1: "#d2a106",   # amber — static LiDAR obstacle
    2: "#e0152b",   # red — human (safety critical)
}
_BG = "#0a0f12"
_RING = "#243038"
_AXIS = "#3a4a54"
_TEXT = "#7d909c"


class SectorFanViz:
    """A minimal Tkinter half-disc radar for the sector-proximity array."""

    def __init__(
        self,
        num_sectors: int = 36,
        max_range: float = 10.0,
        size: int = 460,
        title: str = "Sector Proximity",
    ) -> None:
        self.enabled = False
        self.num_sectors = num_sectors
        self.max_range = float(max_range)

        try:
            import tkinter as tk
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"[sector_viz] Tkinter unavailable, viz disabled: {exc}")
            return

        try:
            self._tk = tk
            self._root = tk.Tk()
            self._root.title(title)
            self._root.configure(bg=_BG)

            self._w = size
            self._h = size // 2 + 30
            self._cx = self._w / 2.0
            self._cy = self._h - 20.0
            self._r_px = min(self._cx, self._cy) - 12.0

            self._canvas = tk.Canvas(
                self._root, width=self._w, height=self._h,
                bg=_BG, highlightthickness=0,
            )
            self._canvas.pack()

            self._draw_static()
            self._wedges: list[int] = []
            for _ in range(num_sectors):
                self._wedges.append(
                    self._canvas.create_polygon(
                        0, 0, 0, 0, 0, 0,
                        fill=_FILL[0], outline=_BG, width=1,
                    )
                )
            self._root.update()
            self.enabled = True
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"[sector_viz] failed to create window, viz disabled: {exc}")
            self.enabled = False

    # ── geometry ──────────────────────────────────────────────────────────
    def _xy(self, ang: float, r_px: float) -> tuple[float, float]:
        """Robot-relative angle (0 = forward/up, + = left) → canvas pixel."""
        return (self._cx - r_px * math.sin(ang), self._cy - r_px * math.cos(ang))

    def _wedge_pts(self, a0: float, a1: float, r_px: float) -> list[float]:
        pts = [self._cx, self._cy]
        steps = 2
        for k in range(steps + 1):
            a = a0 + (a1 - a0) * k / steps
            x, y = self._xy(a, r_px)
            pts += [x, y]
        return pts

    def _draw_static(self) -> None:
        c = self._canvas
        # Range rings at fractions of max range (half-disc arcs).
        for frac in (0.25, 0.5, 0.75, 1.0):
            r = self._r_px * frac
            c.create_arc(
                self._cx - r, self._cy - r, self._cx + r, self._cy + r,
                start=0, extent=180, style=self._tk.ARC, outline=_RING,
            )
            c.create_text(
                self._cx + 3, self._cy - r + 7,
                text=f"{self.max_range * frac:.0f}m", fill=_TEXT, anchor="w",
                font=("TkDefaultFont", 7),
            )
        # Forward axis + side markers.
        c.create_line(self._cx, self._cy, *self._xy(0.0, self._r_px), fill=_AXIS)
        c.create_text(self._cx, self._cy - self._r_px - 8, text="fwd",
                      fill=_TEXT, font=("TkDefaultFont", 8))
        c.create_text(self._cx - self._r_px + 2, self._cy + 10, text="+90 L",
                      fill=_TEXT, anchor="w", font=("TkDefaultFont", 7))
        c.create_text(self._cx + self._r_px - 2, self._cy + 10, text="R -90",
                      fill=_TEXT, anchor="e", font=("TkDefaultFont", 7))

    # ── per-frame update ──────────────────────────────────────────────────
    def update(self, msg: dict[str, Any]) -> None:
        if not self.enabled:
            return
        ranges = msg["ranges"]
        types = msg["object_types"]
        a_min = msg["angle_min"]
        a_inc = msg["angle_increment"]
        try:
            for i, wid in enumerate(self._wedges):
                r = ranges[i]
                t = int(types[i])
                # Sector edge angles; +0.5 keeps a thin gap between wedges.
                a0 = a_min + i * a_inc
                a1 = a_min + (i + 1) * a_inc
                frac = max(0.0, min(1.0, r / self.max_range))
                # Clear sectors drawn faint at full range; obstacles drawn to
                # their detected distance so closer = shorter, brighter wedge.
                r_px = self._r_px if t == 0 else self._r_px * frac
                self._canvas.coords(wid, *self._wedge_pts(a0, a1, r_px))
                self._canvas.itemconfig(wid, fill=_FILL.get(t, _FILL[0]))
            self._root.update()
        except self._tk.TclError:
            # Window closed by the user — stop trying to draw.
            self.enabled = False

    def close(self) -> None:
        if not self.enabled:
            return
        try:
            self._root.destroy()
        except Exception:
            pass
        self.enabled = False


# ANSI SGR colour codes for the terminal renderer (0/1/2 → clear/lidar/human).
_ANSI = {0: "32", 1: "33", 2: "31"}   # green, amber, red
_GLYPH = {0: "·", 1: "#", 2: "!"}
_LABEL = {0: "clear", 1: "lidar", 2: "HUMAN"}


class SectorAsciiViz:
    """A pure-terminal radar: one coloured bar per sector, refreshed in place.

    Emits only text (a few KB/frame) over the SSH channel — no X server — so it
    is the lightest, most link-robust view. Bar length is proportional to the
    free range in that sector (longer bar = more open space); the glyph and
    colour encode the object type.
    """

    def __init__(
        self,
        num_sectors: int = 36,
        max_range: float = 10.0,
        width: int = 36,
        use_color: bool = True,
    ) -> None:
        self.enabled = True
        self.num_sectors = num_sectors
        self.max_range = float(max_range)
        self.width = int(width)
        self.use_color = use_color
        self._cleared = False

    def _row(self, deg: float, r: float, t: int) -> str:
        n = int(round(self.width * max(0.0, min(1.0, r / self.max_range))))
        glyph = _GLYPH.get(t, "·")
        bar = glyph * n + " " * (self.width - n)
        text = f"{deg:+6.1f} |{bar}| {r:5.2f}m {_LABEL.get(t, '?')}"
        if self.use_color:
            return f"\033[{_ANSI.get(t, '37')}m{text}\033[0m"
        return text

    def update(self, msg: dict[str, Any]) -> None:
        if not self.enabled:
            return
        ranges = msg["ranges"]
        types = msg["object_types"]
        a_min = msg["angle_min"]
        a_inc = msg["angle_increment"]

        lines = ["Sector proximity  (deg | range bar | distance / type)"]
        # Highest angle first (left → right on screen, top → bottom in terminal).
        for i in reversed(range(len(ranges))):
            deg = math.degrees(a_min + (i + 0.5) * a_inc)
            lines.append(self._row(deg, float(ranges[i]), int(types[i])))

        out = "\033[H" + "\n".join(line + "\033[K" for line in lines)
        if not self._cleared:
            out = "\033[2J" + out   # clear once so the first frame is clean
            self._cleared = True
        out += "\033[J"             # erase any leftover rows below
        print(out, end="", flush=True)

    def close(self) -> None:
        if not self.enabled:
            return
        # Drop the cursor below the rendered block so the shell prompt is clean.
        print(f"\033[{self.num_sectors + 2}B", flush=True)
        self.enabled = False
