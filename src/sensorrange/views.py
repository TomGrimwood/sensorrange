"""A sensor's field of view drawn round it, from above and from the side, in the range's frame
(ahead the SensorMount's ahead, left positive, up positive): what it covers shaded, where its
points came back as ticks, the way it faces."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from .ranges import angles
from .style import BLUE, GRID, INK, PAPER, RED, _font, _side, _updown


def box(right: float, left: float, low: float, high: float) -> tuple[np.ndarray, np.ndarray]:
    """Azimuths and elevations (degrees) on a 1° grid over a field of view."""
    az, el = np.meshgrid(
        np.linspace(right, left, max(int(left - right) + 1, 2)),
        np.linspace(low, high, max(int(high - low) + 1, 2)),
    )
    return az.ravel(), el.ravel()


def both(
    seen: np.ndarray | None,
    claimed: np.ndarray | None,
    facing: np.ndarray,
    upright: np.ndarray | None = None,
) -> list[Image.Image]:
    """From above and from the side, in the range's frame: directions `seen` (n, 3) as ticks, the
    directions `claimed` (n, 3; or `seen`) covers shaded, `facing` (3) the way it faces. The side
    view is the upright plane through where it faces (ahead when it faces straight up or down):
    elevation in front, 180° less it behind. `upright` (n, 3), if given, is what the side view
    shades instead (a camera's middle column, say, where `claimed` is its middle row)."""
    span = claimed if claimed is not None else seen
    if span is None:
        raise ValueError("A view needs directions seen or claimed.")
    turn, tip = (float(v) for v in angles(facing))
    toward = turn if abs(tip) < 80 else 0.0

    def above_and_side(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        az, el = angles(d)
        behind = np.abs((az - toward + 180) % 360 - 180) > 90
        return az, np.where(behind, 180 - el, el)

    ticks = above_and_side(seen) if seen is not None else (None, None)
    covered = above_and_side(span)
    standing = above_and_side(upright)[1] if upright is not None else covered[1]
    return [
        _view("from above", ticks[0], covered[0], False, turn if abs(tip) < 80 else None),
        _view("from the side", ticks[1], standing, True, tip),
    ]


def _runs(degrees: np.ndarray) -> list[tuple[int, int]]:
    """The whole degrees `degrees` covers, as runs (first, last) round the circle; [] when it
    covers it all."""
    hit = np.zeros(360, bool)
    hit[np.floor(degrees).astype(int) % 360] = True
    for _ in range(2):  # gaps of a degree or two are between samples, not in its view
        hit |= np.roll(hit, 1) & np.roll(hit, -1)
        hit |= np.roll(hit, 2) & np.roll(hit, -2)
    if hit.all():
        return []
    start = int(np.argmin(hit))  # a degree not covered: runs never wrap past it
    runs, first = [], None
    for k in range(start, start + 361):
        on = hit[k % 360]
        if on and first is None:
            first = k
        elif not on and first is not None:
            runs.append((first, k))
            first = None
    return [(a - 360 if a >= 180 else a, b - 360 if a >= 180 else b) for a, b in runs]


def _view(
    title: str,
    angles: np.ndarray | None,
    covered: np.ndarray,
    side: bool,
    facing: float | None = 0.0,
) -> Image.Image:
    """The sensor at the middle, ahead up the panel (from above) or to the right (from the
    side), a circle round it, the directions `covered` (degrees: left or up positive) shaded
    with each edge's angle, `angles` (degrees, its points' directions) as ticks just outside
    the circle, and the way it faces (`facing`, degrees) when that is not ahead."""
    size, margin = 560, 110
    r = size / 2 - margin
    c = size / 2
    panel = Image.new("RGBA", (size, size + 30), (*PAPER, 255))
    font = _font(14)

    def at(angle: float, out: float) -> tuple[float, float]:
        a = math.radians(angle)
        if side:  # ahead to the right, up up
            return c + out * math.cos(a), c - out * math.sin(a)
        return c - out * math.sin(a), c - out * math.cos(a)  # ahead up, left left

    ImageDraw.Draw(panel).ellipse((c - r, c - r, c + r, c + r), outline=GRID, width=2)
    runs = _runs(covered)
    shade = Image.new("RGBA", panel.size, (0, 0, 0, 0))
    if runs:
        for lo, hi in runs:
            arc = [at(a, r) for a in np.linspace(lo, hi, max(hi - lo, 2))]
            ImageDraw.Draw(shade).polygon([(c, c), *arc], fill=(*RED, 45))
    else:
        ImageDraw.Draw(shade).ellipse((c - r, c - r, c + r, c + r), fill=(*RED, 45))
    panel = Image.alpha_composite(panel, shade)
    pen = ImageDraw.Draw(panel)
    if angles is not None and len(angles):
        for a in np.unique(np.round(angles * 2) / 2):  # every half degree it saw
            pen.line([at(a, r + 4), at(a, r + 16)], fill=BLUE, width=2)

    def word(a: float) -> str:
        a = (a + 180) % 360 - 180
        if side and abs(a) > 90:
            return f"{180 - abs(a):.0f}° {'up' if a > 0 else 'down'} behind"
        return _updown(a, 0) if side else _side(a, 0)

    for edge in [e for run in runs for e in run]:
        pen.line([(c, c), at(edge, r)], fill=RED, width=2)
        x, y = at(edge, r + 24)
        said = word(edge)
        box = pen.textbbox((0, 0), said, font=font)
        lean = (x - c) / (r + 24)  # -1 at the left of the circle, 1 at its right: away from it
        left = min(max(x - box[2] * (1 - lean) / 2, 2), size - box[2] - 2)  # kept on the panel
        pen.text((left, y - box[3] / 2), said, fill=RED, font=font)
    if not runs:
        pen.text((c - 30, c + r + 24), "all round", fill=RED, font=font)
    level = at(0, r * 0.45)
    pen.line([(c, c), level], fill=INK, width=2)
    pen.text(
        (level[0] + (6 if side else -18), level[1] - (18 if side else 22)),
        "level" if side else "ahead",
        fill=INK,
        font=font,
    )
    if facing is not None and abs(facing) > 0.5:
        faces = at(facing, r * 0.7)
        pen.line([(c, c), faces], fill=RED, width=3)
        pen.text((faces[0] + 6, faces[1] - 18), f"faces {word(facing)}", fill=RED, font=font)
    pen.ellipse((c - 5, c - 5, c + 5, c + 5), fill=INK)
    pen.text((c, c + r + 34), title, fill=INK, font=_font(16), anchor="mt")  # just under the circle
    return panel.convert("RGB")
