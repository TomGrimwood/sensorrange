"""A lidar's points from above: round it in any scene (`above`), and over a range's targets
with what each target gave back (`lane`)."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from .ranges import Target
from .seen import _ramp
from .style import GRID, INK, MUTED, PAPER, RED, _font

#: How far round a lidar its points are drawn from above in a scene (metres).
ABOVE = 25.0


def above(points: np.ndarray) -> Image.Image:
    """Points (n, 3; metres from the sensor, the range's axes) from above, `ABOVE` metres round
    it, ahead up the panel, coloured by height (low dark, high yellow), rings every 5 m."""
    from .sheets import _timed  # the sheets import this

    size = 560
    scale = (size / 2 - 10) / ABOVE
    c = size / 2
    panel = Image.new("RGB", (size, size + 30), PAPER)
    pen = ImageDraw.Draw(panel)
    for ring in np.arange(5, ABOVE + 0.1, 5):
        pen.ellipse(
            (c - ring * scale, c - ring * scale, c + ring * scale, c + ring * scale), outline=GRID
        )
        pen.text((c + 3, c - ring * scale + 2), f"{ring:g} m", fill=MUTED, font=_font(12))
    near = np.abs(points[:, :2]).max(axis=1) < ABOVE
    p = points[near]
    order = np.argsort(p[:, 2])  # high points drawn last
    p = p[order]
    xy = np.stack([c + p[:, 0] * scale, c - p[:, 1] * scale], axis=-1).astype(int)
    low, high = np.percentile(p[:, 2], [2, 98]) if len(p) else (0.0, 1.0)
    pixels = np.asarray(panel).copy()
    keep = (xy[:, 0] >= 0) & (xy[:, 0] < size) & (xy[:, 1] >= 0) & (xy[:, 1] < size)
    pixels[xy[keep, 1], xy[keep, 0]] = _timed(p[keep, 2] - low, high - low)
    panel = Image.fromarray(pixels)
    pen = ImageDraw.Draw(panel)
    pen.ellipse((c - 5, c - 5, c + 5, c + 5), fill=RED)
    pen.text(
        (10, size + 4), f"from above ({high - low:.1f} m low to high)", fill=INK, font=_font(16)
    )
    return panel


def lane(
    points: np.ndarray, strength: np.ndarray, targets: tuple[Target, ...]
) -> tuple[Image.Image, list[tuple[str, str, int, str, str]]]:
    """The range from above, small (ahead up): rings every 5 m, its targets thin grey lines,
    each point shaded by its intensity; and what every target gave back, a row each (through
    glass: the points beyond it in the directions it covers)."""
    from .sheets import _behind, _on  # the sheets import this

    corners = np.concatenate([t.corners[:, :2] for t in targets])  # the range, all round
    low, high = corners.min(axis=0) - 1, corners.max(axis=0) + 1
    scale = ABOVE_SIZE / max(high - low)
    w, h = int((high[0] - low[0]) * scale), int((high[1] - low[1]) * scale)
    panel = Image.new("RGB", (w, h), PAPER)
    pen = ImageDraw.Draw(panel)
    px = lambda x, y: ((x - low[0]) * scale, (high[1] - y) * scale)  # noqa: E731
    for metres in range(5, int(max(abs(low).max(), abs(high).max()) * 1.5), 5):  # rings round it
        x0, y0 = px(-metres, metres)
        x1, y1 = px(metres, -metres)
        pen.ellipse((x0, y0, x1, y1), outline=GRID)
        pen.text(px(0.3, metres), f"{metres} m", fill=MUTED, font=_font(12))
    for t in targets:
        c = t.corners
        pen.line([px(*c[0][:2]), px(*c[1][:2])], fill=(170, 170, 170), width=2)
    if len(points):
        shade = np.clip(strength / max(np.percentile(strength, 99), 1e-9), 0, 1)
        xy = np.array([px(x, y) for x, y in points[:, :2]]).astype(int)
        keep = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
        pixels = np.asarray(panel).copy()
        colours = np.array([_ramp(f) for f in shade[keep]], np.uint8)
        pixels[xy[keep, 1], xy[keep, 0]] = colours
        panel = Image.fromarray(pixels)
    ImageDraw.Draw(panel).text((6, 4), "range from above", fill=INK, font=_font(14))
    far = np.linalg.norm(points, axis=1)
    every: list[tuple[str, str, int, str, str]] = []
    for t in targets:
        if t.name.startswith(("pole", "board")) or t.name == "wall":
            continue
        on = _on(points, t)
        mean = float(strength[on].mean()) if on.any() else float("nan")
        clear = (
            "glass" in t.base or t.base == "plexiglass" or "visually_transparent" in t.attributes
        )
        beyond = far > np.linalg.norm(t.corners, axis=1).max() + 0.3
        through = str(int((_behind(points, t) & beyond).sum())) if clear else ""
        every.append(
            (t.name, tag(t), int(on.sum()), "" if math.isnan(mean) else f"{mean:.3g}", through)
        )
    return panel, every


#: How big the range from above is drawn (pixels, its longer side).
ABOVE_SIZE = 520


def tag(t: Target) -> str:
    """A target's material as RTX's lidar reads it: base, then coating and attributes if any."""
    said = [
        t.base,
        *([t.coating] if t.coating != "none" else []),
        *(a for a in t.attributes if a != "none"),
    ]
    return " + ".join(said)
