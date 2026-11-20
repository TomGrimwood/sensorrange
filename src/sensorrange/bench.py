"""A lidar on a reflectance bench (a scene labelled so): what each grey card gave back, and how
far it detects each grey, as datasheets put it.

Each card is a target named `grey<percent>_<angle>deg_<metres>m`: its reflectance (% at 905 nm),
the angle it is turned from facing the sensor (its incidence), and its distance. A card is
detected when at least half the scans put a point on it. A point is on a card when RTX says it
came back from the card's prim (the capture's `hits`), so the room, floor and posts never count,
whatever they are made of; for a capture without them, when it lies on the card's face.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .ranges import Target
from .style import GRID, INK, MUTED, PAPER, _font

NAME = re.compile(r"grey(\d+)_(\d+)deg_(\d+)m")
#: The share of scans a card must get a point in to count as detected; how near its face (m) a
#: point must be to count as on it (the cards stand close: a turned card's plane passes by the
#: next one).
DETECTED, SLACK = 0.5, 0.04
#: A colour for each line on the plots, in order.
LINES = ((20, 90, 200), (220, 120, 20), (40, 150, 70), (200, 40, 40), (130, 70, 180), (90, 90, 90))


@dataclass(frozen=True)
class Card:
    """What one card gave back: its reflectance, angle, distance; points a scan, the share of
    scans with any, and their mean intensity (NaN for none); its lidar material."""

    percent: int
    angle: int
    metres: int
    per_scan: float
    seen: float
    intensity: float
    base: str = ""

    @property
    def detected(self) -> bool:
        return self.seen >= DETECTED


def card_of(paths: list[str]) -> np.ndarray:
    """The card each of `paths` (prims a lidar's points came back from) is part of: its name,
    or "" for anything else."""
    names = []
    for path in paths:
        part = next((p for p in path.split("/") if NAME.fullmatch(p)), "")
        names.append(part)
    return np.array([*names, ""], dtype=object)  # last: for format.NO_HIT, mapped by the caller


def cards(
    scans: list[tuple[np.ndarray, np.ndarray, np.ndarray | None]],
    targets: tuple[Target, ...],
    on: Callable[[np.ndarray, Target], np.ndarray],
) -> list[Card]:
    """Each card's returns over `scans` (each its points in the range's frame, their
    intensities, and the card each came back from, `card_of`; None if not recorded, then
    counted on its face by `on`)."""
    found = []
    for t in targets:
        said = NAME.fullmatch(t.name)
        if said is None:
            continue
        counts, strength = [], []
        for points, intensity, came in scans:
            hit = came == t.name if came is not None else on(points, t)
            counts.append(int(hit.sum()))
            strength.append(intensity[hit])
        hits = np.concatenate(strength) if strength else np.zeros(0)
        found.append(
            Card(
                *(int(v) for v in said.groups()),
                float(np.mean(counts)) if counts else 0.0,
                float(np.mean([c > 0 for c in counts])) if counts else 0.0,
                float(hits.mean()) if len(hits) else float("nan"),
                t.base,
            )
        )
    return found


def reach(found: list[Card], angle: int = 0) -> list[tuple[str, str]]:
    """How far each reflectance is detected, cards facing it (`angle`): the furthest distance
    it is detected at, and so at every nearer one ("to 10 m"), or where it drops out."""
    rows = []
    for percent in sorted({c.percent for c in found}):
        mine = sorted(
            (c for c in found if c.percent == percent and c.angle == angle), key=lambda c: c.metres
        )
        far = 0
        for c in mine:
            if not c.detected:
                break
            far = c.metres
        said = f"to {far} m" if far else "not at all"
        if far and far == mine[-1].metres:
            said = f"to {far} m and on"
        base = next((c.base for c in mine), "") if len({c.base for c in found}) > 1 else ""
        rows.append((f"{percent} % {base}".strip(), said))
    return rows


def panels(found: list[Card]) -> list[Image.Image]:
    """The table of every card, and the two plots: intensity against reflectance (a line a
    distance, cards facing it) and against incidence (a line a reflectance, at the nearest
    distance)."""
    if not found:
        return []
    return [_table(found), _plot_reflectance(found), _plot_angle(found)]


def _table(found: list[Card]) -> Image.Image:
    """Points a scan on each card: a row each distance and angle, a column each reflectance;
    green detected, red not (blank: none at all)."""
    percents = sorted({c.percent for c in found})
    rows = sorted({(c.metres, c.angle) for c in found})
    at = {(c.metres, c.angle, c.percent): c for c in found}
    cell, left, top = 88, 130, 58
    picture = Image.new("RGB", (left + cell * len(percents) + 12, top + 28 * len(rows) + 12), PAPER)
    pen = ImageDraw.Draw(picture)
    pen.text((8, 6), "points / scan", fill=INK, font=_font(15))
    bases = {c.percent: c.base for c in found} if len({c.base for c in found}) > 1 else {}
    for i, percent in enumerate(percents):
        pen.text((left + cell * i + 8, 22), f"{percent} %", fill=INK, font=_font(14, bold=True))
        pen.text((left + cell * i + 8, 38), bases.get(percent, "")[:10], fill=MUTED, font=_font(11))
    for j, (metres, angle) in enumerate(rows):
        y = top + 28 * j
        if j and rows[j - 1][0] != metres:
            pen.line((8, y - 2, picture.width - 8, y - 2), fill=GRID, width=2)
        pen.text((8, y + 4), f"{metres:>2} m  {angle:>2}°", fill=INK, font=_font(14))
        for i, percent in enumerate(percents):
            c = at.get((metres, angle, percent))
            if c is None:
                continue
            fill = (210, 240, 210) if c.detected else (245, 210, 205) if c.per_scan else PAPER
            box = (left + cell * i, y, left + cell * (i + 1) - 4, y + 24)
            pen.rectangle(box, fill=fill)
            pen.text(
                (box[0] + 8, y + 4),
                f"{c.per_scan:.1f}" if c.per_scan else "-",
                fill=INK,
                font=_font(14),
            )
    return picture


def _plot(
    title: str,
    xs: list[float],
    x_label: str,
    lines: list[tuple[str, list[float]]],
) -> Image.Image:
    """A small plot of mean intensity (y) against `xs`, one coloured line each of `lines`
    (label, values; NaN: nothing there)."""
    w, h, margin = 620, 400, 64
    picture = Image.new("RGB", (w, h), PAPER)
    pen = ImageDraw.Draw(picture)
    pen.text((10, 8), title, fill=INK, font=_font(15))
    values = [v for _, vs in lines for v in vs if np.isfinite(v)]
    top = max(values) * 1.1 if values else 1.0
    x0, x1, y0, y1 = margin, w - 150, h - margin, 40
    lo, hi = min(xs), max(xs)
    hi = hi if hi > lo else lo + 1.0  # one value: a span to place it in

    def place(x: float, y: float) -> tuple[float, float]:
        return x0 + (x - lo) / (hi - lo) * (x1 - x0), y0 - y / top * (y0 - y1)

    pen.rectangle((x0, y1, x1, y0), outline=GRID)
    for x in xs:
        px, _ = place(x, 0)
        pen.line((px, y0, px, y0 + 5), fill=MUTED)
        pen.text((px - 10, y0 + 8), f"{x:g}", fill=MUTED, font=_font(12))
    pen.text(((x0 + x1) / 2 - 40, h - 26), x_label, fill=MUTED, font=_font(13))
    for k in range(5):
        v = top * k / 4
        _, py = place(lo, v)
        pen.line((x0 - 5, py, x0, py), fill=MUTED)
        pen.text((4, py - 7), f"{v:.3g}", fill=MUTED, font=_font(11))
    pen.text((4, 24), "mean intensity", fill=MUTED, font=_font(12))
    for n, (label, vs) in enumerate(lines):
        colour = LINES[n % len(LINES)]
        dots = [place(x, v) for x, v in zip(xs, vs, strict=True) if np.isfinite(v)]
        if len(dots) > 1:
            pen.line(dots, fill=colour, width=3)
        for x, y in dots:
            pen.ellipse((x - 4, y - 4, x + 4, y + 4), fill=colour)
        pen.text((x1 + 14, y1 + 20 * n), label, fill=colour, font=_font(13, bold=True))
    return picture


def _plot_reflectance(found: list[Card]) -> Image.Image:
    percents = sorted({c.percent for c in found})
    at = {(c.metres, c.percent): c.intensity for c in found if c.angle == 0}
    lines = [
        (f"{m} m", [at.get((m, p), float("nan")) for p in percents])
        for m in sorted({c.metres for c in found})
    ]
    title = "intensity vs reflectance, 0°"
    return _plot(title, [float(p) for p in percents], "reflectance (%)", lines)


def _plot_angle(found: list[Card]) -> Image.Image:
    angles = sorted({c.angle for c in found})
    near = min(c.metres for c in found)
    at = {(c.percent, c.angle): c.intensity for c in found if c.metres == near}
    lines = [
        (f"{p} %", [at.get((p, a), float("nan")) for a in angles])
        for p in sorted({c.percent for c in found})
    ]
    title = f"intensity vs incidence, {near} m"
    return _plot(title, [float(a) for a in angles], "incidence (°)", lines)
