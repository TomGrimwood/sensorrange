"""A rig's extrinsics drawn as an engineer would: from above, from the front and from the side,
each sensor where it is on the rig's mount (its origin) and the way it looks, every place
dimensioned from the origin in millimetres; beside them, the table of each one's place and
turn. No renderer: read from the rig's file (`usdsensors.mounts`).

The rig's frame is its mount's: +X right, +Y ahead, +Z up. A camera looks along its -Z, a lidar
along its +X.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .style import BLUE, GRID, INK, MUTED, PAPER, RED, _font

#: Each kind's colour.
COLOURS = {
    "camera": BLUE,
    "lidar": RED,
    "gnss": (30, 140, 60),
    "imu": (130, 70, 180),
}
#: A view's size (pixels), its margin for the dimensions, the arrow's length (pixels, for a
#: sensor looking across the view).
SIZE, MARGIN, ARROW = 760, 120, 70
#: The views: name, the axes across and up (index into x, y, z and their sign), and their words.
VIEWS = (
    ("top", (0, 1), (1, 1), ("x", "y")),
    ("front", (0, -1), (2, 1), ("x", "z")),  # looking at its front: its right is on the left
    ("side", (1, 1), (2, 1), ("y", "z")),  # from its right
)


def placed(rig: str | Path) -> list[dict]:
    """Each active sensor on rig `rig`: its name, kind, file, place (m) and turn (degrees, as
    `rotateXYZ`), and the way it looks (a unit vector, or None for one that does not look)."""
    import usdsensors
    from pxr import Gf

    found = []
    for m in usdsensors.mounts(rig):
        if not m.active:
            continue
        rx, ry, rz = m.pose.rotate
        turn = Gf.Matrix4d().SetRotate(
            Gf.Rotation(Gf.Vec3d(1, 0, 0), rx)
            * Gf.Rotation(Gf.Vec3d(0, 1, 0), ry)
            * Gf.Rotation(Gf.Vec3d(0, 0, 1), rz)
        )
        axis = {"camera": (0, 0, -1), "lidar": (1, 0, 0)}.get(m.kind)
        looks = np.array(turn.TransformDir(Gf.Vec3d(*axis))) if axis else None
        found.append(
            {
                "name": m.name,
                "kind": m.kind,
                "sensor": m.sensor.name,
                "at": np.array(m.pose.translate, float),
                "turn": tuple(float(v) for v in m.pose.rotate),
                "looks": looks,
            }
        )
    return found


def draw(rig: str | Path, out: str | Path) -> Path:
    """Rig `rig`'s extrinsics drawn, with their table, into `out` (a .png); returns it."""
    from .sheets import _sheet

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sensors = placed(rig)
    if not sensors:
        raise ValueError(f"{rig} has no active sensors.")
    every = np.array([s["at"] for s in sensors])
    span = max(float(np.ptp(every, axis=0).max()), 0.2) * 1.15
    views = [_view(sensors, name, a, b, words, span) for name, a, b, words in VIEWS]
    body = Image.new("RGB", (sum(v.width for v in views) + 20 * (len(views) - 1), SIZE), PAPER)
    x = 0
    for v in views:
        body.paste(v, (x, 0))
        x += v.width + 20
    rows: list[tuple[str, ...]] = [
        (
            str(s["name"]),
            *(_mm(v) for v in s["at"]),
            "  ".join(f"{v + 0.0:.1f}" for v in s["turn"]),
        )
        for s in sensors
    ]
    heads = ("x mm", "y mm", "z mm", "turn x y z °")
    return _sheet(body, (Path(rig).stem, "extrinsics", heads, rows), out)


def _view(
    sensors: list[dict],
    name: str,
    across: tuple[int, int],
    up: tuple[int, int],
    words: tuple[str, str],
    span: float,
) -> Image.Image:
    """One view: the sensors projected on two of the rig's axes (`across`, `up`: index and
    sign), the origin, each sensor's arrow, and ordinate dimensions along the bottom and left."""
    view = Image.new("RGB", (SIZE, SIZE), PAPER)
    pen = ImageDraw.Draw(view)
    small, label, title = _font(18), _font(19), _font(26, bold=True)
    (i, si), (j, sj) = across, up
    every = np.array([s["at"] for s in sensors])
    mid = np.array([(every[:, i] * si).max() + (every[:, i] * si).min(), (every[:, j] * sj).max() + (every[:, j] * sj).min()]) / 2  # fmt: skip
    scale = (SIZE - 2 * MARGIN) / span
    centre = np.array([MARGIN + (SIZE - 2 * MARGIN) / 2, SIZE - MARGIN - (SIZE - 2 * MARGIN) / 2])

    def px(p: np.ndarray) -> tuple[float, float]:
        u, v = p[i] * si - mid[0], p[j] * sj - mid[1]
        return float(centre[0] + u * scale), float(centre[1] - v * scale)

    left, bottom = MARGIN - 40, SIZE - MARGIN + 40
    pen.rectangle((MARGIN - 50, 50, SIZE - MARGIN + 50, SIZE - MARGIN + 50), outline=GRID)
    pen.text((MARGIN - 50, 14), name, fill=INK, font=title)
    # Its axes, at the corner: the way each runs.
    ax, ay = SIZE - MARGIN + 10, 70
    pen.line((ax, ay + 40, ax + 40, ay + 40), fill=INK, width=2)
    pen.line((ax, ay + 40, ax, ay), fill=INK, width=2)
    pen.text((ax + 44, ay + 32), f"{'+' if si > 0 else '-'}{words[0]}", fill=INK, font=small)
    pen.text((ax - 6, ay - 20), f"{'+' if sj > 0 else '-'}{words[1]}", fill=INK, font=small)
    # The origin: the mount.
    ox, oy = px(np.zeros(3))
    pen.line((ox - 12, oy, ox + 12, oy), fill=INK, width=1)
    pen.line((ox, oy - 12, ox, oy + 12), fill=INK, width=1)
    pen.ellipse((ox - 4, oy - 4, ox + 4, oy + 4), outline=INK)
    # Ordinate dimensions: every place's value along the bottom and the left, from 0.
    _ordinates(pen, sorted({round(float(s["at"][i] * si) * 1000) / 1000 for s in sensors} | {0.0}), lambda v: (px(_on(i, v * si))[0], bottom), "x", small)  # fmt: skip
    _ordinates(pen, sorted({round(float(s["at"][j] * sj) * 1000) / 1000 for s in sensors} | {0.0}), lambda v: (left, px(_on(j, v * sj))[1]), "y", small)  # fmt: skip
    # Each sensor: leaders to the dimensions, its arrow, its dot; sensors at one place share
    # one label.
    places: dict[tuple[int, int], list[dict]] = {}
    for s in sensors:
        x, y = px(s["at"])
        places.setdefault((round(x), round(y)), []).append(s)
        pen.line((x, y, x, bottom - 6), fill=GRID, width=1)
        pen.line((left + 6, y, x, y), fill=GRID, width=1)
    for (x, y), here in places.items():
        for s in here:
            colour = COLOURS.get(s["kind"], MUTED)
            if s["looks"] is not None:
                d = np.array([s["looks"][i] * si, -s["looks"][j] * sj])
                tip = (x + d[0] * ARROW, y + d[1] * ARROW)
                pen.line((x, y, *tip), fill=colour, width=3)
                if np.hypot(*d) > 0.2:  # a head where it points across the view
                    a = np.arctan2(d[1], d[0])
                    for side in (-0.45, 0.45):
                        pen.line(
                            (*tip, tip[0] - 12 * np.cos(a + side), tip[1] - 12 * np.sin(a + side)),
                            fill=colour,
                            width=3,
                        )
                else:  # looking out of the view or into it
                    pen.ellipse((x - 14, y - 14, x + 14, y + 14), outline=colour, width=2)
        for s in here:
            colour = COLOURS.get(s["kind"], MUTED)
            pen.ellipse((x - 6, y - 6, x + 6, y + 6), fill=colour, outline=INK)
    taken: list[tuple[float, float, float, float]] = []
    for (x, y), here in sorted(places.items(), key=lambda p: (p[0][1], p[0][0])):
        names = "\n".join(s["name"] for s in here)  # one a line
        box = pen.multiline_textbbox((0, 0), names, font=label, spacing=4)
        w, h = box[2] - box[0], box[3] - box[1] + 6
        for dx, dy in [(14, 6), (14, -h - 6), (-14 - w, 6), (-14 - w, -h - 6)] + [
            (14, 6 + k * h) for k in range(1, 12)
        ]:  # beside it, else further down, clear of every label placed
            spot = (x + dx, y + dy, x + dx + w, y + dy + h)
            inside = spot[0] > MARGIN - 46 and spot[2] < SIZE - 4 and spot[3] < SIZE - MARGIN + 30
            if inside and not any(_overlap(spot, t) for t in taken):
                break
        taken.append(spot)
        if abs(spot[1] - y) > h + 12:  # moved away: a leader back to it
            pen.line((x, y, spot[0], spot[1] + h / 2), fill=MUTED, width=1)
        pen.multiline_text(spot[:2], names, fill=INK, font=label, spacing=4)
    return view


def _overlap(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _mm(metres: float) -> str:
    """Metres as whole millimetres, never "-0"."""
    return f"{round(metres * 1000) + 0:d}"


def _on(axis: int, value: float) -> np.ndarray:
    p = np.zeros(3)
    p[axis] = value
    return p


def _ordinates(
    pen: ImageDraw.ImageDraw,
    values: list[float],
    at: Callable[[float], tuple[float, float]],
    which: str,
    font,
) -> None:
    """Ordinate dimensions: a tick and its value in millimetres at each of `values`, placed by
    `at`, along the bottom (`which` "x") or the left ("y"); a value close to the one before
    it is set a step further out, so none hides another."""
    ticks = [at(v) for v in values]
    gap = 46 if which == "x" else 24
    along = [x if which == "x" else y for x, y in ticks]
    order = np.argsort(along)
    shown = list(np.array(along, float))
    for k in range(1, len(order)):  # spread along the axis, in order, `gap` apart
        a, b = order[k - 1], order[k]
        shown[b] = max(shown[b], shown[a] + gap)
    if len(order):  # centred on where they belong
        shift = (np.mean(along) - np.mean(shown)) if len(order) > 1 else 0.0
        shown = [v + shift for v in shown]
    for v, (x, y), s in zip(values, ticks, shown, strict=True):
        said = _mm(v)
        if which == "x":
            pen.line((x, y - 6, x, y + 6), fill=INK, width=2)
            pen.line((x, y + 6, s, y + 16), fill=MUTED, width=1)
            pen.text((s, y + 18), said, fill=INK, font=font, anchor="mt")
        else:
            pen.line((x - 6, y, x + 6, y), fill=INK, width=2)
            pen.line((x - 6, y, x - 16, s), fill=MUTED, width=1)
            pen.text((x - 20, s), said, fill=INK, font=font, anchor="rm")
