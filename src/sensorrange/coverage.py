"""Where a rig's cameras overlap: each camera's picture painted where it lands on the sphere,
on a map of the sphere's directions from its centre (left at the left, up up), one map for
each distance the sphere was rendered at. Cameras apart on a rig see the same direction only
past some distance; near, the gap between them shows."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
from darkroom import format
from PIL import Image, ImageDraw

from . import sheets, style
from .ranges import direction

#: Map pixels a degree.
PER_DEGREE = 4
#: A colour for each camera, in the rig's order.
COLOURS = ((40, 110, 220), (230, 120, 20), (30, 160, 90), (200, 50, 150), (120, 80, 200))


def draw(captures: list[tuple[float, Path]], out: Path) -> Path:
    """One map for each (radius, capture folder) in `captures`, cropped alike to what any
    camera saw, with a table of how wide each two cameras overlap. Returns `out`."""
    seen = [(radius, _paint(folder, radius)) for radius, folder in captures]
    union = np.any([covered.any(axis=0) for _, (_, covered, _) in seen], axis=0)
    rows, columns = np.nonzero(union)
    m = 5 * PER_DEGREE
    box = (
        max(columns.min() - m, 0),
        max(rows.min() - m, 0),
        min(columns.max() + m, union.shape[1]),
        min(rows.max() + m, union.shape[0]),
    )
    panels = []
    for radius, (picture, covered, names) in seen:
        table: list[tuple[str, ...]] = [
            (f"{a} + {b}", _width(covered[i] & covered[j]))
            for (i, a), (j, b) in itertools.combinations(enumerate(names), 2)
        ]
        card = sheets._label(f"{radius:g} m", "sphere radius", ("overlap",), table)
        crop = _grid(picture, box)
        panel = Image.new(
            "RGB", (card.width + crop.width + 30, max(card.height, crop.height) + 20), style.PAPER
        )
        panel.paste(card, (10, 10))
        panel.paste(crop, (card.width + 25, 10))
        panels.append(panel)
    key = Image.new("RGB", (panels[0].width, 40), style.PAPER)
    pen = ImageDraw.Draw(key)
    x = 10
    for k, name in enumerate(seen[0][1][2]):
        pen.rectangle((x, 12, x + 24, 28), fill=COLOURS[k % len(COLOURS)])
        pen.text((x + 30, 10), name, fill=style.INK, font=style._font(16))
        x += 60 + pen.textbbox((0, 0), name, font=style._font(16))[2]
    sheet = Image.new(
        "RGB", (max(p.width for p in panels), sum(p.height for p in panels) + 40), style.PAPER
    )
    sheet.paste(key, (0, 0))
    y = 40
    for panel in panels:
        sheet.paste(panel, (0, y))
        y += panel.height
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def _paint(folder: Path, radius: float) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """The map (H x W x 3) of capture `folder`'s cameras' first frames on the sphere `radius`
    round, which map cells each camera covers (n x H x W), and the cameras' names."""
    meta = format.read_meta(folder)
    names = [n for n, s in meta.sensors.items() if s.kind == "camera"]
    az, el = np.meshgrid(
        180 - (np.arange(360 * PER_DEGREE) + 0.5) / PER_DEGREE,
        90 - (np.arange(180 * PER_DEGREE) + 0.5) / PER_DEGREE,
    )
    on_sphere = radius * direction(az.ravel(), el.ravel())
    grey = np.full(az.size, 235.0)
    colour = np.zeros((az.size, 3))
    count = np.zeros(az.size)
    covered = []
    for k, name in enumerate(names):
        lens = meta.sensors[name].settings
        poses = format.read_poses(folder, name)
        kept = sheets.first(folder, name)  # its first frame kept
        frame = np.asarray(Image.open(format.frame_path(folder, name, kept)).convert("L"), float)
        rotation = sheets._rotation(poses.rotation[kept])
        pixels = sheets.project((on_sphere - poses.position[kept]) @ rotation, lens)
        u, v = pixels[:, 0], pixels[:, 1]
        inside = np.isfinite(u) & (u >= 0) & (u < lens["width"]) & (v >= 0) & (v < lens["height"])
        here = frame[v[inside].astype(int), u[inside].astype(int)]
        grey[inside] = np.where(count[inside] > 0, (grey[inside] + here) / 2, here)
        colour[inside] += COLOURS[k % len(COLOURS)]
        count[inside] += 1
        covered.append(inside.reshape(az.shape))
    tint = np.where(count[:, None] > 0, colour / np.maximum(count, 1)[:, None], 255)
    strength = np.where(count > 1, 0.55, np.where(count == 1, 0.3, 0.0))[:, None]
    shade = grey[:, None] * (1 - strength) + tint * strength
    shade[count > 1] = np.clip(shade[count > 1] * 1.15 + 20, 0, 255)
    picture = shade.reshape(*az.shape, 3)
    for k, mask in enumerate(covered):  # each camera's edge in its colour
        edge = mask & ~(
            np.roll(mask, 1, 0) & np.roll(mask, -1, 0) & np.roll(mask, 1, 1) & np.roll(mask, -1, 1)
        )
        picture[edge] = COLOURS[k % len(COLOURS)]
    return picture.astype(np.uint8), np.array(covered), names


def _width(both: np.ndarray) -> str:
    """How wide cells `both` are at the horizon (degrees), or at their widest if they miss it."""
    if not both.any():
        return "none"
    middle = both[both.shape[0] // 2]
    if middle.any():
        return f"{middle.sum() / PER_DEGREE:.1f}° at the horizon"
    return f"{both.sum(axis=1).max() / PER_DEGREE:.1f}° at most"


def _grid(picture: np.ndarray, box: tuple[int, int, int, int]) -> Image.Image:
    """The map cropped to `box`, a line every 15° labelled (left/right, up/down)."""
    left, top, right, bottom = box
    image = Image.fromarray(picture[top:bottom, left:right])
    pen = ImageDraw.Draw(image)
    font = style._font(13)
    for degrees in range(-180, 181, 15):
        x = (180 - degrees) * PER_DEGREE - left
        if 0 <= x < image.width:
            pen.line((x, 0, x, image.height), fill=style.GRID)
            if degrees % 30 == 0:
                pen.text(
                    (x + 3, image.height - 16), style._side(degrees, 0), fill=style.MUTED, font=font
                )
    for degrees in range(-90, 91, 15):
        y = (90 - degrees) * PER_DEGREE - top
        if 0 <= y < image.height:
            pen.line((0, y, image.width, y), fill=style.GRID)
            pen.text((3, y + 2), style._updown(degrees, 0), fill=style.MUTED, font=font)
    return image
