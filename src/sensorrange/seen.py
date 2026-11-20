"""A lidar's scan drawn on a camera's picture of it: each point where the camera's lens and pose
put it, coloured by distance or by intensity (a log scale, with its bar)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from darkroom import format
from PIL import Image, ImageDraw

from .style import INK, PAPER, _font

#: The least dot's radius, pixels (dots grow with the picture).
DOT = 0.9


def overlay(
    capture: Path,
    camera: str,
    lens: dict,
    lidar: str,
    least: int = 200,
    by: str = "distance",
    around: np.ndarray | None = None,
    on: Callable[[np.ndarray, format.Scan], np.ndarray] | None = None,
    elsewhere: str = "elsewhere",
) -> Image.Image | None:
    """Camera `camera`'s first frame with lidar `lidar`'s first scan drawn on it (through the
    camera's lens and pose), each point coloured by its distance (near red, far blue) or `by`
    "intensity" (on a log scale, dark to yellow, a bar along the bottom saying the scale: a
    weak return as plain as a strong one); cropped to where `around` (world points: a bench's
    cards) lands, a margin round it, and enlarged; None when fewer than `least` points land in
    the frame. Dots are sized to the picture, ringed dark so they show on anything; with `on`
    (which points, in the range's frame, to colour: a bench's cards'), the rest faint grey."""
    from .sheets import _rotation, first, project  # theirs; imported here, as they import this

    if not format.has_scan(capture, lidar, 0):
        return None
    cam, lid = format.read_poses(capture, camera), format.read_poses(capture, lidar)
    scan = format.read_scan(capture, lidar, 0)
    if not len(scan.points):  # nothing in its reach
        return None
    k = first(capture, camera)
    world = scan.points @ _rotation(lid.rotation[0]).T + lid.position[0]
    pixels = project((world - cam.position[k]) @ _rotation(cam.rotation[k]), lens)
    inside = ~np.isnan(pixels).any(axis=1)
    inside &= (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < lens["width"])
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < lens["height"])
    )
    if inside.sum() < least:
        return None
    frame = Image.open(format.frame_path(capture, camera, k)).convert("RGB")
    if around is not None:  # just the part that matters, enlarged
        box = project((around - cam.position[k]) @ _rotation(cam.rotation[k]), lens)
        box = box[~np.isnan(box).any(axis=1)]
        if len(box):
            pad = 0.08 * max(np.ptp(box[:, 0]), np.ptp(box[:, 1]))
            x0, y0 = np.maximum(box.min(axis=0) - pad, 0)
            x1, y1 = np.minimum(box.max(axis=0) + pad, (lens["width"], lens["height"]))
            scale = max(1.0, 1800 / max(x1 - x0, 1.0))
            frame = frame.crop((int(x0), int(y0), int(x1), int(y1)))
            frame = frame.resize(
                (int(frame.width * scale), int(frame.height * scale)), Image.Resampling.LANCZOS
            )
            pixels = (pixels - (int(x0), int(y0))) * scale
            inside &= (
                (pixels[:, 0] >= 0)
                & (pixels[:, 0] < frame.width)
                & (pixels[:, 1] >= 0)
                & (pixels[:, 1] < frame.height)
            )
    frame = Image.blend(frame, Image.new("RGB", frame.size, (0, 0, 0)), 0.15).convert("RGBA")
    dots = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    pen = ImageDraw.Draw(dots)
    dot = max(DOT, frame.width / 1400)  # a few pixels on any frame
    if by == "intensity":
        level = np.log10(np.maximum(scan.intensity, 1e-6))
        low, high = np.percentile(level[inside], [1, 99]) if inside.any() else (0.0, 1.0)
        t = np.clip((level - low) / max(high - low, 1e-6), 0, 1)
        colours = [(*_ramp(f), 235) for f in t[inside]]
    else:
        far = np.linalg.norm(world - cam.position[k], axis=1)
        near_m, far_m = (far[inside].min(), far[inside].max()) if inside.any() else (0.0, 1.0)
        t = np.clip((far - near_m) / max(far_m - near_m, 1e-6), 0, 1)
        colours = [(*_distance(f), 170) for f in t[inside]]
    lit = on(world, scan)[inside] if on is not None else np.ones(int(inside.sum()), bool)
    for (x, y), colour, keep in zip(pixels[inside], colours, lit, strict=True):
        if not keep:  # not what is looked at: faint, small
            pen.ellipse(
                (x - dot / 2, y - dot / 2, x + dot / 2, y + dot / 2), fill=(200, 200, 200, 70)
            )
            continue
        pen.ellipse((x - dot - 1, y - dot - 1, x + dot + 1, y + dot + 1), fill=(0, 0, 0, 140))
        pen.ellipse((x - dot, y - dot, x + dot, y + dot), fill=colour)
    seen = Image.alpha_composite(frame, dots).convert("RGB")
    if by == "intensity":
        return _scale_bar(
            seen,
            f"{10**low:.3g}",
            f"{10**high:.3g}",
            _ramp,
            "intensity (log)",
            elsewhere if on else None,
        )
    return _scale_bar(
        seen, f"{near_m:.1f} m", f"{far_m:.1f} m", _distance, "distance from the camera"
    )


def _distance(f: float) -> tuple[int, int, int]:
    """Near (0) red, through green, to far (1) blue."""
    return int(255 * (1 - f)), int(255 * (1 - abs(2 * f - 1))), int(255 * f)


#: A viridis-like ramp, dark (weak) to yellow (strong): its stops.
RAMP = ((68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37))


def _ramp(f: float) -> tuple[int, int, int]:
    """The ramp's colour `f` (0 to 1) of the way along."""
    x = min(max(f, 0.0), 1.0) * (len(RAMP) - 1)
    i = min(int(x), len(RAMP) - 2)
    a, b = RAMP[i], RAMP[i + 1]
    r, g, bl = (int(a[c] + (b[c] - a[c]) * (x - i)) for c in range(3))
    return r, g, bl


def _scale_bar(
    picture: Image.Image,
    low: str,
    high: str,
    colour: Callable[[float], tuple[int, int, int]],
    said: str,
    others: str | None = None,
) -> Image.Image:
    """`picture` with `colour`'s ramp along a strip under it, `low` to `high`, `said` after;
    with `others`, a faint grey dot keyed so (the points not looked at)."""
    h = max(28, picture.height // 40)
    out = Image.new("RGB", (picture.width, picture.height + h + 8), PAPER)
    out.paste(picture, (0, 0))
    pen = ImageDraw.Draw(out)
    left, right = picture.width // 4, picture.width * 3 // 4
    for x in range(left, right):
        pen.line(
            (x, picture.height + 4, x, picture.height + h), fill=colour((x - left) / (right - left))
        )
    font = _font(max(20, h // 2 + 6))
    pen.text((left - 10, picture.height + 4), low, fill=INK, font=font, anchor="ra")
    pen.text((right + 10, picture.height + 4), f"{high}  {said}", fill=INK, font=font)
    if others:
        x, y, r = 16 + h // 2, picture.height + 4 + h // 2, h // 4
        pen.ellipse((x - r, y - r, x + r, y + r), fill=(200, 200, 200))
        pen.text((x + r + 8, picture.height + 4), others, fill=INK, font=font)
    return out
