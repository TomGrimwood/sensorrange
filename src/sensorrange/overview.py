"""A scene's corner view: a plain camera on the scene's `Overview` frame (high in a corner, at
what is tested), rendered in every check so its sheets open with a clear picture of the test;
on a reflectance bench, each sector, reflectance and angle labelled on it."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import usdsensors as us
from PIL import Image, ImageDraw
from pxr import Gf, Sdf, Usd, UsdGeom

from .ranges import Range
from .style import _font

#: The camera's name in a capture, its picture (pixels) and its focal length (pixels: about 80
#: degrees across).
NAME = "overview"
SIZE, FOCAL = (1920, 1080), 1150.0


def rig(out: Path) -> Path:
    """`out/Overview.usda`: the corner camera, on its frame as it is (the frame has a camera's
    axes)."""
    out.mkdir(parents=True, exist_ok=True)
    file = out / "OverviewCamera.usda"
    us.make_sensor(file, "camera", about="A plain camera showing the range from its corner.")
    stage = Usd.Stage.Open(str(file))
    cam = UsdGeom.Camera(stage.GetDefaultPrim())
    cam.CreateFocalLengthAttr(10.0)
    cam.CreateHorizontalApertureAttr(10.0 * SIZE[0] / FOCAL)
    cam.CreateVerticalApertureAttr(10.0 * SIZE[1] / FOCAL)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.1, 10000))
    prim = cam.GetPrim()
    prim.AddAppliedSchema("OmniLensDistortionOpenCvPinholeAPI")
    lens = "omni:lensdistortion:opencvPinhole:"
    prim.CreateAttribute("omni:lensdistortion:model", Sdf.ValueTypeNames.Token).Set("opencvPinhole")
    prim.CreateAttribute(lens + "imageSize", Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(*SIZE))
    for key, value in (("fx", FOCAL), ("fy", FOCAL), ("cx", SIZE[0] / 2), ("cy", SIZE[1] / 2)):
        prim.CreateAttribute(lens + key, Sdf.ValueTypeNames.Float).Set(value)
    for key in ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2", "s1", "s2", "s3", "s4"):
        prim.CreateAttribute(lens + key, Sdf.ValueTypeNames.Float).Set(0.0)  # no distortion
    stage.Save()
    made = us.new_rig(out / "Overview.usda", "The range's corner camera.")
    us.add(made, NAME, file, us.Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    return made


def picture(capture: Path, lens: dict, built: Range) -> Image.Image:
    """The corner camera's first frame; on a bench, labelled:
    each sector's distance over it, and the nearest sector's reflectances (across) and angles
    (down)."""
    from darkroom import format

    from .sheets import _rotation, first, project

    k = first(capture, NAME)
    frame = Image.open(format.frame_path(capture, NAME, k)).convert("RGB")
    pose = format.read_poses(capture, NAME)

    def at(xyz: np.ndarray) -> np.ndarray:
        return project((xyz - pose.position[k]) @ _rotation(pose.rotation[k]), lens)

    _marks(frame, built, at)
    if built.kind != "bench":
        return frame

    cards: dict[tuple[int, int, int], np.ndarray] = {}
    for t in built.targets:
        said = re.fullmatch(r"grey(\d+)_(\d+)deg_(\d+)m", t.name)
        if said:
            p, a, m = (int(v) for v in said.groups())
            cards[(p, a, m)] = t.corners.mean(axis=0)
    pen = ImageDraw.Draw(frame)
    big = _font(30, bold=True)

    def label(xy: np.ndarray, text: str, font, anchor: str) -> None:
        if np.isnan(xy).any():
            return
        x, y = float(xy[0]), float(xy[1])
        box = pen.textbbox((x, y), text, font=font, anchor=anchor)
        pen.rectangle((box[0] - 6, box[1] - 4, box[2] + 6, box[3] + 4), fill=(20, 20, 24))
        pen.text((x, y), text, fill=(255, 235, 120), font=font, anchor=anchor)

    percents = sorted({p for p, _, _ in cards})
    angles = sorted({a for _, a, _ in cards})
    distances = sorted({m for _, _, m in cards})
    for m in distances:  # over each sector, its distance
        top = np.array([cards[(p, angles[0], m)] for p in percents]).mean(axis=0)
        label(at(top[None] + np.array([[0.0, 0.0, 0.08 * m]]))[0], f"{m} m", big, "mb")

    def width(m: int) -> float:
        """How far apart the sector's columns are in the picture (pixels)."""
        xs = at(np.array([cards[(p, angles[0], m)] for p in percents]))[:, 0]
        return float(np.nanmin(np.diff(np.sort(xs)))) if len(xs) > 1 else 0.0

    widest = max(distances, key=width)  # the sector seen biggest: room for its labels
    gap = width(widest)
    tag = _font(int(np.clip(gap * 0.32, 12, 22)), bold=True)
    left = min(percents, key=lambda p: at(cards[(p, angles[0], widest)][None])[0, 0])
    rows = at(np.array([cards[(left, a, widest)] for a in angles]))[:, 1]
    step = float(np.nanmin(np.abs(np.diff(rows)))) if len(rows) > 1 else 40.0
    tops = at(np.array([cards[(p, angles[0], widest)] for p in percents]))
    for p, xy in zip(percents, tops, strict=True):  # its reflectances over its columns
        label(xy - np.array([0.0, step * 0.6]), f"{p}%", tag, "mb")
    lefts = at(np.array([cards[(left, a, widest)] for a in angles]))  # the column seen leftmost
    for a, xy in zip(angles, lefts, strict=True):  # and its angles beside its rows
        label(xy - np.array([gap * 0.6, 0.0]), f"{a}°", tag, "rm")
    return frame


#: What a range's measure points are named for, said in words on its test picture (the rest,
#: toes and corners, not said).
SAID = {
    "heap_top": "heap",
    "pothole_bottom": "pothole",
    "manhole": "manhole",
    "cone": "cone",
    "spade": "spade",
    "shard": "scrap steel",
    "pile_top": "heap",
}


def _marks(frame: Image.Image, built: Range, at) -> None:
    """The things the range's measure points are on (`SAID`), each named in words where it is."""
    pen = ImageDraw.Draw(frame)
    font = _font(26, bold=True)
    for name, xyz in built.marks:
        said = next((w for key, w in SAID.items() if name.split("_", 2)[-1] == key), None)
        if said is None:
            continue
        xy = at(np.array([xyz]))[0]
        if np.isnan(xy).any():
            continue
        x, y = float(xy[0]), float(xy[1]) - 18
        box = pen.textbbox((x, y), said, font=font, anchor="mb")
        pen.rectangle((box[0] - 6, box[1] - 4, box[2] + 6, box[3] + 4), fill=(20, 20, 24))
        pen.text((x, y), said, fill=(255, 235, 120), font=font, anchor="mb")
