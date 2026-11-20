"""A scene and what is on record in it, as standard USD (UsdSemantics labels): targets' faces
(taxonomy `TARGET`: a quad, its corners its points, its lidar material its bound material's
`omni:simready:nonvisual` words), AprilTags (`TAG`: "<family>:<id>", corners in OpenCV's order),
an angle sphere (`RANGE`: `SPHERE`: its radius and texture from its mesh and material), a
reflectance bench (`RANGE`: `BENCH`: its targets grey cards), a bay of worksite (`RANGE`:
`SITE`: kind "site", drawn as a scene is). A scene with none of these is of kind "scene": a rig
is put in it where it is told.

How a scene is best rendered for its checks is said in its root layer's
`customLayerData["sensorrange"]` (`Settings`, `SETTINGS`), when it says; else plain defaults.
Nothing beside the scene file is read."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from pxr import Sdf, Usd, UsdGeom, UsdShade

#: Where a scene keeps its measure points: Xforms, each a mark to stake out (the pole along its Z).
MARKS = "/World/Measurement_Points"
#: The semantic taxonomies a scene's record is labelled in, and the angle sphere's label.
TARGET, TAG, RANGE, SPHERE, BENCH, SITE = (
    "target",
    "apriltag",
    "range",
    "angle_sphere",
    "reflectance_bench",
    "worksite_bay",
)
#: The frame a rig goes on (sensorkit's `SENSOR_MOUNT`), found by its name.
MOUNT = "SensorMount"
NV = "omni:simready:nonvisual"
#: Where a rig goes in a scene with no SensorMount on record (`stage` makes it there).
SCENE_FRAME = "/SensorRange/SensorMount"
#: A scene's corner view, if it has one: an Xform with a camera's axes.
OVERVIEW = "/World/Frames/Overview"
#: The key in a scene's root layer's `customLayerData` its settings are under (`Settings`).
SETTINGS = "sensorrange"
#: The looks a scene is rendered in by name, unless it says otherwise: RTX's real time ("ray")
#: and path traced ("path"); each `darkroom.Options` fields, and "lamp" (`Settings.lamp`).
LOOKS: dict[str, dict[str, Any]] = {"ray": {}, "path": {"look": "path_traced_draft"}}


@dataclass(frozen=True)
class Settings:
    """How a scene is rendered for its checks, as its root layer's
    `customLayerData["sensorrange"]` says (a dictionary; every key may be left out):

    - `lamp` (double, default 0: off): a ring light round the rig's lens, this bright, shining
      ahead, so prisms and retroreflective sheeting shine back at it (`stage`).
    - `looks` (dictionary of dictionaries): the looks it is rendered in, by name, each
      `darkroom.Options` fields (and "lamp", for that look alone), laid over `LOOKS`'.
    - `mounted` (bool, default true): whether a rig as mounted is checked in it by `suite`
      (false for a scene meant for each sensor alone, facing it).
    """

    lamp: float = 0.0
    looks: Mapping[str, Mapping[str, Any]] = field(default_factory=lambda: dict(LOOKS))
    mounted: bool = True


def settings(layer: Sdf.Layer) -> Settings:
    """Layer `layer`'s `Settings`, from its `customLayerData[SETTINGS]`; the defaults where it
    says nothing. Raises `ValueError` for a key it does not know."""
    said = dict(layer.customLayerData.get(SETTINGS) or {})
    known = ("lamp", "looks", "mounted")
    if unknown := sorted(set(said) - set(known)):
        raise ValueError(
            f"{layer.identifier}: customLayerData[{SETTINGS!r}] says {', '.join(unknown)}, "
            f"which is not one of {', '.join(known)}."
        )
    looks = {name: dict(how) for name, how in LOOKS.items()}
    for name, how in dict(said.get("looks") or {}).items():
        looks[name] = {**looks.get(name, {}), **dict(how)}
    return Settings(float(said.get("lamp", 0.0)), looks, bool(said.get("mounted", True)))


@dataclass(frozen=True)
class Target:
    """Something on a range facing the sensors: its face's corners (4 x 3, the stage's frame:
    bottom left, bottom right, top right, top left seen from the front) and what it is for a
    lidar."""

    name: str
    corners: np.ndarray
    base: str
    coating: str
    attributes: tuple[str, ...]


@dataclass(frozen=True)
class Tag:
    """An AprilTag on a range: its id and corners (4 x 3, as OpenCV orders them seen from the
    front: top left, top right, bottom right, bottom left)."""

    id: int
    corners: np.ndarray


@dataclass(frozen=True)
class Range:
    """A scene as checks see it: `kind` ("sphere": an angle sphere round its mount; "site": a
    bay of worksite; "bench": a reflectance bench, its targets grey cards; "targets": targets on
    record; "scene": none of these), its file, the frame its rig goes on, the sphere's radius,
    texture and pixels a degree, the targets and AprilTags."""

    kind: str
    scene: Path
    frame: str
    radius: float = 10.0
    texture: Path | None = None
    per_degree: int = 20
    targets: tuple[Target, ...] = ()
    tags: tuple[Tag, ...] = ()
    #: Its corner view's frame, if it has one (`OVERVIEW`).
    overview: str | None = None
    #: Its measure points (`MARKS`): each a name and where it is (x, y, z).
    marks: tuple[tuple[str, tuple[float, float, float]], ...] = ()
    #: The angle sphere's prim, for a sphere.
    sphere: str | None = None
    #: How it is rendered for its checks, as it says (`Settings`).
    settings: Settings = field(default_factory=Settings)


def read(scene: str | Path) -> Range:
    """Scene file `scene` as checks see it: what is on record in it, and how it says it is
    rendered (`Settings`), read from its USD."""
    file = Path(scene)
    if not file.is_file():
        raise FileNotFoundError(f"There is no scene {file}.")
    stage = Usd.Stage.Open(str(file))
    told = settings(stage.GetRootLayer())
    every = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)  # the hidden record too
    mount = next((p for p in stage.Traverse(every) if p.GetName() == MOUNT), None)
    targets, tags, sphere, bench, site = [], [], None, False, False
    for prim in stage.Traverse(every):
        if said := _label(prim, TARGET):
            targets.append(Target(said, _points(prim), *_lidar(prim)))
        elif said := _label(prim, TAG):
            tags.append(Tag(int(said.rsplit(":", 1)[1]), _points(prim)))
        elif _label(prim, RANGE) == SPHERE:
            sphere = prim
        elif _label(prim, RANGE) == BENCH:
            bench = True
        elif _label(prim, RANGE) == SITE:
            site = True
    corner = OVERVIEW if stage.GetPrimAtPath(OVERVIEW) else None
    held = stage.GetPrimAtPath(MARKS)
    marks = tuple(
        (
            m.GetName(),
            tuple(UsdGeom.Xformable(m).ComputeLocalToWorldTransform(0).ExtractTranslation()),
        )
        for m in (held.GetChildren() if held else [])
        if m.IsA(UsdGeom.Xformable)
    )
    if mount is None or (sphere is None and not targets):
        return Range("scene", file, SCENE_FRAME, settings=told)
    frame = str(mount.GetPath())
    if sphere is None:
        kind = "bench" if bench else "site" if site else "targets"
        return Range(
            kind,
            file,
            frame,
            targets=tuple(targets),
            tags=tuple(tags),
            overview=corner,
            marks=marks,
            settings=told,
        )
    centre = np.array(UsdGeom.Xformable(mount).ComputeLocalToWorldTransform(0).ExtractTranslation())
    radius = float(np.linalg.norm(_points(sphere) - centre, axis=1).mean())
    texture = _texture(sphere, file)
    per_degree = 20
    if texture is not None:
        from PIL import Image

        per_degree = round(Image.open(texture).width / 360)
    return Range(
        "sphere",
        file,
        frame,
        radius,
        texture,
        per_degree,
        tuple(targets),
        tuple(tags),
        corner,
        sphere=str(sphere.GetPath()),
        settings=told,
    )


def _label(prim: Usd.Prim, taxonomy: str) -> str | None:
    said = prim.GetAttribute(f"semantics:labels:{taxonomy}")
    value = said.Get() if said else None
    return str(value[0]) if value else None


def _points(prim: Usd.Prim) -> np.ndarray:
    """A mesh's points in the stage's frame."""
    to_world = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0))
    local = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(), float)
    return (np.c_[local, np.ones(len(local))] @ to_world)[:, :3]  # row vectors, as USD's


def _lidar(prim: Usd.Prim) -> tuple[str, str, tuple[str, ...]]:
    """What a prim is to RTX's lidar: its bound material's non-visual base, coating, attributes."""
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    if not material:
        return "none", "none", ("none",)
    at = material.GetPrim()
    base = at.GetAttribute(f"{NV}:base").Get() or "none"
    coating = at.GetAttribute(f"{NV}:coating").Get() or "none"
    attributes = at.GetAttribute(f"{NV}:attributes").Get() or ["none"]
    return str(base), str(coating), tuple(str(a) for a in attributes)


def _texture(prim: Usd.Prim, file: Path) -> Path | None:
    """The picture on a prim's bound material (its UsdUVTexture's file), resolved."""
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    if not material:
        return None
    for inside in Usd.PrimRange(material.GetPrim()):
        if not inside.IsA(UsdShade.Shader):
            continue
        source = UsdShade.Shader(inside).GetInput("file")
        if source and source.Get():
            path = source.Get()
            return Path(path.resolvedPath or (file.parent / path.path))
    return None


def direction(az: np.ndarray | float, el: np.ndarray | float) -> np.ndarray:
    """Unit vectors `az` degrees left of +Y, `el` degrees up."""
    a, e = np.radians(az), np.radians(el)
    return np.stack([-np.sin(a) * np.cos(e), np.cos(a) * np.cos(e), np.sin(e)], axis=-1)


def angles(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Azimuth left of +Y and elevation up of directions `xyz` (n, 3), degrees."""
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    return np.degrees(np.arctan2(-x, y)), np.degrees(np.arctan2(z, np.hypot(x, y)))
