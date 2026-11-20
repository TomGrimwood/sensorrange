"""Checks run on a sensor rig in a scene: the rig put on the scene's SensorMount, rendered for a
moment by darkroom, and a sheet per camera and lidar of what it saw against what its files say
(`sheets`). What the scene holds to check against is read from its own USD (`ranges`): an angle
sphere checkered in degrees (fields of view, lens, lidar pattern), targets of known materials,
AprilTags, grey cards on a reflectance bench; in a scene with none, each sensor's sheet shows
what it saw and what its files say. Its captures are ordinary darkroom capture folders.

    import sensorrange
    sensorrange.check("rig.usda", "out/sphere", "scenes/sphere.usda")      # a check per sensor
    sensorrange.check(rig, "out/town", "town.usd", at=(x, y, 1.6))          # any scene, anywhere
    sensorrange.check_sensors(sensorrange.sensors_of(rig), "out/each", scene)  # each, facing ahead
    sensorrange.overlap(rig, "out/overlap", "scenes/sphere.usda", radii=(1, 2, 5, 20))
    # -> <out>/<sensor>_check.png for each camera and lidar; the capture in <out>/capture
    sensorrange.test("rig.usda", ["sphere.usda", "lab.usda"], "out")  # every check, every scene
    sensorrange.suite("out/rig", scenes, rigs=[rig], looks=("ray", "path"))  # where and how you say
    # -> <out>/results.csv, <out>/<sensor>/<scenario>... (`suites`)
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import usdsensors
from darkroom import Job, Options, format
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

from . import coverage, materials, overview, ranges, sheets, witness
from .ranges import LOOKS, Range, Settings
from .suites import suite, test

#: How fine the scene kept beside a capture is (metres): its charts and props whole, for
#: whatever records the capture.
SCENERY_CELL = 0.005
#: The rendered stage's time codes a second.
PER_SECOND = 30
#: `stage`'s lamp: a ring light round the lens, this wide (metres), facing ahead from the
#: SensorMount, as a total station lights its prism along its own line of sight. A corner cube
#: sends light back where it came from, so only light from round the lens comes back into it.
LAMP = 0.04


def stage(
    rig: str | Path,
    out: str | Path,
    scene: str | Path,
    seconds: float = 0.5,
    radius: float | None = None,
    at: tuple[float, float, float] = (0.0, 0.0, 1.6),
    heading: float = 0.0,
    lamp: float | None = None,
    wide: bool = True,
    segment: bool = True,
) -> Path:
    """`out/stage.usda`: scene file `scene` with rig `rig` on its SensorMount, `seconds` long.
    An angle sphere is made `radius` metres round (its labels stay true: it is scaled about its
    centre, and the scene's lights made brighter to light it as before); in a scene with no
    SensorMount on record, the rig is at `at`. It is turned `heading` degrees left of +Y. With
    `lamp` (its intensity; else the scene's `Settings.lamp`), a ring light round its lens
    (`LAMP`) shines ahead, as a total station lights its prism: what shines back at it (prisms,
    retroreflective sheeting) does so by its mirrors. With `wide`, each lidar has a wide camera
    at it (`witness`) to draw its points on; with `segment`, each camera's frames come
    segmented too, by what each thing is to RTX's lidar (`materials`)."""
    built = ranges.read(scene)
    lamp = built.settings.lamp if lamp is None else lamp
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    layer = Sdf.Layer.CreateNew(str(out / "stage.usda"))
    layer.subLayerPaths.append(str(built.scene.resolve()))
    layer.defaultPrim = "World"
    layer.pseudoRoot.SetInfo("upAxis", "Z")
    layer.pseudoRoot.SetInfo("metersPerUnit", 1.0)
    layer.timeCodesPerSecond = PER_SECOND
    layer.startTimeCode, layer.endTimeCode = 0, math.ceil(seconds * PER_SECOND)
    if radius is not None and built.sphere is not None:
        _rescale(layer, built, radius / built.radius)
    if built.kind == "scene":
        mount = Sdf.CreatePrimInLayer(layer, built.frame)
        mount.specifier = Sdf.SpecifierDef
        mount.typeName = "Xform"
        mount.nameParent.specifier = Sdf.SpecifierDef
        mount.nameParent.typeName = "Xform"
        _xform(mount, translate=at, rotate=(0.0, 0.0, heading))
    elif heading:  # a scene's own SensorMount, turned to look elsewhere in it
        _xform(Sdf.CreatePrimInLayer(layer, built.frame), rotate=(0.0, 0.0, heading))
    if lamp:
        light = Sdf.CreatePrimInLayer(layer, f"{built.frame}/Lamp")
        light.specifier, light.typeName = Sdf.SpecifierDef, "DiskLight"
        Sdf.AttributeSpec(light, "inputs:radius", Sdf.ValueTypeNames.Float).default = LAMP
        Sdf.AttributeSpec(light, "inputs:intensity", Sdf.ValueTypeNames.Float).default = lamp
        _xform(light, rotate=(90.0, 0.0, 0.0))  # its light (along -Z) ahead, along +Y
    usdsensors.attach(layer, rig, built.frame)
    if wide and (seen := witness.rig(rig, out)):
        usdsensors.attach(layer, seen, built.frame)
        # Only the lidars' wide cameras keep each pixel's normal and distance (the return
        # pictures': `returns`), each a camera's worth of GPU memory.
        data = dict(layer.customLayerData)
        data[format.GEOMETRY_CAMERAS] = ",".join(m.name for m in usdsensors.mounts(seen))
        layer.customLayerData = data
    if built.overview:  # the scene's corner view, for the sheets to open with
        usdsensors.attach(layer, overview.rig(out / "overview"), built.overview)
    if segment:
        materials.label(layer, built.scene)
    layer.Save()
    return out / "stage.usda"


def _rescale(layer: Sdf.Layer, built: Range, scale: float) -> None:
    """Scene `built`'s angle sphere scaled `scale` times about its centre (the SensorMount), in
    `layer`, and each of its lights `scale` squared as bright, so the sphere is as lit as it was."""
    scene = Usd.Stage.Open(str(built.scene))
    prim = scene.GetPrimAtPath(str(built.sphere))
    centre = UsdGeom.Xformable(scene.GetPrimAtPath(built.frame)).ComputeLocalToWorldTransform(0)
    about = Gf.Matrix4d().SetTranslate(-centre.ExtractTranslation())
    about *= Gf.Matrix4d().SetScale(Gf.Vec3d(scale, scale, scale))
    about *= Gf.Matrix4d().SetTranslate(centre.ExtractTranslation())
    parent = UsdGeom.Xformable(prim).ComputeParentToWorldTransform(0)
    local = UsdGeom.Xformable(prim).GetLocalTransformation(0)
    spec = Sdf.CreatePrimInLayer(layer, prim.GetPath())
    Sdf.AttributeSpec(spec, "xformOp:transform", Sdf.ValueTypeNames.Matrix4d).default = (
        local * parent * about * parent.GetInverse()  # row vectors, as USD's
    )
    Sdf.AttributeSpec(
        spec, "xformOpOrder", Sdf.ValueTypeNames.TokenArray, Sdf.VariabilityUniform
    ).default = ["xformOp:transform"]
    for light in scene.Traverse():
        if light.HasAPI(UsdLux.LightAPI):
            bright = UsdLux.LightAPI(light).GetIntensityAttr().Get()
            spec = Sdf.CreatePrimInLayer(layer, light.GetPath())
            Sdf.AttributeSpec(spec, "inputs:intensity", Sdf.ValueTypeNames.Float).default = float(
                np.float32(bright * scale * scale)
            )


def check(
    rig: str | Path,
    out: str | Path,
    scene: str | Path,
    seconds: float = 0.5,
    radius: float | None = None,
    at: tuple[float, float, float] = (0.0, 0.0, 1.6),
    heading: float = 0.0,
    lamp: float | None = None,
    **capturing,
) -> list[Path]:
    """Rig `rig`'s cameras and lidars rendered `seconds` in scene file `scene` (`stage`), into
    `out` (new), as `capturing` (`darkroom.Options`' fields) say, and a check picture for each
    (`sheets`). Returns the pictures; the capture is `out/capture`."""
    folder = _capture(rig, out, scene, seconds, radius, at, heading, lamp, **capturing)
    built = ranges.read(scene)
    if radius is not None:
        built = dataclasses.replace(built, radius=radius)
    return sheets.draw(folder, Path(out), built)


def overlap(
    rig: str | Path,
    out: str | Path,
    sphere: str | Path,
    radii: tuple[float, ...] = (1.0, 2.0, 5.0, 20.0),
    seconds: float = 0.1,
    **capturing,
) -> Path:
    """Rig `rig`'s cameras in scene file `sphere`'s angle sphere, made each of `radii` metres
    round, each picture painted where it lands on the sphere: one map a distance, where they
    overlap marked (`coverage.draw`). Returns the picture, `out/overlap.png`."""
    if ranges.read(sphere).kind != "sphere":
        raise ValueError(f"{sphere} has no angle sphere on record to paint the cameras on.")
    capturing.setdefault("kinds", ["camera"])
    folders = [
        (r, _capture(rig, Path(out) / f"{r:g}m", sphere, seconds, r, **capturing)) for r in radii
    ]
    return coverage.draw(folders, Path(out) / "overlap.png")


def _capture(
    rig: str | Path,
    out: str | Path,
    scene: str | Path,
    seconds: float,
    radius: float | None = None,
    at: tuple[float, float, float] = (0.0, 0.0, 1.6),
    heading: float = 0.0,
    lamp: float | None = None,
    on_line: Callable[[str], object] | None = print,
    **capturing,
) -> Path:
    from darkroom import capture

    job = _job(rig, out, scene, seconds, radius, at, heading, lamp, **capturing)
    return capture(job.stage_file, job.out, job.options, on_line=on_line)


def _job(
    rig: str | Path,
    out: str | Path,
    scene: str | Path,
    seconds: float,
    radius: float | None = None,
    at: tuple[float, float, float] = (0.0, 0.0, 1.6),
    heading: float = 0.0,
    lamp: float | None = None,
    **capturing,
) -> Job:
    """Rig `rig` staged in scene file `scene` (`stage`) in `out`, and its capture into
    `out/capture`, made as `capturing` (`darkroom.Options`' fields) say."""
    out = Path(out)
    made = stage(rig, out, scene, seconds, radius, at, heading, lamp)
    capturing.setdefault("kinds", ["camera", "lidar"])
    capturing.setdefault("scenery_cell", SCENERY_CELL)
    if ranges.read(scene).kind == "bench":
        # Which card each point came back from. ovrtx's lidars take a card's reflectance from
        # its material tag, never its visible colour: its grey cards all read alike.
        capturing.setdefault("lidar_hits", True)
    return Job(made, out / "capture", Options(window=(0.0, seconds), **capturing))


def _xform(
    spec: Sdf.PrimSpec,
    translate: tuple[float, float, float] | None = None,
    rotate: tuple[float, float, float] | None = None,
    scale: tuple[float, float, float] | None = None,
) -> None:
    """Prim `spec` moved, turned (degrees about X, Y, Z) and scaled, in that order of ops."""
    order = []
    for name, kind, value in (
        ("xformOp:translate", Sdf.ValueTypeNames.Double3, translate),
        ("xformOp:rotateXYZ", Sdf.ValueTypeNames.Float3, rotate),
        ("xformOp:scale", Sdf.ValueTypeNames.Float3, scale),
    ):
        if value is not None:
            Sdf.AttributeSpec(spec, name, kind).default = (
                Gf.Vec3d(*value) if "translate" in name else Gf.Vec3f(*value)
            )
            order.append(name)
    Sdf.AttributeSpec(
        spec, "xformOpOrder", Sdf.ValueTypeNames.TokenArray, Sdf.VariabilityUniform
    ).default = order


def sensors_of(rig: str | Path) -> list[Path]:
    """The sensor files rig `rig` carries, each once, in its order."""
    found: list[Path] = []
    for mount in usdsensors.mounts(rig):
        if mount.sensor not in found:
            found.append(mount.sensor)
    return found


def check_sensors(
    sensors: Sequence[str | Path],
    out: str | Path,
    scene: str | Path,
    seconds: float = 0.5,
    **capturing,
) -> list[Path]:
    """Each of `sensors` (sensor files) on its own: all at the scene's SensorMount, each facing
    straight ahead, rendered together in scene file `scene`, a check picture each (`check`). The
    same sensor on several rigs, or turned on one, is checked once, as it is."""
    return check(ahead(sensors, out), out, scene, seconds, **capturing)


def ahead(
    sensors: Sequence[str | Path], out: str | Path, names: Sequence[str] | None = None
) -> Path:
    """A rig of `sensors` (sensor files), each at its origin facing ahead, `out/Ahead.usda`,
    each named as in `names` (else for its file)."""
    rig = usdsensors.new_rig(Path(out) / "Ahead.usda", "Each sensor at the origin, facing ahead.")
    taken: set[str] = set()
    for k, sensor in enumerate(map(Path, sensors)):
        name = names[k] if names else sensor.stem
        name = "".join(c if c.isalnum() else "_" for c in name.lower())
        while name in taken:
            name += "_"
        taken.add(name)
        usdsensors.add(rig, name, sensor)  # at the origin, facing ahead
    return rig


__all__ = [
    "LOOKS",
    "Range",
    "Settings",
    "ahead",
    "check",
    "check_sensors",
    "coverage",
    "overlap",
    "ranges",
    "sensors_of",
    "sheets",
    "stage",
    "suite",
    "test",
]
