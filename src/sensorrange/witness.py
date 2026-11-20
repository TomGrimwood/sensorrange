"""Witness cameras for lidars: a wide fisheye at each lidar, looking where it looks, rendered
only so its points can be drawn on a picture that holds them all (a rig's own cameras see a
fraction of a wide lidar's field). Each is a rig of their own, put beside the rig checked."""

from __future__ import annotations

import math
from pathlib import Path

import usdsensors as us
from pxr import Gf, Sdf, Usd, UsdGeom

#: A witness camera's name: its lidar's, and this.
SUFFIX = "_witness"
#: Its picture (pixels) and focal length (pixels, an equidistant fisheye: 88 degrees either
#: side across, 66 up and down).
SIZE, FOCAL = (1600, 1200), 520.0
#: Nearer than this (metres) it does not see: the lidar it sits in.
NEAR = 0.15
#: A camera's axes in its lidar's frame (rows: camera x, y, z): it looks down -Z along the
#: lidar's +X (ahead), its +Y up along the lidar's +Z, its +X to the lidar's right (-Y).
LOOK = Gf.Matrix3d(0, -1, 0, 0, 0, 1, -1, 0, 0)


def rig(checked: str | Path, out: str | Path) -> Path | None:
    """`out/Witness.usda`: a witness camera for each lidar switched on in rig `checked`, at its
    place and facing; None when it has none."""
    lidars = [m for m in us.mounts(checked) if m.kind == "lidar" and m.active]
    if not lidars:
        return None
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    wide = camera(out / "Wide.usda")
    made = us.new_rig(out / "Witness.usda", "A wide camera at each lidar, looking where it looks.")
    for m in lidars:
        us.add(made, m.name + SUFFIX, wide, us.Pose(m.pose.translate, facing(m.pose.rotate)))
    return made


def facing(lidar: tuple[float, float, float]) -> tuple[float, float, float]:
    """The rotation (degrees about X, then Y, then Z, as a rig's poses are) that has a camera
    look where a lidar turned `lidar` looks, its up the lidar's up."""
    turned = Gf.Matrix4d().SetRotate(Gf.Matrix3d(LOOK)) * _xyz(lidar)
    m = [[turned[i][j] for j in range(3)] for i in range(3)]  # row vectors: m[j][i] is A[i][j]
    # As column vectors the rotation is Rz Ry Rx (X first): read its angles off.
    y = math.asin(max(-1.0, min(1.0, -m[0][2])))
    x = math.atan2(m[1][2], m[2][2])
    z = math.atan2(m[0][1], m[0][0])
    return (math.degrees(x), math.degrees(y), math.degrees(z))


def _xyz(degrees: tuple[float, float, float]) -> Gf.Matrix4d:
    """USD's rotateXYZ: about X first, then Y, then Z (row vectors, as Gf multiplies)."""
    x, y, z = (
        Gf.Matrix4d().SetRotate(Gf.Rotation(axis, d))
        for axis, d in zip(
            (Gf.Vec3d.XAxis(), Gf.Vec3d.YAxis(), Gf.Vec3d.ZAxis()), degrees, strict=True
        )
    )
    return x * y * z


def camera(file: Path) -> Path:
    """A wide witness camera's sensor file: an OpenCV fisheye `SIZE`, `FOCAL`, no distortion,
    seeing from `NEAR`."""
    us.make_sensor(file, "camera", about="A wide fisheye for drawing a lidar's points on.")
    stage = Usd.Stage.Open(str(file))
    cam = UsdGeom.Camera(stage.GetDefaultPrim())
    cam.CreateFocalLengthAttr(10.0)
    cam.CreateHorizontalApertureAttr(10.0 * SIZE[0] / FOCAL)
    cam.CreateVerticalApertureAttr(10.0 * SIZE[1] / FOCAL)
    cam.CreateClippingRangeAttr(Gf.Vec2f(NEAR, 10000))
    prim = cam.GetPrim()
    prim.AddAppliedSchema("OmniLensDistortionOpenCvFisheyeAPI")
    lens = "omni:lensdistortion:opencvFisheye:"
    prim.CreateAttribute("omni:lensdistortion:model", Sdf.ValueTypeNames.Token).Set("opencvFisheye")
    prim.CreateAttribute(lens + "imageSize", Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(*SIZE))
    for key, value in (("fx", FOCAL), ("fy", FOCAL), ("cx", SIZE[0] / 2), ("cy", SIZE[1] / 2)):
        prim.CreateAttribute(lens + key, Sdf.ValueTypeNames.Float).Set(value)
    for key in ("k1", "k2", "k3", "k4"):
        prim.CreateAttribute(lens + key, Sdf.ValueTypeNames.Float).Set(0.0)
    stage.Save()
    return file
