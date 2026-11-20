"""What a lidar can get back from what a camera beside it sees, a pixel at a time: each pixel's
material's diffuse albedo at the lidar's wavelength (RTX's own table, `albedo.json`; a
calibration card its grey), the cosine of the angle the lidar's ray meets its surface at (the
camera's ray standing for the lidar's: the camera is at the lidar), and the two multiplied (the
share of light it sends back, before range). From the camera's segmentation by lidar material and
its surfaces' normals (`darkroom.format`: a segmented camera has both)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from darkroom import format
from PIL import Image

from .materials import GREY

#: RTX's diffuse albedo of each base material, by wavelength (nm).
ALBEDO: dict[str, dict[str, float]] = json.loads(
    (Path(__file__).parent / "albedo.json").read_text(encoding="utf-8")
)["albedo"]


def albedo_of(label: str, nm: float) -> float:
    """A segmentation label's albedo ("asphalt + paint": asphalt's; "calibration_lambertian @
    0.20": 0.20) in the table's column nearest `nm`; NaN for none."""
    if not label:
        return float("nan")
    if label.startswith(GREY) and "@" in label:
        return float(label.rsplit("@", 1)[1])
    column = "1550" if nm >= 1200 else "905"
    value = ALBEDO.get(label.split(" + ")[0], {}).get(column)
    return float("nan") if value is None else float(value)


def maps(
    capture: Path, camera: str, lens: dict, nm: float
) -> tuple[Image.Image, np.ndarray, np.ndarray, np.ndarray | None] | None:
    """Camera `camera`'s first frame, and for each of its pixels the albedo of what it saw, the
    cosine of the angle its ray meets it at, and its distance (NaN where unknown; no distances if
    not recorded); None without segments or normals."""
    from .sheets import _rotation, first, unproject

    k = first(capture, camera)
    normals = format.read_normals(capture, camera, k)
    if normals is None or not format.segment_path(capture, camera, k).is_file():
        return None
    index, labels = format.read_segments(capture, camera, k)
    albedo = np.array([albedo_of(label, nm) for label in labels])[
        np.clip(index, 0, len(labels) - 1)
    ]
    h, w = index.shape
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    rays = unproject(lens, np.stack([u.ravel(), v.ravel()], axis=-1)) * np.array([1.0, -1.0, -1.0])
    pose = format.read_poses(capture, camera)
    rays = (rays @ _rotation(pose.rotation[k]).T).reshape(h, w, 3)  # in the world, as the normals
    if normals.shape[:2] != (h, w):
        return None
    cos = np.abs(np.sum(rays * normals, axis=-1))
    cos[np.isnan(albedo)] = np.nan
    frame = Image.open(format.frame_path(capture, camera, k)).convert("RGB")
    far = format.read_distances(capture, camera, k)
    if far is not None and far.shape != (h, w):
        far = None
    return frame, albedo, cos, far


def painted(frame: Image.Image, values: np.ndarray, top: float) -> Image.Image:
    """`frame` greyed, each pixel with a value painted over it by the ramp (0 dark to `top`
    yellow); pixels without one left grey."""
    from .seen import RAMP

    stops = np.array(RAMP, float)
    t = np.clip(np.nan_to_num(values, nan=0.0) / top, 0, 1) * (len(stops) - 1)
    i = np.minimum(t.astype(int), len(stops) - 2)
    colour = stops[i] + (stops[i + 1] - stops[i]) * (t - i)[..., None]
    grey = np.asarray(frame.convert("L").convert("RGB"), float)
    known = ~np.isnan(values)
    out = np.where(known[..., None], 0.8 * colour + 0.2 * grey, 0.6 * grey)
    return Image.fromarray(out.astype(np.uint8))


def sheets(capture: Path, camera: str, meta: format.Meta, lidar: str, out: Path) -> list[Path]:
    """What lidar `lidar` should get back from what its wide camera `camera` sees, a pixel at a
    time, a sheet each: distance, albedo (each pixel's material's, at its wavelength), incidence
    (the cosine of the angle its ray meets it at), and all three combined, the lidar equation's
    signal (albedo x cos / distance^2, for a surface wider than the beam) over the floor the
    profile's own anchor sets (`minReflectance` face on at `minReflectionRangeM`), on a log scale:
    1x and up should be detected. None without the camera's segments and normals."""
    from pxr import Usd

    from .seen import _ramp, _scale_bar
    from .sheets import _sheet

    prim = Usd.Stage.Open(meta.stage).GetPrimAtPath(meta.sensors[lidar].prim)

    def core(name: str, default: float) -> float:
        said = prim.GetAttribute(f"omni:sensor:Core:{name}") if prim else None
        return float(said.Get()) if said and said.Get() else default

    nm = core("waveLengthNm", 905.0)
    got = maps(capture, camera, meta.sensors[camera].settings, nm)
    if got is None:
        return []
    frame, albedo, cos, far = got
    made = []

    def sheet(key: str, values: np.ndarray, top: float, ends: tuple[str, str], legend: str, rows: list[tuple[str, ...]]) -> None:  # fmt: skip
        if not np.isfinite(values).any() or not np.isfinite(top):
            return  # nothing known (no lidar material in sight, say: the sphere's): no picture
        body = _scale_bar(painted(frame, values, top), *ends, _ramp, legend)
        made.append(_sheet(body, (f"{lidar} {key}", key, ("",), rows), out / f"{lidar}_{key}.png"))

    if far is not None:
        top = float(np.nanpercentile(far, 99))
        sheet("distance", far, top, ("0 m", f"{top:.1f} m"), "distance", [("median", f"{np.nanmedian(far):.1f} m")])  # fmt: skip
    top = max(float(np.nanpercentile(albedo, 99)), 0.05)
    column = 1550 if nm >= 1200 else 905
    sheet("albedo", albedo, top, ("0", f"{top:.2g}"), f"albedo at {column} nm", [("median", f"{np.nanmedian(albedo):.3f}")])  # fmt: skip
    sheet("incidence", cos, 1.0, ("0", "1"), "cos(incidence), 1 face on", [("median", f"{np.nanmedian(cos):.3f}")])  # fmt: skip
    if far is None:
        return made
    floor, reach = core("minReflectance", 0.1), core("minReflectionRangeM", 30.0)
    signal = albedo * cos / np.maximum(far, 0.05) ** 2 / (floor / reach**2)
    shown = (np.log10(np.clip(signal, 1e-2, 1e2)) + 2) / 4  # 0.01x to 100x as 0 to 1
    above = float(np.nanmean(signal >= 1))
    sheet(
        "combined",
        shown,
        1.0,
        ("0.01x", "100x"),
        "x the detection limit",
        [
            ("albedo x cos / distance²", ""),
            ("detection limit", f"{floor:g} at {reach:g} m"),
            ("above the limit", f"{100 * above:.0f} %"),
            ("median", f"{np.nanmedian(signal):.2g}x"),
        ],
    )
    return made
