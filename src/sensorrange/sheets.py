"""A picture per sensor of what it saw in a scene against what its files say.

- **Camera, in the sphere:** what it rendered, beside what its lens and pose say it should see
  (the sphere's texture, met by each pixel's ray), and the difference: dark where they agree,
  bright where not; how far the render is shifted from it, in pixels and degrees, beneath.
  Among targets: small corners where each target is, by its lens and pose.
- **Lidar:** its returns by direction from it (azimuth left, elevation up), on a 5° grid, with
  the field of view its profile says as a red box; then from above (X, Y) and from the side
  (Y, Z), the range's circle drawn, so a tilt or a narrow fan shows. What was measured is
  written beside.

Beside the pictures, `RESULTS`: what each sheet says (its sensor, kind and rows of measured
against profile), for analysis; on the bench, `<lidar>_bench.csv`, each card's returns.
"""

from __future__ import annotations

import csv
import json
import math
import threading
from pathlib import Path

import numpy as np
from darkroom import format
from PIL import Image, ImageDraw, ImageFilter

from . import bench, maps, tags, views, witness
from . import overview as corner
from . import seen as drawn
from .ranges import Range, Target, angles
from .style import BLUE, GRID, INK, MUTED, PAPER, RED, _font, _side, _updown

#: What a check's sheets say, beside them: a list of {picture, sensor, kind, headings, rows}.
RESULTS = "results.json"
#: The sheets `draw` has drawn so far, gathered for `RESULTS`: a list a thread, as several
#: captures are drawn at once.
_drawing = threading.local()

#: The lidar pattern's labelled lines, degrees.
LINE = 15


def checked(camera: str) -> bool:
    """Whether a camera is checked: not a lidar's wide witness or the corner view."""
    return not (camera.endswith(witness.SUFFIX) or camera == corner.NAME)


def draw(capture: Path, out: Path, built: Range, pairs: bool = True) -> list[Path]:
    """The check picture of each camera and lidar in `capture` (rendered in scene `built`), into
    `out`; with `pairs`, each lidar on each of the rig's cameras too (not for sensors put
    together only to be checked)."""
    meta = format.read_meta(capture)
    made: list[Path] = []
    _drawing.said = []
    if corner.NAME in meta.sensors:  # the scene from its corner, first: what the test is
        picture = corner.picture(capture, meta.sensors[corner.NAME].settings, built)
        named = "".join(f" {c.lower()}" if c.isupper() else c for c in built.scene.stem).strip()
        title = (named.capitalize(), "test", ("",), [])
        made.append(_sheet(picture, title, out / "overview.png"))
    for name, sensor in meta.sensors.items():
        if sensor.kind == "camera" and not checked(name):
            continue
        if sensor.kind == "camera":
            made.append(_camera(capture, name, sensor.settings, out / f"{name}_check.png", built))
        elif sensor.kind == "lidar":
            made.append(_lidar(capture, name, sensor.settings, out / f"{name}_check.png", built))
    cameras = [n for n, s in meta.sensors.items() if s.kind == "camera" and n != corner.NAME]
    lidars = [n for n, s in meta.sensors.items() if s.kind == "lidar"]
    for camera in cameras:
        for lidar in lidars:
            mine = camera == lidar + witness.SUFFIX  # its own wide witness, else one of the rig's
            if (camera.endswith(witness.SUFFIX) or not pairs) and not mine:
                continue
            bench_ = built.kind == "bench"
            by = "intensity" if bench_ else "distance"
            around = np.concatenate([t.corners for t in built.targets]) if bench_ else None

            card = bench.card_of(format.read_hits(capture, lidar))

            def on_cards(points: np.ndarray, scan: format.Scan, card=card) -> np.ndarray:
                if len(scan.hit) == len(points):  # what RTX says each came back from
                    return card[np.minimum(scan.hit, len(card) - 1)] != ""
                hit = np.zeros(len(points), bool)
                for t in built.targets:
                    hit |= _on(points, t, bench.SLACK)
                return hit

            seen = drawn.overlay(
                capture,
                camera,
                meta.sensors[camera].settings,
                lidar,
                by=by,
                around=around,
                on=on_cards if bench_ else None,
                elsewhere="off the cards",
            )
            if seen is not None:
                title = (
                    (f"{lidar} hit map", "lidar on a wide camera at it")
                    if mine
                    else (f"{lidar} on {camera}", "lidar on camera")
                )
                shown = [lidar] if mine else [camera, lidar]  # the witness is ours: not said
                rows = [row for n in shown for row in _intrinsics(n, meta.sensors[n])]
                made.append(
                    _sheet(
                        seen,
                        (*title, ("",), rows),
                        out / (f"{lidar}_hit_map.png" if mine else f"{camera}_with_{lidar}.png"),
                    )
                )
            if mine:  # what it can get back, a pixel at a time, on its wide camera
                from . import returns

                made += returns.sheets(capture, camera, meta, lidar, out)
    (out / RESULTS).write_text(json.dumps(_drawing.said, indent=1) + "\n", encoding="utf-8")
    return made


def _intrinsics(name: str, sensor: format.Sensor) -> list[tuple[str, ...]]:
    """A sensor's intrinsics, as rows under its name: a camera's lens, image, focal lengths and
    centre; a lidar's field of view and scan rate."""
    s = sensor.settings
    rows: list[tuple[str, ...]] = [(name, "")]
    if sensor.kind == "camera":
        across, down = field_of_view(s)
        rows += [
            *((f"  {k}", v) for k, v in lens_rows(s)),
            ("  image", f"{s['width']} x {s['height']}"),
            ("  fx, fy", f"{s['fx']:.1f}, {s['fy']:.1f}"),
            ("  cx, cy", f"{s['cx']:.1f}, {s['cy']:.1f}"),
            ("  field of view", f"{across:.1f}° x {down:.1f}°"),
        ]
        rows += [(f"  {k}", f"{v:.4g}") for k, v in (s.get("distortion") or {}).items() if v]
    elif sensor.kind == "lidar":
        if "fov" in s:
            a0, a1, e0, e1 = s["fov"]
            rows.append(("  field of view", f"{a1 - a0:.1f}° x {e1 - e0:.1f}°"))
        if s.get("channels"):
            rows.append(("  channels", str(s["channels"])))
        rows.append(("  scan rate", f"{sensor.rate:g} Hz"))
    return rows


# Cameras.


def first(capture: Path, camera: str) -> int:
    """The number of camera `camera`'s first frame kept (a capture may not keep frame 0)."""
    for k in range(1000):
        if format.frame_path(capture, camera, k).is_file():
            return k
    raise FileNotFoundError(f"{camera} has no frames in {capture}.")


def project(xyz: np.ndarray, lens: dict) -> np.ndarray:
    """Pixels (n, 2) of points `xyz` (n, 3) in a camera's frame (USD's: looking down -Z, +Y up),
    through its lens (`darkroom.format.Sensor` settings: OpenCV pinhole or fisheye); NaN behind."""
    x, y, z = xyz[:, 0], -xyz[:, 1], -xyz[:, 2]  # OpenCV's: +x right, +y down, +z ahead
    k = lens.get("distortion") or {}
    with np.errstate(divide="ignore", invalid="ignore"):
        if lens.get("model") == "opencvFisheye":
            r = np.hypot(x, y)
            theta = np.arctan2(r, z)
            t2 = theta**2
            bent = theta * (
                1
                + k.get("k1", 0) * t2
                + k.get("k2", 0) * t2**2
                + k.get("k3", 0) * t2**3
                + k.get("k4", 0) * t2**4
            )
            u, v = np.where(r > 0, bent * x / r, 0.0), np.where(r > 0, bent * y / r, 0.0)
            behind = theta > math.radians(100)
        else:
            u, v = x / z, y / z
            r2 = u * u + v * v
            radial = (1 + k.get("k1", 0) * r2 + k.get("k2", 0) * r2**2 + k.get("k3", 0) * r2**3) / (
                1 + k.get("k4", 0) * r2 + k.get("k5", 0) * r2**2 + k.get("k6", 0) * r2**3
            )
            p1, p2 = k.get("p1", 0), k.get("p2", 0)
            u, v = (
                u * radial + 2 * p1 * u * v + p2 * (r2 + 2 * u * u),
                v * radial + p1 * (r2 + 2 * v * v) + 2 * p2 * u * v,
            )
            behind = z <= 0.01
    pixels = np.stack([lens["fx"] * u + lens["cx"], lens["fy"] * v + lens["cy"]], axis=-1)
    pixels[behind] = np.nan
    return pixels


def field_of_view(lens: dict) -> tuple[float, float]:
    """Degrees across and down from the middle of one image edge to the other's, through the
    middle of the image, as its lens has it."""
    w, h, cx, cy = lens["width"], lens["height"], lens["cx"], lens["cy"]
    left, right, top, bottom = unproject(
        lens, np.array([[0, cy], [w, cy], [cx, 0], [cx, h]], float)
    )
    off_axis = lambda a: math.degrees(math.acos(np.clip(a[2], -1, 1)))  # noqa: E731
    # Each edge's angle from the axis, added: through the front, so a lens wider than 180° is.
    return off_axis(left) + off_axis(right), off_axis(top) + off_axis(bottom)


def unproject(lens: dict, pixels: np.ndarray) -> np.ndarray:
    """Unit rays (n, 3; OpenCV's camera frame: +z ahead, +y down) through `pixels` (n, 2), the
    lens's distortion undone by iteration (as OpenCV's undistortPoints does)."""
    k = lens.get("distortion") or {}
    xd = (pixels[:, 0] - lens["cx"]) / lens["fx"]
    yd = (pixels[:, 1] - lens["cy"]) / lens["fy"]
    if lens.get("model") == "opencvFisheye":
        bent = np.hypot(xd, yd)
        theta = bent.copy()
        for _ in range(20):  # Newton's steps on theta(1 + k1 t^2 + ...) = bent
            t2 = theta * theta
            f = (
                theta
                * (
                    1
                    + k.get("k1", 0) * t2
                    + k.get("k2", 0) * t2**2
                    + k.get("k3", 0) * t2**3
                    + k.get("k4", 0) * t2**4
                )
                - bent
            )
            df = (
                1
                + 3 * k.get("k1", 0) * t2
                + 5 * k.get("k2", 0) * t2**2
                + 7 * k.get("k3", 0) * t2**3
                + 9 * k.get("k4", 0) * t2**4
            )
            theta = theta - f / df
        with np.errstate(divide="ignore", invalid="ignore"):
            across = np.where(bent > 0, np.sin(theta) / bent, 0.0)
        rays = np.stack([xd * across, yd * across, np.cos(theta)], axis=-1)
    else:
        x, y = xd.copy(), yd.copy()
        p1, p2 = k.get("p1", 0), k.get("p2", 0)
        for _ in range(20):
            r2 = x * x + y * y
            radial = (1 + k.get("k1", 0) * r2 + k.get("k2", 0) * r2**2 + k.get("k3", 0) * r2**3) / (
                1 + k.get("k4", 0) * r2 + k.get("k5", 0) * r2**2 + k.get("k6", 0) * r2**3
            )
            x = (xd - (2 * p1 * x * y + p2 * (r2 + 2 * x * x))) / radial
            y = (yd - (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)) / radial
        rays = np.stack([x, y, np.ones_like(x)], axis=-1)
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def expected(
    lens: dict,
    rotation: np.ndarray,
    at: np.ndarray,
    texture: np.ndarray,
    radius: float,
    per_degree: int,
) -> np.ndarray:
    """What a camera of `lens`, turned `rotation` (its axes in the range) at `at`, sees of the
    sphere: each pixel's ray met with the sphere, its texture there (grey, H x W)."""
    w, h = lens["width"], lens["height"]
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    rays = unproject(lens, np.stack([u.ravel(), v.ravel()], axis=-1))
    usd = rays * np.array([1.0, -1.0, -1.0])  # USD's camera: +Y up, looking down -Z
    d = usd @ rotation.T
    b = d @ at
    t = -b + np.sqrt(b * b - (at @ at - radius * radius))
    az, el = angles(at + d * t[:, None])
    rows = np.clip(((90 - el) * per_degree).astype(int), 0, texture.shape[0] - 1)
    columns = np.clip(((180 - az) * per_degree).astype(int), 0, texture.shape[1] - 1)
    return texture[rows, columns].reshape(h, w)


def _shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """How far image `a` is moved from `b`, pixels (x, y): phase correlation, to a tenth."""
    spectrum = np.fft.fft2(a - a.mean()) * np.conj(np.fft.fft2(b - b.mean()))
    peak = np.fft.ifft2(spectrum / (np.abs(spectrum) + 1e-9)).real
    y, x = np.unravel_index(np.argmax(peak), peak.shape)
    h, w = peak.shape
    near = peak[[(y - 1) % h, y, (y + 1) % h]][:, [(x - 1) % w, x, (x + 1) % w]]

    def fine(m: np.ndarray) -> float:  # a parabola's peak through three samples
        bend = m[0] - 2 * m[1] + m[2]
        return float(0.5 * (m[0] - m[2]) / bend) if bend else 0.0

    dx = (x if x <= w // 2 else x - w) + fine(near[1, :])
    dy = (y if y <= h // 2 else y - h) + fine(near[:, 1])
    return float(dx), float(dy)


#: How RTX rendered a fisheye camera (its settings' "rendered"), in words.
RENDERED = {"lut": "LUT-derived", "opencvFisheye": "ovrtx OpenCV"}
#: How many pixels across a camera's picture is compared with what it should see in the sphere.
COMPARED = 1024
#: The full-size crops of a camera's sphere picture (pixels square).
CROP = 540


def _camera(capture: Path, name: str, lens: dict, out: Path, built: Range) -> Path:
    poses = format.read_poses(capture, name)
    k = first(capture, name)
    frame = Image.open(format.frame_path(capture, name, k)).convert("RGB")
    rotation = _rotation(poses.rotation[k])
    across, down = field_of_view(lens)
    rows: list[tuple[str, ...]] = [
        *lens_rows(lens),
        ("image", f"{lens['width']} x {lens['height']}"),
        ("fx, fy", f"{lens['fx']:.1f}, {lens['fy']:.1f}"),
        ("cx, cy", f"{lens['cx']:.1f}, {lens['cy']:.1f}"),
        ("across", f"{across:.1f}°"),
        ("up-down", f"{down:.1f}°"),
    ]
    if built.kind == "targets":
        if built.tags:
            grey = np.asarray(frame.convert("L"))
            rows += tags.check(grey, lens, rotation, poses.position[k], built.tags)
        return _sheet(frame, (name, "camera", ("",), rows), out)
    if built.kind in ("scene", "bench", "site"):  # nothing to compare with: what it saw, its lens
        return _sheet(frame, (name, "camera", ("",), rows), out)
    if built.texture is None:
        raise ValueError(f"{built.scene.name}: its angle sphere has no texture to compare with.")
    texture = np.asarray(Image.open(built.texture).convert("L"), float)
    # Compared at most `COMPARED` pixels across: the same lens, its pixels bigger.
    scale = min(1.0, COMPARED / lens["width"])
    small = {
        **lens,
        **{key: lens[key] * scale for key in ("fx", "fy", "cx", "cy")},
        "width": round(lens["width"] * scale),
        "height": round(lens["height"] * scale),
    }
    rendered = np.asarray(
        frame.convert("L").resize((small["width"], small["height"]), Image.Resampling.BOX), float
    )
    seen = expected(small, rotation, poses.position[k], texture, built.radius, built.per_degree)
    # The lens sees the sphere everywhere but past a fisheye's image circle; the sphere is
    # nowhere black, so a black pixel the lens should see is one the render left out.
    sees = in_lens(small)
    lit = _soft(rendered) > 4.0
    fit = sees & lit
    gain, offset = np.polyfit(seen[fit], rendered[fit], 1)  # its exposure, not its lens
    seen = np.where(sees, gain * seen + offset, 0.0)
    dx, dy = _shift(rendered, seen)
    off = np.abs(_soft(rendered) - _soft(seen))  # softened: hairlines smooth or sharp alike
    # Rendered beside expected, whole; then the middle and the right edge at full size, so the
    # sphere's degree labels read; then the difference and what it covers.
    big = 1100

    def titled(picture: Image.Image, title: str, wide: int) -> Image.Image:
        shown = picture.resize((wide, round(wide * picture.height / picture.width)))
        panel = Image.new("RGB", (wide, shown.height + 26), PAPER)
        panel.paste(shown, (0, 26))
        ImageDraw.Draw(panel).text((0, 2), title, fill=INK, font=_font(16))
        return panel

    def as_grey(values: np.ndarray) -> Image.Image:
        return Image.fromarray(np.clip(values, 0, 255).astype(np.uint8)).convert("RGB")

    zooms = []
    for where, x0 in (("middle", lens["cx"] - CROP / 2), ("right edge", lens["width"] - CROP)):
        x0, y0 = round(x0), round(lens["cy"] - CROP / 2)
        part = {**lens, "cx": lens["cx"] - x0, "cy": lens["cy"] - y0, "width": CROP, "height": CROP}
        wanted = expected(
            part, rotation, poses.position[k], texture, built.radius, built.per_degree
        )
        wanted = np.where(in_lens(part), gain * wanted + offset, 0.0)
        zooms += [
            titled(frame.crop((x0, y0, x0 + CROP, y0 + CROP)), f"rendered, {where}", CROP),
            titled(as_grey(wanted), f"expected, {where}", CROP),
        ]
    top = [titled(frame, "rendered", big), titled(as_grey(seen), "expected", big)]
    # What it covers: across along the image's middle row, up and down along its middle
    # column, as the table's field of view; pixels past a fisheye's image circle left out.
    w, h = lens["width"], lens["height"]
    row = np.stack([np.linspace(0, w, 241), np.full(241, lens["cy"])], axis=-1)
    column = np.stack([np.full(181, lens["cx"]), np.linspace(0, h, 181)], axis=-1)

    def looks(pixels: np.ndarray) -> np.ndarray:
        at = np.clip(
            np.round(pixels * scale).astype(int), 0, [small["width"] - 1, small["height"] - 1]
        )
        pixels = pixels[sees[at[:, 1], at[:, 0]]]
        return (unproject(lens, pixels) * np.array([1.0, -1.0, -1.0])) @ rotation.T  # USD: down -Z

    below = views.both(
        None, looks(row), rotation @ np.array([0.0, 0.0, -1.0]), upright=looks(column)
    )
    full = _rows([top, zooms, [titled(_heat(off), "difference", big), *below]])
    degrees = math.degrees(math.atan(math.hypot(dx, dy) / small["fx"]))
    dx, dy = dx / scale, dy / scale  # in the camera's own pixels
    rows += [
        ("offset", f"{round(dx, 1) + 0:+.1f}, {round(dy, 1) + 0:+.1f} px"),
        ("offset angle", f"{degrees:.2f}°"),
        ("mean diff", f"{off[sees].mean():.1f} / 255"),
        ("dark in the lens", f"{100 * np.mean(~lit[sees]):.1f}%"),
    ]
    return _sheet(full, (name, "camera", ("",), rows), out)


def lens_rows(lens: dict) -> list[tuple[str, str]]:
    """A camera's lens in words: its model, how RTX rendered it ("fisheye, LUT-derived"), and
    the LUT it was rendered through, named for what it was made from."""
    said = lens.get("model", "pinhole").removeprefix("opencv").lower()
    if lens.get("rendered"):
        said += f", {RENDERED.get(lens['rendered'], lens['rendered'])}"
    rows = [("lens", said)]
    if lens.get("lut"):
        rows.append(("LUT", lens["lut"]))
    return rows


def in_lens(lens: dict) -> np.ndarray:
    """Which of camera `lens`'s pixels its lens gathers light for (H x W): all, but past a
    fisheye's image circle (its `circle`, degrees, when known)."""
    w, h = lens["width"], lens["height"]
    if not lens.get("circle"):
        return np.ones((h, w), bool)
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    rays = unproject(lens, np.stack([u.ravel(), v.ravel()], axis=-1))
    off_axis = np.degrees(np.arccos(np.clip(rays[:, 2], -1, 1)))
    return (off_axis <= lens["circle"] / 2).reshape(h, w)


def _soft(grey: np.ndarray) -> np.ndarray:
    """A grey image blurred a little (1.5 px)."""
    picture = Image.fromarray(np.clip(grey, 0, 255).astype(np.uint8))
    return np.asarray(picture.filter(ImageFilter.GaussianBlur(1.5)), float)


def _heat(off: np.ndarray) -> Image.Image:
    """A difference, black where none, through red to yellow where most."""
    t = np.clip(off / 80.0, 0, 1)
    rgb = np.stack(
        [np.clip(3 * t, 0, 1), np.clip(3 * t - 1, 0, 1), np.clip(3 * t - 2, 0, 1)], axis=-1
    )
    return Image.fromarray((rgb * 255).astype(np.uint8))


def _rotation(wxyz: np.ndarray) -> np.ndarray:
    """A unit quaternion (w, x, y, z) as a rotation matrix (columns: its axes in the stage)."""
    w, x, y, z = wxyz
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


# Lidars.


def _lidar(capture: Path, name: str, settings: dict, out: Path, built: Range) -> Path:
    poses = format.read_poses(capture, name)
    numbers = format.scan_numbers(capture, name)
    scans_made = max(len(numbers), 1)
    own, world, strength, when, came = [], [], [], [], []
    card = bench.card_of(format.read_hits(capture, name)) if built.kind == "bench" else None
    for i, scan in format.iter_scans(capture, name):
        near = np.linalg.norm(scan.points, axis=1) < NEAREST  # what is right by it: other sensors
        if card is not None and len(scan.hit) == len(scan.points):  # NO_HIT: the last, ""
            came.append(card[np.minimum(scan.hit, len(card) - 1)][~near])
        else:
            came.append(None)
        own.append(scan.points[~near])
        strength.append(scan.intensity[~near])
        when.append(scan.offset[~near])
        k = min(i, len(poses.rotation) - 1)
        world.append(scan.points[~near] @ _rotation(poses.rotation[k]).T + poses.position[k])
    if not sum(len(o) for o in own):  # nothing in its reach in this range
        rows = [("points / scan", "0", ""), ("returns", "none in reach", "")]
        return _sheet(
            Image.new("RGB", (10, 10), PAPER), (name, "lidar", ("measured", ""), rows), out
        )
    points, placed = np.concatenate(own), np.concatenate(world)
    az = np.degrees(np.arctan2(points[:, 1], points[:, 0]))  # its own: +X ahead, +Y left
    el = np.degrees(np.arctan2(points[:, 2], np.hypot(points[:, 0], points[:, 1])))
    claimed = settings.get("fov")
    pattern = _pattern(az, el, claimed, np.concatenate(when))
    rotation = _rotation(poses.rotation[0])
    if claimed:  # the edges of the box its profile claims, in its own frame: +X ahead, +Y left
        a, e = np.radians(views.box(*claimed))
        box = np.stack([np.cos(a) * np.cos(e), np.sin(a) * np.cos(e), np.sin(e)], -1)
        claimed_looks = box @ rotation.T
    else:
        claimed_looks = None
    # The pattern on its own, big: when each ray fired is what matters most; the views under it.
    small = [
        _trim(v) for v in views.both(placed - poses.position[0], claimed_looks, rotation[:, 0])
    ]
    below: list[Image.Image] = []
    if built.kind == "targets":
        above, every = maps.lane(placed, np.concatenate(strength), built.targets)
        small.append(above)
        with out.with_name(f"{name}_targets.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["target", "material", "points", "intensity", "through"])
            writer.writerows(every)
    elif built.kind in ("scene", "site"):
        small.append(maps.above(placed - poses.position[0]))
    reaches: list[tuple[str, ...]] = []
    if built.kind == "bench":
        scans = list(zip(world, strength, came, strict=True))
        found = bench.cards(scans, built.targets, lambda p, t: _on(p, t, bench.SLACK))
        _cards_csv(found, out.with_name(f"{name}_bench.csv"))
        below = bench.panels(found)
        reaches = [("detected", "", ""), *((p, r, "") for p, r in bench.reach(found))]
    sheet = _rows([[pattern], small, below] if below else [[pattern], small])
    say = lambda d: f"{d:.1f}°"  # noqa: E731
    box = claimed or [math.nan] * 4
    rows: list[tuple[str, ...]] = [
        ("across", say(az.max() - az.min()), say(box[1] - box[0])),
        ("left", say(az.max()), say(box[1])),
        ("right", say(-az.min()), say(-box[0])),
        ("up-down", say(el.max() - el.min()), say(box[3] - box[2])),
        ("up", say(el.max()), say(box[3])),
        ("down", say(-el.min()), say(-box[2])),
        ("points / scan", f"{len(points) // scans_made:,}", ""),
        ("scan time", f"{np.concatenate(when).max() * 1000:.0f} ms", ""),
    ]
    rows = [tuple(v.replace("nan°", "") for v in row) for row in rows] + reaches
    return _sheet(sheet, (name, "lidar", ("measured", "profile"), rows), out)


def _trim(panel: Image.Image, margin: int = 12) -> Image.Image:
    """`panel` cropped to what is drawn on it, `margin` pixels round."""
    drawn = Image.eval(panel.convert("L"), lambda v: 255 if v < 240 else 0).getbbox()
    if drawn is None:
        return panel
    x0, y0, x1, y1 = drawn
    return panel.crop(
        (
            max(x0 - margin, 0),
            max(y0 - margin, 0),
            min(x1 + margin, panel.width),
            min(y1 + margin, panel.height),
        )
    )


def _rows(rows: list[list[Image.Image]]) -> Image.Image:
    """Panels laid out in rows, left to right, top to bottom, 10 pixels apart."""
    width = max(sum(p.width for p in row) + 10 * (len(row) + 1) for row in rows)
    height = sum(max(p.height for p in row) + 10 for row in rows) + 10
    sheet = Image.new("RGB", (width, height), PAPER)
    y = 10
    for row in rows:
        x = 10
        for panel in row:
            sheet.paste(panel, (x, y))
            x += panel.width + 10
        y += max(p.height for p in row) + 10
    return sheet


#: Returns nearer than this (metres) are from what is right by the sensor (another sensor's body).
NEAREST = 0.5


#: How wide a lidar's pattern is drawn (pixels).
PATTERN_WIDTH = 1300


def _pattern(
    az: np.ndarray, el: np.ndarray, claimed: list | None, when: np.ndarray | None = None
) -> Image.Image:
    """Its returns by direction, as it looks out (left at the left, up up), a 5° grid labelled
    every 15° in the margins, the field of view its profile claims as a red box, each point
    coloured by `when` (seconds into its scan it was fired) with a key; titled beneath."""
    left, right = (
        max(az.max(), (claimed or [0, 0])[1]) + 5,
        min(az.min(), (claimed or [0, 0])[0]) - 5,
    )
    low, high = (
        min(el.min(), (claimed or [0, 0, 0, 0])[2]) - 5,
        max(el.max(), (claimed or [0, 0, 0, 0])[3]) + 5,
    )
    gutter, foot = 80, 26
    scale = PATTERN_WIDTH / max(left - right, 1)
    w, h = int((left - right) * scale), int((high - low) * scale)
    panel = Image.new("RGB", (gutter + w + 30, h + foot + 30), PAPER)
    pen = ImageDraw.Draw(panel)
    px = lambda a: gutter + (left - a) * scale  # noqa: E731
    py = lambda e: (high - e) * scale  # noqa: E731
    font = _font(14)
    every = LINE if left - right <= 180 else 3 * LINE  # labels apart on a wide pattern
    for a in range(int(math.ceil(right / 5) * 5), int(left) + 1, 5):
        pen.line([(px(a), 0), (px(a), h)], fill=INK if a % LINE == 0 else GRID, width=1)
        if a % every == 0:
            said = _side(a, 0)
            box = pen.textbbox((0, 0), said, font=font)
            pen.text((px(a) - box[2] / 2, h + 5), said, fill=INK, font=font)
    for e in range(int(math.ceil(low / 5) * 5), int(high) + 1, 5):
        pen.line([(gutter, py(e)), (gutter + w, py(e))], fill=INK if e % LINE == 0 else GRID)
        if e % LINE == 0:
            said = _updown(e, 0)
            box = pen.textbbox((0, 0), said, font=font)
            pen.text((gutter - 6 - box[2], py(e) - box[3] / 2), said, fill=INK, font=font)
    xy = np.stack([px(az), py(el)], axis=-1).astype(int)
    pixels = np.asarray(panel).copy()
    keep = (xy[:, 0] >= gutter) & (xy[:, 0] < gutter + w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    # Each point a square nearly as big as the gap between its directions, so its colour shows.
    seen = max(len(np.unique(xy[keep], axis=0)), 1)
    dot = int(np.clip(0.8 * math.sqrt(w * h / seen), 1, 9))
    colour = BLUE if when is None else _timed(when[keep], when.max())
    for dx in range(dot):
        for dy in range(dot):
            x = np.clip(xy[keep, 0] + dx - dot // 2, 0, pixels.shape[1] - 1)
            y = np.clip(xy[keep, 1] + dy - dot // 2, 0, pixels.shape[0] - 1)
            pixels[y, x] = colour
    panel = Image.fromarray(pixels)
    pen = ImageDraw.Draw(panel)
    if when is not None and when.max() > 0:
        key = np.linspace(0, when.max(), 240)
        strip = np.repeat(_timed(key, when.max())[None], 10, axis=0)
        x0 = gutter + w - 240 - 50  # its last time fits beside it
        panel.paste(Image.fromarray(strip), (x0, h + foot + 10))
        pen.text((x0 - 80, h + foot + 6), "fired 0 ms", fill=INK, font=font)
        pen.text((x0 + 246, h + foot + 6), f"{when.max() * 1000:.0f} ms", fill=INK, font=font)
    if claimed:
        pen.rectangle(
            (px(claimed[1]), py(claimed[3]), px(claimed[0]), py(claimed[2])), outline=RED, width=2
        )
    pen.text((gutter, h + foot + 6), "pattern", fill=INK, font=_font(16))
    return panel


#: Colours for a scan's firing time, early to late (viridis's stops).
TIMES = np.array([(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)], float)


def _timed(when: np.ndarray, last: float) -> np.ndarray:
    """Colours (n, 3; uint8) for times `when` from 0 to `last`."""
    t = np.clip(when / max(last, 1e-9), 0, 1) * (len(TIMES) - 1)
    lo = np.minimum(t.astype(int), len(TIMES) - 2)
    f = (t - lo)[:, None]
    return ((1 - f) * TIMES[lo] + f * TIMES[lo + 1]).astype(np.uint8)


def _behind(points: np.ndarray, t: Target) -> np.ndarray:
    """Which `points` lie in the directions target `t` covers, seen from the sensor (behind it,
    if further): its corners' azimuths and elevations."""
    az, el = angles(t.corners)
    paz, pel = angles(points)
    return (paz >= az.min()) & (paz <= az.max()) & (pel >= el.min()) & (pel <= el.max())


def _on(points: np.ndarray, t: Target, slack: float = 0.15) -> np.ndarray:
    """Which `points` lie on target `t`'s face (within `slack` metres of it)."""
    c = t.corners
    right, up = c[1] - c[0], c[3] - c[0]
    normal = np.cross(right, up)
    normal /= np.linalg.norm(normal)
    rel = points - c[0]
    a, b = rel @ right / (right @ right), rel @ up / (up @ up)
    return (np.abs(rel @ normal) < slack) & (a >= 0) & (a <= 1) & (b >= 0) & (b <= 1)


# The sheet.


#: A sheet's label: the sensor's name, its kind, column headings and rows (name, values...).
Label = tuple[str, str, tuple[str, ...], list[tuple[str, ...]]]


def _cards_csv(found: list[bench.Card], out: Path) -> None:
    """Each bench card's returns, a row each."""
    with out.open("w", newline="", encoding="utf-8") as file:
        rows = csv.writer(file)
        rows.writerow(
            ["percent", "angle", "metres", "points_per_scan", "seen", "intensity", "detected"]
        )
        for c in found:
            rows.writerow(
                [c.percent, c.angle, c.metres, f"{c.per_scan:.2f}", f"{c.seen:.2f}",
                 "" if math.isnan(c.intensity) else f"{c.intensity:.6g}", int(c.detected)]
            )  # fmt: skip


def _sheet(body: Image.Image, label: Label, out: Path) -> Path:
    """`label` (`_label`) at the left of `body`; what it says kept for `RESULTS`."""
    name, kind, heads, rows = label
    getattr(_drawing, "said", []).append(
        {
            "picture": out.name,
            "sensor": name,
            "kind": kind,
            "headings": list(heads),
            "rows": [list(r) for r in rows],
        }
    )
    if body.width > 1600:
        body = body.resize((1600, round(body.height * 1600 / body.width)))
    card = _label(*label)
    sheet = Image.new(
        "RGB", (card.width + body.width + 40, max(card.height, body.height) + 26), PAPER
    )
    sheet.paste(card, (16, 16))
    sheet.paste(body, (card.width + 32, 10))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def _label(
    name: str, kind: str, heads: tuple[str, ...], rows: list[tuple[str, ...]]
) -> Image.Image:
    """The sensor's name, then a plain table: a name column, value columns right aligned under
    `heads`, hairlines between rows. (`kind` is for `RESULTS` only.)"""
    regular, strong, small = _font(17), _font(24, bold=True), _font(14)
    pen = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    wide = lambda text, font: pen.textbbox((0, 0), text, font=font)[2]  # noqa: E731
    first = max([wide(r[0], regular) for r in rows] + [0]) + 28
    columns = [
        max([wide(h, small)] + [wide(r[k + 1], regular) for r in rows if k + 1 < len(r)]) + 24
        for k, h in enumerate(heads)
    ]
    width = int(max(first + sum(columns), wide(name, strong), 300))
    columns[-1] += width - first - sum(columns)  # values flush with the rule
    line = 30
    top = 48 + (line if any(heads) else 0)
    card = Image.new("RGB", (width, top + line * len(rows) + 6), PAPER)
    pen = ImageDraw.Draw(card)
    pen.text((0, 0), name, fill=INK, font=strong)
    pen.line((0, 38, width, 38), fill=INK, width=1)
    right = lambda k: first + sum(columns[: k + 1])  # noqa: E731
    if any(heads):
        for k, head in enumerate(heads):
            pen.text((right(k) - wide(head, small), 46), head, fill=MUTED, font=small)
    y = top
    for row in rows:
        pen.text((0, y + 5), row[0], fill=MUTED, font=regular)
        for k, value in enumerate(row[1:]):
            pen.text((right(k) - wide(value, regular), y + 5), value, fill=INK, font=regular)
        y += line
        pen.line((0, y, width, y), fill=GRID, width=1)
    return card
