"""Rigs put in a range, and the checks' maths, without a renderer or a built range."""

import math

import numpy as np
import pytest
import usdsensors as us
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

import sensorrange
from sensorrange import ranges, sheets

LENS = {"width": 1280, "height": 960, "fx": 948.84, "fy": 948.84, "cx": 640.0, "cy": 480.0}
FRAME = "/World/Frames/SensorMount"


def _range(folder, targets=()):
    """A stand-in for a built range: its scene, a frame a rig goes on and `targets` (name,
    corners, base, attributes) on record as rangegen writes them: labelled quads, each bound to
    a material with its lidar words."""
    folder.mkdir(parents=True)
    stage = Usd.Stage.CreateNew(str(folder / "Range.usda"))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    UsdGeom.Xform.Define(stage, FRAME)
    for name, corners, base, attributes in targets:
        quad = UsdGeom.Mesh.Define(stage, f"/World/Truth/Targets/{name}")
        quad.CreatePointsAttr([Gf.Vec3f(*c) for c in corners])
        quad.CreateVisibilityAttr("invisible")
        quad.GetPrim().CreateAttribute(
            f"semantics:labels:{ranges.TARGET}", Sdf.ValueTypeNames.TokenArray
        ).Set([name])
        material = UsdShade.Material.Define(stage, f"/World/Truth/Looks/{name}")
        material.GetPrim().CreateAttribute(f"{ranges.NV}:base", Sdf.ValueTypeNames.Token).Set(base)
        material.GetPrim().CreateAttribute(
            f"{ranges.NV}:attributes", Sdf.ValueTypeNames.TokenArray
        ).Set(attributes)
        UsdShade.MaterialBindingAPI.Apply(quad.GetPrim()).Bind(material)
    stage.Save()
    return folder / "Range.usda"


def test_a_direction_and_its_angles_are_one_and_the_same():
    az, el = np.array([0.0, 30.0, -45.0, 170.0]), np.array([0.0, 10.0, -20.0, 60.0])
    assert np.allclose(ranges.direction(0.0, 0.0), (0, 1, 0))  # ahead is +Y
    assert np.allclose(ranges.direction(90.0, 0.0), (-1, 0, 0))  # left is -X
    back = ranges.angles(ranges.direction(az, el))
    assert np.allclose(back[0], az) and np.allclose(back[1], el)


def test_a_cameras_field_of_view_is_its_lenses():
    across, down = sheets.field_of_view(LENS)
    assert across == pytest.approx(2 * math.degrees(math.atan(640 / 948.84)), abs=1e-3)
    assert down == pytest.approx(2 * math.degrees(math.atan(480 / 948.84)), abs=1e-3)
    fisheye = {**LENS, "model": "opencvFisheye"}
    assert sheets.field_of_view(fisheye)[0] == pytest.approx(
        math.degrees(2 * 640 / 948.84), abs=1e-3
    )


@pytest.mark.parametrize("model", ["opencvPinhole", "opencvFisheye"])
def test_a_ray_through_a_pixel_projects_back_onto_it(model):
    lens = {**LENS, "model": model, "distortion": {"k1": -0.1, "k2": 0.01}}
    pixels = np.array([(100.0, 50.0), (640.0, 480.0), (1200.0, 900.0)])
    rays = sheets.unproject(lens, pixels)  # OpenCV's frame: +z ahead, +y down
    usd = rays * np.array([1.0, -1.0, -1.0])  # USD's camera: -Z ahead, +Y up
    assert np.allclose(sheets.project(usd, lens), pixels, atol=1e-3)
    assert np.isnan(sheets.project(np.array([[0.0, 0.0, 1.0]]), lens)).all()  # behind it


def test_what_a_camera_should_see_of_the_sphere_is_its_texture_through_its_lens():
    rows, columns = np.mgrid[0 : 180 * 20, 0 : 360 * 20]
    texture = ((rows // 100 + columns // 100) % 2 * 255).astype(float)
    ahead = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])  # its -Z along +Y, +Y up
    seen = sheets.expected(LENS, ahead, np.zeros(3), texture, 10.0, 20)
    assert seen.shape == (960, 1280)
    assert seen[480, 640] == texture[90 * 20, 180 * 20]  # straight ahead: the texture's middle
    moved = np.roll(seen, 3, axis=1)
    assert sheets._shift(moved, seen) == pytest.approx((3.0, 0.0), abs=0.2)


def test_points_on_a_target_and_through_it_are_told_apart(tmp_path):
    pane = (
        "pane",
        [(-1, 15, 0), (1, 15, 0), (1, 15, 2), (-1, 15, 2)],
        "clear_glass",
        ["visually_transparent"],
    )
    wall = ("wall", [(-7, 17, -1), (7, 17, -1), (7, 17, 3), (-7, 17, 3)], "concrete", ["none"])
    built = ranges.read(_range(tmp_path / "lane", [pane, wall]))
    assert built.kind == "targets" and built.targets[0].attributes == ("visually_transparent",)
    points = np.array([[0.0, 15.0, 1.0], [0.0, 17.0, 1.0], [5.0, 17.0, 1.0]])
    glass, behind = built.targets
    assert sheets._on(points, glass).tolist() == [True, False, False]
    assert (sheets._on(points, behind) & sheets._behind(points, glass)).tolist() == [
        False,
        True,
        False,
    ]


def test_a_rig_sits_on_a_ranges_sensor_mount(tmp_path):
    camera = us.make_sensor(tmp_path / "Camera.usda", "camera")
    rig = us.new_rig(tmp_path / "Rig.usda")
    us.add(rig, "front", camera)
    board = ("board", [(-1, 5, 0), (1, 5, 0), (1, 5, 1), (-1, 5, 1)], "paper", ["none"])
    made = sensorrange.stage(
        rig, tmp_path / "out", _range(tmp_path / "range", [board]), seconds=0.2
    )
    stage = Usd.Stage.Open(str(made))
    assert stage.GetPrimAtPath(f"{FRAME}/Rig/front/front").IsA(UsdGeom.Camera)
    assert (stage.GetStartTimeCode(), stage.GetEndTimeCode()) == (0, 6)


def test_the_bench_counts_each_card_s_returns_and_says_how_far_each_grey_is_detected():
    import numpy as np

    from sensorrange import bench, sheets
    from sensorrange.ranges import Target

    def card(name, metres):
        y = float(metres)
        corners = np.array([[-0.1, y, -0.1], [0.1, y, -0.1], [0.1, y, 0.1], [-0.1, y, 0.1]])
        return Target(name, corners, "aluminum", "paint", ("none",))

    targets = (
        card("grey10_00deg_02m", 2),
        card("grey10_00deg_05m", 5),
        card("grey80_00deg_05m", 5),
    )
    on_2m = np.array([[0.0, 2.0, 0.0]] * 3)
    on_5m = np.array([[0.05, 5.0, 0.0]])
    scans = [(np.vstack([on_2m, on_5m]), np.array([0.1, 0.1, 0.1, 0.5]), None)] * 4
    found = bench.cards(scans, targets, sheets._on)
    by = {(c.percent, c.metres): c for c in found}
    assert by[(10, 2)].per_scan == 3 and by[(10, 2)].detected
    assert by[(10, 5)].per_scan == 1 and by[(10, 5)].intensity == 0.5  # the 5 m point: on both
    reach = dict(bench.reach(found))
    assert (
        reach["10 %"] == "to 5 m and on" and reach["80 %"] == "to 5 m and on"
    )  # one material: unnamed
    assert len(bench.panels(found)) == 3


def test_the_bench_counts_only_points_rtx_says_came_back_from_a_card():
    import numpy as np

    from sensorrange import bench, sheets
    from sensorrange.ranges import Target

    corners = np.array([[-0.1, 2, -0.1], [0.1, 2, -0.1], [0.1, 2, 0.1], [-0.1, 2, 0.1]], float)
    targets = (Target("grey10_00deg_02m", corners, "calibration_lambertian", "none", ("none",)),)
    card = bench.card_of(["/World/Targets/grey10_00deg_02m/Card", "/World/Room/Back"])
    hit = np.array([0, 0, 1, 2])  # two on the card, one on the wall, one RTX did not name
    points = np.array([[0.0, 2.0, 0.0]] * 4)  # all on the card's face
    scans = [(points, np.array([0.2, 0.2, 0.9, 0.9]), card[hit])] * 2
    (found,) = bench.cards(scans, targets, sheets._on)
    assert found.per_scan == 2 and found.intensity == np.float32(0.2)


def test_a_suite_s_results_are_plain_numbers_a_row_a_measure(tmp_path):
    import csv

    from sensorrange import suites

    scenarios = {
        "sensors": {
            "sphere_ray": {"pictures": ["sensors/sphere_ray/lidar_check.png"]},
            "bench_ray": {"error": "Failed: no scans"},
        }
    }
    rows = [["across", "128.8°", "129.0°"], ["points / scan", "36,653", ""], ["20 %", "to 20 m and on", ""], ["detected", "", ""]]  # fmt: skip
    said = [{"scenario": "sphere_ray", "run": "sensors", "sensor": "lidar", "kind": "lidar", "rows": rows, "picture": "sensors/sphere_ray/lidar_check.png"}]  # fmt: skip
    suites._write(tmp_path, scenarios, said)
    got = list(csv.DictReader((tmp_path / suites.RESULTS_CSV).open(encoding="utf-8")))
    assert [(r["measure"], r["measured"], r["profile"], r["unit"]) for r in got] == [
        ("across", "128.8", "129.0", "°"),
        ("points / scan", "36653", "", ""),
        ("20 %", "to 20 m and on", "", ""),
        ("error", "Failed: no scans", "", ""),  # the heading row says nothing: left out
    ]


def test_a_scene_says_how_it_is_rendered_in_its_custom_layer_data(tmp_path):
    plain = ranges.read(_range(tmp_path / "plain"))
    assert plain.settings == ranges.Settings() and plain.settings.looks == ranges.LOOKS
    file = _range(tmp_path / "lit")
    layer = Sdf.Layer.FindOrOpen(str(file))
    layer.customLayerData = {
        ranges.SETTINGS: {"lamp": 2e6, "mounted": False, "looks": {"path": {"spp": 64}}}
    }
    layer.Save()
    said = ranges.read(file).settings
    assert said.lamp == 2e6 and not said.mounted
    assert said.looks["path"] == {"look": "path_traced_draft", "spp": 64}  # laid over LOOKS'
    assert said.looks["ray"] == {}
    layer.customLayerData = {ranges.SETTINGS: {"lamps": 1.0}}
    layer.Save()
    with pytest.raises(ValueError, match="lamps"):
        ranges.read(file)


def test_a_scene_s_lamp_is_lit_unless_told_otherwise(tmp_path):
    camera = us.make_sensor(tmp_path / "Camera.usda", "camera")
    rig = us.new_rig(tmp_path / "Rig.usda")
    us.add(rig, "front", camera)
    board = ("board", [(-1, 5, 0), (1, 5, 0), (1, 5, 1), (-1, 5, 1)], "paper", ["none"])
    file = _range(tmp_path / "range", [board])
    layer = Sdf.Layer.FindOrOpen(str(file))
    layer.customLayerData = {ranges.SETTINGS: {"lamp": 5.0}}
    layer.Save()
    lit = Usd.Stage.Open(str(sensorrange.stage(rig, tmp_path / "lit", file, wide=False)))
    assert lit.GetPrimAtPath(f"{FRAME}/Lamp").GetAttribute("inputs:intensity").Get() == 5.0
    dark = Usd.Stage.Open(str(sensorrange.stage(rig, tmp_path / "dark", file, lamp=0.0)))
    assert not dark.GetPrimAtPath(f"{FRAME}/Lamp")


def test_an_angle_sphere_is_scaled_about_its_mount_and_lit_as_before(tmp_path):
    folder = tmp_path / "sphere"
    folder.mkdir()
    stage = Usd.Stage.CreateNew(str(folder / "Sphere.usda"))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    UsdGeom.Xform.Define(stage, FRAME).AddTranslateOp().Set(Gf.Vec3d(1, 2, 3))
    ball = UsdGeom.Mesh.Define(stage, "/World/Ball")
    points = [Gf.Vec3f(1 + 10 * x, 2 + 10 * y, 3 + 10 * z) for x, y, z in np.eye(3)]
    ball.CreatePointsAttr(points)
    ball.GetPrim().CreateAttribute(
        f"semantics:labels:{ranges.RANGE}", Sdf.ValueTypeNames.TokenArray
    ).Set([ranges.SPHERE])
    UsdLux.SphereLight.Define(stage, "/World/Lamp").CreateIntensityAttr(100.0)
    stage.Save()
    built = ranges.read(folder / "Sphere.usda")
    assert built.kind == "sphere" and built.radius == pytest.approx(10.0)
    rig = us.new_rig(tmp_path / "Rig.usda")
    us.add(rig, "front", us.make_sensor(tmp_path / "Camera.usda", "camera"))
    made = Usd.Stage.Open(
        str(sensorrange.stage(rig, tmp_path / "out", built.scene, radius=5.0, segment=False))
    )
    assert ranges.read(made.GetRootLayer().identifier).radius == pytest.approx(5.0)
    scaled = ranges._points(made.GetPrimAtPath("/World/Ball"))
    assert np.allclose(scaled[0], (6, 2, 3))  # 5 m from the mount, as it was 10 m
    lamp = made.GetPrimAtPath("/World/Lamp").GetAttribute("inputs:intensity").Get()
    assert lamp == pytest.approx(25.0)


def test_scenarios_are_named_for_their_scene_files(tmp_path):
    from sensorrange import suites

    one, two, lab = (
        tmp_path / "a" / "sphere.usda",
        tmp_path / "b" / "sphere.usda",
        tmp_path / "lab.usda",
    )
    assert suites._called([one, two, lab]) == {one: "a_sphere", two: "b_sphere", lab: "lab"}
