# sensorrange

Checks run on a sensor rig in a scene. The rig (a `usdsensors` rig, or sensor files) is put on
the scene's `SensorMount`, rendered for a moment by `darkroom` (NVIDIA's ovrtx), and each camera
and lidar gets a sheet of what it saw against what its files say: field of view, lens,
lidar pattern, returns by material and distance, AprilTags, grey cards; the numbers go in a
`results.csv` too.

What a check compares with is read from the scene's own USD, as standard semantic labels: an
angle sphere checkered in degrees (`range: angle_sphere`), targets of known lidar materials
(`target`), AprilTags (`apriltag`), a reflectance bench of grey cards
(`range: reflectance_bench`). Test-range scenes that carry these (envirogen's rangegen makes
some) get the full checks; any other scene gets each sensor's sheet of what it saw. sensorrange
neither needs nor provides any particular scene.

```python
import sensorrange

sensorrange.check("rig.usda", "out/sphere", "scenes/sphere.usda")  # a sheet per sensor
# -> out/sphere/<sensor>_check.png ..., the capture in out/sphere/capture

sensorrange.test("rig.usda", ["scenes/sphere.usda", "scenes/lab.usda"], "checks")
# -> checks/<time>__rig/results.csv, a folder a sensor, the rig as mounted, raw.zip
```

`check(rig, out, scene, ...)`, `check_sensors(sensors, out, scene, ...)` (each facing ahead),
`overlap(rig, out, sphere, radii=...)` (where cameras overlap on the sphere),
`suite(out, scenes, sensors=, rigs=, looks=, ...)` (many at once, rendered together) and
`test(usd, scenes, out)` are in `src/sensorrange/__init__.py` and `suites.py`.

Every capture is an ordinary darkroom capture folder (`<out>/capture`, or in a suite's
`raw.zip`): any exporter of those, to Rerun or MCAP, records it. sensorrange records nothing
itself.

## A scene's settings

How a scene is best rendered for its checks it says in its root layer's
`customLayerData["sensorrange"]`, a dictionary; every key may be left out:

| Key | Type | Default | Means |
|---|---|---|---|
| `lamp` | double | 0 (off) | a ring light round the rig's lens, this bright, shining ahead, so prisms and retroreflective sheeting shine back |
| `looks` | dictionary | `{"ray": {}, "path": {"look": "path_traced_draft"}}` | the looks a suite renders it in, by name: each `darkroom.Options` fields (and `lamp` for that look alone), laid over the defaults |
| `mounted` | bool | true | whether `suite` checks a rig as mounted in it (false for a scene meant for each sensor alone) |

```usda
#usda 1.0
(
    customLayerData = {
        dictionary sensorrange = {
            double lamp = 2000000
        }
    }
)
```

## Checks

```bash
uv sync
uv run python scripts/check.py   # ruff, format, ty, pytest: seconds, no GPU
SENSORRANGE_TEST_SCENES=a.usda:b.usda uv run pytest tests_ovrtx   # a real render in each: the GPU
```

`darkroom` and `usdsensors` come from sibling checkouts (`../darkroom`, `../usdsensors`).
