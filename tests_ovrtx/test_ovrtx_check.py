"""A rig checked in real scenes, rendered by ovrtx for real (the GPU, about a minute a scene):
a plain camera ahead on the SensorMount of each scene in `$SENSORRANGE_TEST_SCENES` (scene
files, separated as `PATH` is), and its check sheet drawn.

    SENSORRANGE_TEST_SCENES=/scenes/sphere.usda:/scenes/lab.usda uv run pytest tests_ovrtx

Not among `pytest`'s own tests (`tests/`): those need no GPU and no scenes. Skipped when the
variable is not set.
"""

import os
from pathlib import Path

import pytest
import usdsensors as us

pytest.importorskip("ovrtx")

from darkroom import format

import sensorrange

SCENES = [Path(s) for s in os.environ.get("SENSORRANGE_TEST_SCENES", "").split(os.pathsep) if s]

pytestmark = pytest.mark.skipif(not SCENES, reason="SENSORRANGE_TEST_SCENES names no scenes")


@pytest.mark.parametrize("scene", SCENES, ids=[s.stem for s in SCENES])
def test_a_camera_is_checked_in_a_scene(tmp_path, scene):
    rig = us.new_rig(tmp_path / "Rig.usda")
    us.add(rig, "front", us.make_sensor(tmp_path / "Camera.usda", "camera"))
    made = sensorrange.check(rig, tmp_path / "out", scene, seconds=0.1)
    assert tmp_path / "out" / "front_check.png" in made
    assert format.finished(tmp_path / "out" / "capture")
