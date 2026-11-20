# sensorrange

Checks run on a sensor rig in a scene, and the sheets and numbers that come of them. Knows no
scene by name: whoever uses it hands it scene files, and what to check against is read from
each scene's own USD (`ranges`). `README.md` has the API and a scene's settings.

## Rules

- **General, not any one user's.** Nothing here names an application, a maker's product, a
  site or a scene. A scene says how it is rendered in its own `customLayerData["sensorrange"]`
  (`ranges.Settings`); a new setting goes there, documented in the README.
- **darkroom renders, sensorrange checks.** Captures are made only through darkroom's front
  door (`darkroom.__all__`, `darkroom.format`) and stay ordinary capture folders; recording
  them (Rerun, MCAP) is an exporter's job, not this one's.
- **Small files.** No source file over 800 lines.
- No environment variables in the library: everything is an argument. Only `tests_ovrtx` reads
  one (`SENSORRANGE_TEST_SCENES`).
- Tests in `tests/` need no GPU and no scenes; anything that renders is in `tests_ovrtx`.
- Comments and docstrings say plainly what the code does, for a reader with only the file.
- Work on a branch; merge to `main` with `--no-ff` when the checks pass. Commit only when asked.

## Commands

```bash
uv sync
uv run python scripts/check.py   # ruff, format, ty, pytest
SENSORRANGE_TEST_SCENES=a.usda:b.usda uv run pytest tests_ovrtx   # real renders, the GPU
```
