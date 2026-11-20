"""Sensors and rigs checked in every scene given, and what came of it laid out in one folder.

    sensorrange.test("lidar.usda", ["sphere.usda", "lab.usda"], "checks")  # a sensor or a rig
    # -> checks/<time>__lidar/


```
<out>/
  results.csv                 # a row per measure: run, scenario, sensor, kind, measure, ...
  scenes/<scenario>.png       # each scene from its corner, if it has a corner view
  <camera>/<scenario>.png     # each camera alone, facing ahead: its check sheet in each scene
  <lidar>/<scenario>/         # each lidar alone, facing ahead, in each scene:
    check.png                 #   its check sheet
    hit_map.png, distance.png, albedo.png, incidence.png, combined.png
    bench.csv, targets.csv    #   its returns from each card, each target
  rig/                        # the rig as mounted, in the scenes that say so (`Settings.mounted`):
    extrinsics.png
    <scenario>/<camera>_with_<lidar>.png   # or <scenario>.png, when there is one
  raw.zip                     # the renders
```

`raw.zip` holds the renders: `<run>/<scenario>/`, each run's stage, rig files, capture and
every sheet drawn. They are made beside `<out>` (`<out>.raw/`) and zipped at the end
(`zipped=False` keeps them there unzipped; `unpack` unzips them there again). Each capture
(`<run>/<scenario>/capture`) is an ordinary darkroom capture folder: any exporter of those
(to Rerun, to MCAP) records it.

A folder that would hold one picture is that picture, named for it. A sensor is named as on its
rig (sensors sharing one file named for what their names share), or for its file. A scenario is
named for its scene file (`<stem>`, made unique), `<stem>_<look>` when more than one look is
asked for.

`sensors` is every sensor file given (a rig's, each once), in one render; `rig` a rig as
mounted. With several rigs, each has its own folder, named for it.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from darkroom import Job, format

from . import extrinsics, ranges, sheets

#: The runs' folders: the sensors alone, a rig as mounted.
SENSORS, RIG = "sensors", "rig"
RESULTS_CSV, SCENES_SHOWN, RAW, RAW_ZIP = "results.csv", "scenes", ".raw", "raw.zip"


def raw(out: str | Path) -> Path:
    """Where run folder `out`'s renders are kept: beside it, `<out>.raw/`, so `out` holds only
    what came of them."""
    out = Path(out)
    return out.with_name(out.name + RAW)


#: A number as a sheet writes it, and its unit ("128.8°", "36,653", "98 ms", "0.3 px").
NUMBER = re.compile(r"(-?[\d,]*\.?\d+)\s*(°|%|ms|px|m)?")


def test(
    usd: str | Path, scenes: Sequence[str | Path], out: str | Path, looks: Sequence[str] = ("ray",)
) -> Path:
    """Sensor file or rig `usd` checked (`suite`) in each of `scenes` (scene files): each sensor
    alone facing ahead in every scene, and a rig as mounted in those that say so, into a new
    folder in `out` named for when and for `usd`. Says where as it starts; returns the folder."""
    import usdsensors

    usd = Path(usd).resolve()
    if not usd.is_file():
        raise FileNotFoundError(f"There is no {usd}.")
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = Path(out) / f"{stamp}__{usd.stem}"
    print(f"sensorrange: {usd.name} -> {out}", flush=True)

    rig = bool(usdsensors.mounts(usd))
    return suite(
        out,
        scenes,
        rigs=[usd] if rig else (),
        sensors=named(usd) if rig else {usd.stem: usd},
        looks=looks,
        step=lambda said: print(f"sensorrange: {said}", flush=True),
    )


def named(rig: str | Path) -> dict[str, Path]:
    """Each sensor file on rig `rig`, once, named as on the rig: for what the names of the
    mounts carrying it share ("lidar", "lidar_down15": "lidar"; "tof_l", "tof_r": "tof")."""
    import usdsensors

    carried: dict[Path, list[str]] = {}
    for mount in usdsensors.mounts(rig):
        carried.setdefault(Path(mount.sensor), []).append(mount.name)
    made: dict[str, Path] = {}
    for file, names in carried.items():
        shared = names[0]
        for other in names[1:]:  # what their names begin with alike
            while not other.startswith(shared):
                shared = shared[:-1]
        name = shared.rstrip("_-") or names[0]
        while name in made:
            name += "_"
        made[name] = file
    return made


def suite(
    out: str | Path,
    scenes: Sequence[str | Path],
    sensors: Sequence[str | Path] | dict[str, Path] = (),
    rigs: Sequence[str | Path] = (),
    looks: Sequence[str] = ("ray",),
    seconds: float = 0.3,
    rig_scenes: Sequence[str | Path] | None = None,
    step: Callable[[str], None] | None = None,
    zipped: bool = True,
    **capturing: Any,
) -> Path:
    """`sensors` (sensor files, or a name for each), each alone facing ahead, in one render in
    each of `scenes` (scene files), and each of `rigs` as mounted in each of `rig_scenes` (else
    those of `scenes` that say so, `Settings.mounted`), every scene rendered in each of `looks`
    (its `Settings.looks`, by name), into `out` laid out as above. Every run is staged first,
    then all rendered together in as few renderer processes as can (`darkroom.captures`), then
    every run's sheets drawn at once. A run that fails is noted in the results and the rest go
    on; `step` is told of each step. Returns `out`."""
    from darkroom import captures

    from . import _job, ahead

    out = Path(out)
    on_line = capturing.pop("on_line", print)
    built = {Path(s): ranges.read(s) for s in [*scenes, *(rig_scenes or ())]}
    called = _called(list(built))
    if rig_scenes is None:
        rig_scenes = [s for s in scenes if built[Path(s)].settings.mounted]
    runs: list[tuple[str, list[Path], bool]] = []  # name, files, as mounted
    names = [str(n) for n in sensors] if isinstance(sensors, dict) else None
    if sensors:
        files = list(sensors.values()) if isinstance(sensors, dict) else sensors
        runs.append((SENSORS, [Path(s) for s in files], False))
    for r in rigs:
        runs.append((RIG if len(rigs) == 1 else Path(r).stem, [Path(r)], True))
    say = step or (lambda said: None)
    scenarios: dict[str, dict[str, Any]] = {}  # run -> scenario -> pictures or error
    planned: list[tuple[str, str, Path, Path, Job]] = []  # run, scenario, scene, work, job
    for name, files, whole in runs:
        scenarios[name] = {}
        for scene in map(Path, rig_scenes if whole else scenes):
            told = built[scene].settings
            for look in looks:
                scenario = called[scene] if len(looks) == 1 else f"{called[scene]}_{look}"
                work = raw(out) / name / scenario
                try:
                    if look not in told.looks:
                        raise KeyError(
                            f"{scene.name} has no look {look!r}: {', '.join(told.looks)}"
                        )
                    how: dict[str, Any] = {**told.looks[look], **capturing, "blind": True}
                    lamp = float(how.pop("lamp", told.lamp))
                    rig = files[0] if whole else ahead(files, work, names)
                    job = _job(rig, work, scene, seconds, lamp=lamp, **how)
                    planned.append((name, scenario, scene, work, job))
                except Exception as exc:
                    scenarios[name][scenario] = {"error": f"{type(exc).__name__}: {exc}"}
                    continue
    for name, files, whole in runs:  # where each sensor is on a rig: no rendering needed
        if whole:
            try:
                extrinsics.draw(files[0], out / name / "extrinsics.png")
            except Exception as exc:
                scenarios[name]["extrinsics"] = {"error": f"{type(exc).__name__}: {exc}"}
    say(f"rendering {len(planned)} scenarios")
    made = captures([job for *_, job in planned], on_line=on_line)
    say("drawing the sheets")
    drawn = _draw_all(
        [(Path(job.out), work, scene, name != SENSORS) for name, _, scene, work, job in planned],
        [isinstance(m, Exception) for m in made],
    )
    said: list[dict[str, Any]] = []
    for (name, scenario, scene, work, _), got, error in zip(planned, made, drawn, strict=True):
        if isinstance(got, Exception) or error:
            failed = error or f"{type(got).__name__}: {got}"
            scenarios[name][scenario] = {"error": failed}
            continue
        shown = _present(work, out, scenario, name != SENSORS)
        scenarios[name][scenario] = {
            "pictures": [p.relative_to(out).as_posix() for p in shown.values()]
        }
        for sheet in _read(work / sheets.RESULTS):
            picture = shown.get(sheet["picture"])
            said.append(
                {"run": name, "scenario": scenario, "scene": scene.as_posix()}
                | sheet
                | {"picture": picture.relative_to(out).as_posix() if picture else ""}
            )
    _write(out, scenarios, said)
    if zipped:
        say(f"zipping the renders into {out / RAW_ZIP}")
        pack(out)
    say(f"done: {out}")
    return out


def _called(scenes: list[Path]) -> dict[Path, str]:
    """A name for each of `scenes` (scene files) in a suite: its file's stem, then its folder's
    name before it while two are alike (`a/sphere.usda`, `b/sphere.usda`: `a_sphere`,
    `b_sphere`), else a number after it."""
    called: dict[Path, str] = {}
    for scene in scenes:
        name = scene.stem
        if any(o.stem == scene.stem for o in scenes if o != scene):
            name = f"{scene.resolve().parent.name}_{name}"
        while name in called.values():
            name += "_"
        called[scene] = name
    return called


def pack(out: str | Path) -> Path:
    """Run folder `out`'s renders (`raw`, beside it) zipped into `out / RAW_ZIP`, then
    removed; returns the zip."""
    import zipfile

    out, renders = Path(out), raw(out)
    made = out / RAW_ZIP
    with zipfile.ZipFile(made.with_suffix(".zip.part"), "w", zipfile.ZIP_DEFLATED, compresslevel=1) as file:  # fmt: skip
        for path in sorted(renders.rglob("*")):
            if path.is_file():
                file.write(path, path.relative_to(renders))
    made.with_suffix(".zip.part").replace(made)
    shutil.rmtree(renders)
    return made


def unpack(out: str | Path) -> Path:
    """Run folder `out`'s renders, unzipped beside it from `out / RAW_ZIP` if they are not
    there already; returns their folder (`raw`)."""
    import zipfile

    renders = raw(out)
    if not renders.is_dir() and (Path(out) / RAW_ZIP).is_file():
        with zipfile.ZipFile(Path(out) / RAW_ZIP) as file:
            file.extractall(renders.with_name(renders.name + ".part"))
        renders.with_name(renders.name + ".part").replace(renders)
    return renders


def _draw_all(todo: list[tuple[Path, Path, Path, bool]], skip: list[bool]) -> list[str | None]:
    """Each capture's sheets (`sheets.draw`: capture, run's folder, scene file, pairs) into its
    run's folder, all at once, a thread each (their numpy and PIL work runs alongside: no
    process to re-run a caller's script); what went wrong with each, or None."""
    from concurrent.futures import ThreadPoolExecutor

    jobs = [
        (str(c), str(w), str(r), p) for (c, w, r, p), s in zip(todo, skip, strict=True) if not s
    ]
    results: dict[tuple[str, str, str, bool], str | None] = {}
    if jobs:
        with ThreadPoolExecutor(min(len(jobs), os.cpu_count() or 4)) as pool:
            for job, error in zip(jobs, pool.map(_draw, jobs), strict=True):
                results[job] = error
    return [
        None if s else results[(str(c), str(w), str(r), p)]
        for (c, w, r, p), s in zip(todo, skip, strict=True)
    ]


def _draw(job: tuple[str, str, str, bool]) -> str | None:
    """One capture's sheets drawn (`_draw_all`): None, or what went wrong."""
    capture, work, scene, pairs = job
    try:
        sheets.draw(Path(capture), Path(work), ranges.read(scene), pairs)
        return None
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def _present(work: Path, out: Path, scenario: str, mounted: bool) -> dict[str, Path]:
    """A run's pictures and tables from its work folder into `out`, laid out as above: the
    corner view as `scenes/<scenario>.png`; each sensor's own, alone, in `<sensor>/<scenario>/`
    (`check.png`, ...); as mounted, only each lidar on each camera, in `rig/<scenario>/`.
    Returns where each work file went, by its name."""
    went: dict[str, Path] = {}
    capture = work / "capture"
    names = (
        sorted(format.read_meta(capture).sensors, key=len, reverse=True)
        if format.finished(capture)
        else []
    )
    placed: dict[Path, Path] = {}
    for file in sorted(work.glob("*.png")) + sorted(work.glob("*.csv")):
        if file.name == "overview.png":
            to = out / SCENES_SHOWN / f"{scenario}.png"
        elif "_with_" in file.name:  # a lidar on one of the rig's cameras
            to = out / RIG / scenario / file.name
        elif mounted:
            continue  # each sensor's own is checked alone
        else:
            sensor = next((n for n in names if file.name.startswith(f"{n}_")), None)
            if sensor is None:
                continue
            to = out / sensor / scenario / file.name[len(sensor) + 1 :]
        placed[file] = to
    for file, to in placed.items():
        alone = sum(t.parent == to.parent for t in placed.values()) == 1
        if alone and to.parent.name == scenario:  # one picture: no folder of its own
            to = to.parent.with_name(f"{scenario}{to.suffix}")
        to.parent.mkdir(parents=True, exist_ok=True)
        went[file.name] = Path(shutil.copy2(file, to))
    return went


def _read(file: Path) -> list[dict[str, Any]]:
    return json.loads(file.read_text(encoding="utf-8")) if file.is_file() else []


def _write(out: Path, scenarios: dict[str, Any], said: list[dict[str, Any]]) -> None:
    """`RESULTS_CSV`: a row a measure, and a row for each scenario that failed ("error")."""
    out.mkdir(parents=True, exist_ok=True)
    with (out / RESULTS_CSV).open("w", newline="", encoding="utf-8") as file:
        rows = csv.writer(file)
        rows.writerow(
            ["run", "scenario", "sensor", "kind", "measure", "measured", "profile", "unit",
             "picture"]
        )  # fmt: skip
        for sheet in said:
            for row in sheet["rows"]:
                if len(row) < 2 or not (row[1] or (len(row) > 2 and row[2])):
                    continue  # a heading row, saying nothing itself
                measured, unit = _number(row[1])
                profile, other = _number(row[2]) if len(row) > 2 else ("", "")
                rows.writerow(
                    [sheet["run"], sheet["scenario"], sheet["sensor"], sheet["kind"], row[0],
                     measured, profile, unit or other, sheet["picture"]]
                )  # fmt: skip
        for run, each in scenarios.items():
            for scenario, got in each.items():
                if "error" in got:
                    rows.writerow([run, scenario, "", "", "error", got["error"], "", "", ""])


def failures(out: str | Path) -> list[str]:
    """What failed in run folder `out`, a sentence each ("rig sphere: ..."), from its
    `RESULTS_CSV`."""
    with (Path(out) / RESULTS_CSV).open(newline="", encoding="utf-8") as file:
        return [
            f"{row['run']} {row['scenario']}: {row['measured']}"
            for row in csv.DictReader(file)
            if row["measure"] == "error"
        ]


def _number(text: str) -> tuple[str, str]:
    """`text` as a plain number and its unit, if it is one ("36,653" -> "36653"); else as it is."""
    said = NUMBER.fullmatch(text.strip())
    if said is None:
        return text, ""
    return said.group(1).replace(",", ""), said.group(2) or ""
