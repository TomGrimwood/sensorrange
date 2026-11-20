def test_a_runs_renders_are_zipped_into_it_and_unzipped_beside_it(tmp_path):
    from sensorrange import suites

    out = tmp_path / "run"
    out.mkdir()
    (suites.raw(out) / "sensors" / "sphere" / "capture").mkdir(parents=True)
    (suites.raw(out) / "sensors" / "sphere" / "capture" / "meta.json").write_text(
        "{}", encoding="utf-8"
    )
    zipped = suites.pack(out)
    assert zipped == out / "raw.zip" and not suites.raw(out).exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run"]
    renders = suites.unpack(out)
    assert (renders / "sensors" / "sphere" / "capture" / "meta.json").read_text(
        encoding="utf-8"
    ) == "{}"
