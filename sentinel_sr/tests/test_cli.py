import json

from sentinel_sr import cli
from sentinel_sr.rgbn import ExportResult, Scene


def test_main_writes_manifest(tmp_path, monkeypatch):
    written = [
        Scene(
            path=tmp_path / "superres_a.tif", cloud_cover=12.5, datetime="2023-01-04"
        ),
        Scene(path=tmp_path / "superres_b.tif", cloud_cover=None),
    ]

    def fake_export(**kwargs):
        assert kwargs["lat"] == 39.49
        assert kwargs["lon"] == -0.43
        assert kwargs["bands"] == ("B02", "B03")
        assert kwargs["interval"] is None
        assert kwargs["overwrite"] is False
        return ExportResult(scenes=written, skipped_bins=[])

    monkeypatch.setattr(cli, "export_superres_tifs", fake_export)

    exit_code = cli.main(
        [
            "--lat",
            "39.49",
            "--lon",
            "-0.43",
            "--start-date",
            "2023-01-01",
            "--end-date",
            "2023-12-31",
            "--model-path",
            "/models/LDSRS2-SEN2SR",
            "--bands",
            "B02,B03",
            "--out",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    manifest = json.loads((tmp_path / cli.MANIFEST_NAME).read_text())
    assert manifest == {
        "scenes": [
            {
                "path": str(written[0].path),
                "cloud_cover": 12.5,
                "datetime": "2023-01-04",
                "aoi_cloud": None,
            },
            {
                "path": str(written[1].path),
                "cloud_cover": None,
                "datetime": None,
                "aoi_cloud": None,
            },
        ],
        "skipped_bins": [],
    }


def test_main_forwards_timeseries_options(tmp_path, monkeypatch):
    seen = {}

    def fake_export(**kwargs):
        seen.update(kwargs)
        return ExportResult(scenes=[], skipped_bins=["2023-02-01"])

    monkeypatch.setattr(cli, "export_superres_tifs", fake_export)

    cli.main(
        [
            "--lat=1",
            "--lon=2",
            "--start-date=2023-01-01",
            "--end-date=2023-03-31",
            "--model-path=/m",
            f"--out={tmp_path}",
            "--interval=month",
            "--every=2",
            "--overwrite",
            "--max-aoi-cloud=12.5",
            "--scl-classes=3,8,9",
            "--skip-dates=2023-01-05,2023-03-02",
        ]
    )

    assert (seen["interval"], seen["every"], seen["overwrite"]) == ("month", 2, True)
    assert (seen["max_aoi_cloud"], seen["scl_classes"]) == (12.5, (3, 8, 9))
    assert seen["skip_dates"] == ["2023-01-05", "2023-03-02"]
    manifest = json.loads((tmp_path / cli.MANIFEST_NAME).read_text())
    assert manifest["skipped_bins"] == ["2023-02-01"]
