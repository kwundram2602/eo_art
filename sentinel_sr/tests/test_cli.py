import json

from sentinel_sr import cli
from sentinel_sr.rgbn import Scene


def test_main_writes_manifest(tmp_path, monkeypatch):
    written = [
        Scene(path=tmp_path / "superres_a.tif", cloud_cover=12.5),
        Scene(path=tmp_path / "superres_b.tif", cloud_cover=None),
    ]

    def fake_export(**kwargs):
        assert kwargs["lat"] == 39.49
        assert kwargs["lon"] == -0.43
        assert kwargs["bands"] == ("B02", "B03")
        return written

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
            {"path": str(written[0].path), "cloud_cover": 12.5},
            {"path": str(written[1].path), "cloud_cover": None},
        ]
    }
