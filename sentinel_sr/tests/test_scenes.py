import pytest

from sentinel_sr.scenes import dedupe, tile_epsg

# The four items Planetary Computer returned for one overflight in the peg run:
# two MGRS tiles (AOI in their overlap) x two processing baselines.
PEG = [
    "S2A_MSIL2A_20230214T105141_R051_T32ULE_20240815T232641",
    "S2A_MSIL2A_20230214T105141_R051_T32ULE_20230214T221915",
    "S2A_MSIL2A_20230214T105141_R051_T31UGV_20240815T232641",
    "S2A_MSIL2A_20230214T105141_R051_T31UGV_20230215T004018",
]
PEG_CLOUDS = [2.776712, 2.776923, 2.075929, 2.075235]


def test_peg_overflight_collapses_to_newest_baseline_of_clearest_tile():
    assert dedupe(PEG, PEG_CLOUDS).kept == [2]


def test_newest_baseline_wins_even_when_cloudier():
    ids = [
        "S2B_MSIL2A_20230301T103019_R108_T32ULE_20230301T150000",
        "S2B_MSIL2A_20230301T103019_R108_T32ULE_20240901T120000",
    ]

    assert dedupe(ids, [1.0, 9.0]).kept == [1]


def test_lowest_cloud_tile_wins_missing_cloud_last():
    ids = [
        "S2B_MSIL2A_20230301T103019_R108_T31UGV_20230301T150000",
        "S2B_MSIL2A_20230301T103019_R108_T32ULE_20230301T150000",
        "S2B_MSIL2A_20230301T103019_R108_T32ULF_20230301T150000",
    ]

    assert dedupe(ids, [None, 5.0, 3.0]).kept == [2]
    assert dedupe(ids, [float("nan"), 5.0, 5.0]).kept == [1]  # tie -> first tile name


def test_different_overflights_are_kept_in_input_order():
    ids = [
        "S2A_MSIL2A_20230214T105141_R051_T32ULE_20240815T232641",
        "S2B_MSIL2A_20230214T105141_R051_T32ULE_20240815T232641",  # other satellite
        "S2A_MSIL2A_20230224T105141_R051_T32ULE_20240815T232641",  # other date
        "S2A_MSIL2A_20230214T105141_R008_T32ULE_20240815T232641",  # other orbit
    ]

    assert dedupe(ids, [1.0, 1.0, 1.0, 1.0]).kept == [0, 1, 2, 3]


def test_unparsable_ids_are_kept():
    ids = ["some-other-item", *PEG]

    assert dedupe(ids, [50.0, *PEG_CLOUDS]).kept == [0, 3]


def test_length_mismatch():
    with pytest.raises(ValueError, match="ids"):
        dedupe(PEG, [1.0])


def test_tile_epsg():
    assert tile_epsg("31UGV") == 32631
    assert tile_epsg("32ULE") == 32632
    assert tile_epsg("33HUD") == 32733  # band H: southern hemisphere


def test_other_utm_zone_is_dropped_in_favour_of_the_grid_zone():
    # peg run: T31UGV had less cloud, but it sits in UTM 31 while the grid
    # is UTM 32; reprojecting it left staircase seams.
    result = dedupe(PEG, PEG_CLOUDS, epsg=32632)

    assert result.kept == [0]
    assert result.other_zone == []


def test_overflight_only_in_another_zone_is_reported():
    ids = [
        *PEG,
        "S2A_MSIL2A_20230909T103631_R008_T31UGV_20241105T003939",
        "S2A_MSIL2A_20230909T103631_R008_T31UGV_20230909T150000",
    ]

    result = dedupe(ids, [*PEG_CLOUDS, 0.02, 0.03], epsg=32632)

    assert result.kept == [0]
    assert result.other_zone == [
        "S2A_MSIL2A_20230909T103631_R008_T31UGV_20241105T003939"
    ]
