from eo_art.forge3d_pipes.acquire.sentinel_bridge import Candidate
from eo_art.forge3d_pipes.acquire.zone_steps import coordinate_steps, describe_step

BINS = ["2024-01-01", "2024-02-01"]


def _c(date, cloud, bin_="2024-01-01"):
    return Candidate(id=f"S2_{date}", datetime=date, cloud_cover=cloud, bin=bin_)


def test_without_offset_every_zone_takes_its_best_scene():
    steps, skipped = coordinate_steps(
        {
            "z00_00": [_c("2024-01-03", 5.0), _c("2024-01-28", 1.0)],
            "z00_01": [_c("2024-01-03", 2.0), _c("2024-01-28", 9.0)],
        },
        BINS,
        None,
    )

    assert skipped == ["2024-02-01"]
    (step,) = steps
    assert {z: p.date for z, p in step.picks.items()} == {
        "z00_00": "2024-01-28",
        "z00_01": "2024-01-03",
    }
    assert step.spread_days == 25
    assert step.window is None
    assert step.date == "2024-01-03"


def test_offset_picks_the_least_cloudy_common_window():
    steps, _ = coordinate_steps(
        {
            "z00_00": [_c("2024-01-03", 5.0), _c("2024-01-28", 1.0)],
            "z00_01": [_c("2024-01-03", 2.0), _c("2024-01-28", 9.0)],
        },
        BINS,
        14,
    )

    (step,) = steps
    # Window at 01-03: mean 3.5; at 01-28: mean 5.0.
    assert {p.date for p in step.picks.values()} == {"2024-01-03"}
    assert step.window == ("2024-01-03", "2024-01-17")
    assert step.spread_days == 0
    assert step.relaxed == {}


def test_offset_prefers_the_window_covering_most_zones():
    steps, _ = coordinate_steps(
        {
            "z00_00": [_c("2024-01-02", 0.0), _c("2024-01-20", 8.0)],
            "z00_01": [_c("2024-01-20", 6.0)],
        },
        BINS,
        7,
    )

    (step,) = steps
    assert step.picks["z00_00"].date == "2024-01-20"
    assert step.relaxed == {}


def test_zone_outside_every_window_is_relaxed_to_the_nearest_scene():
    steps, _ = coordinate_steps(
        {
            "z00_00": [_c("2024-01-02", 1.0)],
            "z00_01": [_c("2024-01-03", 1.0)],
            "z01_00": [_c("2024-01-25", 3.0), _c("2024-01-30", 0.0)],
        },
        BINS,
        7,
    )

    (step,) = steps
    assert step.window == ("2024-01-02", "2024-01-09")
    pick = step.picks["z01_00"]
    # Nearest to the window wins over the less cloudy, later scene.
    assert (pick.date, pick.outside_days) == ("2024-01-25", 16)
    assert step.relaxed == {"z01_00": 16}
    assert step.spread_days == 23
    text = describe_step(step, 7)
    assert "z01_00 2024-01-25 (+16 d outside)" in text
    assert "spread 23 d" in text


def test_zone_without_any_scene_in_the_interval_stays_empty():
    steps, skipped = coordinate_steps(
        {
            "z00_00": [_c("2024-01-02", 1.0), _c("2024-02-10", 1.0, "2024-02-01")],
            "z00_01": [_c("2024-02-11", 1.0, "2024-02-01")],
        },
        BINS,
        7,
    )

    assert skipped == []
    january, february = steps
    assert list(january.picks) == ["z00_00"]
    assert january.empty_zones == ["z00_01"]
    assert february.spread_days == 1
    assert "no scene: z00_01" in describe_step(january, 7)


def test_missing_cloud_cover_ranks_last():
    steps, _ = coordinate_steps(
        {"z00_00": [_c("2024-01-02", None), _c("2024-01-20", 50.0)]}, BINS, None
    )

    assert steps[0].picks["z00_00"].date == "2024-01-20"
