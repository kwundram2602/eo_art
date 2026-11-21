import numpy as np
import pytest

from sentinel_sr.timeseries import Bins, select_timesteps


def _times(*dates):
    return np.array(dates, dtype="datetime64[ns]")


def test_monthly_picks_lowest_cloud_per_month():
    times = _times("2023-01-03", "2023-01-20", "2023-02-07", "2023-02-15")
    clouds = [30.0, 5.0, 1.0, 2.0]

    selection = select_timesteps(times, clouds, "2023-01-01", "2023-02-28", "month")

    assert selection.indices == [1, 2]
    assert selection.skipped_bins == []


def test_missing_cloud_cover_sorts_last():
    times = _times("2023-01-03", "2023-01-20")

    selection = select_timesteps(
        times, [None, 80.0], "2023-01-01", "2023-01-31", "month"
    )

    assert selection.indices == [1]


def test_nan_cloud_cover_sorts_last():
    times = _times("2023-01-03", "2023-01-20")

    selection = select_timesteps(
        times, [float("nan"), 80.0], "2023-01-01", "2023-01-31", "month"
    )

    assert selection.indices == [1]


def test_tie_goes_to_earlier_scene():
    times = _times("2023-01-20", "2023-01-03")

    selection = select_timesteps(times, [5.0, 5.0], "2023-01-01", "2023-01-31", "month")

    assert selection.indices == [1]


def test_empty_intervals_are_reported():
    times = _times("2023-01-10", "2023-04-10")

    selection = select_timesteps(times, [1.0, 1.0], "2023-01-01", "2023-05-31", "month")

    assert selection.indices == [0, 1]
    assert selection.skipped_bins == ["2023-02-01", "2023-03-01", "2023-05-01"]


def test_every_groups_several_intervals():
    times = _times("2023-01-10", "2023-02-10", "2023-03-10", "2023-04-10")
    clouds = [9.0, 3.0, 4.0, 8.0]

    selection = select_timesteps(
        times, clouds, "2023-01-01", "2023-04-30", "month", every=2
    )

    # Bins: Jan+Feb, Mar+Apr.
    assert selection.indices == [1, 2]


def test_weekly_bins_run_monday_to_sunday():
    # 2023-01-01 is a Sunday: the first week bin is just that day.
    times = _times("2023-01-01", "2023-01-02", "2023-01-08")

    selection = select_timesteps(
        times, [1.0, 1.0, 0.5], "2023-01-01", "2023-01-08", "week"
    )

    assert selection.indices == [0, 2]


def test_yearly_and_daily():
    times = _times("2022-06-01", "2022-07-01", "2023-06-01")
    yearly = select_timesteps(
        times, [5.0, 1.0, 2.0], "2022-01-01", "2023-12-31", "year"
    )
    daily = select_timesteps(times, [5.0, 1.0, 2.0], "2022-06-01", "2022-06-02", "day")

    assert yearly.indices == [1, 2]
    assert daily.indices == [0]
    assert daily.skipped_bins == ["2022-06-02"]


def test_invalid_arguments():
    with pytest.raises(ValueError, match="interval"):
        select_timesteps([], [], "2023-01-01", "2023-01-31", "fortnight")
    with pytest.raises(ValueError, match="every"):
        select_timesteps([], [], "2023-01-01", "2023-01-31", "month", every=0)


def test_filled_intervals_get_no_scene_and_are_not_gaps():
    times = _times("2023-01-10", "2023-02-10", "2023-03-10")

    selection = select_timesteps(
        times,
        [1.0, 1.0, 1.0],
        "2023-01-01",
        "2023-04-30",
        "month",
        filled_dates=["2023-02-27", "2023-04-03"],
    )

    assert selection.indices == [0, 2]
    assert selection.filled_bins == ["2023-02-01", "2023-04-01"]
    assert selection.skipped_bins == []


def test_filled_dates_use_the_same_bins_as_every():
    times = _times("2023-01-10", "2023-03-10")

    selection = select_timesteps(
        times,
        [1.0, 1.0],
        "2023-01-01",
        "2023-04-30",
        "month",
        every=2,
        filled_dates=["2023-02-20", "2022-12-01"],  # Jan+Feb bin; out of range
    )

    assert selection.indices == [1]
    assert selection.filled_bins == ["2023-01-01"]


def test_bins_of_ignores_out_of_range_dates():
    bins = Bins("2023-01-01", "2023-03-31", "month")

    assert bins.of(["2022-12-31", "2023-02-01", "2023-04-01"]) == {1}
    assert bins.start(2) == "2023-03-01"
