import numpy as np
import pytest

from sentinel_sr.clouds import DEFAULT_CLOUD_CLASSES, cloud_percent


def test_counts_only_the_given_classes():
    scl = np.array([[4, 8], [9, 3]], dtype="float64")  # veg, cloud, cloud, shadow

    assert cloud_percent(scl, DEFAULT_CLOUD_CLASSES) == pytest.approx(50.0)
    assert cloud_percent(scl, (3, 8, 9)) == pytest.approx(75.0)


def test_nan_counts_as_no_data():
    scl = np.array([[np.nan, 4], [8, 4]])

    assert cloud_percent(scl, (8, 9)) == pytest.approx(25.0)
    assert cloud_percent(scl, (0, 8, 9)) == pytest.approx(50.0)


def test_empty():
    assert cloud_percent(np.array([]), (8, 9)) == 0.0
