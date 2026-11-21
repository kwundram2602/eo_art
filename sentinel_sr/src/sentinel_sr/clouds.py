"""AOI-level cloud check on the Sentinel-2 L2A scene classification (SCL).

``eo:cloud_cover`` describes the whole ~100 km tile; a small AOI can be
clear under a cloudy tile or clouded under a clear one. The SCL band says it
per pixel.
"""

from collections.abc import Sequence

import numpy as np

SCL_BAND = "SCL"
# 8 = cloud medium probability, 9 = cloud high probability.
DEFAULT_CLOUD_CLASSES = (8, 9)


def cloud_percent(scl: np.ndarray, classes: Sequence[int]) -> float:
    """Share of AOI pixels (percent) whose SCL class is in ``classes``.

    Pixels without an SCL value (NaN, e.g. outside the scene footprint)
    count as clear only if 0 ("no data") isn't one of ``classes``.
    """
    values = np.nan_to_num(np.asarray(scl, dtype="float64"), nan=0.0)
    if values.size == 0:
        return 0.0
    return float(np.isin(values, list(classes)).mean() * 100.0)
