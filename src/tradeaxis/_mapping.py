"""Gap detection, datetime-to-index remapping, and tick formatting."""

from __future__ import annotations

from datetime import timedelta
from typing import Callable, Union

import numpy as np

DEFAULT_GAP_MULTIPLIER = 20

GapSpec = Union[str, timedelta, Callable[[np.ndarray], np.ndarray]]


def detect_gaps(timestamps: np.ndarray, gap: GapSpec = "auto") -> np.ndarray:
    """Return a boolean mask of length ``len(timestamps)`` where ``True``
    marks positions immediately *after* a gap.

    Parameters
    ----------
    timestamps : ndarray[datetime64]
        Sorted array of timestamps.
    gap : "auto" | timedelta | callable
        - ``"auto"``: gap where ``dt > DEFAULT_GAP_MULTIPLIER * median(dt)``
          over non-zero diffs.
        - ``timedelta``: gap where ``dt > threshold``.
        - ``callable``: ``f(timestamps) -> bool_mask`` of length
          ``len(timestamps)``.
    """
    n = len(timestamps)
    mask = np.zeros(n, dtype=bool)
    if n <= 1:
        return mask

    if callable(gap) and not isinstance(gap, (str, timedelta)):
        return np.asarray(gap(timestamps), dtype=bool)

    diffs = np.diff(timestamps)

    if isinstance(gap, timedelta):
        threshold = np.timedelta64(int(gap.total_seconds() * 1_000_000), "us")
        gap_indices = np.where(diffs > threshold)[0]
    else:
        nonzero = diffs[diffs > np.timedelta64(0)]
        if len(nonzero) == 0:
            return mask
        median_dt = np.median(nonzero)
        threshold = DEFAULT_GAP_MULTIPLIER * median_dt
        gap_indices = np.where(diffs > threshold)[0]

    mask[gap_indices + 1] = True
    return mask


class TimeMapping:
    """Maps sorted datetimes to a compressed integer index that skips gaps.

    Parameters
    ----------
    timestamps : ndarray[datetime64]
        Sorted array of timestamps.
    gap : "auto" | timedelta | callable
        How to detect gaps (see :func:`detect_gaps`).
    """

    def __init__(self, timestamps: np.ndarray, gap: GapSpec = "auto") -> None:
        self.timestamps = np.asarray(timestamps, dtype="datetime64[ns]")
        self.gap_mask = detect_gaps(self.timestamps, gap)
        self.indices = np.arange(len(self.timestamps))

    # -- tick formatter (for matplotlib FuncFormatter) ----------------------

    def formatter(self, idx: float, pos=None) -> str:
        """Map a compressed index to a human-readable datetime string.

        Returns ``""`` for out-of-range indices.
        """
        i = int(round(idx))
        if i < 0 or i >= len(self.timestamps):
            return ""
        dt64 = self.timestamps[i]
        # Convert to python datetime-like string via numpy str
        # Format: YYYY-MM-DDTHH:MM:SS.sssssssss -> extract HH:MM
        s = str(dt64)
        if "T" in s:
            date_part, time_part = s.split("T")
            hhmm = time_part[:5]
            return hhmm
        return s

    # -- index <-> datetime conversion -------------------------------------

    def to_index(self, dt) -> int:
        """Binary-search a datetime into compressed index space.

        Returns the nearest index, clamped to ``[0, N-1]``.
        """
        dt64 = np.datetime64(dt, "ns")
        pos = int(np.searchsorted(self.timestamps, dt64))
        pos = np.clip(pos, 0, len(self.timestamps) - 1)
        # Check if pos-1 is closer
        if pos > 0:
            d_left = abs(int(dt64 - self.timestamps[pos - 1]))
            d_right = abs(int(dt64 - self.timestamps[pos]))
            if d_left < d_right:
                return pos - 1
        return int(pos)

    def to_datetime(self, idx: int) -> np.datetime64:
        """Direct lookup: index -> stored timestamp."""
        return self.timestamps[idx]
