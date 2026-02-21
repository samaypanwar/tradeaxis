"""Tests for TimeMapping: gap detection, index remapping, formatter."""

import numpy as np
import pytest
from datetime import timedelta

from tradeaxis._mapping import detect_gaps, TimeMapping


# ---------------------------------------------------------------------------
# Helpers to build synthetic timestamp arrays
# ---------------------------------------------------------------------------


def _trading_day(date_str: str, start_h=9, start_m=30, end_h=16, end_m=0, freq_min=1):
    """Return 1-min timestamps for a single US equity trading day."""
    base = np.datetime64(date_str)
    start = base + np.timedelta64(start_h * 60 + start_m, "m")
    end = base + np.timedelta64(end_h * 60 + end_m, "m")
    return np.arange(start, end, np.timedelta64(freq_min, "m"))


def _two_day_timestamps():
    """Two consecutive trading days with a 17.5h overnight gap."""
    day1 = _trading_day("2026-01-05")  # Monday
    day2 = _trading_day("2026-01-06")  # Tuesday
    return np.concatenate([day1, day2])


def _three_day_timestamps():
    day1 = _trading_day("2026-01-05")
    day2 = _trading_day("2026-01-06")
    day3 = _trading_day("2026-01-07")
    return np.concatenate([day1, day2, day3])


# ---------------------------------------------------------------------------
# M1a: detect_gaps
# ---------------------------------------------------------------------------


class TestDetectGaps:
    def test_overnight_gap_detected(self):
        ts = _two_day_timestamps()
        mask = detect_gaps(ts, gap="auto")
        day1_len = len(_trading_day("2026-01-05"))
        assert mask.dtype == bool
        assert len(mask) == len(ts)
        assert mask[day1_len] is np.True_
        assert mask.sum() == 1

    def test_no_gaps_continuous_data(self):
        ts = _trading_day("2026-01-05")
        mask = detect_gaps(ts, gap="auto")
        assert mask.sum() == 0

    def test_multiple_gaps_three_days(self):
        ts = _three_day_timestamps()
        mask = detect_gaps(ts, gap="auto")
        assert mask.sum() == 2

    def test_single_point(self):
        ts = np.array([np.datetime64("2026-01-05T09:30")])
        mask = detect_gaps(ts, gap="auto")
        assert len(mask) == 1
        assert mask.sum() == 0

    def test_explicit_timedelta_threshold(self):
        ts = _two_day_timestamps()
        mask = detect_gaps(ts, gap=timedelta(hours=1))
        assert mask.sum() == 1

    def test_callable_gap(self):
        ts = _two_day_timestamps()
        day1_len = len(_trading_day("2026-01-05"))

        def custom(timestamps):
            m = np.zeros(len(timestamps), dtype=bool)
            m[day1_len] = True
            return m

        mask = detect_gaps(ts, gap=custom)
        assert mask.sum() == 1
        assert mask[day1_len]


# ---------------------------------------------------------------------------
# M1b: TimeMapping index remapping
# ---------------------------------------------------------------------------


class TestTimeMappingInit:
    def test_indices_are_contiguous(self):
        ts = _two_day_timestamps()
        m = TimeMapping(ts)
        expected = np.arange(len(ts))
        np.testing.assert_array_equal(m.indices, expected)

    def test_length_matches_input(self):
        ts = _two_day_timestamps()
        m = TimeMapping(ts)
        assert len(m.indices) == len(ts)
        assert len(m.gap_mask) == len(ts)

    def test_gap_mask_matches_detect_gaps(self):
        ts = _two_day_timestamps()
        m = TimeMapping(ts)
        expected_mask = detect_gaps(ts, gap="auto")
        np.testing.assert_array_equal(m.gap_mask, expected_mask)

    def test_timestamps_stored(self):
        ts = _two_day_timestamps()
        m = TimeMapping(ts)
        np.testing.assert_array_equal(m.timestamps, ts)

    def test_custom_gap_param(self):
        ts = _two_day_timestamps()
        from datetime import timedelta

        m = TimeMapping(ts, gap=timedelta(hours=1))
        assert m.gap_mask.sum() == 1


# ---------------------------------------------------------------------------
# M1c: formatter
# ---------------------------------------------------------------------------


class TestTimeMappingFormatter:
    def test_formatter_first_index(self):
        ts = _two_day_timestamps()
        m = TimeMapping(ts)
        result = m.formatter(0, None)
        assert "09:30" in result

    def test_formatter_gap_boundary(self):
        ts = _two_day_timestamps()
        m = TimeMapping(ts)
        day1_len = len(_trading_day("2026-01-05"))
        result = m.formatter(day1_len, None)
        assert "09:30" in result

    def test_formatter_out_of_range_negative(self):
        ts = _two_day_timestamps()
        m = TimeMapping(ts)
        result = m.formatter(-1, None)
        assert result == ""

    def test_formatter_out_of_range_beyond(self):
        ts = _two_day_timestamps()
        m = TimeMapping(ts)
        result = m.formatter(len(ts) + 10, None)
        assert result == ""


# ---------------------------------------------------------------------------
# M1d: to_index / to_datetime
# ---------------------------------------------------------------------------


class TestTimeMappingLookup:
    def test_to_index_exact_match(self):
        ts = _two_day_timestamps()
        m = TimeMapping(ts)
        assert m.to_index(ts[5]) == 5

    def test_to_index_between_points(self):
        ts = _two_day_timestamps()
        m = TimeMapping(ts)
        midpoint = ts[5] + np.timedelta64(20, "s")
        result = m.to_index(midpoint)
        assert result in (5, 6)

    def test_to_index_out_of_range_clamps(self):
        ts = _two_day_timestamps()
        m = TimeMapping(ts)
        before = ts[0] - np.timedelta64(1, "h")
        after = ts[-1] + np.timedelta64(1, "h")
        assert m.to_index(before) == 0
        assert m.to_index(after) == len(ts) - 1

    def test_to_datetime(self):
        ts = _two_day_timestamps()
        m = TimeMapping(ts)
        assert m.to_datetime(5) == ts[5]

    def test_round_trip(self):
        ts = _two_day_timestamps()
        m = TimeMapping(ts)
        idx = 42
        dt = m.to_datetime(idx)
        assert m.to_index(dt) == idx
