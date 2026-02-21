"""Tests for Chart integration: wrap, separators, LOD wiring, public API."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.ticker import FuncFormatter

import tradeaxis as ta
from tradeaxis.chart import Chart


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trading_day(date_str, start_h=9, start_m=30, end_h=16, end_m=0, freq_min=1):
    base = np.datetime64(date_str)
    start = base + np.timedelta64(start_h * 60 + start_m, "m")
    end = base + np.timedelta64(end_h * 60 + end_m, "m")
    return np.arange(start, end, np.timedelta64(freq_min, "m"))


def _two_day_data():
    day1 = _trading_day("2026-01-05")
    day2 = _trading_day("2026-01-06")
    ts = np.concatenate([day1, day2])
    prices = np.cumsum(np.random.randn(len(ts))) + 100
    return ts, prices


def _three_day_data():
    day1 = _trading_day("2026-01-05")
    day2 = _trading_day("2026-01-06")
    day3 = _trading_day("2026-01-07")
    ts = np.concatenate([day1, day2, day3])
    prices = np.cumsum(np.random.randn(len(ts))) + 100
    return ts, prices


# ---------------------------------------------------------------------------
# M3a: Chart + wrap()
# ---------------------------------------------------------------------------


class TestChartWrap:
    def test_wrap_returns_chart(self):
        fig, ax = plt.subplots()
        ts, prices = _two_day_data()
        chart = ta.wrap(ax, ts, prices)
        assert isinstance(chart, Chart)
        plt.close(fig)

    def test_ax_is_still_standard_axes(self):
        fig, ax = plt.subplots()
        ts, prices = _two_day_data()
        ta.wrap(ax, ts, prices)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_correct_segment_count_two_days(self):
        fig, ax = plt.subplots()
        ts, prices = _two_day_data()
        chart = ta.wrap(ax, ts, prices)
        # 1 gap → 2 segments
        assert len(chart._data_lines) == 2
        plt.close(fig)

    def test_correct_segment_count_three_days(self):
        fig, ax = plt.subplots()
        ts, prices = _three_day_data()
        chart = ta.wrap(ax, ts, prices)
        # 2 gaps → 3 segments
        assert len(chart._data_lines) == 3
        plt.close(fig)

    def test_matplotlib_compat_set_title(self):
        fig, ax = plt.subplots()
        ts, prices = _two_day_data()
        ta.wrap(ax, ts, prices)
        ax.set_title("Test Title")
        assert ax.get_title() == "Test Title"
        plt.close(fig)

    def test_matplotlib_compat_axhline(self):
        fig, ax = plt.subplots()
        ts, prices = _two_day_data()
        ta.wrap(ax, ts, prices)
        line_count_before = len(ax.get_lines())
        ax.axhline(100, color="red")
        assert len(ax.get_lines()) == line_count_before + 1
        plt.close(fig)


# ---------------------------------------------------------------------------
# M3b: Separators
# ---------------------------------------------------------------------------


class TestSeparators:
    def test_separator_count_matches_gaps(self):
        fig, ax = plt.subplots()
        ts, prices = _two_day_data()
        chart = ta.wrap(ax, ts, prices, show_separators=True)
        assert len(chart._separator_lines) == 1
        plt.close(fig)

    def test_separator_count_three_days(self):
        fig, ax = plt.subplots()
        ts, prices = _three_day_data()
        chart = ta.wrap(ax, ts, prices, show_separators=True)
        assert len(chart._separator_lines) == 2
        plt.close(fig)

    def test_separator_x_positions(self):
        fig, ax = plt.subplots()
        ts, prices = _two_day_data()
        chart = ta.wrap(ax, ts, prices, show_separators=True)
        gap_indices = np.where(chart.mapping.gap_mask)[0]
        for sep_line, gap_idx in zip(chart._separator_lines, gap_indices):
            xdata = sep_line.get_xdata()
            assert xdata[0] == pytest.approx(gap_idx - 0.5, abs=1)
        plt.close(fig)

    def test_no_separators_when_disabled(self):
        fig, ax = plt.subplots()
        ts, prices = _two_day_data()
        chart = ta.wrap(ax, ts, prices, show_separators=False)
        assert len(chart._separator_lines) == 0
        plt.close(fig)


# ---------------------------------------------------------------------------
# M3c: Tick formatting
# ---------------------------------------------------------------------------


class TestTickFormatting:
    def test_xaxis_uses_func_formatter(self):
        fig, ax = plt.subplots()
        ts, prices = _two_day_data()
        ta.wrap(ax, ts, prices)
        fmt = ax.xaxis.get_major_formatter()
        assert isinstance(fmt, FuncFormatter)
        plt.close(fig)

    def test_tick_label_contains_time(self):
        fig, ax = plt.subplots()
        ts, prices = _two_day_data()
        chart = ta.wrap(ax, ts, prices)
        label = chart.mapping.formatter(0, None)
        assert ":" in label  # should be HH:MM format
        plt.close(fig)


# ---------------------------------------------------------------------------
# M3d: LOD wiring
# ---------------------------------------------------------------------------


class TestLODWiring:
    def test_large_dataset_decimated(self):
        fig, ax = plt.subplots(figsize=(10, 4))
        n = 100_000
        ts = np.arange(
            np.datetime64("2026-01-05T09:30"),
            np.datetime64("2026-01-05T09:30") + np.timedelta64(n, "s"),
            np.timedelta64(1, "s"),
        )
        prices = np.cumsum(np.random.randn(n)) + 100
        chart = ta.wrap(ax, ts, prices)
        total_points = sum(len(line.get_xdata()) for line in chart._data_lines)
        assert total_points < n
        plt.close(fig)


# ---------------------------------------------------------------------------
# M3e: Public API
# ---------------------------------------------------------------------------


class TestPublicAPI:
    def test_wrap_importable(self):
        assert hasattr(ta, "wrap")
        assert callable(ta.wrap)

    def test_chart_importable(self):
        assert hasattr(ta, "Chart")

    def test_to_index(self):
        fig, ax = plt.subplots()
        ts, prices = _two_day_data()
        chart = ta.wrap(ax, ts, prices)
        idx = chart.to_index(ts[10])
        assert idx == 10
        plt.close(fig)

    def test_plot_intraday_convenience(self):
        assert hasattr(ta, "plot_intraday")
        ts, prices = _two_day_data()
        chart = ta.plot_intraday(ts, prices)
        assert isinstance(chart, Chart)
        plt.close("all")
