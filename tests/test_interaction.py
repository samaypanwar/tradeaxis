"""Tests for InteractionController: pan, zoom, crosshair."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from unittest.mock import MagicMock
from matplotlib.backend_bases import MouseEvent, KeyEvent
from matplotlib.ticker import FuncFormatter

import tradeaxis as ta
from tradeaxis._interaction import InteractionController, _Mode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chart():
    """Create a simple chart for testing interaction."""
    n = 200
    ts = np.arange(
        np.datetime64("2026-01-05T09:30"),
        np.datetime64("2026-01-05T09:30") + np.timedelta64(n, "m"),
        np.timedelta64(1, "m"),
    )
    prices = np.cumsum(np.random.randn(n)) + 100
    fig, ax = plt.subplots()
    chart = ta.wrap(ax, ts, prices, interaction=True)
    return fig, ax, chart


def _mouse_event(fig, ax, x, y, button, name="button_press_event"):
    """Create a synthetic MouseEvent."""
    canvas = fig.canvas
    # Transform data coords to display coords for pixel position
    trans = ax.transData
    display_xy = trans.transform((x, y))
    event = MouseEvent(name, canvas, display_xy[0], display_xy[1], button=button)
    event.inaxes = ax
    event.xdata = x
    event.ydata = y
    return event


def _key_event(fig, key):
    canvas = fig.canvas
    event = KeyEvent("key_press_event", canvas, key)
    return event


# ---------------------------------------------------------------------------
# M4a: Pan (left-drag)
# ---------------------------------------------------------------------------


class TestPan:
    def test_left_drag_shifts_xlim(self):
        fig, ax, chart = _make_chart()
        ic = chart._interaction
        xlim_before = ax.get_xlim()

        # Press at x=50
        press = _mouse_event(fig, ax, 50.0, 100.0, button=1, name="button_press_event")
        ic._on_press(press)
        assert ic._mode == _Mode.PAN

        # Drag to x=70 (delta = +20, so xlim shifts by -20)
        motion = _mouse_event(
            fig, ax, 70.0, 100.0, button=1, name="motion_notify_event"
        )
        ic._on_motion(motion)

        xlim_after = ax.get_xlim()
        expected_shift = -20.0
        assert pytest.approx(xlim_after[0], abs=1) == xlim_before[0] + expected_shift
        assert pytest.approx(xlim_after[1], abs=1) == xlim_before[1] + expected_shift
        plt.close(fig)

    def test_release_returns_to_idle(self):
        fig, ax, chart = _make_chart()
        ic = chart._interaction

        press = _mouse_event(fig, ax, 50.0, 100.0, button=1, name="button_press_event")
        ic._on_press(press)
        assert ic._mode == _Mode.PAN

        release = _mouse_event(
            fig, ax, 70.0, 100.0, button=1, name="button_release_event"
        )
        ic._on_release(release)
        assert ic._mode == _Mode.IDLE
        plt.close(fig)

    def test_motion_after_release_does_nothing(self):
        fig, ax, chart = _make_chart()
        ic = chart._interaction

        press = _mouse_event(fig, ax, 50.0, 100.0, button=1, name="button_press_event")
        ic._on_press(press)
        release = _mouse_event(
            fig, ax, 70.0, 100.0, button=1, name="button_release_event"
        )
        ic._on_release(release)

        xlim_before = ax.get_xlim()
        motion = _mouse_event(
            fig, ax, 90.0, 100.0, button=1, name="motion_notify_event"
        )
        ic._on_motion(motion)
        xlim_after = ax.get_xlim()

        assert xlim_after[0] == pytest.approx(xlim_before[0], abs=0.01)
        assert xlim_after[1] == pytest.approx(xlim_before[1], abs=0.01)
        plt.close(fig)


# ---------------------------------------------------------------------------
# M4b: Zoom (right-drag vertical)
# ---------------------------------------------------------------------------


class TestZoom:
    def test_upward_drag_zooms_in(self):
        fig, ax, chart = _make_chart()
        ic = chart._interaction

        ax.set_xlim(0, 199)
        width_before = ax.get_xlim()[1] - ax.get_xlim()[0]

        press = _mouse_event(fig, ax, 100.0, 100.0, button=3, name="button_press_event")
        # Set pixel y manually since Agg backend may not give us proper coords
        press.y = 200
        ic._on_press(press)
        assert ic._mode == _Mode.ZOOM

        # Upward motion (higher pixel y = zoom in)
        motion = _mouse_event(
            fig, ax, 100.0, 100.0, button=3, name="motion_notify_event"
        )
        motion.y = 300  # 100 pixels up
        motion.inaxes = ax
        ic._on_motion(motion)

        width_after = ax.get_xlim()[1] - ax.get_xlim()[0]
        assert width_after < width_before  # zoomed in
        plt.close(fig)

    def test_downward_drag_zooms_out(self):
        fig, ax, chart = _make_chart()
        ic = chart._interaction

        ax.set_xlim(50, 150)
        width_before = ax.get_xlim()[1] - ax.get_xlim()[0]

        press = _mouse_event(fig, ax, 100.0, 100.0, button=3, name="button_press_event")
        press.y = 300
        ic._on_press(press)

        motion = _mouse_event(
            fig, ax, 100.0, 100.0, button=3, name="motion_notify_event"
        )
        motion.y = 200  # 100 pixels down
        motion.inaxes = ax
        ic._on_motion(motion)

        width_after = ax.get_xlim()[1] - ax.get_xlim()[0]
        assert width_after > width_before  # zoomed out
        plt.close(fig)

    def test_zoom_center_stays_at_cursor(self):
        fig, ax, chart = _make_chart()
        ic = chart._interaction

        center = 100.0
        ax.set_xlim(0, 199)

        press = _mouse_event(
            fig, ax, center, 100.0, button=3, name="button_press_event"
        )
        press.y = 200
        ic._on_press(press)

        motion = _mouse_event(
            fig, ax, center, 100.0, button=3, name="motion_notify_event"
        )
        motion.y = 250
        motion.inaxes = ax
        ic._on_motion(motion)

        lo, hi = ax.get_xlim()
        actual_center = (lo + hi) / 2
        assert pytest.approx(actual_center, abs=1) == center
        plt.close(fig)


# ---------------------------------------------------------------------------
# M4c: Crosshair + snap
# ---------------------------------------------------------------------------


class TestCrosshair:
    def test_snap_to_nearest_point(self):
        fig, ax, chart = _make_chart()
        ic = chart._interaction

        # Middle-click near index 50
        press = _mouse_event(fig, ax, 50.3, 100.0, button=2, name="button_press_event")
        ic._on_press(press)

        # Crosshair should snap to index 50
        xdata = ic._crosshair._vline.get_xdata()
        assert xdata[0] == pytest.approx(50.0, abs=0.5)
        plt.close(fig)

    def test_crosshair_annotation_text(self):
        fig, ax, chart = _make_chart()
        ic = chart._interaction

        press = _mouse_event(fig, ax, 50.0, 100.0, button=2, name="button_press_event")
        ic._on_press(press)

        text = ic._crosshair._annotation.get_text()
        assert ":" in text  # contains time HH:MM
        assert "\n" in text  # has two lines: datetime and value
        plt.close(fig)

    def test_crosshair_visible_after_click(self):
        fig, ax, chart = _make_chart()
        ic = chart._interaction

        assert not ic._crosshair.visible

        press = _mouse_event(fig, ax, 50.0, 100.0, button=2, name="button_press_event")
        ic._on_press(press)

        assert ic._crosshair.visible
        plt.close(fig)


# ---------------------------------------------------------------------------
# M4d: Crosshair toggle/dismiss
# ---------------------------------------------------------------------------


class TestCrosshairToggle:
    def test_second_click_moves_crosshair(self):
        fig, ax, chart = _make_chart()
        ic = chart._interaction

        press1 = _mouse_event(fig, ax, 50.0, 100.0, button=2, name="button_press_event")
        ic._on_press(press1)
        x1 = ic._crosshair._vline.get_xdata()[0]

        press2 = _mouse_event(fig, ax, 80.0, 100.0, button=2, name="button_press_event")
        ic._on_press(press2)
        x2 = ic._crosshair._vline.get_xdata()[0]

        assert x2 != x1
        assert ic._crosshair.visible
        plt.close(fig)

    def test_escape_hides_crosshair(self):
        fig, ax, chart = _make_chart()
        ic = chart._interaction

        press = _mouse_event(fig, ax, 50.0, 100.0, button=2, name="button_press_event")
        ic._on_press(press)
        assert ic._crosshair.visible

        esc = _key_event(fig, "escape")
        ic._on_key(esc)
        assert not ic._crosshair.visible
        plt.close(fig)


# ---------------------------------------------------------------------------
# M4e: chart.detach()
# ---------------------------------------------------------------------------


class TestDetach:
    def test_detach_removes_tradeaxis_ref(self):
        fig, ax, chart = _make_chart()
        assert hasattr(ax, "_tradeaxis")
        chart.detach()
        assert not hasattr(ax, "_tradeaxis")
        plt.close(fig)

    def test_detach_restores_formatter(self):
        fig, ax = plt.subplots()
        original_fmt = ax.xaxis.get_major_formatter()
        ts = np.arange(
            np.datetime64("2026-01-05T09:30"),
            np.datetime64("2026-01-05T09:30") + np.timedelta64(100, "m"),
            np.timedelta64(1, "m"),
        )
        chart = ta.wrap(ax, ts, np.random.randn(100) + 100)
        assert isinstance(ax.xaxis.get_major_formatter(), FuncFormatter)
        chart.detach()
        # After detach, formatter should be the original (not FuncFormatter)
        fmt_after = ax.xaxis.get_major_formatter()
        assert not isinstance(fmt_after, FuncFormatter) or fmt_after is original_fmt
        plt.close(fig)

    def test_detach_removes_separators(self):
        fig, ax, chart = _make_chart()
        chart.detach()
        assert len(chart._separator_lines) == 0
        plt.close(fig)

    def test_detach_removes_data_lines(self):
        fig, ax, chart = _make_chart()
        chart.detach()
        assert len(chart._data_lines) == 0
        plt.close(fig)
