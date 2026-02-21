"""Pan/zoom/crosshair interaction controller."""

from __future__ import annotations

import enum
from math import exp
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from tradeaxis._artists import Crosshair

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.backend_bases import KeyEvent, MouseEvent
    from tradeaxis.chart import Chart


class _Mode(enum.Enum):
    IDLE = "idle"
    PAN = "pan"
    ZOOM = "zoom"


class InteractionController:
    """Mouse-driven pan / zoom / crosshair for a TradeAxis chart.

    Attaches to a matplotlib canvas and translates mouse events into
    axis-limit changes and crosshair updates.
    """

    ZOOM_SENSITIVITY = 0.005

    def __init__(self, ax: Axes, chart: Chart) -> None:
        self._ax = ax
        self._chart = chart
        self._mode = _Mode.IDLE

        # State captured on button press
        self._press_x: Optional[float] = None
        self._press_y: Optional[float] = None  # pixel y
        self._press_xlim: Optional[tuple] = None
        self._press_ylim: Optional[tuple] = None

        # Crosshair
        self._crosshair = Crosshair(ax)

        # Connect canvas events
        canvas = ax.figure.canvas
        self._cids: List[int] = [
            canvas.mpl_connect("button_press_event", self._on_press),
            canvas.mpl_connect("motion_notify_event", self._on_motion),
            canvas.mpl_connect("button_release_event", self._on_release),
            canvas.mpl_connect("key_press_event", self._on_key),
        ]

        # Disable default matplotlib navigation toolbar interactions
        if hasattr(canvas, "toolbar") and canvas.toolbar is not None:
            canvas.toolbar.update()

    # -- Event handlers -----------------------------------------------------

    def _on_press(self, event: MouseEvent) -> None:
        if event.inaxes is not self._ax:
            return

        if event.button == 1:  # left → PAN
            self._mode = _Mode.PAN
            self._press_x = event.xdata
            self._press_xlim = self._ax.get_xlim()

        elif event.button == 3:  # right → ZOOM
            self._mode = _Mode.ZOOM
            self._press_x = event.xdata
            self._press_y = event.y  # pixel y
            self._press_xlim = self._ax.get_xlim()

        elif event.button == 2:  # middle → CROSSHAIR
            self._snap_crosshair(event.xdata)

    def _on_motion(self, event: MouseEvent) -> None:
        if event.inaxes is not self._ax:
            return

        if self._mode == _Mode.PAN and self._press_x is not None:
            dx = event.xdata - self._press_x
            lo, hi = self._press_xlim
            self._ax.set_xlim(lo - dx, hi - dx)
            self._ax.figure.canvas.draw_idle()

        elif self._mode == _Mode.ZOOM and self._press_y is not None:
            dy = event.y - self._press_y
            factor = exp(self.ZOOM_SENSITIVITY * dy)
            lo, hi = self._press_xlim
            half_old = (hi - lo) / 2.0
            center = self._press_x
            half_new = half_old / factor
            new_lo = center - half_new
            new_hi = center + half_new
            # Clamp to data range
            new_lo = max(new_lo, 0)
            new_hi = min(new_hi, len(self._chart._indices) - 1)
            self._ax.set_xlim(new_lo, new_hi)
            self._ax.figure.canvas.draw_idle()

    def _on_release(self, event: MouseEvent) -> None:
        self._mode = _Mode.IDLE
        self._press_x = None
        self._press_y = None
        self._press_xlim = None
        self._press_ylim = None

    def _on_key(self, event: KeyEvent) -> None:
        if event.key == "escape":
            self._crosshair.hide()
            self._ax.figure.canvas.draw_idle()

    # -- Crosshair snap -----------------------------------------------------

    def _snap_crosshair(self, xdata: float) -> None:
        idx = int(round(np.clip(xdata, 0, len(self._chart._indices) - 1)))
        x = self._chart._indices[idx]
        y = self._chart._values[idx]
        dt_str = self._chart.mapping.formatter(idx, None)
        label = f"{dt_str}\n{y:.4f}"
        self._crosshair.update(x, y, label)
        self._ax.figure.canvas.draw_idle()

    # -- Teardown -----------------------------------------------------------

    def detach(self) -> None:
        canvas = self._ax.figure.canvas
        for cid in self._cids:
            canvas.mpl_disconnect(cid)
        self._cids.clear()
        if self._crosshair.visible:
            self._crosshair.hide()
        try:
            self._crosshair.remove()
        except Exception:
            pass
