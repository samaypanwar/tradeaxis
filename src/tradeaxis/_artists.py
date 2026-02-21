"""Session separator and crosshair artist helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np
from matplotlib.lines import Line2D

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class SessionSeparator:
    """Draws dashed vertical lines at gap boundaries."""

    @staticmethod
    def draw(ax: Axes, gap_indices: np.ndarray) -> List[Line2D]:
        lines: List[Line2D] = []
        for idx in gap_indices:
            line = ax.axvline(
                x=idx - 0.5,
                color="gray",
                linestyle="--",
                linewidth=0.8,
                alpha=0.6,
            )
            lines.append(line)
        return lines


class Crosshair:
    """Persistent crosshair (horizontal + vertical lines) with annotation."""

    def __init__(self, ax: Axes) -> None:
        self._ax = ax
        self._vline = ax.axvline(
            x=0, color="gray", linewidth=0.6, alpha=0.8, visible=False
        )
        self._hline = ax.axhline(
            y=0, color="gray", linewidth=0.6, alpha=0.8, visible=False
        )
        self._annotation = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.9),
            fontsize=8,
            visible=False,
        )
        self.visible = False

    def update(self, x: float, y: float, label: str) -> None:
        self._vline.set_xdata([x, x])
        self._hline.set_ydata([y, y])
        self._annotation.xy = (x, y)
        self._annotation.set_text(label)
        self.show()

    def show(self) -> None:
        self._vline.set_visible(True)
        self._hline.set_visible(True)
        self._annotation.set_visible(True)
        self.visible = True

    def hide(self) -> None:
        self._vline.set_visible(False)
        self._hline.set_visible(False)
        self._annotation.set_visible(False)
        self.visible = False

    def remove(self) -> None:
        self._vline.remove()
        self._hline.remove()
        self._annotation.remove()

    @property
    def artists(self) -> list:
        return [self._vline, self._hline, self._annotation]
