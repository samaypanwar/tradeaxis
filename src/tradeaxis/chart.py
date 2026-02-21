"""Chart class orchestrating TimeMapping, LOD, interaction, and artists."""

from __future__ import annotations

from datetime import timedelta
from math import ceil
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from tradeaxis._artists import SessionSeparator
from tradeaxis._decimation import LODRenderer
from tradeaxis._mapping import GapSpec, TimeMapping

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


class Chart:
    """Thin wrapper that composes gap-compression, LOD decimation, and
    interaction onto a standard :class:`matplotlib.axes.Axes`.

    Users should normally call :func:`tradeaxis.wrap` rather than
    instantiating this directly.
    """

    def __init__(
        self,
        ax: Axes,
        timestamps: np.ndarray,
        values: np.ndarray,
        *,
        gap: GapSpec = "auto",
        show_separators: bool = True,
        interaction: bool = True,
    ) -> None:
        self._ax = ax
        self._fig: Figure = ax.figure
        self._original_formatter = ax.xaxis.get_major_formatter()

        # Build the mapping
        self.mapping = TimeMapping(timestamps, gap=gap)
        self._values = np.asarray(values, dtype=np.float64)
        self._indices = self.mapping.indices.astype(np.float64)

        # LOD renderer over compressed-index x-axis
        self._lod = LODRenderer(self._indices, self._values)

        # Plot segmented lines (broken at gaps)
        self._data_lines: List[Line2D] = []
        self._separator_lines: List[Line2D] = []

        self._plot_segments()

        # Separators
        if show_separators:
            gap_where = np.where(self.mapping.gap_mask)[0]
            self._separator_lines = SessionSeparator.draw(ax, gap_where)

        # Apply tick formatter
        ax.xaxis.set_major_formatter(FuncFormatter(self.mapping.formatter))

        # Apply initial LOD and set limits
        ax.set_xlim(0, len(timestamps) - 1)
        self._apply_lod()

        # Hook xlim_changed for LOD refresh
        self._xlim_cid = ax.callbacks.connect("xlim_changed", self._on_xlim_changed)

        # Store ref on ax to prevent GC
        ax._tradeaxis = self  # type: ignore[attr-defined]

        # Interaction (deferred to avoid circular import at module level)
        self._interaction = None
        self._event_cids: List[int] = []
        if interaction:
            from tradeaxis._interaction import InteractionController

            self._interaction = InteractionController(ax, self)

    # -- Segment plotting ---------------------------------------------------

    def _plot_segments(self) -> None:
        gap_mask = self.mapping.gap_mask
        # Find segment boundaries
        gap_indices = np.where(gap_mask)[0]
        boundaries = np.concatenate([[0], gap_indices, [len(self._indices)]])

        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            if end <= start:
                continue
            (line,) = self._ax.plot(
                self._indices[start:end],
                self._values[start:end],
                color="C0",
                linewidth=1.0,
            )
            self._data_lines.append(line)

    # -- LOD ----------------------------------------------------------------

    def _apply_lod(self) -> None:
        xlim = self._ax.get_xlim()
        i0 = max(0, int(xlim[0]))
        i1 = min(len(self._indices), int(ceil(xlim[1])) + 1)

        # Estimate pixel width from figure
        bbox = self._ax.get_window_extent()
        pixel_width = max(1, int(bbox.width)) if bbox.width > 0 else 800

        n_visible = i1 - i0
        if n_visible <= 2 * pixel_width:
            # No decimation needed — restore full-res for visible segments
            self._update_lines_full(i0, i1)
        else:
            xd, yd = self._lod.get_decimated(i0, i1, pixel_width)
            self._update_lines_decimated(xd, yd, i0, i1)

    def _update_lines_full(self, i0: int, i1: int) -> None:
        gap_mask = self.mapping.gap_mask
        gap_indices = np.where(gap_mask)[0]
        boundaries = np.concatenate([[0], gap_indices, [len(self._indices)]])

        line_idx = 0
        for i in range(len(boundaries) - 1):
            seg_start, seg_end = boundaries[i], boundaries[i + 1]
            if seg_end <= seg_start:
                continue
            # Clip to visible range
            vis_start = max(seg_start, i0)
            vis_end = min(seg_end, i1)
            if line_idx < len(self._data_lines):
                if vis_start < vis_end:
                    self._data_lines[line_idx].set_data(
                        self._indices[vis_start:vis_end],
                        self._values[vis_start:vis_end],
                    )
                else:
                    self._data_lines[line_idx].set_data([], [])
            line_idx += 1

    def _update_lines_decimated(
        self, xd: np.ndarray, yd: np.ndarray, i0: int, i1: int
    ) -> None:
        gap_mask = self.mapping.gap_mask
        gap_indices = np.where(gap_mask)[0]
        boundaries = np.concatenate([[0], gap_indices, [len(self._indices)]])

        line_idx = 0
        for i in range(len(boundaries) - 1):
            seg_start, seg_end = boundaries[i], boundaries[i + 1]
            if seg_end <= seg_start:
                continue
            # Filter decimated points belonging to this segment
            seg_mask = (xd >= seg_start) & (xd < seg_end)
            if line_idx < len(self._data_lines):
                self._data_lines[line_idx].set_data(xd[seg_mask], yd[seg_mask])
            line_idx += 1

    def _on_xlim_changed(self, ax: Axes) -> None:
        self._apply_lod()

    # -- Public helpers -----------------------------------------------------

    def to_index(self, dt) -> int:
        """Convert a datetime to the compressed index space."""
        return self.mapping.to_index(dt)

    def to_datetime(self, idx: int) -> np.datetime64:
        """Convert a compressed index to a datetime."""
        return self.mapping.to_datetime(idx)

    def detach(self) -> None:
        """Remove all tradeaxis artifacts and restore the Axes."""
        # Disconnect xlim callback
        self._ax.callbacks.disconnect(self._xlim_cid)

        # Disconnect interaction events
        if self._interaction is not None:
            self._interaction.detach()

        # Remove separator lines
        for line in self._separator_lines:
            line.remove()
        self._separator_lines.clear()

        # Remove data lines
        for line in self._data_lines:
            line.remove()
        self._data_lines.clear()

        # Restore formatter
        self._ax.xaxis.set_major_formatter(self._original_formatter)

        # Remove GC-prevention ref
        if hasattr(self._ax, "_tradeaxis"):
            del self._ax._tradeaxis


def wrap(
    ax: Axes,
    timestamps: np.ndarray,
    values: np.ndarray,
    *,
    gap: GapSpec = "auto",
    show_separators: bool = True,
    interaction: bool = True,
) -> Chart:
    """Enhance a standard matplotlib Axes with gap-compression, LOD, and
    trader-style interaction.

    Parameters
    ----------
    ax : Axes
        A standard matplotlib Axes to enhance.
    timestamps : array-like[datetime64]
        Sorted timestamps.
    values : array-like[float]
        Y-values (prices, volumes, etc.).
    gap : "auto" | timedelta | callable
        Gap detection strategy.
    show_separators : bool
        Draw dashed vertical lines at session breaks.
    interaction : bool
        Enable pan/zoom/crosshair mouse interaction.

    Returns
    -------
    Chart
        Handle for the enhanced chart. The ``ax`` is still a normal
        matplotlib Axes.
    """
    return Chart(
        ax,
        timestamps,
        values,
        gap=gap,
        show_separators=show_separators,
        interaction=interaction,
    )


def plot_intraday(
    timestamps: np.ndarray,
    values: np.ndarray,
    *,
    gap: GapSpec = "auto",
    show_separators: bool = True,
    interaction: bool = True,
    figsize=(14, 6),
    **kwargs,
) -> Chart:
    """Convenience: create a new figure and return a wrapped Chart."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    return wrap(
        ax,
        timestamps,
        values,
        gap=gap,
        show_separators=show_separators,
        interaction=interaction,
    )
