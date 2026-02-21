# TradeAxis

Matplotlib with market-microstructure sanity.

TradeAxis is a thin wrapper that composes gap-compression, level-of-detail decimation, and trader-style mouse interaction onto a **standard matplotlib Axes**. No subclassing, no new API to learn -- your `ax` stays a normal `ax`.

## Features

- **Gap compression** -- Overnight/weekend gaps are detected automatically and removed. Session breaks are marked with clean dashed separators.
- **LOD decimation** -- Millions of points render instantly. On zoom, the visible data is refined to full resolution. Min/max bucket downsampling ensures spikes are never hidden.
- **Trader-style interaction** -- Left-drag to pan, right-drag to zoom, middle-click for a snap-to-data crosshair with datetime + value annotation. Press Escape to dismiss.

## Installation

```bash
pip install tradeaxis
```

Or with Poetry (from source):

```bash
git clone https://github.com/youruser/tradeaxis.git
cd tradeaxis
poetry install
```

## Quick start

```python
import matplotlib.pyplot as plt
import numpy as np
import tradeaxis as ta

# Generate synthetic multi-day 1-min data
def make_day(date):
    base = np.datetime64(date)
    start = base + np.timedelta64(9 * 60 + 30, "m")
    end = base + np.timedelta64(16 * 60, "m")
    return np.arange(start, end, np.timedelta64(1, "m"))

timestamps = np.concatenate([make_day("2026-01-05"), make_day("2026-01-06")])
prices = 150 + np.cumsum(np.random.randn(len(timestamps)) * 0.02)

# One line to enhance the axes
fig, ax = plt.subplots(figsize=(14, 5))
chart = ta.wrap(ax, timestamps, prices)

# Everything below is normal matplotlib
ax.set_title("AAPL 1-min")
ax.set_ylabel("Price ($)")
ax.axhline(prices.mean(), color="red", ls="--", alpha=0.5, label="Mean")
ax.legend()
plt.show()
```

### Interaction controls

| Action | Effect |
|---|---|
| **Left-drag** | Pan along the x-axis |
| **Right-drag up** | Zoom in (centered at cursor) |
| **Right-drag down** | Zoom out (centered at cursor) |
| **Middle-click** | Snap crosshair to nearest data point, show datetime + value |
| **Escape** | Dismiss crosshair |

## API reference

### `tradeaxis.wrap(ax, timestamps, values, *, gap="auto", show_separators=True, interaction=True)`

Enhance a standard matplotlib `Axes` with gap compression, LOD decimation, and interaction.

**Parameters:**

| Name | Type | Default | Description |
|---|---|---|---|
| `ax` | `matplotlib.axes.Axes` | -- | The axes to enhance. |
| `timestamps` | `array[datetime64]` | -- | Sorted timestamps. |
| `values` | `array[float]` | -- | Y-values (prices, volumes, etc.). |
| `gap` | `"auto"` / `timedelta` / `callable` | `"auto"` | Gap detection strategy. `"auto"` uses median heuristic. A `timedelta` sets a fixed threshold. A callable receives the timestamps array and returns a boolean mask. |
| `show_separators` | `bool` | `True` | Draw dashed vertical lines at session breaks. |
| `interaction` | `bool` | `True` | Enable pan/zoom/crosshair mouse handling. |

**Returns:** `Chart` -- a handle for the enhanced chart. The `ax` is still a normal matplotlib `Axes`.

### `tradeaxis.plot_intraday(timestamps, values, *, gap="auto", show_separators=True, interaction=True, figsize=(14, 6))`

Convenience function that creates a new figure + axes and calls `wrap()`.

### `Chart` methods

| Method | Description |
|---|---|
| `chart.to_index(datetime)` | Convert a datetime to compressed index space (for adding custom annotations). |
| `chart.to_datetime(idx)` | Convert a compressed index back to a datetime. |
| `chart.detach()` | Remove all TradeAxis artifacts and restore the axes to vanilla matplotlib. |

### Working with compressed x-axis

The x-axis uses a compressed integer index, not wall-clock time. This means distances are proportional to number of data points, not elapsed time. When adding custom annotations:

```python
# Convert a datetime to index space, then annotate normally
idx = chart.to_index(np.datetime64("2026-01-05T12:00"))
ax.axvline(x=idx, color="green", ls=":")
```

## How it works

1. **TimeMapping** detects gaps in the timestamp sequence (via median-heuristic or user-specified threshold) and remaps datetimes to a contiguous integer index. A `FuncFormatter` maps these integers back to readable time labels on the x-axis.

2. **LODRenderer** decimates large datasets using min/max bucket downsampling. For each bucket of points, only the minimum and maximum values are kept (interleaved in x-order), preserving spikes. An LRU cache avoids recomputation during repeated pans.

3. **InteractionController** implements a simple state machine (`IDLE -> PAN / ZOOM`) over matplotlib canvas events. The crosshair snaps to the nearest data point via binary search.

All of this is composed onto a user-owned `Axes` via callbacks and artist injection -- no subclassing, no monkey-patching.

## Development

```bash
poetry install
poetry run pytest -v
```

Examples notebook:

```bash
poetry run jupyter notebook examples/showcase.ipynb
```

## License

MIT
