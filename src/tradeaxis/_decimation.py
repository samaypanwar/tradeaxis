"""Min/max bucket decimation and LOD rendering with LRU cache."""

from __future__ import annotations

from collections import OrderedDict
from math import ceil
from typing import Tuple

import numpy as np


def decimate_minmax(
    x: np.ndarray,
    y: np.ndarray,
    i0: int,
    i1: int,
    pixel_width: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Downsample ``x[i0:i1], y[i0:i1]`` by keeping the min and max of each
    bucket, interleaved in x-order.

    Returns arrays of length ``<= 2 * pixel_width``.  When ``bucket_size==1``
    the original slice is returned unchanged.
    """
    if i0 >= i1:
        return np.empty(0, dtype=x.dtype), np.empty(0, dtype=y.dtype)

    n = i1 - i0
    bucket_size = max(1, ceil(n / pixel_width))

    if bucket_size <= 1:
        return x[i0:i1].copy(), y[i0:i1].copy()

    xs = x[i0:i1]
    ys = y[i0:i1]

    # Number of full buckets + possible remainder
    n_full = n // bucket_size
    remainder = n - n_full * bucket_size

    out_x = []
    out_y = []

    if n_full > 0:
        # Reshape the full-bucket portion for vectorized min/max
        ys_full = ys[: n_full * bucket_size].reshape(n_full, bucket_size)
        xs_full = xs[: n_full * bucket_size].reshape(n_full, bucket_size)

        idx_min = ys_full.argmin(axis=1)
        idx_max = ys_full.argmax(axis=1)

        bucket_offsets = np.arange(n_full) * bucket_size

        min_positions = bucket_offsets + idx_min
        max_positions = bucket_offsets + idx_max

        for b in range(n_full):
            lo, hi = min_positions[b], max_positions[b]
            if lo <= hi:
                out_x.extend([xs[lo], xs[hi]])
                out_y.extend([ys[lo], ys[hi]])
            else:
                out_x.extend([xs[hi], xs[lo]])
                out_y.extend([ys[hi], ys[lo]])

    if remainder > 0:
        tail_y = ys[n_full * bucket_size :]
        tail_x = xs[n_full * bucket_size :]
        lo = int(tail_y.argmin())
        hi = int(tail_y.argmax())
        if lo <= hi:
            out_x.extend([tail_x[lo], tail_x[hi]])
            out_y.extend([tail_y[lo], tail_y[hi]])
        else:
            out_x.extend([tail_x[hi], tail_x[lo]])
            out_y.extend([tail_y[hi], tail_y[lo]])

    # Deduplicate consecutive identical x values
    result_x = np.array(out_x, dtype=x.dtype)
    result_y = np.array(out_y, dtype=y.dtype)

    # Remove duplicate (x, y) pairs where min==max in a bucket
    keep = np.ones(len(result_x), dtype=bool)
    for k in range(0, len(result_x) - 1, 2):
        if result_x[k] == result_x[k + 1] and result_y[k] == result_y[k + 1]:
            keep[k + 1] = False

    return result_x[keep], result_y[keep]


class LODRenderer:
    """Level-of-detail renderer with an LRU cache for decimation results.

    Parameters
    ----------
    x, y : ndarray
        Full-resolution data arrays (not copied).
    max_cache : int
        Maximum number of cached decimation results.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, max_cache: int = 32) -> None:
        self._x = x
        self._y = y
        self._max_cache = max_cache
        self._cache: OrderedDict[tuple, tuple] = OrderedDict()

    def get_decimated(
        self, i0: int, i1: int, pixel_width: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        key = (i0, i1, pixel_width)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        result = decimate_minmax(self._x, self._y, i0, i1, pixel_width)
        self._cache[key] = result

        if len(self._cache) > self._max_cache:
            self._cache.popitem(last=False)

        return result

    def clear_cache(self) -> None:
        self._cache.clear()
