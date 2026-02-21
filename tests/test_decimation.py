"""Tests for decimate_minmax and LODRenderer."""

import numpy as np
import pytest

from tradeaxis._decimation import decimate_minmax, LODRenderer


# ---------------------------------------------------------------------------
# M2a: decimate_minmax
# ---------------------------------------------------------------------------


class TestDecimateMinmax:
    def test_1m_sine_output_size(self):
        n = 1_000_000
        x = np.arange(n, dtype=np.float64)
        y = np.sin(np.linspace(0, 100 * np.pi, n))
        xd, yd = decimate_minmax(x, y, 0, n, pixel_width=500)
        assert len(xd) == len(yd)
        assert 0 < len(xd) <= 1000

    def test_preserves_global_extrema(self):
        n = 1_000_000
        x = np.arange(n, dtype=np.float64)
        y = np.sin(np.linspace(0, 100 * np.pi, n))
        xd, yd = decimate_minmax(x, y, 0, n, pixel_width=500)
        assert pytest.approx(yd.min(), abs=1e-3) == y.min()
        assert pytest.approx(yd.max(), abs=1e-3) == y.max()

    def test_bucket_size_one_returns_original(self):
        x = np.arange(100, dtype=np.float64)
        y = np.random.randn(100)
        xd, yd = decimate_minmax(x, y, 0, 100, pixel_width=100)
        # bucket_size=1 → should return original data (or very close)
        np.testing.assert_array_equal(xd, x)
        np.testing.assert_array_equal(yd, y)

    def test_empty_range(self):
        x = np.arange(100, dtype=np.float64)
        y = np.random.randn(100)
        xd, yd = decimate_minmax(x, y, 50, 50, pixel_width=500)
        assert len(xd) == 0
        assert len(yd) == 0

    def test_small_range(self):
        x = np.arange(10, dtype=np.float64)
        y = np.array([5, 3, 8, 1, 9, 2, 7, 4, 6, 0], dtype=np.float64)
        xd, yd = decimate_minmax(x, y, 0, 10, pixel_width=5)
        assert 0 < len(xd) <= 10
        assert 0.0 in yd  # global min
        assert 9.0 in yd  # global max


# ---------------------------------------------------------------------------
# M2b: LODRenderer with cache
# ---------------------------------------------------------------------------


class TestLODRenderer:
    def test_cache_hit_returns_same_object(self):
        x = np.arange(10000, dtype=np.float64)
        y = np.random.randn(10000)
        renderer = LODRenderer(x, y, max_cache=32)
        r1 = renderer.get_decimated(0, 1000, 500)
        r2 = renderer.get_decimated(0, 1000, 500)
        assert r1[0] is r2[0]
        assert r1[1] is r2[1]

    def test_cache_eviction_at_capacity(self):
        x = np.arange(10000, dtype=np.float64)
        y = np.random.randn(10000)
        renderer = LODRenderer(x, y, max_cache=32)
        for i in range(33):
            renderer.get_decimated(0, 1000 + i, 500)
        # First entry should have been evicted
        assert (0, 1000, 500) not in renderer._cache

    def test_clear_cache(self):
        x = np.arange(10000, dtype=np.float64)
        y = np.random.randn(10000)
        renderer = LODRenderer(x, y, max_cache=32)
        renderer.get_decimated(0, 1000, 500)
        assert len(renderer._cache) == 1
        renderer.clear_cache()
        assert len(renderer._cache) == 0
