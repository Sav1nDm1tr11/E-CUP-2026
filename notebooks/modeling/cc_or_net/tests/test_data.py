import numpy as np
import pytest

from src.data import (
    assign_buckets,
    denorm_stats_from_array,
    fit_bucket_denorm_stats,
    fit_bucket_threshold,
    threshold_from_array,
)


# ---------------------------------------------------------------------------
# threshold_from_array
# ---------------------------------------------------------------------------

def test_threshold_from_array_known_median():
    y = np.array([0.0, 0.0, 10.0, 20.0, 30.0, 40.0])
    # положительные: [10, 20, 30, 40] -> медиана 25
    assert threshold_from_array(y) == pytest.approx(25.0)


def test_threshold_from_array_odd_count_of_positives():
    y = np.array([0.0, 5.0, 15.0, 100.0])
    # положительные: [5, 15, 100] -> медиана 15
    assert threshold_from_array(y) == pytest.approx(15.0)


def test_threshold_from_array_raises_without_positives():
    y = np.zeros(5)
    with pytest.raises(ValueError):
        threshold_from_array(y)


# ---------------------------------------------------------------------------
# assign_buckets
# ---------------------------------------------------------------------------

def test_assign_buckets_boundary_cases():
    tau2 = 25.0
    y = np.array([0.0, 1.0, 25.0, 25.001, 1000.0])
    buckets = assign_buckets(y, tau2)
    # y=0 -> bucket 0
    # 0 < y <= tau2 -> bucket 1 (включая ровно tau2)
    # y > tau2 -> bucket 2
    np.testing.assert_array_equal(buckets, [0, 1, 1, 2, 2])


def test_assign_buckets_dtype_is_int64():
    buckets = assign_buckets(np.array([0.0, 10.0]), tau2=5.0)
    assert buckets.dtype == np.int64


# ---------------------------------------------------------------------------
# denorm_stats_from_array
# ---------------------------------------------------------------------------

def test_denorm_stats_from_array_center_is_median():
    log_y = np.log1p(np.array([10.0, 20.0, 30.0, 40.0, 50.0]))
    r_b, c_b = denorm_stats_from_array(log_y)
    assert c_b == pytest.approx(np.median(log_y))
    assert r_b > 0


def test_denorm_stats_from_array_degenerate_bucket_no_nan_or_zero():
    # константный бакет: P10 == P90 == медиана -- без floor'а r_b был бы 0.
    log_y = np.full(10, np.log1p(50.0))
    r_b, c_b = denorm_stats_from_array(log_y)
    assert np.isfinite(r_b)
    assert np.isfinite(c_b)
    assert r_b > 0
    assert r_b == pytest.approx(1e-3)
    assert c_b == pytest.approx(np.log1p(50.0))


def test_denorm_stats_from_array_single_value_bucket():
    log_y = np.array([np.log1p(7.0)])
    r_b, c_b = denorm_stats_from_array(log_y)
    assert np.isfinite(r_b) and r_b > 0
    assert c_b == pytest.approx(np.log1p(7.0))


# ---------------------------------------------------------------------------
# fit_bucket_threshold / fit_bucket_denorm_stats -- сборка поверх нескольких
# cutoff'ов с диска (data_dir/<cutoff>/y.npy), fold-safe чтение только
# переданных cutoff'ов.
# ---------------------------------------------------------------------------

def _write_y(data_dir, cutoff, y):
    cutoff_dir = data_dir / cutoff
    cutoff_dir.mkdir(parents=True, exist_ok=True)
    np.save(cutoff_dir / "y.npy", np.asarray(y, dtype=np.float64))


def test_fit_bucket_threshold_pools_multiple_cutoffs(tmp_path):
    _write_y(tmp_path, "cutoff_a", [0.0, 10.0, 20.0])
    _write_y(tmp_path, "cutoff_b", [0.0, 30.0, 40.0])

    tau2 = fit_bucket_threshold(["cutoff_a", "cutoff_b"], data_dir=tmp_path)

    # положительные, объединённые: [10, 20, 30, 40] -> медиана 25
    assert tau2 == pytest.approx(25.0)


def test_fit_bucket_threshold_ignores_cutoffs_not_passed(tmp_path):
    _write_y(tmp_path, "train_cutoff", [0.0, 10.0, 20.0])
    _write_y(tmp_path, "holdout_cutoff", [0.0, 1000.0, 2000.0])

    tau2 = fit_bucket_threshold(["train_cutoff"], data_dir=tmp_path)

    # holdout_cutoff не передан -- не должен влиять на tau2.
    # положительные в train_cutoff: [10, 20] -> медиана 15.
    assert tau2 == pytest.approx(15.0)


def test_fit_bucket_denorm_stats_returns_only_positive_buckets(tmp_path):
    _write_y(tmp_path, "cutoff_a", [0.0, 0.0, 5.0, 10.0, 50.0, 100.0])

    tau2 = 10.0
    stats = fit_bucket_denorm_stats(["cutoff_a"], tau2, data_dir=tmp_path)

    assert set(stats.keys()) == {1, 2}
    for r_b, c_b in stats.values():
        assert np.isfinite(r_b) and r_b > 0
        assert np.isfinite(c_b)
