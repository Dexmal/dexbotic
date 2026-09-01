import numpy as np

from dexbotic.data.utils.normalize import RunningStats


def _assert_histogram_counts_match(stats: RunningStats) -> None:
    for histogram in stats._histograms:
        assert histogram.sum() == stats._count


def test_rebin_preserves_min_bin_counts_and_quantiles() -> None:
    stats = RunningStats()
    stats.update(np.concatenate((np.zeros(100_000), np.ones(1))))

    stats.update(np.array([10.0]))

    _assert_histogram_counts_match(stats)
    result = stats.get_statistics()
    np.testing.assert_array_equal(result.q01, np.array([0.0]))
    np.testing.assert_array_equal(result.q99, np.array([0.0]))


def test_one_dimension_expanding_preserves_counts_in_every_dimension() -> None:
    stats = RunningStats()
    values = np.concatenate((np.zeros(1_000), np.ones(1)))
    stats.update(np.column_stack((values, values)))

    stats.update(np.array([[10.0, 0.5]]))

    _assert_histogram_counts_match(stats)


def test_repeated_min_and_max_expansion_preserves_counts() -> None:
    stats = RunningStats()
    stats.update(np.array([0.0, 0.0, 1.0]))

    for new_values in (
        np.array([2.0]),
        np.array([-3.0]),
        np.array([5.0]),
        np.array([-8.0]),
    ):
        stats.update(new_values)
        _assert_histogram_counts_match(stats)
