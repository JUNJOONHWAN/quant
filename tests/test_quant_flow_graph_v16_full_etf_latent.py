from __future__ import annotations

import numpy as np

from training.quant_flow_graph_v16.full_etf_latent import (
    LATENT_COMPONENTS,
    STATE_NAMES,
    _date_block_shuffle,
    _masked_rolling_mean,
    _query_features,
)


def test_masked_rolling_mean_keeps_missing_separate_from_true_zero() -> None:
    values = np.asarray([[2.0], [0.0], [0.0], [8.0]], dtype=np.float32)
    observed = np.asarray([[True], [True], [False], [True]])
    result = _masked_rolling_mean(values, observed, 3)
    np.testing.assert_allclose(result[:, 0], [2.0, 1.0, 1.0, 4.0])


def test_query_features_are_stock_conditioned_and_dimension_fixed() -> None:
    scores = np.arange(
        3 * len(STATE_NAMES) * LATENT_COMPONENTS, dtype=np.float32
    ).reshape(3, -1)
    exposures = np.ones((4, LATENT_COMPONENTS), dtype=np.float32)
    exposures[1] *= -1.0
    date_codes = np.asarray([0, 0, 1, 2], dtype=np.int32)
    values, names = _query_features(
        scores=scores, exposures=exposures, date_codes=date_codes
    )
    assert values.shape == (
        4,
        len(STATE_NAMES) * LATENT_COMPONENTS + len(STATE_NAMES) * 3,
    )
    assert len(names) == values.shape[1]
    np.testing.assert_allclose(values[0, :LATENT_COMPONENTS], -values[1, :LATENT_COMPONENTS])


def test_date_shuffle_is_split_local_and_deterministic() -> None:
    scores = np.arange(12, dtype=np.float32).reshape(6, 2)
    train = np.asarray([0, 1, 2], dtype=np.int32)
    test = np.asarray([3, 4, 5], dtype=np.int32)
    first = _date_block_shuffle(scores=scores, train_date_codes=train, test_date_codes=test, seed=7)
    second = _date_block_shuffle(scores=scores, train_date_codes=train, test_date_codes=test, seed=7)
    np.testing.assert_array_equal(first, second)
    assert {tuple(row) for row in first[train]} == {tuple(row) for row in scores[train]}
    assert {tuple(row) for row in first[test]} == {tuple(row) for row in scores[test]}
