"""Preregistered full-ETF identity latent Drift/Diffusion canary.

The v11 stock path compressed indirect ETF Flow into 44 static cluster-family
states before fitting a model.  This canary keeps every strictly eligible ETF
identity until a fold-local, target-free TruncatedSVD is fitted.  A stock query
is then formed by combining the latent all-ETF Flow state with its point-in-time
ETF holdings exposure.  The resulting query features measure propagation,
alignment, convergence, and divergence without date-centering absolute common
Flow.

The primary estimator remains the fixed v12 CatBoost residual adapter so that
the only intended treatment is the representation.  Sealed v12 price-only and
current aggregate-Flow predictions are reused after exact row/date/target
identity checks.  Lagged, ETF-axis-shuffled, date-shuffled, and global-only
variants have identical estimator capacity.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from training.quant_flow_graph_v11_r2.contracts import (
    DEFAULT_SOURCE_DATABASE,
    TIMING_CONTRACT,
)
from training.quant_flow_graph_v11_r2.phase_a import (
    readonly_connection,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from training.quant_flow_graph_v11_r2.phase_b_stock import (
    DEFAULT_GRAPH_DATASET_ROOT,
    DEFAULT_PHASE_A_ROOT,
    OUTER_YEARS,
    PURGE_SESSIONS,
    TARGET_NAMES,
    build_stock_matrix_from_sources,
    fold_indices,
    lag_flow_by_symbol,
    regression_metrics,
    stock_cross_sectional_metrics,
)
from training.quant_flow_graph_v12.residual_canary import (
    CATBOOST_PARAMETERS,
    CATBOOST_VERSION,
    capped_residual_prediction,
    date_balanced_weights,
    fit_predict_multioutput,
    residual_caps,
)


SCHEMA_VERSION = "quant.etf_flow_v16.full_etf_identity_latent.v1"
PREREGISTRATION_SCHEMA_VERSION = (
    "quant.etf_flow_v16.full_etf_identity_latent_preregistration.v1"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/flow_graph_v16/"
    "full_etf_identity_latent_walk_forward"
)
DEFAULT_V12_ROOT = Path(
    "/home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/flow_graph_v12/"
    "catboost_residual_canary"
)

RANDOM_SEED = 20260829
LATENT_COMPONENTS = 32
SVD_ITERATIONS = 7
RATE_CLIP_PCT = 25.0
ROLLING_WINDOWS = (5, 20)
STATE_NAMES = ("current", "mean5", "mean20", "innovation", "convergence")

PRICE_MODEL = "price_only"
V12_CURRENT_MODEL = "v12_current_flow"
RAW_PRIMARY_MODEL = "full_etf_query_raw"
PRIMARY_MODEL = "full_etf_query"
GLOBAL_MODEL = "full_etf_global_only"
LAG5_MODEL = "full_etf_query_lag5"
AXIS_SHUFFLE_MODEL = "full_etf_axis_shuffle"
DATE_SHUFFLE_MODEL = "full_etf_date_shuffle"
MODEL_NAMES = (
    PRICE_MODEL,
    V12_CURRENT_MODEL,
    RAW_PRIMARY_MODEL,
    PRIMARY_MODEL,
    GLOBAL_MODEL,
    LAG5_MODEL,
    AXIS_SHUFFLE_MODEL,
    DATE_SHUFFLE_MODEL,
)

REFERENCE_PAPERS = (
    {
        "title": "Deep Sets",
        "url": "https://papers.nips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html",
        "implication": "ETF inputs are a set; ticker row order must not carry signal.",
    },
    {
        "title": "Set Transformer",
        "url": "https://proceedings.mlr.press/v97/lee19d.html",
        "implication": "Inducing-point attention is the next neural stage only after an identity-information gate.",
    },
    {
        "title": "Diffusion Convolutional Recurrent Neural Network",
        "url": "https://openreview.net/forum?id=SJiHXGWAZ",
        "implication": "Diffusion is graph propagation, not generative denoising.",
    },
    {
        "title": "A Flow-Based Explanation for Return Predictability",
        "url": "https://academic.oup.com/rfs/article-abstract/25/12/3457/1594242",
        "implication": "Aggregate flow-induced demand through holdings and test temporary pressure separately from price-only forecasts.",
    },
)


def _progress(payload: Mapping[str, Any]) -> None:
    print(json.dumps(dict(payload), sort_keys=True), flush=True)


def _array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.view(np.uint8))
    return digest.hexdigest()


def _write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _masked_rolling_mean(
    values: np.ndarray, observed: np.ndarray, window: int
) -> np.ndarray:
    """Causal rolling mean where missing is not silently converted to zero."""

    value64 = np.asarray(values, dtype=np.float64)
    mask64 = np.asarray(observed, dtype=np.float64)
    value_prefix = np.vstack(
        [np.zeros((1, value64.shape[1]), dtype=np.float64), np.cumsum(value64, axis=0)]
    )
    mask_prefix = np.vstack(
        [np.zeros((1, mask64.shape[1]), dtype=np.float64), np.cumsum(mask64, axis=0)]
    )
    end = np.arange(1, len(value64) + 1, dtype=np.int64)
    start = np.maximum(end - int(window), 0)
    numerator = value_prefix[end] - value_prefix[start]
    denominator = mask_prefix[end] - mask_prefix[start]
    result = np.zeros_like(numerator, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result.astype(np.float32)


def _row_scale(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Keep breadth comparable while leaving absolute market Flow in v12 fields."""

    counts = np.sum(observed, axis=1, dtype=np.float64)
    scale = np.sqrt(np.maximum(counts, 1.0))
    return (np.asarray(values, dtype=np.float64) / scale[:, None]).astype(np.float32)


def build_full_etf_panel(
    *,
    event: Any,
    date_values: Sequence[str],
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Build two typed ETF-identity channels without stock/date sampling."""

    dates = tuple(str(value) for value in date_values)
    date_to_code = {value: index for index, value in enumerate(dates)}
    first_date, last_date = dates[0], dates[-1]
    tickers = tuple(
        str(row[0])
        for row in event.execute(
            """
            SELECT DISTINCT ticker
            FROM etf_flow_events
            WHERE signal_date BETWEEN ? AND ? AND strict_eligible=1
            ORDER BY ticker
            """,
            (first_date, last_date),
        )
    )
    ticker_to_code = {ticker: index for index, ticker in enumerate(tickers)}
    date_count = len(dates)
    etf_count = len(tickers)
    clean = np.zeros((date_count, etf_count), dtype=np.float32)
    special = np.zeros_like(clean)
    clean_observed = np.zeros((date_count, etf_count), dtype=bool)
    special_observed = np.zeros_like(clean_observed)
    counters = defaultdict(int)

    query = """
        SELECT signal_date,ticker,clean_eligible,special_eligible,
               observed_exact_t2,true_zero,missing_exact_t2,stale_visible_state,
               flow_rate_pct,effective_sign,target_multiple
        FROM etf_flow_events
        WHERE signal_date BETWEEN ? AND ? AND strict_eligible=1
        ORDER BY signal_date,ticker
    """
    for row_number, row in enumerate(event.execute(query, (first_date, last_date)), 1):
        date_code = date_to_code.get(str(row[0]))
        ticker_code = ticker_to_code.get(str(row[1]))
        if date_code is None or ticker_code is None:
            continue
        counters["strict_eligible_rows"] += 1
        exact = bool(row[4]) or bool(row[5])
        if bool(row[6]):
            counters["missing_rows"] += 1
        if bool(row[7]):
            counters["stale_rows"] += 1
        if bool(row[5]):
            counters["true_zero_rows"] += 1
        if not exact:
            continue
        value = 0.0 if row[8] is None else float(row[8])
        if bool(row[2]):
            clean[date_code, ticker_code] = float(
                np.clip(value, -RATE_CLIP_PCT, RATE_CLIP_PCT)
            )
            clean_observed[date_code, ticker_code] = True
            counters["clean_observed_rows"] += 1
        if bool(row[3]):
            effective = value * float(row[9]) * abs(float(row[10]))
            special[date_code, ticker_code] = float(
                np.clip(effective, -2.0 * RATE_CLIP_PCT, 2.0 * RATE_CLIP_PCT)
            )
            special_observed[date_code, ticker_code] = True
            counters["special_observed_rows"] += 1
        if progress and row_number % 1_000_000 == 0:
            progress(
                {
                    "stage": "v16_full_etf_panel",
                    "rows_read": row_number,
                    "signal_date": row[0],
                    "at_utc": utc_now(),
                }
            )

    raw = np.column_stack([clean, special]).astype(np.float32)
    observed = np.column_stack([clean_observed, special_observed])
    mean5_raw = _masked_rolling_mean(raw, observed, ROLLING_WINDOWS[0])
    mean20_raw = _masked_rolling_mean(raw, observed, ROLLING_WINDOWS[1])
    observed5 = _masked_rolling_mean(observed.astype(np.float32), observed, 5) > 0
    observed20 = _masked_rolling_mean(observed.astype(np.float32), observed, 20) > 0
    current = _row_scale(raw, observed)
    mean5 = _row_scale(mean5_raw, observed5)
    mean20 = _row_scale(mean20_raw, observed20)
    return {
        "dates": dates,
        "tickers": tickers,
        "ticker_to_code": ticker_to_code,
        "current": current,
        "mean5": mean5,
        "mean20": mean20,
        "observed": observed,
        "audit": {
            **dict(counters),
            "signal_date_count": date_count,
            "etf_identity_count": etf_count,
            "typed_column_count": 2 * etf_count,
            "clean_channel": "strict clean ETF flow_rate_pct",
            "special_channel": "typed effective_sign*abs(target_multiple)*flow_rate_pct",
            "small_etfs_removed_from_denominator": True,
            "absolute_common_flow_date_centered": False,
            "selection_biased_table_48_used": False,
        },
    }


def fit_fold_latent_state(
    *, panel: Mapping[str, Any], train_date_codes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit target-free SVD on outer-train dates and project five causal states."""

    try:
        from sklearn.decomposition import TruncatedSVD
    except ImportError as exc:  # pragma: no cover - environment gate
        raise RuntimeError("scikit-learn is required in the isolated container") from exc

    train_dates = np.unique(np.asarray(train_date_codes, dtype=np.int64))
    current = np.asarray(panel["current"], dtype=np.float32)
    active = np.any(np.abs(current[train_dates]) > 0, axis=0)
    active_count = int(np.sum(active))
    components = min(LATENT_COMPONENTS, len(train_dates) - 1, active_count - 1)
    if components != LATENT_COMPONENTS:
        raise ValueError(
            f"fold cannot support fixed {LATENT_COMPONENTS} components: {components}"
        )
    model = TruncatedSVD(
        n_components=LATENT_COMPONENTS,
        algorithm="randomized",
        n_iter=SVD_ITERATIONS,
        random_state=RANDOM_SEED,
    )
    model.fit(current[train_dates][:, active])
    full_components = np.zeros(
        (LATENT_COMPONENTS, current.shape[1]), dtype=np.float32
    )
    full_components[:, active] = np.asarray(model.components_, dtype=np.float32)

    state_values = {
        "current": current,
        "mean5": np.asarray(panel["mean5"], dtype=np.float32),
        "mean20": np.asarray(panel["mean20"], dtype=np.float32),
    }
    state_values["innovation"] = state_values["current"] - state_values["mean5"]
    state_values["convergence"] = state_values["mean5"] - state_values["mean20"]
    scores = np.column_stack(
        [state_values[name] @ full_components.T for name in STATE_NAMES]
    ).astype(np.float32)

    etf_count = len(panel["tickers"])
    clean_loading = full_components[:, :etf_count].T
    special_loading = full_components[:, etf_count:].T
    identity_loading = ((clean_loading + special_loading) / math.sqrt(2.0)).astype(
        np.float32
    )
    diagnostics = {
        "component_count": LATENT_COMPONENTS,
        "active_typed_column_count": active_count,
        "outer_train_date_count": int(len(train_dates)),
        "explained_variance_ratio_sum": float(
            np.sum(model.explained_variance_ratio_)
        ),
        "singular_values": [float(value) for value in model.singular_values_],
        "state_names": list(STATE_NAMES),
        "factor_state_sha256": _array_sha256(scores),
        "identity_loading_sha256": _array_sha256(identity_loading),
    }
    return scores, identity_loading, diagnostics


def _graph_exposures(
    *,
    matrix: Mapping[str, Any],
    graph_manifest: Mapping[str, Any],
    ticker_to_code: Mapping[str, int],
    identity_loading: np.ndarray,
    shuffled_loading: np.ndarray,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Map fold-local ETF factors through every PIT ETF-stock edge."""

    try:
        from scipy.sparse import csr_matrix
    except ImportError as exc:  # pragma: no cover - environment gate
        raise RuntimeError("SciPy is required in the isolated container") from exc

    rows = len(matrix["targets"])
    components = identity_loading.shape[1]
    actual = np.zeros((rows, components), dtype=np.float32)
    shuffled = np.zeros_like(actual)
    coverage = np.zeros(rows, dtype=np.float32)
    snapshots = {str(item["signal_date"]): item for item in graph_manifest["snapshots"]}
    vocabulary = tuple(str(value) for value in graph_manifest["etf_vocabulary"])
    symbol_values = tuple(str(value) for value in matrix["symbol_values"])
    counters = defaultdict(int)

    for position, signal_date in enumerate(matrix["date_values"], 1):
        ref = snapshots.get(str(signal_date))
        if ref is None:
            raise ValueError(f"missing graph snapshot for {signal_date}")
        with np.load(ref["path"], allow_pickle=False) as item:
            stock_symbols = tuple(str(value) for value in item["stock_symbols"])
            targets = np.asarray(item["targets"], dtype=np.float32)
            target_mask = np.asarray(item["target_mask"], dtype=bool)
            local_global_etf = np.asarray(item["etf_ids"], dtype=np.int64)
            edge_index = np.asarray(item["edge_index"], dtype=np.int64)
            edge_attr = np.asarray(item["edge_attr"], dtype=np.float32)
        complete = np.all(target_mask, axis=1) & np.all(np.isfinite(targets), axis=1)
        row_indices = np.flatnonzero(matrix["date_codes"] == position - 1)
        expected_symbols = tuple(
            symbol_values[index] for index in matrix["symbol_codes"][row_indices]
        )
        observed_symbols = tuple(np.asarray(stock_symbols, dtype=object)[complete])
        if expected_symbols != observed_symbols:
            raise ValueError(f"stock row alignment mismatch on {signal_date}")

        local_tickers = tuple(vocabulary[index] for index in local_global_etf)
        local_codes = np.asarray(
            [ticker_to_code.get(ticker, -1) for ticker in local_tickers],
            dtype=np.int64,
        )
        local_actual = np.zeros((len(local_tickers), components), dtype=np.float32)
        local_shuffled = np.zeros_like(local_actual)
        mapped = local_codes >= 0
        local_actual[mapped] = identity_loading[local_codes[mapped]]
        local_shuffled[mapped] = shuffled_loading[local_codes[mapped]]
        weights = np.asarray(edge_attr[:, 0], dtype=np.float32)
        graph = csr_matrix(
            (weights, (edge_index[0], edge_index[1])),
            shape=(len(stock_symbols), len(local_tickers)),
            dtype=np.float32,
        )
        total_mass = np.asarray(graph.sum(axis=1)).ravel().astype(np.float32)
        mapped_mass = np.asarray(graph[:, mapped].sum(axis=1)).ravel().astype(np.float32)
        denominator = np.maximum(total_mass, 1e-6)
        actual_local = np.asarray(graph @ local_actual, dtype=np.float32)
        shuffled_local = np.asarray(graph @ local_shuffled, dtype=np.float32)
        actual_local /= denominator[:, None]
        shuffled_local /= denominator[:, None]
        actual[row_indices] = actual_local[complete]
        shuffled[row_indices] = shuffled_local[complete]
        coverage[row_indices] = (mapped_mass / denominator)[complete]
        counters["edge_count"] += int(len(weights))
        counters["mapped_edge_count"] += int(np.sum(mapped[edge_index[1]]))
        counters["snapshot_count"] += 1
        if progress and (position == 1 or position % 100 == 0 or position == len(matrix["date_values"])):
            progress(
                {
                    "stage": "v16_graph_exposure",
                    "completed_snapshots": position,
                    "total_snapshots": len(matrix["date_values"]),
                    "signal_date": signal_date,
                    "at_utc": utc_now(),
                }
            )
    audit = {
        **dict(counters),
        "mean_weight_coverage": float(np.mean(coverage)),
        "minimum_weight_coverage": float(np.min(coverage)),
        "complete_weight_coverage_ratio": float(np.mean(coverage >= 0.999999)),
        "actual_exposure_sha256": _array_sha256(actual),
        "shuffled_exposure_sha256": _array_sha256(shuffled),
    }
    return actual, shuffled, coverage, audit


def _factor_names(prefix: str) -> tuple[str, ...]:
    return tuple(
        f"{prefix}::{state}::{component:02d}"
        for state in STATE_NAMES
        for component in range(LATENT_COMPONENTS)
    )


def _query_features(
    *, scores: np.ndarray, exposures: np.ndarray, date_codes: np.ndarray
) -> tuple[np.ndarray, tuple[str, ...]]:
    row_scores = np.asarray(scores, dtype=np.float32)[date_codes]
    blocks = []
    scalars = []
    scalar_names = []
    for state_index, state_name in enumerate(STATE_NAMES):
        start = state_index * LATENT_COMPONENTS
        end = start + LATENT_COMPONENTS
        state = row_scores[:, start:end]
        product = state * exposures
        blocks.append(product)
        dot = np.sum(product, axis=1)
        norm = np.linalg.norm(state, axis=1) * np.linalg.norm(exposures, axis=1)
        cosine = np.divide(
            dot,
            norm,
            out=np.zeros_like(dot, dtype=np.float32),
            where=norm > 1e-9,
        )
        sign_agreement = np.mean(
            np.sign(state) == np.sign(exposures), axis=1, dtype=np.float32
        )
        scalars.extend([dot, cosine, sign_agreement])
        scalar_names.extend(
            [
                f"relation::{state_name}::dot",
                f"relation::{state_name}::cosine",
                f"relation::{state_name}::sign_agreement",
            ]
        )
    result = np.column_stack([*blocks, *scalars]).astype(np.float32)
    names = _factor_names("alignment") + tuple(scalar_names)
    return result, names


def _date_block_shuffle(
    *, scores: np.ndarray, train_date_codes: np.ndarray, test_date_codes: np.ndarray, seed: int
) -> np.ndarray:
    result = np.asarray(scores, dtype=np.float32).copy()
    rng = np.random.default_rng(seed)
    for codes in (np.unique(train_date_codes), np.unique(test_date_codes)):
        if len(codes) > 1:
            result[codes] = scores[rng.permutation(codes)]
    return result


def _load_v12_predictions(
    *, v12_root: Path, year: int, actual: np.ndarray, date_codes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    npz_path = v12_root / f"fold_{year}.npz"
    metadata_path = v12_root / f"fold_{year}.json"
    if not npz_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"sealed v12 fold is missing for {year}")
    with np.load(npz_path, allow_pickle=False) as item:
        sealed_actual = np.asarray(item["actual"], dtype=np.float32)
        sealed_dates = np.asarray(item["date_codes"], dtype=np.int32)
        price = np.asarray(item["price_only"], dtype=np.float32)
        current = np.asarray(item["capped_flow_residual"], dtype=np.float32)
    if not np.array_equal(sealed_actual, actual):
        raise ValueError(f"v12 target identity mismatch for {year}")
    if not np.array_equal(sealed_dates, date_codes):
        raise ValueError(f"v12 date identity mismatch for {year}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return price, current, {
        "npz_path": str(npz_path),
        "npz_sha256": sha256_file(npz_path),
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "outer_year": int(metadata["outer_year"]),
    }


def _model_metrics(
    *, actual: np.ndarray, predictions: Mapping[str, np.ndarray], date_codes: np.ndarray
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for target_index, target_name in enumerate(TARGET_NAMES):
        models: dict[str, Any] = {}
        for model_name, prediction in predictions.items():
            metrics = regression_metrics(actual[:, target_index], prediction[:, target_index])
            metrics.update(
                stock_cross_sectional_metrics(
                    date_codes=date_codes,
                    target=actual[:, target_index],
                    prediction=prediction[:, target_index],
                    loss_target=target_name.startswith("loss_"),
                )
            )
            models[model_name] = metrics
        result[target_name] = models
    return result


def _fold_checkpoint(
    *, output_root: Path, year: int, preregistration_sha256: str
) -> tuple[dict[str, np.ndarray], dict[str, Any]] | None:
    npz_path = output_root / f"fold_{year}.npz"
    json_path = output_root / f"fold_{year}.json"
    if not npz_path.exists() or not json_path.exists():
        return None
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    if metadata.get("preregistration_sha256") != preregistration_sha256:
        raise ValueError(f"fold {year} preregistration mismatch")
    with np.load(npz_path, allow_pickle=False) as item:
        arrays = {name: np.asarray(item[name]) for name in item.files}
    expected = {"actual", "date_codes", *MODEL_NAMES}
    if set(arrays) != expected:
        raise ValueError(f"fold {year} checkpoint keys mismatch")
    if sha256_file(npz_path) != metadata.get("prediction_sha256"):
        raise ValueError(f"fold {year} checkpoint hash mismatch")
    return arrays, metadata


def _save_fold_checkpoint(
    *,
    output_root: Path,
    year: int,
    preregistration_sha256: str,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    npz_path = output_root / f"fold_{year}.npz"
    json_path = output_root / f"fold_{year}.json"
    _write_npz_atomic(npz_path, arrays)
    result = {
        **dict(metadata),
        "preregistration_sha256": preregistration_sha256,
        "prediction_path": str(npz_path),
        "prediction_sha256": sha256_file(npz_path),
    }
    write_json_atomic(json_path, result)
    return result


def _fit_variant(
    *,
    price: np.ndarray,
    base_flow: np.ndarray,
    extra: np.ndarray,
    targets: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    weights: np.ndarray,
    names: Sequence[str],
    thread_count: int,
) -> tuple[np.ndarray, list[dict[str, Any]], float]:
    features = np.column_stack([price, base_flow, extra]).astype(np.float32)
    prediction, top, elapsed = fit_predict_multioutput(
        features=features,
        targets=targets,
        train=train,
        test=test,
        weights=weights,
        feature_names=names,
        thread_count=thread_count,
    )
    del features
    gc.collect()
    return prediction, top, elapsed


def evaluate(
    *,
    matrix: Mapping[str, Any],
    panel: Mapping[str, Any],
    graph_manifest: Mapping[str, Any],
    v12_root: Path,
    output_root: Path,
    preregistration_sha256: str,
    thread_count: int,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    price = np.asarray(matrix["price_matrix"], dtype=np.float32)
    base_flow = np.asarray(matrix["flow_matrix"], dtype=np.float32)
    targets = np.asarray(matrix["targets"], dtype=np.float32)
    price_names = tuple(f"price::{name}" for name in matrix["price_names"])
    base_names = tuple(f"v12::{name}" for name in matrix["flow_names"])
    global_names = _factor_names("global")
    query_names_template = _factor_names("alignment") + tuple(
        f"relation::{state}::{kind}"
        for state in STATE_NAMES
        for kind in ("dot", "cosine", "sign_agreement")
    )
    prediction_parts: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    actual_parts: list[np.ndarray] = []
    date_parts: list[np.ndarray] = []
    folds: list[dict[str, Any]] = []

    for year in OUTER_YEARS:
        train, test = fold_indices(matrix, year)
        if len(train) < 50_000 or len(test) < 10_000:
            continue
        checkpoint = _fold_checkpoint(
            output_root=output_root,
            year=year,
            preregistration_sha256=preregistration_sha256,
        )
        if checkpoint is not None:
            arrays, metadata = checkpoint
            if not np.array_equal(arrays["actual"], targets[test]):
                raise ValueError(f"fold {year} target mismatch on resume")
            if not np.array_equal(arrays["date_codes"], matrix["date_codes"][test]):
                raise ValueError(f"fold {year} date mismatch on resume")
            predictions = {name: arrays[name] for name in MODEL_NAMES}
            metadata = {**metadata, "resumed": True}
            if progress:
                progress({"stage": "v16_fold_resumed", "outer_year": year, "at_utc": utc_now()})
        else:
            started = time.monotonic()
            train_date_codes = matrix["date_codes"][train]
            test_date_codes = matrix["date_codes"][test]
            scores, loading, latent_diagnostics = fit_fold_latent_state(
                panel=panel, train_date_codes=train_date_codes
            )
            rng = np.random.default_rng(RANDOM_SEED + int(year))
            shuffled_loading = loading[rng.permutation(len(loading))]
            exposure, shuffled_exposure, coverage, graph_audit = _graph_exposures(
                matrix=matrix,
                graph_manifest=graph_manifest,
                ticker_to_code=panel["ticker_to_code"],
                identity_loading=loading,
                shuffled_loading=shuffled_loading,
                progress=progress,
            )
            global_rows = scores[matrix["date_codes"]]
            primary_query, query_names = _query_features(
                scores=scores,
                exposures=exposure,
                date_codes=matrix["date_codes"],
            )
            if query_names != query_names_template:
                raise ValueError("query feature-name contract mismatch")
            axis_query, _ = _query_features(
                scores=scores,
                exposures=shuffled_exposure,
                date_codes=matrix["date_codes"],
            )
            date_shuffled_scores = _date_block_shuffle(
                scores=scores,
                train_date_codes=train_date_codes,
                test_date_codes=test_date_codes,
                seed=RANDOM_SEED + 10_000 + int(year),
            )
            date_query, _ = _query_features(
                scores=date_shuffled_scores,
                exposures=exposure,
                date_codes=matrix["date_codes"],
            )
            date_global_rows = date_shuffled_scores[matrix["date_codes"]]
            full_new = np.column_stack([global_rows, primary_query]).astype(np.float32)
            axis_new = np.column_stack([global_rows, axis_query]).astype(np.float32)
            date_new = np.column_stack([date_global_rows, date_query]).astype(np.float32)
            lag_new = lag_flow_by_symbol(
                full_new,
                matrix["date_codes"],
                matrix["symbol_codes"],
                len(matrix["date_values"]),
                len(matrix["symbol_values"]),
                5,
            )
            weights = date_balanced_weights(matrix["date_codes"], train)
            price_prediction, current_prediction, v12_source = _load_v12_predictions(
                v12_root=v12_root,
                year=year,
                actual=targets[test],
                date_codes=matrix["date_codes"][test],
            )
            caps = residual_caps(targets[train])
            raw_predictions: dict[str, np.ndarray] = {}
            top_features: dict[str, list[dict[str, Any]]] = {}
            fit_seconds: dict[str, float] = {}
            variants = (
                (GLOBAL_MODEL, global_rows, global_names),
                (PRIMARY_MODEL, full_new, global_names + query_names),
                (LAG5_MODEL, lag_new, global_names + query_names),
                (AXIS_SHUFFLE_MODEL, axis_new, global_names + query_names),
                (DATE_SHUFFLE_MODEL, date_new, global_names + query_names),
            )
            feature_names_prefix = price_names + base_names
            for model_name, extra, extra_names in variants:
                raw, top, elapsed = _fit_variant(
                    price=price,
                    base_flow=base_flow,
                    extra=extra,
                    targets=targets,
                    train=train,
                    test=test,
                    weights=weights,
                    names=feature_names_prefix + tuple(extra_names),
                    thread_count=thread_count,
                )
                raw_predictions[model_name] = raw
                top_features[model_name] = top
                fit_seconds[model_name] = elapsed
                if progress:
                    progress(
                        {
                            "stage": "v16_model_fit",
                            "outer_year": year,
                            "model": model_name,
                            "fit_seconds": elapsed,
                            "at_utc": utc_now(),
                        }
                    )
            predictions = {
                PRICE_MODEL: price_prediction,
                V12_CURRENT_MODEL: current_prediction,
                RAW_PRIMARY_MODEL: raw_predictions[PRIMARY_MODEL],
                PRIMARY_MODEL: capped_residual_prediction(
                    price_prediction, raw_predictions[PRIMARY_MODEL], caps
                ),
                GLOBAL_MODEL: capped_residual_prediction(
                    price_prediction, raw_predictions[GLOBAL_MODEL], caps
                ),
                LAG5_MODEL: capped_residual_prediction(
                    price_prediction, raw_predictions[LAG5_MODEL], caps
                ),
                AXIS_SHUFFLE_MODEL: capped_residual_prediction(
                    price_prediction, raw_predictions[AXIS_SHUFFLE_MODEL], caps
                ),
                DATE_SHUFFLE_MODEL: capped_residual_prediction(
                    price_prediction, raw_predictions[DATE_SHUFFLE_MODEL], caps
                ),
            }
            metadata = {
                "outer_year": int(year),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "train_date_count": int(len(np.unique(train_date_codes))),
                "test_date_count": int(len(np.unique(test_date_codes))),
                "train_end_signal_date": matrix["date_values"][int(np.max(train_date_codes))],
                "test_start_signal_date": matrix["date_values"][int(np.min(test_date_codes))],
                "test_end_signal_date": matrix["date_values"][int(np.max(test_date_codes))],
                "latent": latent_diagnostics,
                "graph_exposure": graph_audit,
                "mean_row_etf_weight_coverage": float(np.mean(coverage)),
                "v12_source": v12_source,
                "fit_seconds": fit_seconds,
                "total_fold_seconds": time.monotonic() - started,
                "top_features": top_features,
                "target_metrics": _model_metrics(
                    actual=targets[test],
                    predictions=predictions,
                    date_codes=matrix["date_codes"][test],
                ),
                "resumed": False,
            }
            arrays = {
                "actual": targets[test],
                "date_codes": matrix["date_codes"][test],
                **predictions,
            }
            metadata = _save_fold_checkpoint(
                output_root=output_root,
                year=year,
                preregistration_sha256=preregistration_sha256,
                arrays=arrays,
                metadata=metadata,
            )
            del scores, loading, shuffled_loading, exposure, shuffled_exposure
            del global_rows, primary_query, axis_query, date_query
            del full_new, axis_new, date_new, lag_new, raw_predictions
            gc.collect()

        actual_parts.append(targets[test])
        date_parts.append(matrix["date_codes"][test])
        for model_name, prediction in predictions.items():
            prediction_parts[model_name].append(prediction)
        folds.append(metadata)
        write_json_atomic(
            output_root / "run_state.json",
            {
                "status": "RUNNING",
                "stage": "outer_folds",
                "completed_outer_years": [int(item["outer_year"]) for item in folds],
                "updated_at_utc": utc_now(),
            },
        )
        if progress:
            progress({"stage": "v16_outer_fold_complete", "outer_year": year, "at_utc": utc_now()})

    actual_all = np.concatenate(actual_parts)
    date_all = np.concatenate(date_parts)
    prediction_all = {name: np.concatenate(parts) for name, parts in prediction_parts.items()}
    targets_receipt: dict[str, Any] = {}
    for target_index, target_name in enumerate(TARGET_NAMES):
        folds_for_target = []
        for fold in folds:
            folds_for_target.append(
                {
                    "outer_year": int(fold["outer_year"]),
                    "train_rows": int(fold["train_rows"]),
                    "test_rows": int(fold["test_rows"]),
                    "test_start_signal_date": fold["test_start_signal_date"],
                    "test_end_signal_date": fold["test_end_signal_date"],
                    "models": fold["target_metrics"][target_name],
                }
            )
        pooled = {}
        for model_name, prediction in prediction_all.items():
            metrics = regression_metrics(actual_all[:, target_index], prediction[:, target_index])
            metrics.update(
                stock_cross_sectional_metrics(
                    date_codes=date_all,
                    target=actual_all[:, target_index],
                    prediction=prediction[:, target_index],
                    loss_target=target_name.startswith("loss_"),
                )
            )
            pooled[model_name] = metrics
        pooled[PRIMARY_MODEL]["relative_mae_improvement_vs_price_pct"] = (
            (pooled[PRICE_MODEL]["mae"] - pooled[PRIMARY_MODEL]["mae"])
            / pooled[PRICE_MODEL]["mae"]
            * 100.0
        )
        pooled[PRIMARY_MODEL]["relative_mae_improvement_vs_v12_current_pct"] = (
            (pooled[V12_CURRENT_MODEL]["mae"] - pooled[PRIMARY_MODEL]["mae"])
            / pooled[V12_CURRENT_MODEL]["mae"]
            * 100.0
        )
        targets_receipt[target_name] = {
            "folds": folds_for_target,
            "pooled": pooled,
            "rows": int(len(actual_all)),
        }
    return targets_receipt, folds


def summarize_gate(targets: Mapping[str, Any]) -> dict[str, Any]:
    counters = defaultdict(int)
    improvements = []
    improvements_price = []
    positive_fold_targets = 0
    outer_fold_targets = 0
    yearly: defaultdict[int, list[float]] = defaultdict(list)
    core_names = {
        "loss_5d_pct",
        "loss_20d_pct",
        "benchmark_downside_defense_5d_pct",
        "benchmark_downside_defense_20d_pct",
    }
    core = defaultdict(int)
    for target_name, target in targets.items():
        pooled = target["pooled"]
        primary = pooled[PRIMARY_MODEL]
        v12 = pooled[V12_CURRENT_MODEL]
        price = pooled[PRICE_MODEL]
        controls = {
            "global": pooled[GLOBAL_MODEL],
            "lag5": pooled[LAG5_MODEL],
            "axis_shuffle": pooled[AXIS_SHUFFLE_MODEL],
            "date_shuffle": pooled[DATE_SHUFFLE_MODEL],
        }
        improvement = (v12["mae"] - primary["mae"]) / v12["mae"] * 100.0
        improvements.append(improvement)
        improvements_price.append((price["mae"] - primary["mae"]) / price["mae"] * 100.0)
        counters["mae_beats_price"] += primary["mae"] < price["mae"]
        counters["mae_beats_v12_current"] += primary["mae"] < v12["mae"]
        for name, control in controls.items():
            counters[f"mae_beats_{name}"] += primary["mae"] < control["mae"]
        counters["rank_ic_beats_v12_current"] += (
            primary["mean_daily_rank_ic"] > v12["mean_daily_rank_ic"]
        )
        counters["economic_basket_beats_v12_current"] += (
            primary["economic_basket_value"] > v12["economic_basket_value"]
        )
        if target_name in core_names:
            core["mae_beats_v12_current"] += primary["mae"] < v12["mae"]
            core["mae_beats_axis_shuffle"] += primary["mae"] < controls["axis_shuffle"]["mae"]
            core["rank_ic_beats_v12_current"] += (
                primary["mean_daily_rank_ic"] > v12["mean_daily_rank_ic"]
            )
            core["economic_basket_beats_v12_current"] += (
                primary["economic_basket_value"] > v12["economic_basket_value"]
            )
        for fold in target["folds"]:
            outer_fold_targets += 1
            primary_mae = fold["models"][PRIMARY_MODEL]["mae"]
            v12_mae = fold["models"][V12_CURRENT_MODEL]["mae"]
            positive_fold_targets += primary_mae < v12_mae
            yearly[int(fold["outer_year"])].append(
                (v12_mae - primary_mae) / v12_mae * 100.0
            )

    mean_improvement = float(np.mean(improvements))
    worst_improvement = float(np.min(improvements))
    yearly_mean = {str(year): float(np.mean(values)) for year, values in yearly.items()}
    forecast = (
        counters["mae_beats_price"] >= 8
        and counters["mae_beats_v12_current"] >= 8
        and counters["mae_beats_global"] >= 8
        and counters["mae_beats_lag5"] >= 8
        and counters["mae_beats_axis_shuffle"] >= 8
        and counters["mae_beats_date_shuffle"] >= 8
        and mean_improvement > 0
        and worst_improvement >= -0.5
        and positive_fold_targets >= outer_fold_targets / 2
        and yearly_mean.get("2025", -math.inf) >= 0
        and yearly_mean.get("2026", -math.inf) >= 0
    )
    basket = (
        counters["rank_ic_beats_v12_current"] >= 8
        and counters["economic_basket_beats_v12_current"] >= 8
        and counters["mae_beats_axis_shuffle"] >= 8
        and counters["mae_beats_date_shuffle"] >= 8
    )
    avoidance = (
        core["mae_beats_v12_current"] >= 3
        and core["mae_beats_axis_shuffle"] >= 3
        and core["rank_ic_beats_v12_current"] >= 3
        and core["economic_basket_beats_v12_current"] >= 3
    )
    passed_paths = [name for name, passed in (("FORECAST", forecast), ("BASKET", basket), ("AVOIDANCE", avoidance)) if passed]
    return {
        "status": "V16_FULL_ETF_IDENTITY_PASS" if passed_paths else "V16_FULL_ETF_IDENTITY_FAIL",
        "passed_paths": passed_paths,
        "fixed_before_results": True,
        "historical_oos_not_clean_forward_lockbox": True,
        "checks": {
            "forecast_path_pass": forecast,
            "basket_path_pass": basket,
            "avoidance_path_pass": avoidance,
            "mae_beats_price_8_of_12": counters["mae_beats_price"] >= 8,
            "mae_beats_v12_current_8_of_12": counters["mae_beats_v12_current"] >= 8,
            "query_beats_global_8_of_12": counters["mae_beats_global"] >= 8,
            "query_beats_lag5_8_of_12": counters["mae_beats_lag5"] >= 8,
            "query_beats_axis_shuffle_8_of_12": counters["mae_beats_axis_shuffle"] >= 8,
            "query_beats_date_shuffle_8_of_12": counters["mae_beats_date_shuffle"] >= 8,
            "mean_incremental_mae_improvement_positive": mean_improvement > 0,
            "worst_incremental_target_degradation_at_most_0_5pct": worst_improvement >= -0.5,
            "positive_half_outer_fold_targets": positive_fold_targets >= outer_fold_targets / 2,
            "2025_and_2026_mean_incremental_improvement_nonnegative": yearly_mean.get("2025", -math.inf) >= 0 and yearly_mean.get("2026", -math.inf) >= 0,
        },
        "counters": {
            **dict(counters),
            "mean_relative_mae_improvement_vs_v12_current_pct": mean_improvement,
            "mean_relative_mae_improvement_vs_price_pct": float(np.mean(improvements_price)),
            "worst_relative_mae_improvement_vs_v12_current_pct": worst_improvement,
            "positive_outer_fold_target_count": int(positive_fold_targets),
            "outer_fold_target_count": int(outer_fold_targets),
            "yearly_mean_improvement_vs_v12_current_pct": yearly_mean,
            "target_count": len(targets),
            "avoidance_core": dict(core),
        },
    }


def preregistration(
    *, event_path: Path, graph_manifest_path: Path, v12_root: Path
) -> dict[str, Any]:
    return {
        "schema_version": PREREGISTRATION_SCHEMA_VERSION,
        "frozen_before_results": True,
        "purpose": "test whether ETF identity-level Flow propagation adds beyond v12 aggregate current Flow",
        "timing_contract": TIMING_CONTRACT,
        "scope": {
            "outer_years": list(OUTER_YEARS),
            "purge_sessions": PURGE_SESSIONS,
            "targets": list(TARGET_NAMES),
            "no_date_or_stock_sampling": True,
            "strict_pit_eligibility": True,
            "small_etf_action": "REMOVE_FROM_DENOMINATOR_NOT_DOWNWEIGHT",
        },
        "representation": {
            "typed_channels": ["clean_flow_rate_pct", "special_effective_flow_rate_pct"],
            "rate_clip_pct": RATE_CLIP_PCT,
            "missing_not_zero": True,
            "true_zero_preserved": True,
            "rolling_windows": list(ROLLING_WINDOWS),
            "states": list(STATE_NAMES),
            "truncated_svd_components": LATENT_COMPONENTS,
            "truncated_svd_iterations": SVD_ITERATIONS,
            "svd_fit": "outer_train_dates_only_without_targets",
            "absolute_common_flow_date_centered": False,
            "selection_biased_table_48_used": False,
            "stock_query": "PIT holding-weight exposure times all-ETF latent states",
        },
        "estimator": {
            "library": "catboost",
            "version": CATBOOST_VERSION,
            "parameters": CATBOOST_PARAMETERS,
            "date_balanced_total_weight": True,
            "residual_adapter_identical_to_v12": True,
        },
        "models": list(MODEL_NAMES),
        "controls": {
            "sealed_v12_price_and_current_predictions": True,
            "global_factor_without_stock_query": True,
            "flow_lag_sessions": 5,
            "etf_identity_axis_shuffle": True,
            "train_and_test_date_block_shuffle_separate": True,
            "same_model_capacity": True,
        },
        "gate_thresholds": {
            "targets_required": 8,
            "controls_required": [
                "price_only",
                "v12_current_flow",
                "full_etf_global_only",
                "full_etf_query_lag5",
                "full_etf_axis_shuffle",
                "full_etf_date_shuffle",
            ],
            "mean_incremental_mae_improvement_positive": True,
            "worst_incremental_target_degradation_at_most_pct": 0.5,
            "positive_outer_fold_targets": 36,
            "2025_and_2026_mean_incremental_improvement_nonnegative": True,
            "basket_rank_and_economic_targets": 8,
            "avoidance_core_targets": 3,
        },
        "activation": {
            "pass": "ELIGIBLE_FOR_BF16_INDUCED_SET_CROSS_ATTENTION_CANARY_NOT_DEPLOYMENT",
            "fail": "NO_FULL_ETF_IDENTITY_ACTIVATION_FROM_LINEAR_LATENT_GATE",
            "deployment_forbidden": True,
            "nvfp4_conversion_forbidden": True,
        },
        "references": list(REFERENCE_PAPERS),
        "frozen_inputs": {
            "source_sha256": sha256_file(Path(__file__)),
            "event_cube_path": str(event_path),
            "event_cube_sha256": sha256_file(event_path),
            "graph_manifest_path": str(graph_manifest_path),
            "graph_manifest_sha256": sha256_file(graph_manifest_path),
            "v12_receipt_sha256": sha256_file(v12_root / "v12_residual_canary_receipt.json"),
            "v12_preregistration_sha256": sha256_file(v12_root / "v12_residual_canary_preregistration.json"),
        },
    }


def run(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    phase_a_root = Path(args.phase_a_root)
    event_path = phase_a_root / "v11_r2_flow_event_cube.sqlite3"
    graph_root = Path(args.graph_dataset_root)
    graph_manifest_path = graph_root / "manifest.json"
    v12_root = Path(args.v12_root)
    for required in (
        event_path,
        graph_manifest_path,
        v12_root / "v12_residual_canary_receipt.json",
        v12_root / "v12_residual_canary_preregistration.json",
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    frozen = preregistration(
        event_path=event_path,
        graph_manifest_path=graph_manifest_path,
        v12_root=v12_root,
    )
    preregistration_path = output_root / "v16_full_etf_identity_preregistration.json"
    if preregistration_path.exists():
        existing = json.loads(preregistration_path.read_text(encoding="utf-8"))
        if existing != frozen:
            raise ValueError("existing v16 preregistration does not match frozen source/input contract")
    else:
        write_json_atomic(preregistration_path, frozen)
    preregistration_sha256 = sha256_file(preregistration_path)
    if args.preregister_only:
        return preregistration_path, {
            "status": "PREREGISTERED",
            "preregistration_sha256": preregistration_sha256,
        }

    receipt_path = output_root / "v16_full_etf_identity_receipt.json"
    if receipt_path.exists() and not args.replace:
        raise FileExistsError(receipt_path)
    started_at = utc_now()
    write_json_atomic(
        output_root / "run_state.json",
        {
            "status": "RUNNING",
            "stage": "stock_matrix",
            "started_at_utc": started_at,
            "preregistration_sha256": preregistration_sha256,
        },
    )
    with readonly_connection(event_path) as event, readonly_connection(
        Path(args.source_database)
    ) as source:
        matrix = build_stock_matrix_from_sources(
            event=event,
            source=source,
            graph_dataset_root=graph_root,
            progress=_progress,
        )
        panel = build_full_etf_panel(
            event=event,
            date_values=matrix["date_values"],
            progress=_progress,
        )
    if tuple(panel["dates"]) != tuple(matrix["date_values"]):
        raise ValueError("ETF panel date identity mismatch")
    graph_manifest = json.loads(graph_manifest_path.read_text(encoding="utf-8"))
    write_json_atomic(
        output_root / "run_state.json",
        {
            "status": "RUNNING",
            "stage": "outer_folds",
            "started_at_utc": started_at,
            "preregistration_sha256": preregistration_sha256,
            "scope": {
                "signal_date_count": len(matrix["date_values"]),
                "stock_row_count": len(matrix["targets"]),
                "stock_symbol_count": len(matrix["symbol_values"]),
                "etf_identity_count": len(panel["tickers"]),
            },
        },
    )
    targets, folds = evaluate(
        matrix=matrix,
        panel=panel,
        graph_manifest=graph_manifest,
        v12_root=v12_root,
        output_root=output_root,
        preregistration_sha256=preregistration_sha256,
        thread_count=args.thread_count,
        progress=_progress,
    )
    gate = summarize_gate(targets)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "started_at_utc": started_at,
        "generated_at_utc": utc_now(),
        "timing_contract": TIMING_CONTRACT,
        "preregistration_sha256": preregistration_sha256,
        "source_sha256": sha256_file(Path(__file__)),
        "source_event_cube_sha256": sha256_file(event_path),
        "source_graph_manifest_sha256": sha256_file(graph_manifest_path),
        "scope": {
            "signal_date_start": matrix["date_values"][0],
            "signal_date_end": matrix["date_values"][-1],
            "signal_date_count": len(matrix["date_values"]),
            "stock_symbol_count": len(matrix["symbol_values"]),
            "stock_row_count": len(matrix["targets"]),
            "etf_identity_count": len(panel["tickers"]),
            "target_count": len(TARGET_NAMES),
            "stock_matrix_audit": matrix["audit"],
            "stock_matrix_excluded": matrix["excluded"],
            "etf_panel_audit": panel["audit"],
            "timing_violation_count": matrix["timing_violation_count"],
            "no_date_or_stock_sampling": True,
        },
        "catboost": {
            "version": CATBOOST_VERSION,
            "parameters": CATBOOST_PARAMETERS,
            "thread_count": int(args.thread_count),
            "gpu_used": False,
        },
        "representation": {
            "latent_components": LATENT_COMPONENTS,
            "states": list(STATE_NAMES),
            "typed_channels": 2,
            "fold_local_target_free_fit": True,
            "full_etf_axis_retained_until_svd": True,
        },
        "folds": folds,
        "targets": targets,
        "gate": gate,
        "next_activation": (
            "ELIGIBLE_FOR_BF16_INDUCED_SET_CROSS_ATTENTION_CANARY_NOT_DEPLOYMENT"
            if gate["passed_paths"]
            else "NO_FULL_ETF_IDENTITY_ACTIVATION_FROM_LINEAR_LATENT_GATE"
        ),
        "implementation_validity": {
            "price_flow_lag_contract_preserved": True,
            "date_balanced": True,
            "absolute_common_flow_date_centered": False,
            "small_etfs_removed_from_denominator": True,
            "true_zero_separate_from_missing": True,
            "table_48_breadth_used": False,
            "target_information_used_in_svd": False,
            "existing_v11_v12_v13_v14_v15_outputs_modified": False,
        },
        "references": list(REFERENCE_PAPERS),
        "limitations": [
            "2021-2026 historical OOS informed prior designs and is not a clean new lockbox",
            "TruncatedSVD is a target-free information gate, not the final BF16 set-attention model",
            "ETF identity-to-stock propagation uses disclosed PIT holdings where available; global factors still include every strictly eligible ETF",
            "a PASS cannot activate trading, deployment, BF16 production training, or NVFP4 conversion without a new forward lockbox",
        ],
    }
    write_json_atomic(receipt_path, receipt)
    write_json_atomic(
        output_root / "run_state.json",
        {
            "status": "COMPLETE",
            "gate_status": gate["status"],
            "passed_paths": gate["passed_paths"],
            "receipt_path": str(receipt_path),
            "receipt_sha256": sha256_file(receipt_path),
            "completed_at_utc": utc_now(),
        },
    )
    return receipt_path, receipt


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--phase-a-root", type=Path, default=DEFAULT_PHASE_A_ROOT)
    result.add_argument("--graph-dataset-root", type=Path, default=DEFAULT_GRAPH_DATASET_ROOT)
    result.add_argument("--source-database", type=Path, default=DEFAULT_SOURCE_DATABASE)
    result.add_argument("--v12-root", type=Path, default=DEFAULT_V12_ROOT)
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    result.add_argument("--thread-count", type=int, default=10)
    result.add_argument("--preregister-only", action="store_true")
    result.add_argument("--replace", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    path, payload = run(args)
    if payload.get("status") == "PREREGISTERED":
        print(json.dumps({"path": str(path), **payload}, indent=2, sort_keys=True))
        return 0
    summary = {
        "status": payload["gate"]["status"],
        "path": str(path),
        "sha256": sha256_file(path),
        "scope": payload["scope"],
        "gate": payload["gate"],
        "next_activation": payload["next_activation"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if payload["gate"]["status"] == "V16_FULL_ETF_IDENTITY_PASS" else 3
