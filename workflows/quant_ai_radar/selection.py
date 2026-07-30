"""Evidence-led daily selection for expensive trained-model interpretation."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from workflows.quant_ai_radar.universe import Candidate


SELECTION_SCHEMA_VERSION = "quant.ai_radar_dynamic_selection.v1"
_TICKER = re.compile(r"^[A-Z][A-Z0-9.\-]{0,14}$")
_RELATED = re.compile(
    r"(?:^|,\s*)([A-Za-z][A-Za-z0-9.\-]{0,14})"
    r"(?:\s*:\s*([-+]?\d+(?:\.\d+)?)%)?"
)


@dataclass(frozen=True)
class SelectionResult:
    selected: tuple[Candidate, ...]
    manifest: dict[str, Any]
    coverage_ledger: tuple[dict[str, Any], ...]


def _symbol(value: Any) -> str | None:
    candidate = str(value or "").strip().upper()
    return candidate if _TICKER.fullmatch(candidate) else None


def _symbols(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        values = value
    else:
        values = re.split(r"[,;|\s]+", str(value))
    result = []
    for value_item in values:
        normalized = _symbol(value_item)
        if normalized and normalized not in result:
            result.append(normalized)
    return result


def _related(value: Any) -> list[tuple[str, float]]:
    result = []
    for match in _RELATED.finditer(str(value or "")):
        ticker = _symbol(match.group(1))
        if ticker:
            result.append((ticker, abs(float(match.group(2) or 0.0))))
    return result


def _number(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key) if row.get(key) is not None else default)
    except (TypeError, ValueError):
        return default


def _rank_points(row: Mapping[str, Any], key: str = "rank") -> float:
    rank = max(_number(row, key, 9999.0), 1.0)
    return 100.0 / rank


def _add(
    scores: dict[str, float],
    reasons: dict[str, set[str]],
    symbol: str | None,
    points: float,
    reason: str,
) -> None:
    if not symbol:
        return
    scores[symbol] += max(float(points), 0.0)
    reasons[symbol].add(reason)


def _score_release(
    features: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, set[str]], dict[str, float], dict[str, set[str]]]:
    etf_scores: dict[str, float] = defaultdict(float)
    etf_reasons: dict[str, set[str]] = defaultdict(set)
    stock_scores: dict[str, float] = defaultdict(float)
    stock_reasons: dict[str, set[str]] = defaultdict(set)

    for row in features.get("etfs", ()):
        state = str(row.get("state") or "")
        reasons = {
            "oracle_pit_flow_price_materiality",
            "oracle_flow_price_divergence"
            if state == "flow_price_divergence"
            else state,
        }
        base = _number(row, "priority_score") + _rank_points(row)
        ticker = _symbol(row.get("ticker"))
        for reason in reasons:
            _add(etf_scores, etf_reasons, ticker, base, reason)

    for row in features.get("stocks", ()):
        symbol = _symbol(row.get("symbol"))
        base = _number(row, "priority_score") + _rank_points(row)
        _add(
            stock_scores,
            stock_reasons,
            symbol,
            base,
            "oracle_pit_etf_constituent_flow_exposure",
        )

    return etf_scores, etf_reasons, stock_scores, stock_reasons


def _ranked_candidates(
    candidates: Iterable[Candidate],
    *,
    task_type: str,
    scores: Mapping[str, float],
    reasons: Mapping[str, set[str]],
) -> list[tuple[Candidate, float, tuple[str, ...]]]:
    ranked = []
    for candidate in candidates:
        if candidate.proxy_task_type != task_type:
            continue
        score = float(scores.get(candidate.symbol, 0.0))
        evidence = tuple(sorted(reasons.get(candidate.symbol, set())))
        if score > 0 and evidence:
            ranked.append((candidate, score, evidence))
    return sorted(ranked, key=lambda item: (-item[1], item[0].symbol))


def select_daily_inference(
    candidates: Sequence[Candidate],
    features: Mapping[str, Any],
    *,
    max_etfs: int = 64,
    max_stocks: int = 192,
) -> SelectionResult:
    """Select dynamic material evidence, never a fixed ticker list."""

    if max_etfs < 1 or max_stocks < 1:
        raise ValueError("daily model budgets must be >= 1 for each bucket")
    etf_scores, etf_reasons, stock_scores, stock_reasons = _score_release(
        features
    )
    ranked_etfs = _ranked_candidates(
        candidates,
        task_type="etf_own_flow_analysis",
        scores=etf_scores,
        reasons=etf_reasons,
    )
    ranked_stocks = _ranked_candidates(
        candidates,
        task_type="stock_constituent_flow_analysis",
        scores=stock_scores,
        reasons=stock_reasons,
    )
    chosen = ranked_etfs[:max_etfs] + ranked_stocks[:max_stocks]
    selected_by_symbol = {
        candidate.symbol: {
            "priority_score": round(score, 6),
            "selection_reasons": list(reasons),
        }
        for candidate, score, reasons in chosen
    }
    candidate_by_symbol = {candidate.symbol: candidate for candidate in candidates}
    selected = tuple(
        candidate_by_symbol[symbol] for symbol in sorted(selected_by_symbol)
    )
    ledger = []
    for candidate in sorted(candidates, key=lambda item: item.symbol):
        detail = selected_by_symbol.get(candidate.symbol)
        if detail:
            ledger.append(
                {
                    **candidate.to_dict(),
                    "model_inference_selected": True,
                    "selection_status": "selected_material_daily_evidence",
                    **detail,
                }
            )
        else:
            score_map = (
                etf_scores
                if candidate.proxy_task_type == "etf_own_flow_analysis"
                else stock_scores
            )
            reason_map = (
                etf_reasons
                if candidate.proxy_task_type == "etf_own_flow_analysis"
                else stock_reasons
            )
            score = float(score_map.get(candidate.symbol, 0.0))
            evidence = sorted(reason_map.get(candidate.symbol, set()))
            ledger.append(
                {
                    **candidate.to_dict(),
                    "model_inference_selected": False,
                    "selection_status": (
                        "capacity_ranked_below_daily_budget"
                        if score > 0
                        else "no_material_oracle_evidence_today"
                    ),
                    "priority_score": round(score, 6),
                    "selection_reasons": evidence,
                }
            )
    manifest = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "policy": "full_quant_scan_then_dynamic_material_evidence_model_inference",
        "fixed_ticker_list_used": False,
        "full_candidate_count": len(candidates),
        "selected_count": len(selected),
        "selected_etf_count": min(len(ranked_etfs), max_etfs),
        "selected_stock_count": min(len(ranked_stocks), max_stocks),
        "eligible_scored_etf_count": len(ranked_etfs),
        "eligible_scored_stock_count": len(ranked_stocks),
        "max_etfs": max_etfs,
        "max_stocks": max_stocks,
        "selection_inputs": [
            "sealed_oracle_daily_price",
            "sealed_oracle_massive_etf_flow",
            "sealed_oracle_fmp_etf_constituents",
            "oracle_derived_rotation_clusters",
        ],
        "nonselected_policy": (
            "retain full quantitative coverage and an explicit exclusion reason; "
            "do not spend generative inference on immaterial daily evidence"
        ),
    }
    return SelectionResult(
        selected=selected,
        manifest=manifest,
        coverage_ledger=tuple(ledger),
    )
