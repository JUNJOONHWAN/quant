#!/usr/bin/env python3
"""Classify every FMP stable endpoint into an auditable training collection plan."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


COVERED_EXISTING = {
    "company_search_stock_screener": "fmp_universe_snapshot",
    "stock_directory_company_symbols_list": "fmp_universe_snapshot",
    "stock_directory_symbol_changes_list": "fmp_universe_snapshot",
    "stock_directory_etf_symbol_search": "fmp_universe_snapshot",
    "stock_directory_actively_trading_list": "fmp_universe_snapshot",
    "company_information_delisted_companies": "fmp_universe_snapshot",
    "charts_stock_price_and_volume_data": "fmp_historical_price_eod_full",
    "etf_and_mutual_funds_etf_and_fund_holdings": "fmp_v4_historical_etf_holdings",
}

LOOKUP_ONLY = {
    "company_search_stock_symbol_search",
    "company_search_company_name_search",
    "company_search_cik",
    "company_search_cusip",
    "company_search_isin",
    "company_search_exchange_variants",
    "company_information_company_profile_by_cik",
    "company_information_search_mergers_and_acquisitions",
    "sec_filings_sec_filings_by_cik",
    "sec_filings_sec_filings_by_name",
    "sec_filings_sec_filings_company_search_by_cik",
    "sec_filings_industry_classification_search",
    "insider_trades_search_insider_trades_by_reporting_name",
    "senate_senate_trades_by_name",
    "senate_house_trades_by_name",
    "fundraisers_crowdfunding_campaign_search",
    "fundraisers_equity_offering_search",
}

BINARY_DUPLICATE = {
    "statements_financial_reports_form_10_k_xlsx": "statements_financial_reports_form_10_k_json"
}

DOMAIN_DISCOVERY = {
    "Indexes": ("indexes_stock_market_indexes_list", ["symbol", "ticker"]),
    "Commodity": ("commodity_commodities_list", ["symbol", "ticker"]),
    "Forex": ("forex_forex_currency_pairs", ["symbol", "ticker"]),
    "Crypto": ("crypto_cryptocurrency_list", ["symbol", "ticker"]),
}

VALUE_DIMENSIONS = {
    "exchange": ["NASDAQ", "NYSE", "AMEX", "OTC"],
    "sector": [
        "Basic Materials",
        "Communication Services",
        "Consumer Cyclical",
        "Consumer Defensive",
        "Energy",
        "Financial Services",
        "Healthcare",
        "Industrials",
        "Real Estate",
        "Technology",
        "Utilities",
    ],
    "name": [
        "GDP",
        "realGDP",
        "nominalPotentialGDP",
        "realGDPPerCapita",
        "federalFunds",
        "CPI",
        "inflationRate",
        "inflation",
        "retailSales",
        "consumerSentiment",
        "durableGoods",
        "unemploymentRate",
        "totalNonfarmPayroll",
        "initialClaims",
        "industrialProductionTotalIndex",
        "newPrivatelyOwnedHousingUnitsStartedTotalUnits",
        "totalVehicleSales",
    ],
}


def _load_registry(stock_repo: Path) -> Sequence[Mapping[str, Any]]:
    path = stock_repo / "mcp_servers" / "fmp_endpoint_registry.py"
    spec = importlib.util.spec_from_file_location("fmp_endpoint_registry", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load FMP endpoint registry")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.FMP_STABLE_ENDPOINTS)


def _params(endpoint: Mapping[str, Any]) -> List[str]:
    return [str(item.get("name")) for item in endpoint.get("query_parameters", [])]


def _collection(endpoint: Mapping[str, Any], start_year: int, end_year: int) -> dict:
    endpoint_id = str(endpoint["id"])
    category = str(endpoint["category"])
    params = _params(endpoint)
    sample = dict(endpoint.get("sample_params") or {})
    collection: Dict[str, Any] = {
        "mode": "global",
        "static_params": {},
        "variants": [],
        "pagination": "page" in params,
        "page_size": int(sample.get("limit") or 100),
        "max_pages": 10000,
    }

    if category == "ETF And Mutual Funds" and "symbol" in params:
        collection.update({"mode": "per_etf", "dimension_param": "symbol"})
    elif category in DOMAIN_DISCOVERY and "symbol" in params:
        source, keys = DOMAIN_DISCOVERY[category]
        collection.update(
            {
                "mode": "per_discovered",
                "dimension_param": "symbol",
                "source_endpoint_id": source,
                "source_keys": keys,
            }
        )
    elif "symbol" in params:
        collection.update({"mode": "per_symbol", "dimension_param": "symbol"})
    elif "symbols" in params:
        # The runner currently expands one symbol per call so every checkpoint has
        # one unambiguous entity. FMP still receives the documented plural key.
        collection.update({"mode": "per_symbol", "dimension_param": "symbols"})
    elif "cik" in params:
        cik_source = (
            "sec_filings_sec_company_full_profile"
            if category == "Fundraisers"
            else "stock_directory_cik_list"
        )
        collection.update(
            {
                "mode": "per_discovered",
                "dimension_param": "cik",
                "source_endpoint_id": cik_source,
                "source_keys": ["cik"],
            }
        )
    elif "industry" in params:
        collection.update(
            {
                "mode": "per_discovered",
                "dimension_param": "industry",
                "source_endpoint_id": "stock_directory_available_industries",
                "source_keys": ["industry", "name"],
            }
        )
    elif "sector" in params:
        collection.update(
            {
                "mode": "per_value",
                "dimension_param": "sector",
                "values": VALUE_DIMENSIONS["sector"],
            }
        )
    elif "exchange" in params:
        collection.update(
            {
                "mode": "per_value",
                "dimension_param": "exchange",
                "values": VALUE_DIMENSIONS["exchange"],
            }
        )
    elif "name" in params and category == "Economics":
        collection.update(
            {
                "mode": "per_value",
                "dimension_param": "name",
                "values": VALUE_DIMENSIONS["name"],
            }
        )

    if "from" in params and "to" in params:
        collection["date_windows"] = "year"
    elif endpoint_id.startswith("charts_") or "historical" in endpoint_id:
        collection["include_date_range"] = True

    if "year" in params and "quarter" in params:
        collection["variants"] = [
            {"year": year, "quarter": quarter}
            for year in range(start_year, end_year + 1)
            for quarter in range(1, 5)
        ]
    elif "year" in params and "period" in params:
        collection["variants"] = [
            {"year": year, "period": period}
            for year in range(start_year, end_year + 1)
            for period in ("FY", "Q1", "Q2", "Q3", "Q4")
        ]
    elif "year" in params:
        collection["variants"] = [{"year": year} for year in range(start_year, end_year + 1)]
    elif "period" in params:
        collection["variants"] = [{"period": "annual"}, {"period": "quarter"}]
    elif category == "Statements" and endpoint_id not in {
        "statements_financial_reports_dates",
        "statements_financial_reports_form_10_k_json",
        "statements_financial_reports_form_10_k_xlsx",
    }:
        collection["variants"] = [
            {"period": "annual", "limit": 1000},
            {"period": "quarter", "limit": 1000},
        ]

    if category == "Technical Indicators":
        collection["static_params"].update({"periodLength": 14, "timeframe": "1day"})
        collection["include_date_range"] = True

    if "date" in params and "from" not in params and "to" not in params:
        # Snapshot-by-date endpoints are scheduled separately as daily refreshes.
        collection["static_params"]["date"] = date.today().isoformat()

    return collection


def build_plan(
    registry: Sequence[Mapping[str, Any]],
    probe: Mapping[str, Any],
    start_date: str,
    end_date: str,
) -> dict:
    probe_records = {
        str(item["id"]): item
        for item in probe["providers"]["FMP"]["records"]
    }
    endpoints = []
    for endpoint in registry:
        endpoint_id = str(endpoint["id"])
        observed = dict(probe_records.get(endpoint_id) or {})
        status = str(observed.get("status") or "not_probed")
        record = {
            "id": endpoint_id,
            "category": endpoint.get("category"),
            "title": endpoint.get("title"),
            "path": endpoint.get("path"),
            "plan_access": endpoint.get("plan_access"),
            "probe_status": status,
            "probe_http_status": observed.get("status_code"),
        }
        if (
            endpoint_id in LOOKUP_ONLY
            and status == "api_error"
            and observed.get("status_code") == 400
        ):
            record.update(
                {
                    "action": "lookup_only",
                    "reason": (
                        "required_query_route_live_probe_400_then_symbol_probe_200;"
                        "not_distinct_training_dataset"
                    ),
                }
            )
        elif status not in {"accessible", "empty_success"}:
            record.update(
                {
                    "action": "not_entitled" if observed.get("status_code") in {402, 403} else "probe_error",
                    "reason": "live_access_probe_{}".format(observed.get("status_code") or status),
                }
            )
        elif endpoint_id in COVERED_EXISTING:
            record.update(
                {
                    "action": "covered_existing",
                    "reason": COVERED_EXISTING[endpoint_id],
                }
            )
        elif endpoint_id in BINARY_DUPLICATE:
            record.update(
                {
                    "action": "covered_by_json_equivalent",
                    "reason": BINARY_DUPLICATE[endpoint_id],
                }
            )
        elif endpoint_id in LOOKUP_ONLY:
            record.update(
                {
                    "action": "lookup_only",
                    "reason": "query_route_not_distinct_training_dataset",
                }
            )
        else:
            record.update(
                {
                    "action": "snapshot" if endpoint["category"] in {"Quote", "Market Hours"} else "backfill",
                    "collection": _collection(
                        endpoint,
                        date.fromisoformat(start_date).year,
                        date.fromisoformat(end_date).year,
                    ),
                }
            )
        endpoints.append(record)

    action_counts = Counter(item["action"] for item in endpoints)
    category_actions: Dict[str, Counter[str]] = defaultdict(Counter)
    for item in endpoints:
        category_actions[str(item["category"])][str(item["action"])] += 1
    return {
        "schema_version": "quant.fmp_training_plan.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "date_range": {"from": start_date, "to": end_date},
        "source_probe_generated_at_utc": probe.get("generated_at_utc"),
        "endpoint_count": len(endpoints),
        "action_counts": dict(sorted(action_counts.items())),
        "category_action_counts": {
            category: dict(sorted(counts.items()))
            for category, counts in sorted(category_actions.items())
        },
        "completion_contract": {
            "every_catalog_endpoint_classified": True,
            "not_entitled_is_terminal_only_with_live_402_or_403": True,
            "covered_existing_requires_separate_dataset_completion_gate": True,
            "lookup_only_means_no_distinct_rows_beyond_canonical_route": True,
            "backfill_and_snapshot_require_zero_failed_checkpoints": True,
        },
        "endpoints": endpoints,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock-repo", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--from", dest="start_date", required=True)
    parser.add_argument("--to", dest="end_date", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registry = _load_registry(args.stock_repo.expanduser())
    probe = json.loads(args.probe.expanduser().read_text(encoding="utf-8"))
    plan = build_plan(registry, probe, args.start_date, args.end_date)
    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.output.expanduser().write_text(
        json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: plan[key] for key in ("endpoint_count", "action_counts")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
