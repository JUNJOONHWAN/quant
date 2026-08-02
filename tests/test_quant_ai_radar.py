from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from training.quant_llm.create_model_release import build_release
from training.quant_llm.evaluate_frozen_test import adapter_artifact_set, evaluate
from quant_dataset.etf_flows import EtfFlowStore
from quant_dataset.fmp_etf_constituents import FmpEtfConstituentStore
from quant_dataset.shared_market import (
    SharedMarketStoreError,
    SharedReadOnlyDatabase,
    load_shared_market_binding,
)
from quant_dataset.storage import Database, canonical_json
from workflows.quant_ai_radar.action_assessment import (
    ACTION_PROMPT_CONTRACT,
    ACTION_VIEWS,
    build_action_assessment,
)
from workflows.quant_ai_radar.decision_support import build_market_dashboard
from workflows.quant_ai_radar.analyze_on_demand import (
    OnDemandError,
    _analysis_packet as on_demand_analysis_packet,
    _symbols as on_demand_symbols,
)
from workflows.quant_ai_radar.market_report import (
    MAX_MARKET_CATALOG_CHARS,
    MARKET_REPAIR_MAX_TOKENS,
    MARKET_SYNTHESIS_MAX_TOKENS,
    _evidence_catalog,
    aggregate_judgements,
    market_contract_repair_instruction,
    market_guided_json_schema,
    normalize_market_synthesis_confidence,
    strip_renderer_owned_numbers,
    synthesize_market,
)
from workflows.quant_ai_radar.model_runtime import (
    ModelGateError,
    ModelResponseParseError,
    ModelRelease,
    ResponseContractError,
    TrainedQuantClient,
    contract_repair_instruction,
    judgement_prohibited_violations,
    load_model_release,
    symbol_guided_json_schema,
    validate_symbol_judgement,
)
from workflows.quant_ai_radar.run_queue import RadarQueue
from workflows.quant_ai_radar.run_daily_cycle import build_stage_commands
from workflows.quant_ai_radar.prepare_shared_data import (
    build_parser as prepare_shared_parser,
)
from workflows.quant_ai_radar.run_quant_ai_radar import (
    SUCCESSFUL_RUN_STATUSES,
    build_parser as radar_parser,
)
from workflows.quant_ai_radar.relation_index import (
    load_verified_relation_index,
    refresh_relation_index,
)
from workflows.quant_ai_radar.selection import select_daily_inference
from workflows.quant_ai_radar.report_renderer import (
    render_reports,
    render_single_security_html,
)
from workflows.quant_ai_radar.report_narratives import (
    _unsupported_cluster_terms,
)
from workflows.quant_ai_radar.validate_shadow_run import validate_shadow_run
from workflows.quant_ai_radar.validate_runtime_readiness import (
    classify_runtime_evidence,
    summarize_kernel_events,
)
from workflows.quant_ai_radar.universe import (
    Candidate,
    resolve_as_of_date,
    scan_universe,
)
from workflows.market_structure_oracle.incremental_store import (
    latest_closed_nyse_session,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QuantAiRadarTest(unittest.TestCase):
    def test_oracle_target_is_close_aware_and_not_release_driven(self):
        from datetime import datetime
        from zoneinfo import ZoneInfo

        et = ZoneInfo("America/New_York")
        self.assertEqual(
            latest_closed_nyse_session(
                datetime(2026, 7, 29, 17, 59, tzinfo=et)
            ),
            "2026-07-28",
        )
        self.assertEqual(
            latest_closed_nyse_session(
                datetime(2026, 7, 30, 0, 15, tzinfo=et)
            ),
            "2026-07-29",
        )

    @staticmethod
    def _shared_store_fixture(root: Path) -> tuple[Path, Path, Path]:
        base_root = root / "base"
        incremental_root = root / "incremental"
        base_store = Database(base_root)
        incremental_store = Database(incremental_root)
        EtfFlowStore(base_store)
        EtfFlowStore(incremental_store)
        FmpEtfConstituentStore(base_store)
        FmpEtfConstituentStore(incremental_store)

        def provenance(connection: sqlite3.Connection, captured: str) -> None:
            connection.execute(
                """
                INSERT INTO raw_artifacts VALUES(
                    1,'test','daily','partition','request','payload',
                    'raw/test.json.gz','raw/test.metadata.json','{}','{}',
                    ?,200,2,2
                )
                """,
                (captured,),
            )
            connection.execute(
                """
                INSERT INTO capture_events VALUES(
                    1,1,'test','daily','partition','request',?,200,2,'{}','{}'
                )
                """,
                (captured,),
            )

        with sqlite3.connect(base_store.db_path) as connection:
            provenance(connection, "2026-07-15T00:00:00+00:00")
            daily_values = (
                "fmp",
                "AAA",
                "2026-07-14",
                100.0,
                101.0,
                99.0,
                100.0,
                100.0,
                1000.0,
                100.0,
                10,
                1,
                None,
                1,
                1,
                0,
                "2026-07-15T00:00:00+00:00",
                "{}",
            )
            connection.execute(
                """
                INSERT INTO daily_observations VALUES(
                    ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
                )
                """,
                daily_values,
            )
            connection.execute(
                """
                INSERT INTO daily_observation_versions VALUES(
                    1,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
                )
                """,
                daily_values,
            )
            connection.execute(
                """
                INSERT INTO quality_checks VALUES(
                    'AAA','2026-07-14','single_source','["fmp"]','{}',
                    '["missing_source:massive"]','{}',
                    '2026-07-15T00:00:00+00:00'
                )
                """
            )
            connection.execute(
                """
                INSERT INTO etf_constituent_snapshots VALUES(
                    'fmp','AAA','2026-05-31','2026-07-07',1,0,1,1,
                    '2026-07-07T00:00:00+00:00'
                )
                """
            )
            connection.execute(
                """
                INSERT INTO etf_constituent_observations VALUES(
                    'fmp','AAA','AAPL','AAPL','Apple',NULL,NULL,NULL,NULL,
                    '2026-05-31',NULL,'2026-07-07','acceptanceTime','provider',
                    1,100,100,'USD','shares','equity','US',1,1,0,
                    '2026-07-07T00:00:00+00:00','{}'
                )
                """
            )

        with sqlite3.connect(incremental_store.db_path) as connection:
            provenance(connection, "2026-07-28T00:00:00+00:00")
            daily_values = (
                "massive",
                "AAA",
                "2026-07-27",
                110.0,
                111.0,
                109.0,
                110.0,
                110.0,
                1200.0,
                110.0,
                11,
                1,
                None,
                1,
                1,
                0,
                "2026-07-28T00:00:00+00:00",
                "{}",
            )
            connection.execute(
                """
                INSERT INTO daily_observations VALUES(
                    ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
                )
                """,
                daily_values,
            )
            connection.execute(
                """
                INSERT INTO daily_observation_versions VALUES(
                    1,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
                )
                """,
                daily_values,
            )
            connection.execute(
                """
                INSERT INTO quality_checks VALUES(
                    'AAA','2026-07-27','single_source','["massive"]','{}',
                    '["missing_source:fmp"]','{}',
                    '2026-07-28T00:00:00+00:00'
                )
                """
            )
            flow_values = (
                "massive",
                "partners_etf_fund_flows",
                "AAA",
                "2026-07-24",
                "2026-07-24",
                100.0,
                10.0,
                10.0,
                100.0,
                "USD",
                "2026-07-24",
                "provider",
                "provider",
                "record",
                "source",
                1,
                1,
                0,
                "2026-07-28T00:00:00+00:00",
                "2026-07-28T00:00:00+00:00",
                "{}",
            )
            connection.execute(
                """
                INSERT INTO etf_flow_versions(
                    provider,endpoint_id,ticker,effective_date,processed_date,
                    fund_flow,nav,shares_outstanding,assets,currency,
                    available_at_date,availability_basis,pit_confidence,
                    record_hash,source_record_id,raw_artifact_id,capture_event_id,
                    source_row_index,captured_at_utc,ingested_at_utc,extra_json
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                flow_values,
            )
            connection.execute(
                """
                INSERT INTO etf_flow_observations VALUES(
                    'massive','partners_etf_fund_flows','AAA','2026-07-24',
                    '2026-07-24',100,10,10,100,'USD','2026-07-24',
                    'provider','provider','record','source',1,1,1,0,
                    '2026-07-28T00:00:00+00:00',
                    '2026-07-28T00:00:00+00:00','{}'
                )
                """
            )
            receipt = {
                "schema": "quant.market_structure_oracle.incremental.v2",
                "source_contract": (
                    "oracle_owned_fmp_massive_no_etf_radar_dependency"
                ),
                "target_as_of_date": "2026-07-27",
            }
            receipt_json = canonical_json(receipt)
            receipt_sha = hashlib.sha256(receipt_json.encode()).hexdigest()
            connection.execute(
                """
                CREATE TABLE oracle_snapshot_seals(
                    target_as_of_date TEXT PRIMARY KEY,
                    schema_version TEXT,source_contract TEXT,
                    receipt_sha256 TEXT,payload_json TEXT,sealed_at_utc TEXT
                )
                """
            )
            connection.execute(
                """
                INSERT INTO oracle_snapshot_seals VALUES(
                    '2026-07-27','quant.oracle_snapshot_seal.v1',?,?,?,
                    '2026-07-28T00:00:00+00:00'
                )
                """,
                (
                    "oracle_owned_fmp_massive_no_etf_radar_dependency",
                    receipt_sha,
                    receipt_json,
                ),
            )
        status_path = incremental_root / "state" / "oracle_incremental_status.json"
        status_path.parent.mkdir(parents=True)
        status_payload = {
                    "status": "COMPLETE",
                    "base_history_end": "2026-07-14",
                    "target_as_of_date": "2026-07-27",
                    "missing_sessions": [],
                    "database": str(incremental_store.db_path),
                    "market_row_gate": {"minimum_rows": 1},
                    "etf_flow": {
                        "expected_effective_date_at_least": "2026-07-23"
                    },
                    "snapshot_seal": {"receipt_sha256": receipt_sha},
                }
        status_path.write_text(
            json.dumps(status_payload),
            encoding="utf-8",
        )
        return base_store.db_path, incremental_store.db_path, status_path

    def test_shared_oracle_overlay_is_read_only_and_current(self):
        with tempfile.TemporaryDirectory() as temporary:
            base, incremental, status = self._shared_store_fixture(Path(temporary))
            binding = load_shared_market_binding(
                base_database=base,
                incremental_database=incremental,
                oracle_status_path=status,
            )
            database = SharedReadOnlyDatabase(binding)
            self.assertEqual(resolve_as_of_date(database), "2026-07-27")
            history_rows = database.history_payload_rows(
                "AAA",
                "2026-07-27",
                21,
            )
            self.assertEqual(
                [row["trade_date"] for row in history_rows],
                ["2026-07-14", "2026-07-27"],
            )
            with database.connect() as connection:
                rows = connection.execute(
                    """
                    SELECT source,trade_date,raw_artifact_id
                    FROM daily_observations ORDER BY trade_date
                    """
                ).fetchall()
                self.assertEqual(
                    [(row["source"], row["trade_date"]) for row in rows],
                    [("fmp", "2026-07-14"), ("massive", "2026-07-27")],
                )
                self.assertNotEqual(rows[0]["raw_artifact_id"], rows[1]["raw_artifact_id"])
                joined = connection.execute(
                    """
                    SELECT COUNT(*) FROM etf_flow_versions v
                    JOIN raw_artifacts r ON r.id=v.raw_artifact_id
                    """
                ).fetchone()[0]
                self.assertEqual(joined, 1)
                with self.assertRaises(sqlite3.OperationalError):
                    connection.execute(
                        "INSERT INTO quality_checks VALUES(NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL)"
                    )
            candidates, manifest = scan_universe(database, "2026-07-27")
            self.assertEqual([item.symbol for item in candidates], ["AAA"])
            self.assertEqual(manifest["all_stock_control_symbols"], 0)
            constituent_packet = FmpEtfConstituentStore(
                database,
                initialize_schema=False,
            ).packet_for_symbol("AAPL", "2026-07-27")
            self.assertIn("etf_memberships", constituent_packet)
            flow_packet = EtfFlowStore(
                database,
                initialize_schema=False,
            ).packet_for_ticker("AAA", "2026-07-27", 20)
            self.assertIsNone(flow_packet["latest"])

    def test_shared_oracle_binding_fails_closed_on_stale_constituents(self):
        with tempfile.TemporaryDirectory() as temporary:
            base, incremental, status = self._shared_store_fixture(Path(temporary))
            with self.assertRaises(SharedMarketStoreError):
                load_shared_market_binding(
                    base_database=base,
                    incremental_database=incremental,
                    oracle_status_path=status,
                    max_constituent_available_lag_days=10,
                )

    def test_relation_index_scans_history_once_then_reuses_markers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, incremental, status = self._shared_store_fixture(root)
            binding = load_shared_market_binding(
                base_database=base,
                incremental_database=incremental,
                oracle_status_path=status,
            )
            index_path = root / "relations.sqlite3"
            first = refresh_relation_index(binding, index_path)
            self.assertEqual(
                set(first["full_scan_sources"]),
                {
                    "base_flow_versions",
                    "incremental_flow_versions",
                    "base_constituent_snapshots",
                    "incremental_constituent_snapshots",
                    "base_constituent_members",
                    "incremental_constituent_members",
                },
            )
            self.assertEqual(
                first["relation_counts"],
                {
                    "fmp_etf_constituents": 1,
                    "fmp_etf_membership": 1,
                    "massive_etf_flow": 1,
                },
            )
            second = refresh_relation_index(binding, index_path)
            self.assertEqual(second["full_scan_sources"], [])
            self.assertTrue(
                all(
                    count == 0
                    for count in second[
                        "source_rows_processed_this_refresh"
                    ].values()
                )
            )
            verified = load_verified_relation_index(binding, index_path)
            self.assertEqual(
                verified["shared_source_fingerprint_sha256"],
                binding.source_fingerprint_sha256,
            )
            database = SharedReadOnlyDatabase(binding)
            candidates, manifest = scan_universe(
                database,
                "2026-07-27",
                relation_index_path=index_path,
            )
            self.assertEqual([item.symbol for item in candidates], ["AAA"])
            self.assertEqual(
                manifest["relation_source"],
                "persistent_incremental_relation_index",
            )
            self.assertFalse(
                manifest["historical_relation_tables_scanned_this_run"]
            )

    def test_dynamic_selection_keeps_full_coverage_ledger(self):
        candidates = [
            Candidate(
                symbol="AAA",
                proxy_task_type="etf_own_flow_analysis",
                quality_status="pass",
                relation_types=("massive_etf_flow",),
            ),
            Candidate(
                symbol="BBB",
                proxy_task_type="etf_own_flow_analysis",
                quality_status="pass",
                relation_types=("massive_etf_flow",),
            ),
            Candidate(
                symbol="AAPL",
                proxy_task_type="stock_constituent_flow_analysis",
                quality_status="pass",
                relation_types=("fmp_etf_membership",),
            ),
            Candidate(
                symbol="MSFT",
                proxy_task_type="stock_constituent_flow_analysis",
                quality_status="pass",
                relation_types=("fmp_etf_membership",),
            ),
        ]
        features = {
            "etfs": [
                {
                    "rank": 1,
                    "priority_score": 200,
                    "ticker": "AAA",
                    "state": "confirmed_accumulation",
                },
                {
                    "rank": 2,
                    "priority_score": 100,
                    "ticker": "BBB",
                    "state": "flow_price_divergence",
                },
            ],
            "stocks": [
                {"rank": 1, "priority_score": 80, "symbol": "AAPL"},
                {"rank": 2, "priority_score": 40, "symbol": "MSFT"},
            ],
        }
        selected = select_daily_inference(
            candidates,
            features,
            max_etfs=1,
            max_stocks=1,
        )
        self.assertEqual(len(selected.selected), 2)
        self.assertEqual(
            {item.symbol for item in selected.selected},
            {"AAA", "AAPL"},
        )
        self.assertEqual(len(selected.coverage_ledger), 4)
        ledger = {
            item["symbol"]: item for item in selected.coverage_ledger
        }
        self.assertTrue(ledger["AAA"]["model_inference_selected"])
        self.assertFalse(ledger["BBB"]["model_inference_selected"])
        self.assertEqual(
            ledger["BBB"]["selection_status"],
            "capacity_ranked_below_daily_budget",
        )
        self.assertFalse(selected.manifest["fixed_ticker_list_used"])

    def test_on_demand_symbols_are_not_limited_by_daily_selection(self):
        self.assertEqual(
            on_demand_symbols(["aapl,NVDA", "AAPL"]),
            ["AAPL", "NVDA"],
        )
        with self.assertRaises(OnDemandError):
            on_demand_symbols(["bad symbol"])

    def test_on_demand_packet_uses_same_corporate_action_adjustment(self):
        class Pipeline:
            def analysis_packet_for_pair(
                self,
                symbol,
                as_of_date,
                *,
                lookback_days,
                recompute_quality,
            ):
                self.call = {
                    "symbol": symbol,
                    "as_of_date": as_of_date,
                    "lookback_days": lookback_days,
                    "recompute_quality": recompute_quality,
                }
                return {
                    "symbol": symbol,
                    "packet_id": "before",
                    "history": [
                        {
                            "trade_date": "2026-07-14",
                            "sources": [
                                {
                                    "source": "massive",
                                    "close": 3.0,
                                    "volume": 1000.0,
                                }
                            ],
                        }
                    ],
                }

        pipeline = Pipeline()
        ledger = {
            "sha256": "oracle-ledger-sha",
            "events_by_symbol": {
                "TZA": [
                    {
                        "symbol": "TZA",
                        "action_type": "reverse_split",
                        "effective_date": "2026-07-15",
                        "available_date": "2026-06-10",
                        "old_shares": 10.0,
                        "new_shares": 1.0,
                        "price_factor_for_prior_rows": 10.0,
                        "volume_factor_for_prior_rows": 0.1,
                        "verification_status": "official",
                    }
                ]
            },
        }
        packet = on_demand_analysis_packet(
            pipeline,
            symbol="TZA",
            as_of_date="2026-07-29",
            corporate_actions=ledger,
        )
        source = packet["history"][0]["sources"][0]
        self.assertEqual(source["close"], 30.0)
        self.assertEqual(source["volume"], 100.0)
        self.assertEqual(
            packet["verified_corporate_actions"]["ledger_sha256"],
            "oracle-ledger-sha",
        )
        self.assertEqual(
            pipeline.call,
            {
                "symbol": "TZA",
                "as_of_date": "2026-07-29",
                "lookback_days": 21,
                "recompute_quality": False,
            },
        )

    def test_shadow_mode_is_explicit_and_opt_in(self):
        parser = radar_parser()
        self.assertFalse(parser.parse_args(["--prepare-only"]).shadow)
        self.assertTrue(
            parser.parse_args(["--prepare-only", "--shadow"]).shadow
        )
        self.assertIn("shadow_complete_not_published", SUCCESSFUL_RUN_STATUSES)

    def test_on_demand_renderer_has_no_broken_daily_navigation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "AAPL.html"
            rendered = render_single_security_html(
                path,
                result={
                    "symbol": "AAPL",
                    "task_type": "all_stock_control_analysis",
                    "judgement": {
                        "facts": {"symbol": "AAPL"},
                        "interpretation": {
                            "price_signal": "positive",
                            "etf_flow_signal": "unknown",
                        },
                        "counter_evidence": [],
                        "unknowns": [],
                        "regime": "mixed_or_flat",
                        "confidence": 0.5,
                        "conclusion": "온디맨드 분석",
                    },
                },
                as_of_date="2026-07-27",
            )
            html = path.read_text(encoding="utf-8")
            self.assertNotIn("../security_index.html", html)
            self.assertNotIn("../market_report.html", html)
            self.assertEqual(rendered["sha256"], sha256(path))

    def test_renderer_writes_hashed_market_and_security_reports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = {
                "as_of_date": "2026-07-27",
                "generated_at_kst": "2026-07-28T09:15:00+09:00",
                "selection": {
                    "full_candidate_count": 100,
                    "selected_count": 1,
                },
                "aggregate": {"analyzed_security_count": 1},
                "market_judgement": {
                    "market_state": "rotation",
                    "confidence": 0.7,
                    "summary": "<evidence bounded>",
                    "confirmations": [
                        {
                            "evidence_id": "etf.AAA",
                            "interpretation": "확인",
                        }
                    ],
                    "contradictions": [
                        {
                            "evidence_id": "aggregate.regime_counts",
                            "interpretation": "혼재",
                        }
                    ],
                    "unknowns": ["뉴스 원인 미확인"],
                },
            }
            results = [
                {
                    "symbol": "AAA",
                    "task_type": "etf_own_flow_analysis",
                    "judgement": {
                        "facts": {"symbol": "AAA"},
                        "interpretation": {
                            "price_signal": "positive",
                            "etf_flow_signal": "positive",
                            "relationship": "confirmation",
                        },
                        "counter_evidence": [],
                        "unknowns": [],
                        "regime": "confirmation",
                        "confidence": 0.7,
                        "conclusion": "확인",
                    },
                }
            ]
            rendered = render_reports(
                run_dir=root,
                report=report,
                results=results,
                coverage_ledger=[
                    {
                        "symbol": "AAA",
                        "priority_score": 9,
                        "selection_reasons": ["test"],
                    }
                ],
            )
            self.assertEqual(rendered["security_report_count"], 1)
            self.assertTrue(root.joinpath("market_report.html").is_file())
            self.assertTrue(root.joinpath("security_reports", "AAA.html").is_file())
            self.assertIn(
                "&lt;evidence bounded&gt;",
                root.joinpath("market_report.html").read_text(encoding="utf-8"),
            )
            self.assertEqual(len(rendered["content_sha256"]), 64)

    def test_shadow_gate_validates_queue_pit_hashes_and_nonpublication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue = RadarQueue(root / "selected_run_queue.sqlite3")
            queue.bind_metadata({"as_of_date": "2026-07-27"})
            queue.seed(
                [
                    Candidate(
                        symbol="AAA",
                        proxy_task_type="etf_own_flow_analysis",
                        quality_status="pass",
                        relation_types=("massive_etf_flow",),
                    )
                ]
            )
            queue.mark_done(
                symbol="AAA",
                actual_task_type="etf_own_flow_analysis",
                packet_id="packet",
                eligibility={"eligible": True},
                prompt_sha256="a" * 64,
                response_sha256="b" * 64,
                trace={"contract_attempts": 1},
                result={
                    "facts": {
                        "symbol": "AAA",
                        "as_of_date": "2026-07-27",
                    },
                    "interpretation": {
                        "price_signal": "positive",
                        "etf_flow_signal": "positive",
                        "relationship": "confirmation",
                        "scope": "data_interpretation_not_trade_execution",
                        "task_type": "etf_own_flow_analysis",
                    },
                    "counter_evidence": [],
                    "unknowns": [],
                    "regime": "confirmation",
                    "confidence": 0.7,
                    "conclusion": "visible evidence confirms the relation",
                },
            )
            results = queue.done_results()
            report = {
                "as_of_date": "2026-07-27",
                "generated_at_kst": "2026-07-28T09:15:00+09:00",
                "deployment_mode": "shadow",
                "full_universe_quantitative_scan_complete": True,
                "selected_model_scope_complete": True,
                "selection": {
                    "full_candidate_count": 100,
                    "selected_count": 1,
                },
                "aggregate": {"analyzed_security_count": 1},
                "market_judgement": {
                    "market_state": "rotation",
                    "confidence": 0.7,
                    "summary": "evidence bounded",
                    "confirmations": [],
                    "contradictions": [],
                    "unknowns": [],
                    "scope": "market_and_security_analysis_not_trade_execution",
                },
                "source_status": {
                    "quant_dataset": {
                        "manifest_sha256": "c" * 64,
                        "source_fingerprint": {"sha256": "d" * 64},
                    },
                    "shared_oracle_store": {
                        "source_fingerprint_sha256": "e" * 64,
                    },
                    "oracle_market_features": {
                        "snapshot_sha256": "f" * 64,
                    },
                },
            }
            root.joinpath("market_report.json").write_text(
                json.dumps(report),
                encoding="utf-8",
            )
            root.joinpath("run_state.json").write_text(
                json.dumps(
                    {
                        "status": "shadow_complete_not_published",
                        "production_scope_complete": True,
                        "production_latest_published": False,
                    }
                ),
                encoding="utf-8",
            )
            root.joinpath("security_judgements.jsonl").write_text(
                json.dumps(results[0]) + "\n",
                encoding="utf-8",
            )
            render_reports(
                run_dir=root,
                report=report,
                results=results,
                coverage_ledger=[
                    {
                        "symbol": "AAA",
                        "priority_score": 9,
                        "selection_reasons": ["test"],
                    }
                ],
            )
            audit = validate_shadow_run(
                run_dir=root,
                latest_path=root / "latest.json",
                elapsed_seconds=60,
                operating_window_seconds=120,
            )
            self.assertEqual(audit["status"], "pass")
            self.assertTrue(all(audit["gates"].values()))
            self.assertFalse(
                audit["activation_policy"][
                    "this_shadow_counts_toward_required_consecutive_runs"
                ]
            )

    def test_runtime_readiness_separates_nvrm_watchpoint_from_vllm_oom(self):
        evidence = {
            "expected_model": "qwen3-8b-quant-lora-v1",
            "served_models": ["qwen3-8b-quant-lora-v1"],
            "kernel_events": {
                "nvrm_nv_err_no_memory_lines": 1,
                "linux_oom_kill_lines": 0,
                "nvidia_xid_lines": 0,
            },
            "docker": {"enabled": "enabled", "active": "active"},
            "vllm": {
                "running": True,
                "oom_killed": False,
                "restart_count": 0,
                "restart_policy": "unless-stopped",
            },
            "user_linger": True,
            "radar_units": {
                unit: {"enabled": "disabled", "active": "inactive"}
                for unit in (
                    "quant-ai-radar-daily.service",
                    "quant-ai-radar-daily.timer",
                    "quant-ai-radar-relations-weekly.service",
                    "quant-ai-radar-relations-weekly.timer",
                )
            },
            "fmp_backfill": {
                "unit": {"enabled": "disabled", "active": "inactive"},
                "processes": [],
            },
        }
        result = classify_runtime_evidence(evidence)
        self.assertEqual(
            result["status"], "pass_with_resource_contention_watchpoint"
        )
        self.assertTrue(result["manual_reference_ready"])
        self.assertFalse(result["timer_activation_eligible"])
        evidence["vllm"]["oom_killed"] = True
        self.assertEqual(classify_runtime_evidence(evidence)["status"], "fail")

    def test_runtime_kernel_event_summary_keeps_failure_classes_separate(self):
        summary = summarize_kernel_events(
            "2026-07-30T00:19:10+09:00 host kernel: "
            "NVRM: Out of memory [NV_ERR_NO_MEMORY]\n"
            "2026-07-30T00:19:11+09:00 host kernel: NVRM: Xid 31\n"
            "2026-07-30T00:19:12+09:00 host kernel: oom-kill: Killed process"
        )
        self.assertEqual(summary["nvrm_nv_err_no_memory_lines"], 1)
        self.assertEqual(summary["nvidia_xid_lines"], 1)
        self.assertEqual(summary["linux_oom_kill_lines"], 1)

    def test_daily_cycle_uses_oracle_store_not_duplicate_source_refresh(self):
        commands = build_stage_commands(
            model_endpoint="http://127.0.0.1:8018/v1/chat/completions",
            release_manifest="/release.json",
            workers="8",
            max_ai_etfs="17",
            max_ai_stocks="29",
        )
        self.assertIn("workflows.quant_ai_radar.prepare_shared_data", commands[0])
        self.assertIn("workflows.quant_ai_radar.run_quant_ai_radar", commands[1])
        self.assertNotIn("refresh_daily_data", " ".join(sum(commands, [])))
        self.assertNotIn("--max-ai-etfs", commands[0])
        self.assertNotIn("--max-ai-stocks", commands[0])
        self.assertEqual(
            commands[1][commands[1].index("--max-ai-etfs") + 1],
            "17",
        )
        self.assertEqual(
            commands[1][commands[1].index("--max-ai-stocks") + 1],
            "29",
        )
        prepare_shared_parser().parse_args(commands[0][3:])
        radar_parser().parse_args(commands[1][3:])

    def test_full_universe_scan_has_no_fixed_list(self):
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "daily.sqlite3"
            connection = sqlite3.connect(database)
            connection.executescript(
                """
                CREATE TABLE quality_checks(
                    symbol TEXT, trade_date TEXT, status TEXT
                );
                CREATE TABLE daily_observations(
                    symbol TEXT, trade_date TEXT, close REAL, volume REAL
                );
                CREATE TABLE etf_flow_observations(
                    ticker TEXT, effective_date TEXT, processed_date TEXT, fund_flow REAL
                );
                CREATE TABLE etf_constituent_snapshots(
                    etf_ticker TEXT, effective_date TEXT, available_date TEXT
                );
                CREATE TABLE etf_constituent_observations(
                    etf_ticker TEXT, constituent_ticker TEXT,
                    effective_date TEXT, available_date TEXT
                );
                """
            )
            for symbol in ("AAA", "AAPL", "CONTROL"):
                connection.execute(
                    "INSERT INTO quality_checks VALUES(?,?,?)",
                    (symbol, "2026-07-16", "pass"),
                )
                connection.execute(
                    "INSERT INTO daily_observations VALUES(?,?,?,?)",
                    (symbol, "2026-07-16", 100.0, 1000.0),
                )
            connection.execute(
                "INSERT INTO etf_flow_observations VALUES(?,?,?,?)",
                ("AAA", "2026-07-14", "2026-07-15", 10.0),
            )
            connection.execute(
                "INSERT INTO etf_constituent_snapshots VALUES(?,?,?)",
                ("AAA", "2026-06-30", "2026-07-01"),
            )
            connection.execute(
                "INSERT INTO etf_constituent_observations VALUES(?,?,?,?)",
                ("AAA", "AAPL", "2026-06-30", "2026-07-01"),
            )
            connection.commit()
            connection.close()

            candidates, manifest = scan_universe(database, "2026-07-16")
            by_symbol = {item.symbol: item for item in candidates}
            self.assertEqual(set(by_symbol), {"AAA", "AAPL"})
            self.assertEqual(by_symbol["AAA"].proxy_task_type, "etf_own_flow_analysis")
            self.assertEqual(
                by_symbol["AAPL"].proxy_task_type,
                "stock_constituent_flow_analysis",
            )
            self.assertEqual(manifest["all_stock_control_symbols"], 1)
            self.assertFalse(manifest["fixed_ticker_list_used"])
            self.assertFalse(manifest["top_n_selection_used"])

    def test_queue_is_restartable_and_metadata_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            queue = RadarQueue(Path(temporary) / "queue.sqlite3")
            queue.bind_metadata({"as_of_date": "2026-07-16"})
            queue.seed(
                [
                    Candidate(
                        symbol="AAA",
                        proxy_task_type="etf_own_flow_analysis",
                        quality_status="pass",
                        relation_types=("massive_etf_flow",),
                    )
                ]
            )
            queue.mark_running("AAA")
            reopened = RadarQueue(queue.path)
            self.assertEqual(reopened.counts(), {"pending": 1})
            with self.assertRaises(ValueError):
                reopened.bind_metadata({"as_of_date": "2026-07-17"})
            reopened.mark_done(
                symbol="AAA",
                actual_task_type="etf_own_flow_analysis",
                packet_id="packet",
                eligibility={"eligible": True},
                prompt_sha256="prompt",
                response_sha256="response",
                trace={"usage": {"completion_tokens": 12}},
                result={"regime": "mixed"},
            )
            self.assertEqual(
                reopened.done_results()[0]["trace"]["usage"][
                    "completion_tokens"
                ],
                12,
            )

    def test_model_release_binds_adapter_dataset_and_green_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter = root / "adapter"
            adapter.mkdir()
            weights = adapter / "adapter_model.safetensors"
            config = adapter / "adapter_config.json"
            weights.write_bytes(b"weights")
            config.write_text("{}\n", encoding="utf-8")
            dataset = root / "dataset.json"
            dataset.write_text('{"schema_version":"dataset"}\n', encoding="utf-8")
            frozen_test = root / "test.jsonl"
            frozen_test.write_text('{"example_id":"one"}\n', encoding="utf-8")
            predictions = root / "predictions.jsonl"
            predictions.write_text('{"example_id":"one"}\n', encoding="utf-8")
            evaluation = root / "evaluation.json"
            adapter_set_sha, _ = adapter_artifact_set(adapter, [weights, config])
            evaluation.write_text(
                json.dumps(
                    {
                        "schema_version": "quant.frozen_test_evaluation.v1",
                        "endpoint_model": "quant-v1",
                        "adapter_set_sha256": adapter_set_sha,
                        "dataset_manifest": {"sha256": sha256(dataset)},
                        "frozen_test": {
                            "path": str(frozen_test),
                            "sha256": sha256(frozen_test),
                        },
                        "predictions": {
                            "path": str(predictions),
                            "sha256": sha256(predictions),
                        },
                        "status": "green",
                        "prohibited_violation_count": 0,
                        "required_gates": {"full_test": True, "no_lookahead": True},
                    }
                ),
                encoding="utf-8",
            )
            release_value = build_release(
                model_id="quant-v1",
                endpoint_model="quant-v1",
                base_model="Qwen3-8B",
                adapter_root=adapter,
                artifacts=[weights, config],
                dataset_manifest=dataset,
                evaluation_report=evaluation,
            )
            release_path = root / "release.json"
            release_path.write_text(json.dumps(release_value), encoding="utf-8")
            loaded = load_model_release(release_path)
            self.assertEqual(loaded.model_id, "quant-v1")
            weights.write_bytes(b"tampered")
            with self.assertRaises(ModelGateError):
                load_model_release(release_path)

    def test_merged_bf16_release_binds_evaluated_adapter_and_model_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter = root / "adapter"
            adapter.mkdir()
            weights = adapter / "adapter_model.safetensors"
            config = adapter / "adapter_config.json"
            weights.write_bytes(b"weights")
            config.write_text("{}\n", encoding="utf-8")
            dataset = root / "dataset.json"
            dataset.write_text('{"schema_version":"dataset"}\n', encoding="utf-8")
            frozen_test = root / "test.jsonl"
            frozen_test.write_text('{"example_id":"one"}\n', encoding="utf-8")
            predictions = root / "predictions.jsonl"
            predictions.write_text('{"example_id":"one"}\n', encoding="utf-8")
            adapter_set_sha, _ = adapter_artifact_set(
                adapter, [weights, config]
            )
            evaluation = root / "evaluation.json"
            evaluation.write_text(
                json.dumps(
                    {
                        "schema_version": "quant.frozen_test_evaluation.v1",
                        "endpoint_model": "qwen3-8b-quant-lora-v1",
                        "adapter_set_sha256": adapter_set_sha,
                        "dataset_manifest": {"sha256": sha256(dataset)},
                        "frozen_test": {
                            "path": str(frozen_test),
                            "sha256": sha256(frozen_test),
                        },
                        "predictions": {
                            "path": str(predictions),
                            "sha256": sha256(predictions),
                        },
                        "status": "green",
                        "prohibited_violation_count": 0,
                        "required_gates": {
                            "full_test": True,
                            "no_lookahead": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            merged_root = root / "Qwen3-8B-FLOW-BF16"
            merged_root.mkdir()
            model_file = merged_root / "model.safetensors"
            model_file.write_bytes(b"merged-bf16-weights")
            merge_core = {
                "schema_version": "quant.merged_hf_model.v1",
                "status": "complete",
                "model_name": "Qwen3-8B-FLOW",
                "precision": "bfloat16",
                "adapter_artifacts": [
                    {"path": str(weights), "sha256": sha256(weights)},
                    {"path": str(config), "sha256": sha256(config)},
                ],
                "files": [
                    {
                        "path": model_file.name,
                        "bytes": model_file.stat().st_size,
                        "sha256": sha256(model_file),
                    }
                ],
            }
            merge_core["content_sha256"] = hashlib.sha256(
                json.dumps(
                    merge_core,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            merge_manifest = merged_root / "merge_manifest.json"
            merge_manifest.write_text(
                json.dumps(merge_core), encoding="utf-8"
            )
            release_value = build_release(
                model_id="Qwen3-8B-FLOW",
                endpoint_model="Qwen3-8B-FLOW",
                base_model=str(merged_root),
                adapter_root=adapter,
                artifacts=[weights, config],
                dataset_manifest=dataset,
                evaluation_report=evaluation,
                merged_manifest=merge_manifest,
                merged_model_root=merged_root,
            )
            release_path = root / "release.json"
            release_path.write_text(json.dumps(release_value), encoding="utf-8")
            loaded = load_model_release(release_path)
            self.assertEqual(loaded.endpoint_model, "Qwen3-8B-FLOW")
            self.assertEqual(
                loaded.public_metadata()["merged_model"]["precision"],
                "bfloat16",
            )
            model_file.write_bytes(b"tampered")
            with self.assertRaises(ModelGateError):
                load_model_release(release_path)

    def test_frozen_evaluation_fails_coverage_and_prohibited_output(self):
        target = {
            "facts": {
                "symbol": "AAA",
                "as_of_date": "2026-07-16",
                "etf_flow_to_constituent": {
                    "eligible_etf_count": 1,
                    "net_weighted_flow_rate_contribution_pct": 0.000099,
                },
            },
            "interpretation": {
                "price_signal": "positive",
                "etf_flow_signal": "positive",
                "etf_flow_signal_source": "own_etf_flow",
                "relationship": "price_flow_positive_confirmation",
                "scope": "data_interpretation_not_trade_execution",
                "task_type": "etf_own_flow_analysis",
            },
            "counter_evidence": [],
            "unknowns": ["historical_backfill_not_true_as_observed_point_in_time"],
            "regime": "price_flow_positive_confirmation",
            "confidence": 0.7,
            "conclusion": "2026-07-16 현재 증거가 일치한다.",
        }
        precision_equivalent = json.loads(json.dumps(target))
        precision_equivalent["facts"]["etf_flow_to_constituent"][
            "net_weighted_flow_rate_contribution_pct"
        ] = 0.0000988
        green = evaluate({"one": target}, {"one": precision_equivalent})
        self.assertEqual(green["status"], "green")
        self.assertEqual(green["counts"]["facts_exact"], 1)
        mutated_fact = json.loads(json.dumps(precision_equivalent))
        mutated_fact["facts"]["etf_flow_to_constituent"][
            "eligible_etf_count"
        ] = 2
        fact_red = evaluate({"one": target}, {"one": mutated_fact})
        self.assertEqual(fact_red["status"], "red")
        self.assertEqual(fact_red["counts"]["facts_exact"], 0)
        violating = json.loads(json.dumps(precision_equivalent))
        violating["conclusion"] = "2026-07-17에 매수해야 한다."
        red = evaluate({"one": target, "two": target}, {"one": violating})
        self.assertEqual(red["status"], "red")
        self.assertFalse(red["required_gates"]["full_frozen_test_coverage"])
        self.assertGreater(red["prohibited_violation_count"], 0)
        malformed = json.loads(json.dumps(precision_equivalent))
        malformed["interpretation"] = "not-an-object"
        malformed["counter_evidence"] = "not-an-array"
        malformed["unknowns"] = "not-an-array"
        malformed_report = evaluate({"one": target}, {"one": malformed})
        self.assertEqual(malformed_report["status"], "red")
        self.assertEqual(malformed_report["counts"]["schema_valid"], 0)
        self.assertEqual(malformed_report["counts"]["signals_exact"], 0)
        self.assertEqual(malformed_report["metrics"]["counter_evidence_recall"], 0)
        self.assertEqual(malformed_report["metrics"]["unknown_recall"], 0)

    def test_symbol_judgement_preserves_deterministic_signals_and_facts(self):
        expected = {
            "facts": {
                "symbol": "AAA",
                "as_of_date": "2026-07-16",
                "etf_flow_to_constituent": {
                    "eligible_etf_count": 1,
                    "net_weighted_flow_rate_contribution_pct": 0.000099,
                },
            },
            "interpretation": {
                "price_signal": "positive",
                "etf_flow_signal": "negative",
                "etf_flow_signal_source": "own_etf_flow",
                "relationship": "price_up_flow_out_divergence",
                "scope": "data_interpretation_not_trade_execution",
                "task_type": "etf_own_flow_analysis",
            },
            "regime": "price_up_flow_out_divergence",
        }
        value = {
            "facts": expected["facts"],
            "interpretation": {
                "price_signal": "positive",
                "etf_flow_signal": "negative",
                "etf_flow_signal_source": "own_etf_flow",
                "relationship": "price_up_flow_out_divergence",
                "scope": "data_interpretation_not_trade_execution",
                "task_type": "etf_own_flow_analysis",
            },
            "counter_evidence": ["price_and_etf_flow_signals_diverge"],
            "unknowns": [],
            "regime": "price_up_flow_out_divergence",
            "confidence": 0.6,
            "conclusion": "2026-07-16 현재 증거가 엇갈린다.",
        }
        value["facts"] = json.loads(json.dumps(value["facts"]))
        value["facts"]["etf_flow_to_constituent"][
            "net_weighted_flow_rate_contribution_pct"
        ] = 0.0000988
        validated = validate_symbol_judgement(value, expected)
        self.assertEqual(validated["facts"], expected["facts"])
        mutated = json.loads(json.dumps(value))
        mutated["facts"]["symbol"] = "BBB"
        with self.assertRaises(ResponseContractError):
            validate_symbol_judgement(mutated, expected)
        mutated = json.loads(json.dumps(value))
        mutated["facts"]["etf_flow_to_constituent"]["eligible_etf_count"] = 2
        with self.assertRaises(ResponseContractError):
            validate_symbol_judgement(mutated, expected)
        mutated = json.loads(json.dumps(value))
        mutated["quality_status"] = "unexpected"
        with self.assertRaises(ResponseContractError):
            validate_symbol_judgement(mutated, expected)

    def test_client_repairs_one_hard_contract_failure_with_exact_facts(self):
        expected = {
            "facts": {"symbol": "PTCO", "as_of_date": "2025-11-03"},
            "interpretation": {
                "price_signal": "positive",
                "etf_flow_signal": "unknown",
                "etf_flow_signal_source": "none",
                "relationship": "insufficient_joint_evidence",
                "scope": "data_interpretation_not_trade_execution",
                "task_type": "all_stock_control_analysis",
            },
            "counter_evidence": [],
            "unknowns": [],
            "regime": "insufficient_joint_evidence",
            "confidence": 0.5,
            "conclusion": "Evidence as of 2025-11-03.",
        }
        invalid = json.loads(json.dumps(expected))
        invalid["facts"]["as_of_date"] = "2025-11-13"
        responses = iter((invalid, expected))
        payloads = []

        def transport(payload, headers, timeout):
            payloads.append(payload)
            return {
                "model": "quant-v1",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(next(responses), ensure_ascii=False)
                        },
                    }
                ],
                "usage": {"completion_tokens": 10},
            }

        release = ModelRelease(
            manifest_path=Path("/tmp/release.json"),
            manifest_sha256="a" * 64,
            model_id="quant-v1",
            endpoint_model="quant-v1",
            base_model="Qwen3-8B",
            adapter_root=Path("/tmp/adapter"),
            adapter_set_sha256="b" * 64,
            dataset_manifest_sha256="c" * 64,
            evaluation_sha256="d" * 64,
            raw={},
        )
        client = TrainedQuantClient(
            endpoint="http://127.0.0.1:8018/v1/chat/completions",
            release=release,
            transport=transport,
        )
        instruction = "지정된 구조로 답하라. /no_think"
        value, trace = client.complete_validated(
            system="EVIDENCE_JSON={}",
            user=instruction,
            expected_response=expected,
        )
        self.assertEqual(value["facts"], expected["facts"])
        self.assertEqual(trace["contract_attempts"], 2)
        self.assertTrue(trace["contract_repair_applied"])
        self.assertEqual(len(payloads), 2)
        self.assertEqual(payloads[0]["messages"][1]["content"], instruction)
        self.assertEqual(
            payloads[0]["response_format"]["type"], "json_schema"
        )
        guided = payloads[0]["response_format"]["json_schema"]["schema"]
        self.assertEqual(
            guided["required"],
            [
                "interpretation",
                "counter_evidence",
                "unknowns",
                "regime",
                "confidence",
                "conclusion",
            ],
        )
        self.assertEqual(
            guided["properties"]["regime"]["enum"],
            ["insufficient_joint_evidence"],
        )
        repair = payloads[1]["messages"][-1]["content"]
        self.assertEqual(repair, contract_repair_instruction(
            expected, "trained model changed deterministic facts"
        ))
        self.assertTrue(repair.endswith("/no_think"))
        self.assertIn(
            '"as_of_date":"2025-11-03"',
            repair,
        )
        self.assertIn(
            '"etf_flow_signal_source":"none"',
            repair,
        )
        self.assertIn(
            '"task_type":"all_stock_control_analysis"',
            repair,
        )
        self.assertIn(
            '"scope":"data_interpretation_not_trade_execution"',
            repair,
        )
        self.assertIn(
            '"allowed_etf_flow_signal_values":["flat","negative","positive","unknown"]',
            repair,
        )
        self.assertNotIn("DETERMINISTIC_FACTS_JSON", repair)
        self.assertNotIn('"facts"', guided["properties"])

    def test_client_accepts_compact_model_response_and_reattaches_exact_facts(self):
        expected = {
            "facts": {
                "symbol": "AAPL",
                "as_of_date": "2026-07-31",
                "etf_relations": {"membership_count": 128},
            },
            "interpretation": {
                "price_signal": "positive",
                "etf_flow_signal": "positive",
                "etf_flow_signal_source": "constituent_etf_flow_exposure",
                "relationship": "price_flow_positive_confirmation",
                "scope": "data_interpretation_not_trade_execution",
                "task_type": "stock_constituent_flow_analysis",
            },
            "counter_evidence": ["concentration_risk"],
            "unknowns": ["future_outcome_unknown"],
            "regime": "price_flow_positive_confirmation",
            "confidence": 0.7,
            "conclusion": "현재 증거에서는 가격과 ETF Flow가 함께 확인됩니다.",
        }
        compact = {
            key: value for key, value in expected.items() if key != "facts"
        }
        payloads = []

        def transport(payload, headers, timeout):
            payloads.append(payload)
            return {
                "model": "quant-v1",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(compact, ensure_ascii=False)
                        },
                    }
                ],
                "usage": {"completion_tokens": 120},
            }

        release = ModelRelease(
            manifest_path=Path("/tmp/release.json"),
            manifest_sha256="a" * 64,
            model_id="quant-v1",
            endpoint_model="quant-v1",
            base_model="Qwen3-8B",
            adapter_root=Path("/tmp/adapter"),
            adapter_set_sha256="b" * 64,
            dataset_manifest_sha256="c" * 64,
            evaluation_sha256="d" * 64,
            raw={},
        )
        client = TrainedQuantClient(
            endpoint="http://127.0.0.1:8018/v1/chat/completions",
            release=release,
            transport=transport,
        )
        value, trace = client.complete_validated(
            system="EVIDENCE_JSON={}",
            user="Return interpretation JSON only. /no_think",
            expected_response=expected,
        )
        self.assertEqual(value["facts"], expected["facts"])
        self.assertEqual(value["conclusion"], compact["conclusion"])
        self.assertEqual(trace["contract_attempts"], 1)
        schema = payloads[0]["response_format"]["json_schema"]["schema"]
        self.assertNotIn("facts", schema["properties"])
        self.assertNotIn("facts", schema["required"])

    def test_buy_sell_opinion_language_is_preserved(self):
        for opinion in (
            "시장 참여자들이 자산을 매도하고 있음을 나타낸다.",
            "지금 매도",
            "매도하라",
            "즉시 청산",
            "매수해야 한다",
        ):
            self.assertEqual(
                judgement_prohibited_violations(
                    {"conclusion": opinion}, "2026-07-31"
                ),
                [],
            )

    def test_verified_cluster_label_substrings_are_not_forbidden(self):
        forbidden = _unsupported_cluster_terms("Consumer Cyclical")
        self.assertNotIn("경기소비재", forbidden)
        self.assertNotIn("소비재", forbidden)
        self.assertIn("금융", forbidden)

    def test_action_assessment_uses_bounded_judgement_and_five_views(self):
        captured = {}

        class FakeClient:
            def complete_messages(self, **kwargs):
                captured.update(kwargs)
                return (
                    {
                        "symbol": "AAPL",
                        "action_view": "관망",
                        "horizon": "단기",
                        "historical_pattern": "기간별 가격과 자금 흐름이 엇갈리는 과거 패턴과 유사합니다.",
                        "reason": "단기 가격 약세와 자금 흐름 강세가 충돌하여 추가 확인이 필요합니다.",
                        "supporting_evidence": "중기 가격과 전체 자금 흐름의 방향은 함께 양수로 확인됩니다.",
                        "counter_evidence": "최근 가격 약세와 상위 기여 항목의 음수 방향이 반대 근거입니다.",
                        "invalidation_condition": "기간별 가격과 자금 흐름이 같은 방향으로 재확인되면 판단을 바꿉니다.",
                    },
                    {"finish_reason": "stop"},
                )

        result = {
            "symbol": "AAPL",
            "judgement": {
                "facts": {
                    "symbol": "AAPL",
                    "as_of_date": "2026-07-31",
                    "price": {"return_20_session_pct": 1.2},
                    "etf_flow_to_constituent": {
                        "net_weighted_flow_rate_contribution_pct": 0.1,
                    },
                    "large_unused_payload": {"rows": list(range(1000))},
                },
                "interpretation": {
                    "price_signal": "positive",
                    "etf_flow_signal": "positive",
                    "etf_flow_signal_source": "constituent_etf_flow_exposure",
                    "relationship": "price_flow_positive_confirmation",
                    "scope": "data_interpretation_not_trade_execution",
                    "task_type": "stock_constituent_flow_analysis",
                },
                "counter_evidence": [],
                "unknowns": [],
                "regime": "price_flow_positive_confirmation",
                "confidence": 0.7,
                "conclusion": "현재 증거는 두 방향의 일치를 보여 줍니다.",
            },
        }
        assessment, trace = build_action_assessment(
            client=FakeClient(), result=result
        )
        self.assertEqual(len(ACTION_VIEWS), 5)
        self.assertEqual(assessment["action_view"], "관망")
        self.assertEqual(
            assessment["prompt_contract"], ACTION_PROMPT_CONTRACT
        )
        self.assertEqual(trace["contract_attempts"], 1)
        user = captured["messages"][1]["content"]
        bounded = user.split("CURRENT_JUDGEMENT=", 1)[1].split(
            "\nEXACT_SECURITY_BRIEF=", 1
        )[0]
        self.assertNotIn("large_unused_payload", bounded)
        self.assertNotIn('"facts"', bounded)
        self.assertNotIn("SUPPORT_RULE", user)
        self.assertNotIn("양수 regime만으로", user)

    def test_client_repairs_missing_regime_on_third_model_turn(self):
        expected = {
            "facts": {"symbol": "SKY", "as_of_date": "2026-07-30"},
            "interpretation": {
                "price_signal": "negative",
                "etf_flow_signal": "negative",
                "etf_flow_signal_source": "constituent_etf_flow_exposure",
                "relationship": "price_flow_negative_confirmation",
                "scope": "data_interpretation_not_trade_execution",
                "task_type": "stock_constituent_flow_analysis",
            },
            "counter_evidence": [],
            "unknowns": [],
            "regime": "price_flow_negative_confirmation",
            "confidence": 0.5,
            "conclusion": "Evidence as of 2026-07-30.",
        }
        missing_regime = json.loads(json.dumps(expected))
        missing_regime.pop("regime")
        responses = iter((missing_regime, missing_regime, expected))
        payloads = []

        def transport(payload, headers, timeout):
            payloads.append(payload)
            return {
                "model": "quant-v1",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                next(responses), ensure_ascii=False
                            )
                        },
                    }
                ],
                "usage": {"completion_tokens": 10},
            }

        release = ModelRelease(
            manifest_path=Path("/tmp/release.json"),
            manifest_sha256="a" * 64,
            model_id="quant-v1",
            endpoint_model="quant-v1",
            base_model="Qwen3-8B",
            adapter_root=Path("/tmp/adapter"),
            adapter_set_sha256="b" * 64,
            dataset_manifest_sha256="c" * 64,
            evaluation_sha256="d" * 64,
            raw={},
        )
        client = TrainedQuantClient(
            endpoint="http://127.0.0.1:8018/v1/chat/completions",
            release=release,
            transport=transport,
        )
        value, trace = client.complete_validated(
            system="EVIDENCE_JSON={}",
            user="지정된 구조로 답하라. /no_think",
            expected_response=expected,
        )
        self.assertEqual(value, expected)
        self.assertEqual(trace["contract_attempts"], 3)
        self.assertIn("missing=['regime']", trace["second_contract_error"])
        self.assertEqual(len(payloads), 3)
        for payload in payloads:
            schema = payload["response_format"]["json_schema"]["schema"]
            self.assertEqual(
                schema["properties"]["regime"]["enum"],
                ["price_flow_negative_confirmation"],
            )

    def test_client_preserves_final_invalid_json_for_failure_audit(self):
        expected = {
            "facts": {"symbol": "SKY", "as_of_date": "2026-07-30"},
            "interpretation": {
                "price_signal": "negative",
                "etf_flow_signal": "positive",
                "etf_flow_signal_source": "constituent_etf_flow_exposure",
                "relationship": "price_down_flow_in_divergence",
                "scope": "data_interpretation_not_trade_execution",
                "task_type": "stock_constituent_flow_analysis",
            },
            "counter_evidence": [],
            "unknowns": [],
            "regime": "price_down_flow_in_divergence",
            "confidence": 0.5,
            "conclusion": "Evidence as of 2026-07-30.",
        }
        responses = iter(("not json", "still not json", "final broken json"))

        def transport(payload, headers, timeout):
            return {
                "model": "quant-v1",
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": next(responses)},
                    }
                ],
                "usage": {"completion_tokens": 1400},
            }

        release = ModelRelease(
            manifest_path=Path("/tmp/release.json"),
            manifest_sha256="a" * 64,
            model_id="quant-v1",
            endpoint_model="quant-v1",
            base_model="Qwen3-8B",
            adapter_root=Path("/tmp/adapter"),
            adapter_set_sha256="b" * 64,
            dataset_manifest_sha256="c" * 64,
            evaluation_sha256="d" * 64,
            raw={},
        )
        client = TrainedQuantClient(
            endpoint="http://127.0.0.1:8018/v1/chat/completions",
            release=release,
            transport=transport,
        )
        with self.assertRaises(ModelResponseParseError) as caught:
            client.complete_validated(
                system="EVIDENCE_JSON={}",
                user="Return JSON. /no_think",
                expected_response=expected,
            )
        self.assertEqual(caught.exception.raw_content, "final broken json")
        self.assertEqual(caught.exception.trace["contract_attempts"], 3)
        self.assertEqual(caught.exception.trace["finish_reason"], "length")
        self.assertIn(
            "ModelResponseParseError",
            caught.exception.trace["final_contract_error"],
        )

    def test_queue_error_preserves_failure_trace_and_raw_response(self):
        with tempfile.TemporaryDirectory() as temporary:
            queue = RadarQueue(Path(temporary) / "queue.sqlite3")
            queue.seed(
                [
                    Candidate(
                        symbol="SKY",
                        proxy_task_type="stock_constituent_flow_analysis",
                        quality_status="pass",
                        relation_types=("fmp_etf_membership",),
                    )
                ]
            )
            queue.mark_error(
                "SKY",
                "ModelResponseParseError: invalid JSON",
                trace={
                    "contract_attempts": 3,
                    "finish_reason": "length",
                    "failed_response_text": "broken",
                },
            )
            with queue.connect() as connection:
                row = connection.execute(
                    "SELECT status,error,trace_json FROM items WHERE symbol='SKY'"
                ).fetchone()
        self.assertEqual(row["status"], "error")
        self.assertIn("invalid JSON", row["error"])
        trace = json.loads(row["trace_json"])
        self.assertEqual(trace["contract_attempts"], 3)
        self.assertEqual(trace["failed_response_text"], "broken")

    def test_symbol_guided_schema_rejects_invalid_expected_contract(self):
        with self.assertRaises(ResponseContractError):
            symbol_guided_json_schema(
                {"interpretation": {}, "regime": "not-a-regime"}
            )

    def test_aggregate_uses_all_selected_results_and_limits_leaders(self):
        rows = []
        for index in range(30):
            rows.append(
                {
                    "symbol": f"ETF{index:02d}",
                    "task_type": "etf_own_flow_analysis",
                    "judgement": {
                        "regime": "mixed_or_flat",
                        "confidence": 0.5,
                        "interpretation": {
                            "price_signal": "flat",
                            "etf_flow_signal": "flat",
                        },
                        "facts": {
                            "etf_flow": {"latest_robust_zscore": float(index)}
                        },
                    },
                }
            )
        aggregate = aggregate_judgements(rows)
        self.assertEqual(aggregate["analyzed_security_count"], 30)
        self.assertEqual(len(aggregate["etf_leaders"]), 25)
        self.assertEqual(
            len(
                aggregate["candidate_rankings"]["etfs_by_regime"][
                    "mixed_or_flat"
                ]
            ),
            10,
        )
        self.assertIn(
            "dynamically selected",
            aggregate["presentation_policy"],
        )

    def test_market_dashboard_keeps_every_deterministic_rotation_cluster(self):
        clusters = [
            {
                "integrated_cluster": f"Sector {index}",
                "integrated_state": "mixed",
                "integrated_score": float(index),
            }
            for index in range(11)
        ]
        dashboard = build_market_dashboard(
            {"analyzed_security_count": 0},
            {"integrated_rotation_clusters": clusters},
        )
        self.assertEqual(
            [row["cluster"] for row in dashboard["rotation_clusters"]],
            [row["integrated_cluster"] for row in clusters],
        )

    def test_market_synthesis_catalog_is_bounded_and_excludes_verbose_notes(self):
        verbose = "x" * (MAX_MARKET_CATALOG_CHARS * 2)
        aggregate = {
            "analyzed_security_count": 2,
            "task_type_counts": {"etf_own_flow_analysis": 1},
            "regime_counts": {"mixed_or_flat": 2},
            "price_signal_counts": {"flat": 2},
            "etf_flow_signal_counts": {"flat": 2},
            "mean_model_confidence": 0.5,
            "etf_leaders": [
                {
                    "symbol": "AAA",
                    "regime": "mixed_or_flat",
                    "confidence": 0.5,
                    "price_signal": "flat",
                    "etf_flow_signal": "flat",
                    "evidence_note": verbose,
                }
            ],
            "stock_leaders": [],
        }
        radar = {
            "release_binding": {
                "release_id": "release",
                "trade_date_us": "2026-07-27",
            },
            "master_eligibility_counts": {"eligible": 1},
            "master_flow_status_counts": {"visible": 1},
            "accumulation_clusters": [
                {
                    "rank": 1,
                    "accum_cluster": "AI",
                    "cluster_score": 99,
                    "evidence_note": verbose,
                }
            ],
            "integrated_rotation_clusters": [],
        }
        catalog = _evidence_catalog(aggregate, radar)
        encoded = json.dumps(catalog, sort_keys=True, separators=(",", ":"))
        self.assertLessEqual(len(encoded), MAX_MARKET_CATALOG_CHARS)
        self.assertNotIn(verbose, encoded)

    def test_market_contract_repair_lists_only_catalog_evidence(self):
        self.assertLess(
            MARKET_REPAIR_MAX_TOKENS,
            MARKET_SYNTHESIS_MAX_TOKENS,
        )
        repair = market_contract_repair_instruction(
            contract_error="cited unknown evidence",
            catalog={
                "aggregate.regime_counts": {"mixed": 2},
                "etf.AAA": {"symbol": "AAA"},
                "stock.AAPL": {"symbol": "AAPL"},
            },
            schema={"market_state": "mixed"},
        )
        self.assertIn(
            'ALLOWED_EVIDENCE_IDS_JSON=["aggregate.regime_counts","etf.AAA","stock.AAPL"]',
            repair,
        )
        self.assertIn(
            'MANDATORY_EVIDENCE_IDS_JSON={"confirmations":["aggregate.regime_counts","etf.AAA"],"contradictions":["stock.AAPL"]}',
            repair,
        )
        self.assertIn('ALLOWED_LEADING_ETFS_JSON=["AAA"]', repair)
        self.assertIn('ALLOWED_AFFECTED_STOCKS_JSON=["AAPL"]', repair)
        self.assertTrue(repair.endswith("/no_think"))

    def test_market_guided_schema_enums_only_catalog_ids(self):
        schema = market_guided_json_schema(
            {
                "aggregate.regime_counts": {"mixed": 1},
                "etf.AAA": {"symbol": "AAA"},
                "stock.AAPL": {"symbol": "AAPL"},
            },
            minimum_confirmations=3,
            minimum_contradictions=2,
        )
        properties = schema["properties"]
        self.assertEqual(
            properties["confirmations"]["items"]["properties"]["evidence_id"][
                "enum"
            ],
            ["aggregate.regime_counts", "etf.AAA", "stock.AAPL"],
        )
        self.assertEqual(properties["leading_etfs"]["items"]["enum"], ["AAA"])
        self.assertEqual(properties["affected_stocks"]["items"]["enum"], ["AAPL"])

    def test_market_confidence_normalizes_percent_scale_only(self):
        original = {
            "confidence": 82,
            "market_state": "mixed",
            "confirmations": [{"evidence_id": "x"}],
        }
        normalized, changed = normalize_market_synthesis_confidence(original)
        self.assertTrue(changed)
        self.assertEqual(normalized["confidence"], 0.82)
        self.assertEqual(normalized["market_state"], original["market_state"])
        self.assertEqual(normalized["confirmations"], original["confirmations"])
        self.assertEqual(original["confidence"], 82)

    def test_market_natural_language_leaves_exact_numbers_to_renderer(self):
        original = {
            "summary": "양수 77개와 음수 40개가 엇갈립니다.",
            "confirmations": [
                {"evidence_id": "x", "interpretation": "점수 82.5점입니다."}
            ],
            "contradictions": [],
            "unknowns": ["향후 5일은 미확인입니다."],
        }
        normalized, changed = strip_renderer_owned_numbers(original)
        self.assertTrue(changed)
        self.assertNotRegex(
            json.dumps(normalized, ensure_ascii=False),
            r"\d",
        )
        self.assertEqual(original["summary"], "양수 77개와 음수 40개가 엇갈립니다.")

    def test_market_synthesis_repairs_unknown_evidence_once(self):
        class Client:
            def complete(self, **kwargs):
                self.initial_max_tokens = kwargs["max_tokens"]
                return {
                    "market_state": "mixed",
                    "confidence": 0.5,
                    "summary": "initial",
                    "confirmations": [
                        {"evidence_id": "invented", "interpretation": "bad"}
                    ],
                    "contradictions": [
                        {"evidence_id": "invented", "interpretation": "bad"}
                    ],
                    "unknowns": [],
                    "leading_etfs": [],
                    "affected_stocks": [],
                    "scope": "market_and_security_analysis_not_trade_execution",
                }, {
                    "request_sha256": "a" * 64,
                    "response_sha256": "b" * 64,
                }

            def complete_messages(self, **kwargs):
                self.repair_max_tokens = kwargs["max_tokens"]
                self.repair_messages = kwargs["messages"]
                self.repair_prompt = kwargs["messages"][-1]["content"]
                return {
                    "market_state": "mixed",
                    "confidence": 0.5,
                    "summary": (
                        "시장 가격 폭은 한 방향으로 수렴하지 않고 ETF 자금 흐름과 "
                        "섹터 회전이 엇갈리며, 괴리 신호가 남아 혼조 국면의 지속성을 "
                        "더 확인해야 합니다."
                    ),
                    "confirmations": [
                        {
                            "evidence_id": "aggregate.regime_counts",
                            "interpretation": "국면 분포를 확인했습니다.",
                        }
                    ],
                    "contradictions": [
                        {
                            "evidence_id": "aggregate.price_signal_counts",
                            "interpretation": "가격 신호는 혼조입니다.",
                        }
                    ],
                    "unknowns": [],
                    "leading_etfs": [],
                    "affected_stocks": [],
                    "scope": "market_and_security_analysis_not_trade_execution",
                }, {
                    "request_sha256": "c" * 64,
                    "response_sha256": "d" * 64,
                }

        client = Client()
        aggregate = {
            "analyzed_security_count": 1,
            "task_type_counts": {"etf_own_flow_analysis": 1},
            "regime_counts": {"mixed_or_flat": 1},
            "price_signal_counts": {"flat": 1},
            "etf_flow_signal_counts": {"flat": 1},
            "mean_model_confidence": 0.5,
            "etf_leaders": [],
            "stock_leaders": [],
        }
        radar = {
            "release_binding": {
                "release_id": "release",
                "trade_date_us": "2026-07-27",
            },
            "master_eligibility_counts": {"eligible": 1},
            "master_flow_status_counts": {"visible": 1},
            "accumulation_clusters": [],
            "integrated_rotation_clusters": [],
        }
        synthesis, trace, _ = synthesize_market(
            client=client,
            as_of_date="2026-07-27",
            aggregate=aggregate,
            radar=radar,
        )
        self.assertIn("섹터 회전", synthesis["summary"])
        self.assertEqual(trace["contract_attempts"], 2)
        self.assertTrue(trace["contract_repair_applied"])
        self.assertEqual(
            client.initial_max_tokens,
            MARKET_SYNTHESIS_MAX_TOKENS,
        )
        self.assertEqual(
            client.repair_max_tokens,
            MARKET_REPAIR_MAX_TOKENS,
        )
        self.assertEqual(
            [message["role"] for message in client.repair_messages],
            ["system", "user", "user"],
        )
        self.assertIn("ALLOWED_EVIDENCE_IDS_JSON", client.repair_prompt)

    def test_market_synthesis_repairs_invalid_json_response(self):
        class Client:
            def complete(self, **kwargs):
                error = ModelResponseParseError("invalid JSON")
                error.trace = {
                    "request_sha256": "a" * 64,
                    "response_sha256": "b" * 64,
                }
                raise error

            def complete_messages(self, **kwargs):
                return {
                    "market_state": "mixed",
                    "confidence": 0.5,
                    "summary": (
                        "시장 가격 폭은 한 방향으로 수렴하지 않고 ETF 자금 흐름과 "
                        "섹터 회전이 엇갈리며, 괴리 신호가 남아 혼조 국면의 지속성을 "
                        "더 확인해야 합니다."
                    ),
                    "confirmations": [
                        {
                            "evidence_id": "aggregate.regime_counts",
                            "interpretation": "국면 분포를 확인했습니다.",
                        }
                    ],
                    "contradictions": [
                        {
                            "evidence_id": "aggregate.price_signal_counts",
                            "interpretation": "가격 신호는 혼조입니다.",
                        }
                    ],
                    "unknowns": [],
                    "leading_etfs": [],
                    "affected_stocks": [],
                    "scope": "market_and_security_analysis_not_trade_execution",
                }, {
                    "request_sha256": "c" * 64,
                    "response_sha256": "d" * 64,
                }

        synthesis, trace, _ = synthesize_market(
            client=Client(),
            as_of_date="2026-07-27",
            aggregate={
                "analyzed_security_count": 1,
                "task_type_counts": {"etf_own_flow_analysis": 1},
                "regime_counts": {"mixed_or_flat": 1},
                "price_signal_counts": {"flat": 1},
                "etf_flow_signal_counts": {"flat": 1},
                "mean_model_confidence": 0.5,
                "etf_leaders": [],
                "stock_leaders": [],
            },
            radar={
                "release_binding": {
                    "release_id": "release",
                    "trade_date_us": "2026-07-27",
                },
                "master_eligibility_counts": {"eligible": 1},
                "master_flow_status_counts": {"visible": 1},
                "accumulation_clusters": [],
                "integrated_rotation_clusters": [],
            },
        )
        self.assertIn("ETF 자금 흐름", synthesis["summary"])
        self.assertEqual(trace["contract_attempts"], 2)
        self.assertIn("ModelResponseParseError", trace["initial_contract_error"])


if __name__ == "__main__":
    unittest.main()
