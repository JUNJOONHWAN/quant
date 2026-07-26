from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from training.quant_llm.create_model_release import build_release
from training.quant_llm.evaluate_frozen_test import adapter_artifact_set, evaluate
from workflows.quant_ai_radar.etfradar_release import REQUIRED_TABLES, verify_release
from workflows.quant_ai_radar.market_report import aggregate_judgements
from workflows.quant_ai_radar.model_runtime import (
    ModelGateError,
    ModelRelease,
    ResponseContractError,
    TrainedQuantClient,
    contract_repair_instruction,
    load_model_release,
    validate_symbol_judgement,
)
from workflows.quant_ai_radar.run_queue import RadarQueue
from workflows.quant_ai_radar.universe import Candidate, scan_universe


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QuantAiRadarTest(unittest.TestCase):
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

    def test_symbol_judgement_may_interpret_but_cannot_change_facts(self):
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
                "task_type": "etf_own_flow_analysis",
            },
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
        repair = payloads[1]["messages"][-1]["content"]
        self.assertEqual(repair, contract_repair_instruction(
            expected, "trained model changed deterministic facts"
        ))
        self.assertTrue(repair.endswith("/no_think"))
        self.assertIn(
            '"as_of_date":"2025-11-03"',
            repair,
        )

    def test_etfradar_release_requires_every_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            release = Path(temporary) / "release"
            release.mkdir()
            release.joinpath("COMPLETE").touch()
            tables = []
            for name in REQUIRED_TABLES:
                table_dir = release / "tables" / name
                table_dir.mkdir(parents=True)
                files = []
                for filename, payload in (
                    ("data.parquet", b"parquet"),
                    ("meta.json", b'{"row_count":1}'),
                    ("preview.csv", b"ticker\nAAA\n"),
                    ("schema.json", b"{}"),
                ):
                    path = table_dir / filename
                    path.write_bytes(payload)
                    files.append(
                        {
                            "relative_path": filename,
                            "sha256": sha256(path),
                            "bytes": len(payload),
                        }
                    )
                tables.append({"sheet_name": name, "row_count": 1, "files": files})
            release.joinpath("release_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "etfradar-release-v1",
                        "release_id": "test",
                        "trade_date_us": "2026-07-16",
                        "complete": True,
                        "tables": tables,
                    }
                ),
                encoding="utf-8",
            )
            binding = verify_release(release)
            self.assertTrue(binding["complete"])
            release.joinpath("tables", REQUIRED_TABLES[0], "preview.csv").write_text(
                "changed", encoding="utf-8"
            )
            with self.assertRaises(Exception):
                verify_release(release)

    def test_aggregate_uses_all_results_and_only_limits_display(self):
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
        self.assertIn("display-only", aggregate["presentation_policy"])


if __name__ == "__main__":
    unittest.main()
