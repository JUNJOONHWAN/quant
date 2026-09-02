from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

from quant_dataset.point_in_time import ETF_CONSTITUENT_POLICY_ID, ETF_FLOW_POLICY_ID
from quant_dataset.etf_flow_exposure import (
    ETF_CONSTITUENT_FLOW_POLICY_ID,
    build_constituent_flow_exposure,
)
from training.quant_llm.build_sft_dataset import (
    build_dataset,
    build_example,
    packet_eligibility,
    validate_packet,
)
from training.quant_llm.build_balanced_training_set import _select
from training.quant_llm.audit_token_lengths import _percentile, _token_count
from training.quant_llm.export_packet_shards import _monthly_ranges
from training.quant_llm.select_training_pairs import _consider, _pair_hash, _proxy_task
from training.quant_llm.materialization_status import inspect_status
from training.quant_llm.service_gate import _mem_available_mib
from training.quant_llm.build_selected_sft_from_db import (
    _seed_state_from_completed_output,
    _verify_seed_pair_subset,
)
from training.quant_llm.complete_training_release import (
    ENDPOINT_MODEL,
    EXPECTED_FINAL_CHECKPOINT_STEP,
    VLLM_CONTAINER,
    build_checkpoint_permission_command,
    build_vllm_command,
    candidate_adapter_sha,
    container_matches_candidate,
    discover_final_adapter,
    ensure_candidate_artifacts_readable,
)
from training.quant_llm.collect_frozen_predictions import (
    PredictionBindingError,
    completed_ids,
    request_prediction,
    validated_completed_ids,
)
from training.quant_llm.evaluate_frozen_test import read_predictions
from training.quant_llm.validate_sft_dataset import validate_dataset
from workflows.quant_ai_radar.model_runtime import ResponseContractError, canonical_json


def packet(as_of_date: str, packet_id: str) -> dict:
    available = as_of_date
    end = date.fromisoformat(as_of_date)
    history = [
        {
            "trade_date": (end - timedelta(days=offset)).isoformat(),
            "sources": [
                {
                    "source": "fmp",
                    "close": 400.0 + (4 - offset),
                    "adjusted_close": 400.0 + (4 - offset),
                    "volume": 1_000_000.0,
                }
            ],
        }
        for offset in range(4, -1, -1)
    ]
    document = {
        "schema_version": "quant.analysis_packet.v3",
        "packet_id": packet_id,
        "symbol": "QQQ",
        "as_of_date": as_of_date,
        "history": history,
        "etf_flow": {
            "availability_policy": {"policy_id": ETF_FLOW_POLICY_ID},
            "observations": [
                {
                    "effective_date": (end - timedelta(days=2)).isoformat(),
                    "processed_date": (end - timedelta(days=1)).isoformat(),
                    "available_at_date": available,
                    "training_available_session_date": available,
                    "training_availability_policy_id": ETF_FLOW_POLICY_ID,
                    "fund_flow": 100.0,
                    "assets": 10_000.0,
                    "currency": "USD",
                }
            ],
        },
        "etf_constituents": {
            "availability_policy": {"policy_id": ETF_CONSTITUENT_POLICY_ID},
            "constituents": [],
            "etf_memberships": [],
        },
        "etf_flow_to_constituent": {
            "policy_id": ETF_CONSTITUENT_FLOW_POLICY_ID,
            "eligible_etf_count": 0,
            "excluded_etf_count": 0,
            "rows": [],
        },
        "quality": {"status": "pass", "metrics": {}, "reasons": []},
        "provenance": {"raw_artifacts": []},
    }
    return document


class QuantLlmTrainingTest(unittest.TestCase):
    def test_completion_lane_selects_only_complete_final_adapter(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            early = root / "epoch_0_step_499" / "model"
            early.mkdir(parents=True)
            early.joinpath("adapter_model.safetensors").write_bytes(b"early")
            early.joinpath("adapter_config.json").write_text("{}", encoding="utf-8")
            final = root / "epoch_1_step_14027" / "model"
            final.mkdir(parents=True)
            final.joinpath("adapter_model.safetensors").write_bytes(b"final")
            final.joinpath("adapter_config.json").write_text("{}", encoding="utf-8")
            incomplete = root / "epoch_2_step_15000" / "model"
            incomplete.mkdir(parents=True)
            incomplete.joinpath("adapter_model.safetensors").touch()
            incomplete.joinpath("adapter_config.json").write_text("{}", encoding="utf-8")

            candidate = discover_final_adapter(root, EXPECTED_FINAL_CHECKPOINT_STEP)
            self.assertIsNotNone(candidate)
            assert candidate is not None
            self.assertEqual(candidate.step, EXPECTED_FINAL_CHECKPOINT_STEP)
            self.assertEqual(candidate.weights.read_bytes(), b"final")
            self.assertIsNone(discover_final_adapter(root, 14028))

    def test_completion_lane_builds_isolated_exact_lora_server(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "epoch_1_step_14027" / "model"
            model.mkdir(parents=True)
            model.joinpath("adapter_model.safetensors").write_bytes(b"final")
            model.joinpath("adapter_config.json").write_text("{}", encoding="utf-8")
            candidate = discover_final_adapter(root, EXPECTED_FINAL_CHECKPOINT_STEP)
            assert candidate is not None
            command = build_vllm_command(candidate)
            self.assertIn(VLLM_CONTAINER, command)
            self.assertIn("127.0.0.1:8018:8000", command)
            self.assertIn("--enable-lora", command)
            self.assertIn(f"{ENDPOINT_MODEL}=/adapter", command)
            self.assertIn(f"{candidate.model_dir}:/adapter:ro", command)

            inspection = {
                "Config": {
                    "Image": "vllm/vllm-openai:v0.25.0",
                    "Cmd": ["--lora-modules", f"{ENDPOINT_MODEL}=/adapter"],
                },
                "Mounts": [
                    {"Destination": "/model", "Source": "/home/zooh/models/Qwen3-8B-bf16"},
                    {"Destination": "/adapter", "Source": str(candidate.model_dir)},
                ],
            }
            self.assertTrue(container_matches_candidate(inspection, candidate))
            inspection["Mounts"][1]["Source"] = str(root / "old-adapter")
            self.assertFalse(container_matches_candidate(inspection, candidate))

    def test_completion_lane_repairs_root_only_adapter_permissions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "epoch_1_step_14027" / "model"
            model.mkdir(parents=True)
            model.joinpath("adapter_model.safetensors").write_bytes(b"final")
            model.joinpath("adapter_config.json").write_text("{}", encoding="utf-8")
            candidate = discover_final_adapter(root, EXPECTED_FINAL_CHECKPOINT_STEP)
            assert candidate is not None

            expected = build_checkpoint_permission_command(
                candidate, [candidate.weights]
            )
            self.assertEqual(
                expected[:7],
                [
                    "docker",
                    "run",
                    "--rm",
                    "--user",
                    "0:0",
                    "-v",
                    f"{candidate.checkpoint_dir}:/checkpoint",
                ],
            )
            self.assertIn("/checkpoint/model/adapter_model.safetensors", expected)

            with mock.patch(
                "training.quant_llm.complete_training_release.os.access",
                side_effect=[False, True, True, True],
            ), mock.patch(
                "training.quant_llm.complete_training_release.run"
            ) as run_mock:
                ensure_candidate_artifacts_readable(candidate)
            run_mock.assert_called_once_with(expected)

    def test_completion_lane_fails_closed_when_adapter_stays_unreadable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "epoch_1_step_14027" / "model"
            model.mkdir(parents=True)
            model.joinpath("adapter_model.safetensors").write_bytes(b"final")
            model.joinpath("adapter_config.json").write_text("{}", encoding="utf-8")
            candidate = discover_final_adapter(root, EXPECTED_FINAL_CHECKPOINT_STEP)
            assert candidate is not None

            with mock.patch(
                "training.quant_llm.complete_training_release.os.access",
                return_value=False,
            ), mock.patch("training.quant_llm.complete_training_release.run"):
                with self.assertRaisesRegex(PermissionError, "remain unreadable"):
                    ensure_candidate_artifacts_readable(candidate)

    def test_frozen_predictions_are_bound_to_adapter_and_test_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "epoch_1_step_14027" / "model"
            model.mkdir(parents=True)
            model.joinpath("adapter_model.safetensors").write_bytes(b"final")
            model.joinpath("adapter_config.json").write_text("{}", encoding="utf-8")
            candidate = discover_final_adapter(root, EXPECTED_FINAL_CHECKPOINT_STEP)
            assert candidate is not None
            adapter_sha = candidate_adapter_sha(candidate)
            test_sha = "b" * 64
            predictions = root / "predictions.jsonl"
            response = {"facts": {}}
            predictions.write_text(
                json.dumps(
                    {
                        "example_id": "one",
                        "endpoint_model": ENDPOINT_MODEL,
                        "adapter_set_sha256": adapter_sha,
                        "frozen_test_sha256": test_sha,
                        "response": response,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(
                completed_ids(predictions, ENDPOINT_MODEL, adapter_sha, test_sha),
                {"one"},
            )
            self.assertEqual(
                read_predictions(predictions, ENDPOINT_MODEL, adapter_sha, test_sha),
                {"one": response},
            )
            with self.assertRaises(PredictionBindingError):
                completed_ids(predictions, ENDPOINT_MODEL, "c" * 64, test_sha)
            with self.assertRaisesRegex(ValueError, "different frozen test"):
                read_predictions(predictions, ENDPOINT_MODEL, adapter_sha, "d" * 64)

    def test_resume_quarantines_only_contract_invalid_predictions(self):
        expected = {
            "facts": {"symbol": "AAA", "as_of_date": "2025-11-03"},
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
        endpoint = ENDPOINT_MODEL
        adapter_sha = "a" * 64
        test_sha = "b" * 64
        with tempfile.TemporaryDirectory() as temporary:
            predictions = Path(temporary) / "predictions.jsonl"
            rows = [
                {
                    "example_id": "valid",
                    "endpoint_model": endpoint,
                    "adapter_set_sha256": adapter_sha,
                    "frozen_test_sha256": test_sha,
                    "response": expected,
                },
                {
                    "example_id": "invalid",
                    "endpoint_model": endpoint,
                    "adapter_set_sha256": adapter_sha,
                    "frozen_test_sha256": test_sha,
                    "response": invalid,
                },
            ]
            predictions.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            predictions.with_suffix(".jsonl.state.json").write_text(
                "{}\n", encoding="utf-8"
            )
            done, invalid_ids, archived = validated_completed_ids(
                predictions,
                endpoint,
                adapter_sha,
                test_sha,
                {"valid": expected, "invalid": expected},
            )
            self.assertEqual(done, {"valid"})
            self.assertEqual(invalid_ids, ["invalid"])
            self.assertIsNotNone(archived)
            assert archived is not None
            self.assertTrue(archived.is_file())
            self.assertTrue(
                archived.with_suffix(archived.suffix + ".state.json").is_file()
            )
            retained = [json.loads(line) for line in predictions.read_text().splitlines()]
            self.assertEqual([row["example_id"] for row in retained], ["valid"])

    def test_collector_preserves_nonprohibited_error_but_blocks_future_date(self):
        expected = {
            "facts": {"symbol": "AAA", "as_of_date": "2025-11-03"},
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
        malformed = json.loads(json.dumps(expected))
        malformed["quality_status"] = "unexpected"
        repair_failed = {
            "facts": expected["facts"],
            "interpretation": "invalid",
            "counter_evidence": [],
            "unknowns": [],
            "regime": "insufficient_joint_evidence",
            "confidence": 0.5,
            "conclusion": "Evidence as of 2025-11-03.",
        }
        traces = [
            {
                "request_sha256": "a" * 64,
                "response_sha256": "b" * 64,
                "finish_reason": "stop",
            },
            {
                "request_sha256": "c" * 64,
                "response_sha256": "d" * 64,
                "finish_reason": "stop",
            },
        ]
        with mock.patch(
            "training.quant_llm.collect_frozen_predictions.request_messages",
            side_effect=[
                (canonical_json(malformed), traces[0]),
                (canonical_json(repair_failed), traces[1]),
            ],
        ):
            response, trace = request_prediction(
                endpoint="http://127.0.0.1:8018/v1/chat/completions",
                endpoint_model=ENDPOINT_MODEL,
                context="EVIDENCE_JSON={}",
                instruction="분석하라. /no_think",
                expected_response=expected,
                token=None,
                timeout=180,
                max_tokens=1400,
            )
        self.assertEqual(response, malformed)
        self.assertTrue(trace["contract_repair_failed_preserved_for_evaluation"])
        self.assertEqual(trace["preserved_response_source"], "initial")

        with mock.patch(
            "training.quant_llm.collect_frozen_predictions.request_messages",
            side_effect=[
                ("not-json", traces[0]),
                (canonical_json(repair_failed), traces[1]),
            ],
        ):
            response, trace = request_prediction(
                endpoint="http://127.0.0.1:8018/v1/chat/completions",
                endpoint_model=ENDPOINT_MODEL,
                context="EVIDENCE_JSON={}",
                instruction="분석하라. /no_think",
                expected_response=expected,
                token=None,
                timeout=180,
                max_tokens=1400,
            )
        self.assertEqual(response, repair_failed)
        self.assertTrue(trace["contract_repair_failed_preserved_for_evaluation"])
        self.assertEqual(trace["preserved_response_source"], "repair")

        future = json.loads(json.dumps(expected))
        future["facts"]["as_of_date"] = "2025-11-13"
        with mock.patch(
            "training.quant_llm.collect_frozen_predictions.request_messages",
            side_effect=[
                (canonical_json(future), traces[0]),
                (canonical_json(repair_failed), traces[1]),
            ],
        ):
            with self.assertRaisesRegex(
                ResponseContractError, "release-blocking response"
            ):
                request_prediction(
                    endpoint="http://127.0.0.1:8018/v1/chat/completions",
                    endpoint_model=ENDPOINT_MODEL,
                    context="EVIDENCE_JSON={}",
                    instruction="분석하라. /no_think",
                    expected_response=expected,
                    token=None,
                    timeout=180,
                    max_tokens=1400,
                )

    def test_nemo_configs_bind_explicit_train_and_validation_splits(self):
        config_root = Path(__file__).parents[1] / "training" / "quant_llm" / "configs"
        for name in ("qwen3_8b_lora_spark.yaml", "qwen3_8b_lora_smoke_spark.yaml"):
            text = (config_root / name).read_text(encoding="utf-8")
            self.assertRegex(text, r"(?ms)^dataset:.*?^  split: train$")
            self.assertRegex(text, r"(?ms)^validation_dataset:.*?^  split: validation$")

    def test_memory_gate_reads_proc_meminfo_without_login_shell(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "meminfo"
            path.write_text(
                "MemTotal:       131072000 kB\nMemAvailable:   50331648 kB\n",
                encoding="utf-8",
            )
            self.assertEqual(_mem_available_mib(path), 49152)

    def test_completed_state_can_seed_an_unchanged_pair_extension(self):
        base_pair = {
            "pair_hash": "a" * 64,
            "symbol": "QQQ",
            "as_of_date": "2023-01-03",
            "split": "train",
            "proxy_task_type": "etf_own_flow_analysis",
        }
        extra_pair = {
            "pair_hash": "b" * 64,
            "symbol": "SPY",
            "as_of_date": "2023-01-04",
            "split": "train",
            "proxy_task_type": "etf_own_flow_analysis",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seed_root = root / "seed"
            seed_root.mkdir()
            seed_pairs = root / "seed_pairs.jsonl"
            seed_pairs.write_text(json.dumps(base_pair) + "\n", encoding="utf-8")
            extended_pairs = root / "extended_pairs.jsonl"
            extended_pairs.write_text(
                json.dumps(base_pair) + "\n" + json.dumps(extra_pair) + "\n",
                encoding="utf-8",
            )
            split_file = seed_root / "train.jsonl"
            split_file.write_text('{"example_id":"e1"}\n', encoding="utf-8")
            seed_manifest = {
                "split_contract": {"train": {"start": "2023-01-01"}},
                "materialization_contract_sha256": "old-contract",
                "input_pair_selection": {
                    "pairs": str(seed_pairs),
                    "pairs_sha256": hashlib.sha256(seed_pairs.read_bytes()).hexdigest(),
                    "selected_pairs": 1,
                },
                "files": {
                    "train": {
                        "filename": "train.jsonl",
                        "rows": 1,
                        "sha256": hashlib.sha256(split_file.read_bytes()).hexdigest(),
                    }
                },
            }
            (seed_root / "manifest.json").write_text(
                json.dumps(seed_manifest), encoding="utf-8"
            )
            state = sqlite3.connect(seed_root / "materialization_state.sqlite3")
            state.executescript(
                """
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE pair_results (
                    pair_hash TEXT PRIMARY KEY, symbol TEXT NOT NULL,
                    as_of_date TEXT NOT NULL, declared_split TEXT NOT NULL,
                    proxy_task_type TEXT NOT NULL, status TEXT NOT NULL,
                    actual_task_type TEXT, example_id TEXT, reasons_json TEXT NOT NULL
                );
                CREATE TABLE examples (
                    example_id TEXT PRIMARY KEY, split TEXT NOT NULL,
                    task_type TEXT NOT NULL, packet_content_sha256 TEXT UNIQUE NOT NULL,
                    encoded_json TEXT NOT NULL
                );
                INSERT INTO metadata VALUES ('contract_sha256','old-contract');
                INSERT INTO pair_results VALUES
                    ('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
                     'QQQ','2023-01-03','train','etf_own_flow_analysis','eligible',
                     'etf_own_flow_analysis','e1','[]');
                INSERT INTO examples VALUES
                    ('e1','train','etf_own_flow_analysis','packet-sha','{}');
                """
            )
            state.commit()
            state.close()
            extended_manifest = root / "extended_manifest.json"
            extended_manifest.write_text(
                json.dumps({"split_contract": seed_manifest["split_contract"]}),
                encoding="utf-8",
            )
            destination = root / "extended_state.sqlite3"
            result = _seed_state_from_completed_output(
                seed_output_root=seed_root,
                extended_pair_manifest_path=extended_manifest,
                extended_pairs_path=extended_pairs,
                destination_state_path=destination,
                contract_sha="new-contract",
            )
            self.assertEqual(result["seed_pairs"], 1)
            self.assertEqual(result["additional_pairs"], 1)
            copied = sqlite3.connect(destination)
            self.assertEqual(
                copied.execute(
                    "SELECT value FROM metadata WHERE key='contract_sha256'"
                ).fetchone()[0],
                "new-contract",
            )
            self.assertEqual(copied.execute("SELECT COUNT(*) FROM pair_results").fetchone()[0], 1)
            copied.close()

    def test_extended_pair_selection_preserves_sealed_seed_rows(self):
        seed_rows = [
            {
                "pair_hash": "a" * 64,
                "symbol": "QQQ",
                "as_of_date": "2023-01-03",
                "split": "train",
                "proxy_task_type": "etf_own_flow_analysis",
            },
            {
                "pair_hash": "b" * 64,
                "symbol": "AAPL",
                "as_of_date": "2023-01-03",
                "split": "train",
                "proxy_task_type": "stock_constituent_flow_analysis",
            },
        ]
        extra = {
            "pair_hash": "c" * 64,
            "symbol": "SPY",
            "as_of_date": "2023-01-04",
            "split": "train",
            "proxy_task_type": "etf_own_flow_analysis",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seed = root / "seed.jsonl"
            extended = root / "extended.jsonl"
            seed.write_text(
                "".join(json.dumps(row) + "\n" for row in seed_rows),
                encoding="utf-8",
            )
            extended.write_text(
                "".join(json.dumps(row) + "\n" for row in seed_rows + [extra]),
                encoding="utf-8",
            )
            result = _verify_seed_pair_subset(seed, extended)
            self.assertEqual(result["matched_unchanged_seed_pairs"], 2)
            self.assertEqual(result["additional_pairs"], 1)

            changed = [dict(seed_rows[0], symbol="CHANGED"), seed_rows[1], extra]
            extended.write_text(
                "".join(json.dumps(row) + "\n" for row in changed),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "changed sealed seed pair"):
                _verify_seed_pair_subset(seed, extended)

    def test_materialization_status_reports_exact_progress(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            connection = sqlite3.connect(root / "materialization_state.sqlite3")
            connection.executescript(
                """
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE pair_results (
                    pair_hash TEXT PRIMARY KEY, symbol TEXT, as_of_date TEXT,
                    declared_split TEXT, proxy_task_type TEXT, status TEXT,
                    actual_task_type TEXT, example_id TEXT, reasons_json TEXT
                );
                CREATE TABLE examples (
                    example_id TEXT PRIMARY KEY, split TEXT, task_type TEXT,
                    packet_content_sha256 TEXT UNIQUE, encoded_json TEXT
                );
                INSERT INTO metadata VALUES ('contract_sha256','abc');
                INSERT INTO metadata VALUES ('expected_pairs','10');
                INSERT INTO metadata VALUES ('seed_pair_count','1');
                INSERT INTO metadata VALUES ('run_status','running');
                INSERT INTO pair_results VALUES
                    ('p1','QQQ','2023-01-03','train','etf_own_flow_analysis',
                     'eligible','etf_own_flow_analysis','e1','[]'),
                    ('p2','BAD','2023-01-03','train','all_stock_control_analysis',
                     'excluded',NULL,NULL,'["bad"]');
                INSERT INTO examples VALUES
                    ('e1','train','etf_own_flow_analysis','sha','{}');
                """
            )
            connection.commit()
            connection.close()
            result = inspect_status(root)
            self.assertEqual(result["run_status"], "running")
            self.assertEqual(result["expected_pairs"], 10)
            self.assertEqual(result["processed_pairs"], 2)
            self.assertEqual(result["remaining_pairs"], 8)
            self.assertEqual(result["progress_percent"], 20.0)
            self.assertEqual(result["extension_expected_pairs"], 9)
            self.assertEqual(result["extension_processed_pairs"], 1)
            self.assertEqual(result["extension_progress_percent"], 11.1111)
            self.assertEqual(result["pair_status_counts"], {"eligible": 1, "excluded": 1})

    def test_token_audit_percentiles_are_deterministic(self):
        values = [1, 2, 3, 4, 5]
        self.assertEqual(_percentile(values, 0.50), 3)
        self.assertEqual(_percentile(values, 0.99), 5)
        self.assertEqual(_token_count({"input_ids": [1, 2, 3]}), 3)

    def test_monthly_packet_ranges_use_observed_sessions(self):
        sessions = ("2023-01-30", "2023-01-31", "2023-02-01", "2023-02-03")
        self.assertEqual(
            _monthly_ranges(sessions, "2023-01-31", "2023-02-02"),
            [("2023-01-31", "2023-01-31"), ("2023-02-01", "2023-02-01")],
        )

    def test_training_pair_proxy_uses_only_historical_dates(self):
        flow_first = {"QQQ": "2020-01-03"}
        membership_first = {"AAPL": "2019-05-01"}
        self.assertEqual(
            _proxy_task("QQQ", "2020-01-02", flow_first, membership_first),
            "all_stock_control_analysis",
        )
        self.assertEqual(
            _proxy_task("QQQ", "2020-01-03", flow_first, membership_first),
            "etf_own_flow_analysis",
        )
        self.assertEqual(
            _proxy_task("AAPL", "2020-01-03", flow_first, membership_first),
            "stock_constituent_flow_analysis",
        )
        self.assertEqual(
            _pair_hash("train", "all_stock_control_analysis", "HMBL", "2023-06-30"),
            _pair_hash("train", "all_stock_control_analysis", "HMBL", "2023-06-30"),
        )
        heap = []
        _consider(heap, 1, ("f" * 64, "ZZZ", "2023-06-30", "train", "all_stock_control_analysis"))
        _consider(heap, 1, ("0" * 64, "AAA", "2023-06-30", "train", "all_stock_control_analysis"))
        self.assertEqual(heap[0][1], "0" * 64)

    def test_v1_packet_is_rejected(self):
        document = packet("2023-12-27", "v1")
        document["schema_version"] = "quant.analysis_packet.v1"
        with self.assertRaisesRegex(ValueError, "v3"):
            validate_packet(document)

    def test_future_flow_visibility_is_rejected(self):
        document = packet("2023-12-27", "future")
        row = document["etf_flow"]["observations"][0]
        row["available_at_date"] = "2023-12-28"
        row["training_available_session_date"] = "2023-12-28"
        with self.assertRaisesRegex(ValueError, "not available"):
            build_example(document)

    def test_future_constituent_exposure_dates_are_rejected(self):
        document = packet("2023-12-27", "future-exposure")
        document["etf_flow_to_constituent"] = {
            "policy_id": ETF_CONSTITUENT_FLOW_POLICY_ID,
            "eligible_etf_count": 1,
            "excluded_etf_count": 0,
            "rows": [
                {
                    "membership_effective_date": "2023-10-31",
                    "membership_available_date": "2023-11-01",
                    "flow_effective_date": "2023-12-28",
                    "flow_processed_date": "2023-12-27",
                    "flow_training_available_session_date": "2023-12-27",
                    "flow_availability_policy_id": ETF_FLOW_POLICY_ID,
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "effective_date exceeds"):
            build_example(document)

    def test_prompt_and_target_use_same_declared_exposure_precision(self):
        document = packet("2023-12-27", "exposure-precision")
        document["etf_flow_to_constituent"][
            "net_weighted_flow_rate_contribution_pct"
        ] = 0.0000988
        example = build_example(document)
        response = json.loads(example["response"])
        evidence = json.loads(example["context"].split("EVIDENCE_JSON=", 1)[1])

        expected = 0.000099
        self.assertEqual(
            response["facts"]["etf_flow_to_constituent"][
                "net_weighted_flow_rate_contribution_pct"
            ],
            expected,
        )
        self.assertEqual(
            evidence["etf_flow_to_constituent"]["summary"][
                "net_weighted_flow_rate_contribution_pct"
            ],
            expected,
        )

    def test_builds_purged_splits_and_validates_hashes(self):
        sessions = (
            "2023-12-27",
            "2023-12-28",
            "2023-12-29",
            "2024-01-02",
            "2024-01-03",
            "2024-12-27",
            "2024-12-30",
            "2024-12-31",
            "2025-01-02",
        )
        documents = [
            packet("2023-12-27", "train"),
            packet("2023-12-28", "purged-train"),
            packet("2024-01-02", "validation"),
            packet("2024-12-30", "purged-validation"),
            packet("2025-01-02", "test"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packets = root / "packets.jsonl"
            packets.write_text(
                "".join(json.dumps(item) + "\n" for item in documents), encoding="utf-8"
            )
            output = root / "dataset"
            manifest = build_dataset(
                [packets],
                output,
                sessions,
                embargo_sessions=2,
                min_etf_observed_sessions=1,
                min_etf_nonzero_volume_ratio=1.0,
                min_etf_median_dollar_volume=1.0,
            )
            self.assertEqual(manifest["purged_embargo_or_out_of_range_rows"], 2)
            self.assertEqual(
                {key: value["rows"] for key, value in manifest["files"].items()},
                {"train": 1, "validation": 1, "test": 1},
            )
            result = validate_dataset(output)
            self.assertTrue(result["ok"], result)

    def test_zero_volume_etf_is_excluded_without_current_active_list(self):
        document = packet("2025-01-02", "illiquid")
        document["history"][-1]["sources"][0]["volume"] = 0
        result = packet_eligibility(
            document,
            min_etf_observed_sessions=1,
            min_etf_nonzero_volume_ratio=1.0,
            min_etf_median_dollar_volume=1.0,
        )
        self.assertFalse(result["eligible"])
        self.assertIn("security_zero_or_missing_latest_volume", result["reasons"])
        self.assertIn("never filter historical rows", result["delisting_policy"])

    def test_duplicate_membership_positions_apply_etf_flow_once(self):
        memberships = [
            {
                "etf_ticker": "QQQ",
                "effective_date": "2024-03-31",
                "available_date": "2024-05-15",
                "weight_percent": 10.0,
                "direct_equity_proxy_eligible": True,
                "direct_equity_proxy_reasons": [],
            },
            {
                "etf_ticker": "QQQ",
                "effective_date": "2024-03-31",
                "available_date": "2024-05-15",
                "weight_percent": 2.0,
                "direct_equity_proxy_eligible": True,
                "direct_equity_proxy_reasons": [],
            },
        ]
        flow_packets = {
            "QQQ": {
                "latest": {
                    "effective_date": "2024-05-10",
                    "processed_date": "2024-05-10",
                    "training_available_session_date": "2024-05-14",
                    "training_availability_policy_id": ETF_FLOW_POLICY_ID,
                    "fund_flow": 1000.0,
                    "nav": 100.0,
                    "shares_outstanding": 1000.0,
                    "currency": None,
                }
            }
        }
        result = build_constituent_flow_exposure(
            "AAPL", "2024-05-15", memberships, flow_packets
        )
        self.assertEqual(result["eligible_etf_count"], 1)
        self.assertEqual(result["rows"][0]["source_position_count"], 2)
        self.assertEqual(result["rows"][0]["membership_weight_percent"], 12.0)
        self.assertEqual(result["rows"][0]["allocated_flow_reported_units"], 120.0)

    def test_constituent_flow_excludes_stale_and_implausible_latest_rows(self):
        memberships = [
            {
                "etf_ticker": ticker,
                "effective_date": "2026-06-30",
                "available_date": "2026-07-01",
                "weight_percent": 5.0,
                "direct_equity_proxy_eligible": True,
                "direct_equity_proxy_reasons": [],
            }
            for ticker in ("STALE", "EXTREME", "CURRENT")
        ]
        packets = {
            "STALE": {
                "latest": {
                    "effective_date": "2024-03-20",
                    "processed_date": "2024-03-20",
                    "training_available_session_date": "2024-03-22",
                    "training_availability_policy_id": ETF_FLOW_POLICY_ID,
                    "fund_flow": 1_000.0,
                    "nav": 100.0,
                    "shares_outstanding": 1_000.0,
                }
            },
            "EXTREME": {
                "latest": {
                    "effective_date": "2026-07-27",
                    "processed_date": "2026-07-27",
                    "training_available_session_date": "2026-07-29",
                    "training_availability_policy_id": ETF_FLOW_POLICY_ID,
                    "fund_flow": -2_000_000.0,
                    "nav": 10.0,
                    "shares_outstanding": 10_000.0,
                }
            },
            "CURRENT": {
                "latest": {
                    "effective_date": "2026-07-27",
                    "processed_date": "2026-07-27",
                    "training_available_session_date": "2026-07-29",
                    "training_availability_policy_id": ETF_FLOW_POLICY_ID,
                    "fund_flow": 10_000.0,
                    "nav": 10.0,
                    "shares_outstanding": 10_000.0,
                }
            },
        }
        result = build_constituent_flow_exposure(
            "AAPL", "2026-07-29", memberships, packets
        )
        self.assertEqual(result["eligible_etf_count"], 1)
        self.assertEqual(result["rows"][0]["etf_ticker"], "CURRENT")
        self.assertEqual(
            result["exclusion_counts"]["etf_flow_stale_latest_observation"],
            1,
        )
        self.assertEqual(
            result["exclusion_counts"][
                "etf_flow_rate_outside_plausibility_gate"
            ],
            1,
        )

    def test_balanced_selector_keeps_each_task_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "candidate.jsonl"
            rows = []
            for task_index, task in enumerate(
                (
                    "etf_own_flow_analysis",
                    "stock_constituent_flow_analysis",
                    "all_stock_control_analysis",
                )
            ):
                for row_index in range(2):
                    example_id = hashlib.sha256(
                        "{}:{}".format(task, row_index).encode("utf-8")
                    ).hexdigest()
                    rows.append(
                        {
                            "example_id": example_id,
                            "metadata": {"task_type": task},
                        }
                    )
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            selected, candidates, counts = _select(
                path,
                {
                    "etf_own_flow_analysis": 1,
                    "stock_constituent_flow_analysis": 1,
                    "all_stock_control_analysis": 1,
                },
            )
            self.assertEqual(len(selected), 3)
            self.assertTrue(all(value == 2 for value in candidates.values()))
            self.assertTrue(all(value == 1 for value in counts.values()))


if __name__ == "__main__":
    unittest.main()
