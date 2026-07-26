from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from .cli import _load_completed_receipt, _operations_request_workspace
from .manager import AppManager, AppManagerError, SCHEMA


class AppManagerContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.home = Path(self.temp.name)
        self.script = self.home / "ok.py"
        self.script.write_text("print('ok')\n", encoding="utf-8")
        self.manager = AppManager(home=self.home)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def manifest(self, **execution) -> dict:
        return {
            "schema": SCHEMA,
            "app_id": "test-app",
            "name": "Test App",
            "managed": True,
            "runtime": {
                "kind": "script",
                "entrypoint": str(self.script),
                "workdir": str(self.home),
                "timeout_seconds": 30,
            },
            "execution": execution,
            "capabilities": {"required": []},
            "final_gates": [{"type": "exit_code", "equals": 0}],
        }

    def install(self, payload: dict) -> None:
        path = self.manager.apps_dir / "test-app/app.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_default_dry_run_requires_operations_worker(self) -> None:
        self.install(
            self.manifest(
                default_worker="hermes-worker-general",
                bypass_operations_worker=False,
                worker_skills=[],
            )
        )
        with self.assertRaisesRegex(
            AppManagerError,
            "live Operations Role Shell worker",
        ):
            self.manager.run("test-app", dry_run=True)

    def test_completed_app_receipt_survives_worker_pid_reap(self) -> None:
        request_id = "request-1"
        task_id = "t_worker"
        receipt_dir = self.manager.runs_dir / "test-app"
        receipt_dir.mkdir(parents=True, exist_ok=True)
        (receipt_dir / f"{request_id}.json").write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "exit_code": 0,
                    "operations_worker_task_id": task_id,
                    "run_id": "run-1",
                }
            ),
            encoding="utf-8",
        )
        receipt = _load_completed_receipt(
            self.manager,
            app_id="test-app",
            request_id=request_id,
            task_id=task_id,
            expected_status="PASS",
        )
        self.assertEqual(receipt["run_id"], "run-1")

    def test_operations_request_workspace_isolated_from_runtime_workdir(self) -> None:
        live_checkout = self.home / "hermes-agent"
        live_checkout.mkdir()
        workspace = _operations_request_workspace(self.manager, "test-app")
        self.assertTrue(workspace.is_dir())
        self.assertEqual(
            workspace,
            self.manager.root / "request-workspaces" / "test-app",
        )
        self.assertNotEqual(workspace, self.home)
        self.assertNotEqual(workspace, live_checkout)

    def test_operations_worker_calls_manager_without_worker_dispatch(self) -> None:
        self.install(
            self.manifest(
                default_worker="hermes-worker-general",
                bypass_operations_worker=False,
                worker_skills=[],
            )
        )
        context = SimpleNamespace(
            shell_key="operations",
            shell_id="shell_operations_v4",
            executor_id="executor_hermes_worker_general",
            task_id="t_operations",
            run_id=42,
        )
        with patch.object(
            self.manager,
            "_bound_operations_context",
            return_value=context,
        ):
            receipt = self.manager.run(
                "test-app",
                request_id="c" * 32,
                dry_run=True,
            )
        self.assertTrue(receipt["operations_worker_required"])
        self.assertEqual(
            receipt["operations_worker"],
            "hermes-worker-general",
        )
        self.assertTrue(receipt["operations_worker_context"])
        self.assertTrue(receipt["operations_worker_dispatched"])
        self.assertEqual(
            receipt["operations_worker_dispatch_owner"],
            "hermes-role-shell",
        )
        self.assertTrue(receipt["operations_worker_routed_by_hermes"])
        self.assertFalse(receipt["app_manager_created_kanban_card"])
        self.assertFalse(receipt["app_manager_created_worker"])
        self.assertEqual(
            receipt["operations_role_shell_id"],
            "shell_operations_v4",
        )
        self.assertEqual(
            receipt["operations_worker_task_id"],
            "t_operations",
        )

    def test_app_runtime_uses_manager_home_not_worker_profile(self) -> None:
        self.script.write_text(
            "import os\nprint(os.environ.get('HERMES_HOME', ''))\n",
            encoding="utf-8",
        )
        self.install(
            self.manifest(
                default_worker="hermes-worker-general",
                bypass_operations_worker=False,
                worker_skills=[],
            )
        )
        context = SimpleNamespace(
            shell_key="operations",
            shell_id="shell_operations_v4",
            executor_id="executor_hermes_worker_general",
            task_id="t_operations",
            run_id=42,
        )
        worker_profile = self.home / "profiles" / "hermes-worker-general"
        with (
            patch.dict(os.environ, {"HERMES_HOME": str(worker_profile)}),
            patch.object(
                self.manager,
                "_bound_operations_context",
                return_value=context,
            ),
        ):
            receipt = self.manager.run(
                "test-app",
                request_id="d" * 32,
            )
        self.assertEqual(receipt["stdout"].strip(), str(self.home))

    def test_worker_bypass_requires_reason(self) -> None:
        self.install(self.manifest(bypass_operations_worker=True))
        with self.assertRaises(AppManagerError):
            self.manager.run("test-app", dry_run=True)

    def test_explicit_worker_bypass_with_reason_is_allowed(self) -> None:
        self.install(
            self.manifest(
                bypass_operations_worker=True,
                bypass_reason="explicit operator exception",
            )
        )
        receipt = self.manager.run("test-app", dry_run=True)
        self.assertEqual(receipt["status"], "DRY_RUN")
        self.assertFalse(receipt["operations_worker_required"])
        self.assertIsNone(receipt["operations_worker"])
        self.assertFalse(receipt["operations_worker_context"])

    def test_reconcile_adds_default_worker(self) -> None:
        self.install(self.manifest())
        result = self.manager.reconcile_manifests()
        self.assertEqual(result["updated_count"], 1)
        manifest = self.manager._load_manifest("test-app")
        self.assertEqual(
            manifest["execution"]["default_worker"],
            "hermes-worker-general",
        )
        self.assertFalse(manifest["execution"]["bypass_operations_worker"])

    def test_internal_execute_without_operations_context_is_rejected(self) -> None:
        self.install(
            self.manifest(
                default_worker="hermes-worker-general",
                bypass_operations_worker=False,
                worker_skills=[],
            )
        )
        request_id = "a" * 32
        with self.assertRaisesRegex(
            AppManagerError,
            "requires Operations Role Shell provenance",
        ):
            self.manager.execute_direct(
                "test-app",
                request_id=request_id,
                trigger="worker",
            )

    def test_preflight_is_non_production_completion(self) -> None:
        payload = self.manifest(
            default_worker="hermes-worker-general",
            bypass_operations_worker=False,
            worker_skills=[],
        )
        payload["preflight"] = {
            "env": {"TEST_APP_PREFLIGHT_ONLY": "1"},
            "success_status": "PREFLIGHT_PASS",
            "production_completion_claim_allowed": False,
        }
        self.install(payload)
        request_id = "b" * 32
        context = SimpleNamespace(
            shell_key="operations",
            shell_id="shell_operations_v4",
            executor_id="executor_hermes_worker_general",
            task_id="t_preflight",
            run_id=43,
        )
        with patch.object(
            self.manager,
            "_bound_operations_context",
            return_value=context,
        ):
            receipt = self.manager.run(
                "test-app",
                request_id=request_id,
                trigger="worker",
                preflight_only=True,
            )
        self.assertEqual(receipt["status"], "PREFLIGHT_PASS")
        self.assertTrue(receipt["preflight_only"])
        self.assertFalse(receipt["managed_completion_claim_allowed"])

    def test_request_input_is_sealed_and_passed_by_file(self) -> None:
        self.script.write_text(
            "import json, os\n"
            "path = os.environ['OPERATIONS_APP_INPUT_FILE']\n"
            "print(json.dumps(json.load(open(path, encoding='utf-8')), "
            "sort_keys=True))\n",
            encoding="utf-8",
        )
        self.install(
            self.manifest(
                default_worker="hermes-worker-general",
                bypass_operations_worker=False,
                worker_skills=[],
            )
        )
        context = SimpleNamespace(
            shell_key="operations",
            shell_id="shell_operations_v4",
            executor_id="executor_hermes_worker_general",
            task_id="t_input",
            run_id=44,
        )
        request = {"query": "반도체 섹터", "scope": "semiconductors"}
        with patch.object(
            self.manager,
            "_bound_operations_context",
            return_value=context,
        ):
            receipt = self.manager.run(
                "test-app",
                request_id="e" * 32,
                request_input=request,
            )
        self.assertEqual(json.loads(receipt["stdout"]), request)
        self.assertTrue(receipt["request_input_present"])
        self.assertEqual(receipt["request_input_bytes"], 53)
        sealed = Path(receipt["request_input_file"])
        self.assertEqual(json.loads(sealed.read_text(encoding="utf-8")), request)
        self.assertNotIn("반도체", json.dumps(receipt, ensure_ascii=False))

    def test_request_input_must_be_object_and_has_size_limit(self) -> None:
        self.install(
            self.manifest(
                bypass_operations_worker=True,
                bypass_reason="test-only direct execution",
            )
        )
        with self.assertRaisesRegex(AppManagerError, "one JSON object"):
            self.manager.run("test-app", request_input=["not", "object"])
        with self.assertRaisesRegex(AppManagerError, "exceeds"):
            self.manager.run("test-app", request_input={"query": "x" * 33000})


if __name__ == "__main__":
    unittest.main()
