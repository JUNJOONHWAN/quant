"""Collect resumable candidate-adapter predictions for the frozen test split.

This command deliberately does not require an accepted model release: it runs
before release creation.  The later evaluator binds these predictions to the
exact endpoint model, test file, dataset manifest, and adapter artifact set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from workflows.quant_ai_radar.model_runtime import (
    InferenceError,
    ResponseContractError,
    canonical_json,
    contract_repair_instruction,
    judgement_prohibited_violations,
    parse_json_object,
    validate_symbol_judgement,
)


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class PredictionBindingError(ValueError):
    """Raised when resumable output belongs to another immutable candidate."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"test row {line_number} is not an object")
            example_id = str(value.get("example_id") or "")
            if not example_id or example_id in seen:
                raise ValueError(f"missing/duplicate test example_id at line {line_number}")
            if not value.get("context") or not value.get("instruction"):
                raise ValueError(f"test row {line_number} has no prompt fields")
            seen.add(example_id)
            yield value


def completed_ids(
    path: Path,
    endpoint_model: str,
    adapter_set_sha256: str,
    frozen_test_sha256: str,
) -> set[str]:
    if not path.is_file():
        return set()
    result: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            example_id = str(row.get("example_id") or "")
            if not example_id or example_id in result:
                raise ValueError(f"duplicate prediction example_id at line {line_number}")
            if row.get("endpoint_model") != endpoint_model:
                raise PredictionBindingError("existing output is bound to a different endpoint model")
            if row.get("adapter_set_sha256") != adapter_set_sha256:
                raise PredictionBindingError("existing output is bound to a different adapter set")
            if row.get("frozen_test_sha256") != frozen_test_sha256:
                raise PredictionBindingError("existing output is bound to a different frozen test")
            result.add(example_id)
    return result


def quarantine_stale_output(path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archived = path.with_name(f"{path.name}.stale.{stamp}")
    os.replace(path, archived)
    state = path.with_suffix(path.suffix + ".state.json")
    if state.exists():
        os.replace(state, archived.with_suffix(archived.suffix + ".state.json"))
    return archived


def write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".frozen-predictions-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(canonical_json(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validated_completed_ids(
    path: Path,
    endpoint_model: str,
    adapter_set_sha256: str,
    frozen_test_sha256: str,
    expected_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[set[str], list[str], Path | None]:
    """Retain only resumable rows that already satisfy the live response contract."""

    if not path.is_file():
        return set(), [], None
    valid_rows: list[dict[str, Any]] = []
    valid_ids: set[str] = set()
    invalid_ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            example_id = str(row.get("example_id") or "")
            if not example_id or example_id in valid_ids or example_id in invalid_ids:
                raise ValueError(f"duplicate prediction example_id at line {line_number}")
            if row.get("endpoint_model") != endpoint_model:
                raise PredictionBindingError(
                    "existing output is bound to a different endpoint model"
                )
            if row.get("adapter_set_sha256") != adapter_set_sha256:
                raise PredictionBindingError(
                    "existing output is bound to a different adapter set"
                )
            if row.get("frozen_test_sha256") != frozen_test_sha256:
                raise PredictionBindingError(
                    "existing output is bound to a different frozen test"
                )
            expected = expected_by_id.get(example_id)
            if expected is None:
                raise PredictionBindingError(
                    f"existing output contains unknown frozen example: {example_id}"
                )
            response = row.get("response")
            if isinstance(response, str):
                response = json.loads(response)
            try:
                validated = validate_symbol_judgement(response, expected)
            except (ResponseContractError, TypeError, ValueError):
                as_of_date = str((expected.get("facts") or {}).get("as_of_date") or "")
                violations = (
                    judgement_prohibited_violations(response, as_of_date)
                    if isinstance(response, Mapping) and as_of_date
                    else ["unreadable_contract_response"]
                )
                trace = row.get("trace") or {}
                if (
                    isinstance(trace, Mapping)
                    and trace.get("contract_repair_failed_preserved_for_evaluation")
                    is True
                    and not violations
                ):
                    valid_rows.append(row)
                    valid_ids.add(example_id)
                    continue
                invalid_ids.append(example_id)
                continue
            row["response"] = validated
            valid_rows.append(row)
            valid_ids.add(example_id)
    if not invalid_ids:
        return valid_ids, [], None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archived = path.with_name(f"{path.name}.contract-invalid.{stamp}")
    os.replace(path, archived)
    state = path.with_suffix(path.suffix + ".state.json")
    if state.exists():
        os.replace(state, archived.with_suffix(archived.suffix + ".state.json"))
    write_jsonl_atomic(path, valid_rows)
    return valid_ids, invalid_ids, archived


def request_messages(
    *,
    endpoint: str,
    endpoint_model: str,
    messages: Sequence[Mapping[str, str]],
    token: str | None,
    timeout: int,
    max_tokens: int,
) -> tuple[str, dict[str, Any]]:
    payload = {
        "model": endpoint_model,
        "messages": [dict(message) for message in messages],
        "temperature": 0,
        "seed": 1111,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    body = canonical_json(payload).encode("utf-8")
    headers = {"content-type": "application/json", "user-agent": "quant-frozen-eval/1.0"}
    if token:
        headers["authorization"] = f"Bearer {token}"
    request = urllib.request.Request(endpoint, data=body, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            status = response.status
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        raise RuntimeError(f"candidate endpoint request failed: {exc}") from exc
    if status != 200:
        raise RuntimeError(f"candidate endpoint returned HTTP {status}")
    envelope = json.loads(raw)
    returned_model = str(envelope.get("model") or "")
    if returned_model != endpoint_model:
        raise RuntimeError(
            f"endpoint served a different model: expected={endpoint_model!r} got={returned_model!r}"
        )
    try:
        content = envelope["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("candidate endpoint response has no assistant content") from exc
    if not isinstance(content, str):
        raise RuntimeError("candidate endpoint assistant content is not text")
    return content, {
        "endpoint_model": returned_model,
        "request_sha256": hashlib.sha256(body).hexdigest(),
        "response_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "finish_reason": (envelope.get("choices") or [{}])[0].get("finish_reason"),
        "usage": envelope.get("usage"),
    }


def request_prediction(
    *,
    endpoint: str,
    endpoint_model: str,
    context: str,
    instruction: str,
    expected_response: Mapping[str, Any],
    token: str | None,
    timeout: int,
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    initial_messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": instruction},
    ]
    content, initial_trace = request_messages(
        endpoint=endpoint,
        endpoint_model=endpoint_model,
        messages=initial_messages,
        token=token,
        timeout=timeout,
        max_tokens=max_tokens,
    )
    initial: dict[str, Any] | None = None
    try:
        initial = parse_json_object(content)
        validated = validate_symbol_judgement(initial, expected_response)
    except (InferenceError, ResponseContractError) as exc:
        repaired_content, final_trace = request_messages(
            endpoint=endpoint,
            endpoint_model=endpoint_model,
            messages=[
                *initial_messages,
                {
                    "role": "assistant",
                    "content": (
                        canonical_json(initial) if initial is not None else content
                    ),
                },
                {
                    "role": "user",
                    "content": contract_repair_instruction(
                        expected_response, str(exc)
                    ),
                },
            ],
            token=token,
            timeout=timeout,
            max_tokens=max_tokens,
        )
        repaired: dict[str, Any] | None = None
        try:
            repaired = parse_json_object(repaired_content)
            validated = validate_symbol_judgement(repaired, expected_response)
        except (InferenceError, ResponseContractError) as repair_exc:
            preserved = initial if initial is not None else repaired
            as_of_date = str(
                (expected_response.get("facts") or {}).get("as_of_date") or ""
            )
            violations = (
                judgement_prohibited_violations(preserved, as_of_date)
                if preserved is not None and as_of_date
                else ["unreadable_contract_response"]
            )
            if violations:
                raise ResponseContractError(
                    "release-blocking response could not be contract-repaired: "
                    f"violations={violations} repair_error={repair_exc}"
                ) from repair_exc
            if preserved is None:
                raise
            return preserved, {
                **initial_trace,
                "contract_attempts": 2,
                "contract_repair_applied": False,
                "contract_repair_failed_preserved_for_evaluation": True,
                "preserved_response_source": (
                    "initial" if initial is not None else "repair"
                ),
                "repair_request_sha256": final_trace["request_sha256"],
                "repair_response_sha256": final_trace["response_sha256"],
                "initial_contract_error": f"{type(exc).__name__}: {exc}",
                "repair_contract_error": (
                    f"{type(repair_exc).__name__}: {repair_exc}"
                ),
            }
        return validated, {
            **final_trace,
            "contract_attempts": 2,
            "contract_repair_applied": True,
            "initial_request_sha256": initial_trace["request_sha256"],
            "initial_response_sha256": initial_trace["response_sha256"],
            "initial_contract_error": f"{type(exc).__name__}: {exc}",
        }
    return validated, {
        **initial_trace,
        "contract_attempts": 1,
        "contract_repair_applied": False,
    }


def append_durable(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_state(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".frozen-predictions-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--endpoint-model", required=True)
    parser.add_argument("--adapter-set-sha256", required=True)
    parser.add_argument("--frozen-test-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=1400)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not args.endpoint.startswith(("http://", "https://")):
        raise ValueError("--endpoint must be an absolute HTTP URL")
    if not SHA256_RE.fullmatch(args.adapter_set_sha256):
        raise ValueError("--adapter-set-sha256 must be a lowercase SHA256")
    if not SHA256_RE.fullmatch(args.frozen_test_sha256):
        raise ValueError("--frozen-test-sha256 must be a lowercase SHA256")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    token = None
    if args.token_file:
        token = args.token_file.expanduser().read_text(encoding="utf-8").strip()
    test_file = args.test_file.expanduser().resolve()
    observed_test_sha256 = sha256_file(test_file)
    if observed_test_sha256 != args.frozen_test_sha256:
        raise ValueError("--frozen-test-sha256 does not match --test-file")
    output = args.output.expanduser().resolve()
    rows = list(iter_rows(test_file))
    expected_by_id = {
        str(row["example_id"]): json.loads(str(row.get("response") or ""))
        for row in rows
    }
    try:
        done, invalid_ids, archived = validated_completed_ids(
            output,
            args.endpoint_model,
            args.adapter_set_sha256,
            args.frozen_test_sha256,
            expected_by_id,
        )
    except PredictionBindingError as exc:
        archived = quarantine_stale_output(output)
        print(f"quarantined stale predictions at {archived}: {exc}", flush=True)
        done = set()
    else:
        if archived is not None:
            print(
                "quarantined contract-invalid predictions at "
                f"{archived}: invalid={len(invalid_ids)}",
                flush=True,
            )
    total = len(rows)
    pending = [row for row in rows if row["example_id"] not in done]

    def fetch(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        return request_prediction(
            endpoint=args.endpoint,
            endpoint_model=args.endpoint_model,
            context=str(row["context"]),
            instruction=str(row["instruction"]),
            expected_response=expected_by_id[str(row["example_id"])],
            token=token,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for offset in range(0, len(pending), args.workers):
            batch = pending[offset : offset + args.workers]
            futures = [executor.submit(fetch, row) for row in batch]
            for row, future in zip(batch, futures):
                response, trace = future.result()
                append_durable(
                    output,
                    {
                        "example_id": row["example_id"],
                        "endpoint_model": args.endpoint_model,
                        "adapter_set_sha256": args.adapter_set_sha256,
                        "frozen_test_sha256": args.frozen_test_sha256,
                        "response": response,
                        "trace": trace,
                    },
                )
                done.add(row["example_id"])
                write_state(
                    output.with_suffix(output.suffix + ".state.json"),
                    {
                        "status": "running",
                        "completed": len(done),
                        "total": total,
                        "endpoint_model": args.endpoint_model,
                        "adapter_set_sha256": args.adapter_set_sha256,
                        "frozen_test_sha256": args.frozen_test_sha256,
                    },
                )
    if len(done) != total:
        raise RuntimeError(f"frozen prediction coverage mismatch: completed={len(done)} total={total}")
    write_state(
        output.with_suffix(output.suffix + ".state.json"),
        {
            "status": "complete",
            "completed": len(done),
            "total": total,
            "endpoint_model": args.endpoint_model,
            "adapter_set_sha256": args.adapter_set_sha256,
            "frozen_test_sha256": args.frozen_test_sha256,
        },
    )
    print(json.dumps({"status": "complete", "predictions": len(done), "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
