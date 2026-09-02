"""Single inference entrypoint for the released training-native contract."""

from __future__ import annotations

import json
from typing import Any, Mapping

from .model_runtime import TrainedQuantClient


TRAINING_NATIVE_INPUT_SCHEMA = "quant.analysis_packet.v3"
TRAINING_NATIVE_PROMPT_CONTRACT = (
    "quant.analysis_packet.v3.build_example.context_instruction.v1"
)


def complete_training_native_judgement(
    *,
    client: TrainedQuantClient,
    example: Mapping[str, Any],
    max_tokens: int = 900,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Call the model with exactly the context and instruction used for SFT."""

    metadata = example.get("metadata") or {}
    if metadata.get("input_packet_schema") != TRAINING_NATIVE_INPUT_SCHEMA:
        raise ValueError(
            "training-native inference requires quant.analysis_packet.v3"
        )
    expected = json.loads(str(example["response"]))
    return client.complete_validated(
        system=str(example["context"]),
        user=str(example["instruction"]),
        expected_response=expected,
        max_tokens=max_tokens,
    )
