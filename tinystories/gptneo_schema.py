"""Validation for the supported TinyStories GPT-Neo hardware contract."""

from __future__ import annotations

from collections.abc import Mapping


EXPECTED = {
    "schema_version": 1,
    "model_type": "gpt_neo",
    "n_layer": 8,
    "hidden_size": 64,
    "n_head": 16,
    "vocab_size": 50257,
    "source_revision": "ac533fb8b4f69c71894bf96badfe11e6294d9fcf",
    "max_context": 32,
    "tie_word_embeddings": True,
    "activation_function": "gelu_new",
}


def validate_manifest(value: Mapping[str, object]) -> dict[str, object]:
    """Return a normalized manifest or reject unsupported model semantics."""

    if not isinstance(value, Mapping):
        raise ValueError("manifest must be a mapping")

    normalized = dict(value)
    max_context = normalized.get("max_context")
    local_window = normalized.get("local_window")
    if not isinstance(max_context, int) or max_context <= 0:
        raise ValueError("max_context must be a positive integer")
    if not isinstance(local_window, int) or local_window <= 0:
        raise ValueError("local_window must be a positive integer")
    if max_context > local_window:
        raise ValueError("max_context <= local_window is required")

    for field, expected in EXPECTED.items():
        if field not in normalized:
            raise ValueError(f"missing field: {field}")
        if normalized[field] != expected:
            raise ValueError(
                f"{field}: expected {expected!r}, got {normalized[field]!r}"
            )

    hidden_size = int(normalized["hidden_size"])
    n_head = int(normalized["n_head"])
    if hidden_size % n_head:
        raise ValueError("hidden_size must be divisible by n_head")
    normalized["head_dim"] = hidden_size // n_head
    return normalized
