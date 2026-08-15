"""Strict adapter from the pinned Hugging Face GPT-Neo checkpoint."""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import numpy as np
import torch

from .gptneo_schema import validate_manifest


@dataclass(frozen=True)
class ImportedGPTNeo:
    tensors: dict[str, np.ndarray]
    manifest: dict[str, object]
    source_config: dict[str, object]
    tokenizer_files: tuple[str, ...]


def _required_tensor(state: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    try:
        tensor = state[name]
    except KeyError as error:
        raise ValueError(f"missing source tensor: {name}") from error
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"source tensor is not a torch.Tensor: {name}")
    return tensor.detach().cpu()


def _array(tensor: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(tensor.to(torch.float32).numpy())


def _matrix(tensor: torch.Tensor, rows: int, columns: int, name: str) -> np.ndarray:
    array = _array(tensor)
    if array.shape == (rows, columns):
        return array
    if array.shape == (columns, rows):
        return np.ascontiguousarray(array.T)
    raise ValueError(f"{name}: expected {(rows, columns)}, got {array.shape}")


def import_model(source_dir: pathlib.Path) -> ImportedGPTNeo:
    """Load and canonicalize only the supported TinyStories-1M architecture."""

    source_dir = pathlib.Path(source_dir)
    source_config = json.loads((source_dir / "config.json").read_text())
    semantic_manifest = validate_manifest({
        "schema_version": 1,
        "model_type": source_config.get("model_type"),
        "n_layer": source_config.get("num_layers"),
        "hidden_size": source_config.get("hidden_size"),
        "n_head": source_config.get("num_heads"),
        "vocab_size": source_config.get("vocab_size"),
        "source_revision": "ac533fb8b4f69c71894bf96badfe11e6294d9fcf",
        "max_context": 32,
        "local_window": source_config.get("window_size"),
        "tie_word_embeddings": True,
        "activation_function": source_config.get("activation_function"),
    })
    checkpoint = source_dir / "pytorch_model.bin"
    with torch.no_grad():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("checkpoint must contain a state dictionary")

    embedding_tensor = _required_tensor(state, "transformer.wte.weight")
    head_tensor = state.get("lm_head.weight", embedding_tensor)
    if not isinstance(head_tensor, torch.Tensor):
        raise ValueError("source tensor is not a torch.Tensor: lm_head.weight")
    head_tensor = head_tensor.detach().cpu()
    if embedding_tensor.shape != head_tensor.shape or not torch.equal(embedding_tensor, head_tensor):
        raise ValueError("lm_head.weight must be tied to transformer.wte.weight")
    embedding = _array(embedding_tensor)
    if embedding.shape != (50257, 64):
        raise ValueError(f"transformer.wte.weight: unexpected shape {embedding.shape}")

    positions = _array(_required_tensor(state, "transformer.wpe.weight"))
    if positions.shape[0] < 32 or positions.shape[1:] != (64,):
        raise ValueError(f"transformer.wpe.weight: unexpected shape {positions.shape}")
    tensors: dict[str, np.ndarray] = {
        "token_embedding.weight": embedding,
        "lm_head.weight": embedding,
        "position_embedding.weight": np.ascontiguousarray(positions[:32]),
        "final_ln.weight": _array(_required_tensor(state, "transformer.ln_f.weight")),
        "final_ln.bias": _array(_required_tensor(state, "transformer.ln_f.bias")),
    }

    for layer in range(8):
        source = f"transformer.h.{layer}"
        target = f"blocks.{layer}"
        direct = {
            "ln_1.weight": "ln1.weight",
            "ln_1.bias": "ln1.bias",
            "attn.attention.q_proj.weight": "attn.q.weight",
            "attn.attention.k_proj.weight": "attn.k.weight",
            "attn.attention.v_proj.weight": "attn.v.weight",
            "attn.attention.out_proj.weight": "attn.out.weight",
            "attn.attention.out_proj.bias": "attn.out.bias",
            "ln_2.weight": "ln2.weight",
            "ln_2.bias": "ln2.bias",
            "mlp.c_fc.bias": "mlp.fc.bias",
            "mlp.c_proj.bias": "mlp.proj.bias",
        }
        for source_suffix, target_suffix in direct.items():
            name = f"{source}.{source_suffix}"
            tensors[f"{target}.{target_suffix}"] = _array(_required_tensor(state, name))
        for source_suffix, target_suffix, rows, columns in (
            ("mlp.c_fc.weight", "mlp.fc.weight", 256, 64),
            ("mlp.c_proj.weight", "mlp.proj.weight", 64, 256),
        ):
            name = f"{source}.{source_suffix}"
            tensors[f"{target}.{target_suffix}"] = _matrix(
                _required_tensor(state, name), rows, columns, name
            )

    tokenizer_files = tuple(sorted(
        path.name for path in source_dir.iterdir()
        if path.name in {
            "merges.txt", "special_tokens_map.json", "tokenizer.json",
            "tokenizer_config.json", "vocab.json",
        }
    ))
    return ImportedGPTNeo(tensors, semantic_manifest, source_config, tokenizer_files)
