"""Regenerate and qualify the canonical TinyStories-1M FPGA package."""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil

import numpy as np
import torch
from transformers import AutoTokenizer, GPTNeoForCausalLM

from .import_gptneo import import_model
from .package_io import write_package
from .quantize import (
    fake_quantize_state_dict,
    measure_logits_quality,
    quantize_model,
    select_quality_candidate,
)


FIXTURE_TEXT = """Once upon a time, a little girl named Lily found a bright red ball in the garden.
She showed it to her brother, and together they rolled it under the old apple tree.
The ball bumped a tiny door. A mouse opened the door and asked them to help find his blue hat.
Lily looked beside the flowers while her brother looked behind the stones. At last they saw the hat
floating in a puddle. They used a long stick to pull it out, dried it in the sun, and gave it back.
The mouse thanked them with three warm cookies. They went home happy and told their mother everything.
The next morning, Lily heard a soft sound by the window. A small yellow bird had hurt its wing.
She made a safe nest from a box and a towel, brought water, and waited patiently. Soon the bird felt
strong enough to fly. It circled the garden twice, sang a cheerful song, and disappeared over the trees.
"""

BRAM36_COUNT = 955
BRAM36_BYTES = 36 * 1024 // 8
BRAM_RESERVE_BYTES = BRAM36_COUNT * BRAM36_BYTES // 10
KV_CACHE_BYTES = 8 * 2 * 32 * 64
MODEL_BUDGET_BYTES = BRAM36_COUNT * BRAM36_BYTES - BRAM_RESERVE_BYTES


def _fixture_ids(tokenizer) -> np.ndarray:
    ids = tokenizer(FIXTURE_TEXT, add_special_tokens=False)["input_ids"]
    if len(ids) < 64:
        raise ValueError("quality fixture unexpectedly contains fewer than 64 tokens")
    return np.asarray(ids[:256], dtype=np.int32)


def _logits_for_fixture(model, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    logits = []
    labels = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(ids) - 1, 32):
            window = ids[start:start + 32]
            if len(window) < 2:
                continue
            input_ids = torch.from_numpy(window.astype(np.int64))[None, :]
            output = model(input_ids=input_ids).logits[0, :-1]
            logits.append(output.cpu().numpy())
            labels.append(window[1:])
    return np.concatenate(logits), np.concatenate(labels)


def _matrix_modules(model):
    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.ndim == 2:
            yield name, module


def _calibrate_activation_scales(model, ids: np.ndarray) -> dict[str, list[float]]:
    maxima: dict[str, torch.Tensor] = {}
    handles = []

    def observe(key: str, value: torch.Tensor) -> None:
        detached = value.detach().abs().reshape(-1, value.shape[-1]).amax(dim=0).cpu()
        maxima[key] = detached if key not in maxima else torch.maximum(maxima[key], detached)

    for name, module in _matrix_modules(model):
        if isinstance(module, torch.nn.Embedding):
            continue
        if not isinstance(module, torch.nn.Embedding):
            handles.append(module.register_forward_pre_hook(
                lambda _module, inputs, key=f"{name}.input": observe(key, inputs[0])
            ))
        if name != "lm_head":
            handles.append(module.register_forward_hook(
                lambda _module, _inputs, output, key=f"{name}.output": observe(key, output)
            ))
    _logits_for_fixture(model, ids)
    for handle in handles:
        handle.remove()
    return {
        name: torch.clamp(value / 127.0, min=1.0e-12).tolist()
        for name, value in sorted(maxima.items())
    }


def _install_activation_fake_quant(model, scales: dict[str, list[float]]):
    handles = []

    def quantize(value: torch.Tensor, scale_values: list[float]) -> torch.Tensor:
        scale = torch.as_tensor(scale_values, dtype=value.dtype, device=value.device)
        codes = torch.clamp(torch.round(value / scale), -128, 127)
        return codes * scale

    for name, module in _matrix_modules(model):
        if isinstance(module, torch.nn.Embedding):
            continue
        input_key = f"{name}.input"
        output_key = f"{name}.output"
        if input_key in scales:
            handles.append(module.register_forward_pre_hook(
                lambda _module, inputs, scale=scales[input_key]:
                    (quantize(inputs[0], scale), *inputs[1:])
            ))
        if output_key in scales:
            handles.append(module.register_forward_hook(
                lambda _module, _inputs, output, scale=scales[output_key]:
                    quantize(output, scale)
            ))
    return handles


def _state_bits(state: dict[str, torch.Tensor], groups: set[str]) -> dict[str, int]:
    result = {}
    for name, value in state.items():
        if value.ndim != 2 or not name.endswith("weight"):
            continue
        bits = 4
        if "embedding" in groups and name in {"transformer.wte.weight", "lm_head.weight"}:
            bits = 8
        if "position" in groups and name == "transformer.wpe.weight":
            bits = 8
        if "attention" in groups and ".attn." in name:
            bits = 8
        if "mlp" in groups and ".mlp." in name:
            bits = 8
        if "all" in groups:
            bits = 8
        for embedding_bits in (5, 6, 7):
            if f"embedding{embedding_bits}" in groups and name in {
                "transformer.wte.weight", "lm_head.weight"
            }:
                bits = embedding_bits
        result[name] = bits
    return result


def _package_overrides(imported, groups: set[str]) -> dict[str, int]:
    result = {}
    for name, value in imported.tensors.items():
        if value.ndim != 2 or not name.endswith(".weight") or name == "lm_head.weight":
            continue
        use_int8 = (
            "all" in groups
            or ("embedding" in groups and name == "token_embedding.weight")
            or ("position" in groups and name == "position_embedding.weight")
            or ("attention" in groups and ".attn." in name)
            or ("mlp" in groups and ".mlp." in name)
        )
        if use_int8:
            result[name] = 8
        for embedding_bits in (5, 6, 7):
            if f"embedding{embedding_bits}" in groups and name == "token_embedding.weight":
                result[name] = embedding_bits
    return result


def build_qualified_package(source_dir: pathlib.Path, out_dir: pathlib.Path) -> dict[str, object]:
    tokenizer = AutoTokenizer.from_pretrained(source_dir, local_files_only=True)
    imported = import_model(source_dir)
    ids = _fixture_ids(tokenizer)
    fp32_model = GPTNeoForCausalLM.from_pretrained(source_dir, local_files_only=True)
    fp32_logits, labels = _logits_for_fixture(fp32_model, ids)
    activation_scales = _calibrate_activation_scales(fp32_model, ids)
    source_state = fp32_model.state_dict()
    candidate_groups = [
        ("w4", set()),
        ("w4_embedding8", {"embedding"}),
        ("w4_embedding_position8", {"embedding", "position"}),
        ("w4_attention8", {"attention"}),
        ("w4_mlp8", {"mlp"}),
        ("w4_attention_mlp8", {"attention", "mlp"}),
        ("w4_embedding_attention8", {"embedding", "attention"}),
        ("w4_embedding_mlp8", {"embedding", "mlp"}),
        ("w8_embedding5", {"all", "embedding5"}),
        ("w8_embedding6", {"all", "embedding6"}),
        ("w8_embedding7", {"all", "embedding7"}),
        ("w8", {"all"}),
    ]
    candidates = []
    packages = {}
    for name, groups in candidate_groups:
        overrides = _package_overrides(imported, groups)
        package = quantize_model(
            imported,
            ids,
            bits_by_tensor=overrides,
            activation_scales=activation_scales,
        )
        packed_bytes = sum(len(t.data) + len(t.scales) for t in package.tensors.values())
        quantized_state = fake_quantize_state_dict(source_state, _state_bits(source_state, groups))
        candidate_model = GPTNeoForCausalLM(fp32_model.config)
        candidate_model.load_state_dict(quantized_state)
        _install_activation_fake_quant(candidate_model, activation_scales)
        quantized_logits, candidate_labels = _logits_for_fixture(candidate_model, ids)
        if not np.array_equal(labels, candidate_labels):
            raise ValueError("candidate and FP32 labels differ")
        metrics = measure_logits_quality(fp32_logits, quantized_logits, labels)
        candidate = {
            "name": name,
            **metrics,
            "packed_bytes": packed_bytes,
            "non_int4_tensors": len(overrides),
            "fits_model_budget": packed_bytes <= MODEL_BUDGET_BYTES,
            "formats": dict(sorted(overrides.items())),
        }
        candidates.append(candidate)
        packages[name] = package
    fitting = [candidate for candidate in candidates if candidate["fits_model_budget"]]
    print(json.dumps({"candidates": candidates}, indent=2, sort_keys=True))
    selected = select_quality_candidate(fitting)
    package = packages[str(selected["name"])]
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = write_package(package, out_dir)
    for name in imported.tokenizer_files:
        shutil.copyfile(source_dir / name, out_dir / name)
    quality = {
        "schema_version": 1,
        "fixture_token_count": int(ids.size),
        "candidates": candidates,
        "selected": selected,
        "memory_budget": {
            "bram36_count": BRAM36_COUNT,
            "raw_bytes": BRAM36_COUNT * BRAM36_BYTES,
            "reserve_bytes": BRAM_RESERVE_BYTES,
            "kv_cache_bytes": KV_CACHE_BYTES,
            "reserve_after_kv_bytes": BRAM_RESERVE_BYTES - KV_CACHE_BYTES,
            "model_budget_bytes": MODEL_BUDGET_BYTES,
        },
    }
    (out_dir / "quality.json").write_text(json.dumps(quality, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    build_qualified_package(args.source, args.output)


if __name__ == "__main__":
    main()
