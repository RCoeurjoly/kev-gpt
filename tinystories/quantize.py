"""Deterministic NumPy quantization primitives for the canonical package."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .import_gptneo import ImportedGPTNeo


@dataclass(frozen=True)
class QuantizedTensor:
    data: bytes
    scales: bytes
    logical_shape: tuple[int, ...]
    packed_shape: tuple[int, ...]
    bits: int
    signed: bool
    format: str


@dataclass(frozen=True)
class QuantizedPackage:
    tensors: dict[str, QuantizedTensor]
    manifest: dict[str, object]
    calibration_ids: np.ndarray


def fake_quantize_activation(values: np.ndarray, *, scale: float | np.ndarray) -> np.ndarray:
    """Apply the signed-A8 round/saturate boundary used by matrix engines."""

    scale_array = np.asarray(scale, dtype=np.float32)
    if not np.all(np.isfinite(scale_array)) or np.any(scale_array <= 0):
        raise ValueError("activation scale must be finite and positive")
    values = np.asarray(values, dtype=np.float32)
    codes = np.clip(np.rint(values / scale_array), -128, 127)
    return codes.astype(np.float32) * scale_array


def measure_logits_quality(
    fp32_logits: np.ndarray, quantized_logits: np.ndarray, labels: np.ndarray
) -> dict[str, float | int]:
    """Measure perplexity and top-1 agreement on one shared token fixture."""

    fp32_logits = np.asarray(fp32_logits, dtype=np.float64)
    quantized_logits = np.asarray(quantized_logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if fp32_logits.shape != quantized_logits.shape:
        raise ValueError("FP32 and quantized logits must have identical shapes")
    if fp32_logits.ndim != 2 or fp32_logits.shape[0] != labels.size:
        raise ValueError("logits must be [tokens, vocabulary] and match labels")

    def perplexity(logits: np.ndarray) -> float:
        maxima = np.max(logits, axis=1)
        log_sums = maxima + np.log(np.exp(logits - maxima[:, None]).sum(axis=1))
        nll = log_sums - logits[np.arange(labels.size), labels]
        return float(np.exp(np.mean(nll)))

    fp32_perplexity = perplexity(fp32_logits)
    quantized_perplexity = perplexity(quantized_logits)
    matches = int(np.count_nonzero(
        np.argmax(fp32_logits, axis=1) == np.argmax(quantized_logits, axis=1)
    ))
    total = int(labels.size)
    return {
        "fp32_perplexity": fp32_perplexity,
        "quantized_perplexity": quantized_perplexity,
        "perplexity_ratio": quantized_perplexity / fp32_perplexity,
        "top1_matches": matches,
        "top1_total": total,
        "top1_agreement": matches / total,
    }


def fake_quantize_state_dict(
    state: dict[str, torch.Tensor], bits_by_name: dict[str, int]
) -> dict[str, torch.Tensor]:
    """Return a dequantized state dict matching canonical package scales."""

    result: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        bits = bits_by_name.get(name)
        if bits is None or value.ndim != 2:
            result[name] = value.detach().clone()
            continue
        output_axis = 1 if ".mlp.c_fc.weight" in name or ".mlp.c_proj.weight" in name else 0
        quantized, scales = quantize_symmetric(
            value.detach().cpu().numpy(), bits=bits, axis=output_axis
        )
        shape = [1] * quantized.ndim
        shape[output_axis] = quantized.shape[output_axis]
        restored = quantized.astype(np.float32) * scales.reshape(shape)
        result[name] = torch.from_numpy(restored).to(dtype=value.dtype, device=value.device)
    return result


def select_quality_candidate(
    candidates: list[dict[str, object]],
) -> dict[str, object]:
    """Choose a passing format by bytes, then by mixed-precision count."""

    passing = [
        candidate for candidate in candidates
        if float(candidate["perplexity_ratio"]) <= 1.10
        and float(candidate["top1_agreement"]) >= 0.90
    ]
    if not passing:
        if not candidates:
            raise ValueError("no quantization candidates were evaluated")
        best = min(
            candidates,
            key=lambda candidate: (
                max(0.0, float(candidate["perplexity_ratio"]) - 1.10)
                + max(0.0, 0.90 - float(candidate["top1_agreement"])),
                int(candidate["packed_bytes"]),
            ),
        )
        raise ValueError(
            f"no candidate passed; best {best['name']}: "
            f"perplexity_ratio={best['perplexity_ratio']}, "
            f"top1_agreement={best['top1_agreement']}"
        )
    return min(
        passing,
        key=lambda candidate: (
            int(candidate["packed_bytes"]),
            int(candidate["non_int4_tensors"]),
            str(candidate["name"]),
        ),
    )


def quantize_symmetric(
    values: np.ndarray, *, bits: int, axis: int
) -> tuple[np.ndarray, np.ndarray]:
    """Symmetrically quantize with one scale for each index along ``axis``."""

    values = np.asarray(values, dtype=np.float32)
    if bits < 2 or bits > 8:
        raise ValueError("bits must be between 2 and 8")
    if axis < 0:
        axis += values.ndim
    if axis < 0 or axis >= values.ndim:
        raise np.AxisError(axis, ndim=values.ndim)
    reduction_axes = tuple(index for index in range(values.ndim) if index != axis)
    maximum = np.max(np.abs(values), axis=reduction_axes)
    positive_limit = (1 << (bits - 1)) - 1
    scale = maximum / np.float32(positive_limit)
    scale = np.where(scale == 0, np.float32(1.0), scale).astype(np.float32)
    broadcast_shape = [1] * values.ndim
    broadcast_shape[axis] = values.shape[axis]
    quantized = np.rint(values / scale.reshape(broadcast_shape))
    quantized = np.clip(quantized, -positive_limit, positive_limit).astype(np.int8)
    return quantized, scale


def pack_int4(values: np.ndarray) -> np.ndarray:
    """Pack signed INT4 values, putting the earlier value in the low nibble."""

    flat = np.asarray(values, dtype=np.int8).reshape(-1)
    if np.any(flat < -8) or np.any(flat > 7):
        raise ValueError("INT4 value outside [-8, 7]")
    if flat.size % 2:
        flat = np.pad(flat, (0, 1))
    nibbles = flat.astype(np.uint8) & np.uint8(0x0F)
    return nibbles[0::2] | (nibbles[1::2] << np.uint8(4))


def pack_signed(values: np.ndarray, *, bits: int) -> np.ndarray:
    """Densely pack signed values LSB-first, with logical index increasing first."""

    if bits < 2 or bits > 8:
        raise ValueError("bits must be between 2 and 8")
    flat = np.asarray(values, dtype=np.int8).reshape(-1)
    lower = -(1 << (bits - 1))
    upper = (1 << (bits - 1)) - 1
    if np.any(flat < lower) or np.any(flat > upper):
        raise ValueError(f"INT{bits} value outside [{lower}, {upper}]")
    unsigned = flat.astype(np.uint8) & np.uint8((1 << bits) - 1)
    shifts = np.arange(bits, dtype=np.uint8)
    bitstream = ((unsigned[:, None] >> shifts) & np.uint8(1)).reshape(-1)
    return np.packbits(bitstream, bitorder="little")


def quantize_model(
    model: ImportedGPTNeo,
    calibration_ids: np.ndarray,
    max_context: int = 32,
    bits_by_tensor: dict[str, int] | None = None,
    activation_scales: dict[str, float | list[float] | np.ndarray] | None = None,
) -> QuantizedPackage:
    """Create the deterministic weight package; activation calibration is receipted."""

    calibration_ids = np.asarray(calibration_ids, dtype=np.int32).reshape(-1)
    if max_context <= 0 or calibration_ids.size == 0:
        raise ValueError("max_context and calibration_ids must be non-empty")
    bits_by_tensor = {} if bits_by_tensor is None else dict(bits_by_tensor)
    raw_activation_scales = {} if activation_scales is None else dict(activation_scales)
    activation_scales = {}
    for name, scale in raw_activation_scales.items():
        scale_array = np.asarray(scale, dtype=np.float32)
        if not np.all(np.isfinite(scale_array)) or np.any(scale_array <= 0):
            raise ValueError(f"{name}: activation scale must be finite and positive")
        activation_scales[name] = (
            float(scale_array) if scale_array.ndim == 0 else scale_array.tolist()
        )
    tensors: dict[str, QuantizedTensor] = {}
    roles: dict[str, dict[str, object]] = {}
    for name in sorted(model.tensors):
        if name == "lm_head.weight":
            roles[name] = {"alias_of": "token_embedding.weight"}
            continue
        value = np.asarray(model.tensors[name], dtype=np.float32)
        if name.endswith(".weight") and value.ndim == 2:
            bits = bits_by_tensor.get(name, 4)
            if bits < 4 or bits > 8:
                raise ValueError(f"{name}: weight bits must be in [4, 8]")
            quantized, scales = quantize_symmetric(value, bits=bits, axis=0)
            packed = quantized.reshape(-1) if bits == 8 else pack_signed(quantized, bits=bits)
            tensor = QuantizedTensor(
                data=packed.tobytes(),
                scales=np.asarray(scales, dtype="<f4").tobytes(),
                logical_shape=tuple(value.shape),
                packed_shape=tuple(packed.shape),
                bits=bits,
                signed=True,
                format=f"symmetric_int{bits}_per_output",
            )
        else:
            full = np.asarray(value, dtype="<f4")
            tensor = QuantizedTensor(
                data=full.tobytes(),
                scales=b"",
                logical_shape=tuple(value.shape),
                packed_shape=tuple(full.shape),
                bits=32,
                signed=True,
                format="float32",
            )
        tensors[name] = tensor
        roles[name] = {"storage": name}
    manifest = {
        "schema_version": 1,
        "model": model.manifest,
        "max_context": max_context,
        "activation_format": "symmetric_int8",
        "activation_scales": dict(sorted(activation_scales.items())),
        "weight_overrides": dict(sorted(bits_by_tensor.items())),
        "tensor_roles": roles,
        "calibration": {
            "token_count": int(calibration_ids.size),
            "token_ids_sha256": __import__("hashlib").sha256(
                np.asarray(calibration_ids, dtype="<i4").tobytes()
            ).hexdigest(),
        },
    }
    return QuantizedPackage(tensors, manifest, calibration_ids.copy())
