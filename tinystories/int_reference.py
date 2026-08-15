"""Package-only NumPy reference primitives for the Kintex GPT-Neo engine."""

from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np


def trace_sha256(trace: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(trace):
        digest.update(name.encode())
        value = np.asarray(trace[name], dtype="<f4")
        digest.update(np.asarray(value.shape, dtype="<u4").tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def round_shift_signed(values: np.ndarray, shift: int) -> np.ndarray:
    """Round signed integers to nearest while shifting right."""

    values = np.asarray(values, dtype=np.int64)
    if shift < 0:
        raise ValueError("shift must be non-negative")
    if shift == 0:
        return values.copy()
    magnitude = np.abs(values)
    rounded = (magnitude + (1 << (shift - 1))) >> shift
    return np.where(values < 0, -rounded, rounded)


def saturate_int8(values: np.ndarray) -> np.ndarray:
    return np.clip(values, -128, 127).astype(np.int8)


def layer_norm(
    values: np.ndarray,
    *,
    gamma: np.ndarray,
    beta: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean(axis=-1, keepdims=True)
    variance = np.square(values - mean).mean(axis=-1, keepdims=True)
    normalized = (values - mean) / np.sqrt(variance + epsilon)
    return normalized * np.asarray(gamma) + np.asarray(beta)


def gelu_new(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    coefficient = np.float32(0.7978845608028654)
    cubic = np.power(values, np.float32(3.0))
    inner = values + np.float32(0.044715) * cubic
    return np.float32(0.5) * values * (
        np.float32(1.0) + np.tanh(coefficient * inner)
    )


def causal_softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim < 2 or scores.shape[-2] != scores.shape[-1]:
        raise ValueError("causal softmax expects square score matrices")
    length = scores.shape[-1]
    masked = np.where(np.triu(np.ones((length, length), dtype=bool), 1), -np.inf, scores)
    maxima = np.max(masked, axis=-1, keepdims=True)
    exponentials = np.exp(masked - maxima)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def split_heads(values: np.ndarray, *, n_head: int) -> np.ndarray:
    values = np.asarray(values)
    if values.shape[-1] % n_head:
        raise ValueError("hidden width must be divisible by head count")
    length = values.shape[-2]
    head_dim = values.shape[-1] // n_head
    return values.reshape(length, n_head, head_dim).transpose(1, 0, 2)


def merge_heads(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    n_head, length, head_dim = values.shape
    return values.transpose(1, 0, 2).reshape(length, n_head * head_dim)


class KVCache:
    def __init__(self, *, n_layer: int, max_context: int, n_head: int, head_dim: int):
        self.n_layer = n_layer
        self.max_context = max_context
        self.n_head = n_head
        self.head_dim = head_dim
        self.reset()

    def reset(self) -> None:
        self._keys: list[list[np.ndarray]] = [[] for _ in range(self.n_layer)]
        self._values: list[list[np.ndarray]] = [[] for _ in range(self.n_layer)]

    def append(self, layer: int, key: np.ndarray, value: np.ndarray) -> None:
        expected = (self.n_head, self.head_dim)
        if np.shape(key) != expected or np.shape(value) != expected:
            raise ValueError(f"KV entry must have shape {expected}")
        if len(self._keys[layer]) >= self.max_context:
            raise ValueError("context capacity exceeded")
        self._keys[layer].append(np.array(key, copy=True))
        self._values[layer].append(np.array(value, copy=True))

    def read(self, layer: int) -> tuple[np.ndarray, np.ndarray]:
        if not self._keys[layer]:
            empty = np.empty((self.n_head, 0, self.head_dim), dtype=np.float32)
            return empty, empty.copy()
        keys = np.stack(self._keys[layer], axis=1)
        values = np.stack(self._values[layer], axis=1)
        return keys, values


class IntegerGPTNeo:
    """Validated, package-only GPT-Neo reference with calibrated A8 boundaries."""

    def __init__(self, package_dir: pathlib.Path):
        self.package_dir = pathlib.Path(package_dir)
        receipt = json.loads((self.package_dir / "receipt.json").read_text())
        for name, expected in receipt["files"].items():
            data = (self.package_dir / name).read_bytes()
            actual = hashlib.sha256(data).hexdigest()
            if actual != expected["sha256"]:
                raise ValueError(f"sha256 mismatch for {name}: {actual}")
        self.manifest = json.loads((self.package_dir / "manifest.json").read_text())
        self._weight_image = (self.package_dir / "weights.bin").read_bytes()
        self._scale_image = (self.package_dir / "scales.bin").read_bytes()
        self.tensors: dict[str, np.ndarray] = {}
        self.tensor_codes: dict[str, np.ndarray] = {}
        self.tensor_scales: dict[str, np.ndarray] = {}
        for name in sorted(self.manifest["tensors"]):
            self.tensors[name] = self._decode_tensor(name)
        self.tensors["lm_head.weight"] = self.tensors["token_embedding.weight"]
        self.activation_scales = {
            name: np.asarray(scale, dtype=np.float32)
            for name, scale in self.manifest["activation_scales"].items()
        }
        self.reset()

    @staticmethod
    def _unpack_signed(data: bytes, bits: int, count: int) -> np.ndarray:
        if bits == 8:
            return np.frombuffer(data, dtype=np.int8, count=count).copy()
        packed = np.frombuffer(data, dtype=np.uint8)
        bitstream = np.unpackbits(packed, bitorder="little")[:count * bits]
        shifts = np.arange(bits, dtype=np.uint32)
        unsigned = (bitstream.reshape(count, bits).astype(np.uint32) << shifts).sum(axis=1)
        sign = np.uint32(1 << (bits - 1))
        signed = ((unsigned ^ sign).astype(np.int32) - int(sign)).astype(np.int8)
        return signed

    def _decode_tensor(self, name: str) -> np.ndarray:
        descriptor = self.manifest["tensors"][name]
        shape = tuple(descriptor["logical_shape"])
        offset = int(descriptor["offset"])
        nbytes = int(descriptor["nbytes"])
        data = self._weight_image[offset:offset + nbytes]
        bits = int(descriptor["bits"])
        if descriptor["format"] == "float32":
            return np.frombuffer(data, dtype="<f4").copy().reshape(shape)
        if descriptor["format"] == "float16":
            return np.frombuffer(data, dtype="<f2").astype(np.float32).reshape(shape)
        count = int(np.prod(shape))
        codes = self._unpack_signed(data, bits, count).reshape(shape)
        scale_offset = int(descriptor["scale_offset"])
        scale_nbytes = int(descriptor["scale_nbytes"])
        scales = np.frombuffer(
            self._scale_image[scale_offset:scale_offset + scale_nbytes], dtype="<f4"
        ).copy()
        self.tensor_codes[name] = codes
        self.tensor_scales[name] = scales
        return codes.astype(np.float32) * scales.reshape((shape[0],) + (1,) * (len(shape) - 1))

    def reset(self) -> None:
        self._tokens: list[int] = []
        self._last_logits: np.ndarray | None = None
        self._last_trace: dict[str, np.ndarray] = {}

    def _a8(self, values: np.ndarray, key: str) -> np.ndarray:
        scale = self.activation_scales.get(key)
        if scale is None:
            return np.asarray(values, dtype=np.float32)
        codes = np.clip(np.rint(np.asarray(values, dtype=np.float32) / scale), -128, 127)
        return (codes * scale).astype(np.float32)

    def _a8_codes(self, values: np.ndarray, key: str) -> tuple[np.ndarray, np.ndarray]:
        scale = self.activation_scales[key]
        codes = np.clip(
            np.rint(np.asarray(values, dtype=np.float32) / scale), -128, 127
        ).astype(np.int8)
        return codes, scale

    def _gemv(
        self,
        values: np.ndarray,
        weight_name: str,
        bias_name: str | None,
        module_name: str,
    ) -> np.ndarray:
        input_codes, input_scale = self._a8_codes(values, f"{module_name}.input")
        weight_codes = self.tensor_codes[weight_name]
        weight_scale = self.tensor_scales[weight_name]
        # Define the MAC independently of host BLAS reduction order.  Per-channel
        # A8 scales prevent factoring the dot product into one integer accumulator,
        # so each exact signed-code product is scaled and accumulated in float64.
        products = (
            input_codes.astype(np.int32)[:, None, :]
            * weight_codes.astype(np.int32)[None, :, :]
        )
        result = np.sum(
            products.astype(np.float64) * input_scale.astype(np.float64)[None, None, :],
            axis=-1,
            dtype=np.float64,
        ) * weight_scale.astype(np.float64)[None, :]
        if bias_name is not None:
            result = result + self.tensors[bias_name]
        return self._a8(result, f"{module_name}.output")

    def _forward(self, token_ids: list[int]) -> np.ndarray:
        if not token_ids or len(token_ids) > 32:
            raise ValueError("context length must be between 1 and 32")
        ids = np.asarray(token_ids, dtype=np.int64)
        x = (
            self.tensors["token_embedding.weight"][ids]
            + self.tensors["position_embedding.weight"][:len(ids)]
        ).astype(np.float32)
        trace: dict[str, np.ndarray] = {"embedding": x.copy()}
        for layer in range(8):
            block = f"blocks.{layer}"
            source = f"transformer.h.{layer}"
            normalized = layer_norm(
                x,
                gamma=self.tensors[f"{block}.ln1.weight"],
                beta=self.tensors[f"{block}.ln1.bias"],
                epsilon=1.0e-5,
            ).astype(np.float32)
            trace[f"block.{layer}.ln1"] = normalized[-1].copy()
            query = self._gemv(
                normalized, f"{block}.attn.q.weight", None,
                f"{source}.attn.attention.q_proj",
            )
            key = self._gemv(
                normalized, f"{block}.attn.k.weight", None,
                f"{source}.attn.attention.k_proj",
            )
            value = self._gemv(
                normalized, f"{block}.attn.v.weight", None,
                f"{source}.attn.attention.v_proj",
            )
            trace[f"block.{layer}.q"] = query[-1].copy()
            trace[f"block.{layer}.k"] = key[-1].copy()
            trace[f"block.{layer}.v"] = value[-1].copy()
            query_heads = split_heads(query, n_head=16)
            key_heads = split_heads(key, n_head=16)
            value_heads = split_heads(value, n_head=16)
            scores = np.matmul(query_heads, key_heads.transpose(0, 2, 1))
            probabilities = causal_softmax(scores).astype(np.float32)
            context = merge_heads(np.matmul(probabilities, value_heads))
            attention = self._gemv(
                context,
                f"{block}.attn.out.weight",
                f"{block}.attn.out.bias",
                f"{source}.attn.attention.out_proj",
            )
            trace[f"block.{layer}.attention"] = attention[-1].copy()
            x = (x + attention).astype(np.float32)
            normalized = layer_norm(
                x,
                gamma=self.tensors[f"{block}.ln2.weight"],
                beta=self.tensors[f"{block}.ln2.bias"],
                epsilon=1.0e-5,
            ).astype(np.float32)
            trace[f"block.{layer}.ln2"] = normalized[-1].copy()
            hidden = self._gemv(
                normalized,
                f"{block}.mlp.fc.weight",
                f"{block}.mlp.fc.bias",
                f"{source}.mlp.c_fc",
            )
            trace[f"block.{layer}.fc"] = hidden[-1].copy()
            hidden = gelu_new(hidden).astype(np.float32)
            trace[f"block.{layer}.gelu"] = hidden[-1].copy()
            projected = self._gemv(
                hidden,
                f"{block}.mlp.proj.weight",
                f"{block}.mlp.proj.bias",
                f"{source}.mlp.c_proj",
            )
            trace[f"block.{layer}.projected"] = projected[-1].copy()
            x = (x + projected).astype(np.float32)
            trace[f"block.{layer}.hidden"] = x[-1].copy()
            trace[f"block.{layer}.key"] = key_heads[:, -1].copy()
            trace[f"block.{layer}.value"] = value_heads[:, -1].copy()
        normalized = layer_norm(
            x,
            gamma=self.tensors["final_ln.weight"],
            beta=self.tensors["final_ln.bias"],
            epsilon=1.0e-5,
        ).astype(np.float32)
        normalized = self._a8(normalized, "lm_head.input")
        logits = normalized @ self.tensors["token_embedding.weight"].T
        trace["final_hidden"] = normalized[-1].copy()
        trace["logits"] = logits[-1].copy()
        self._last_trace = trace
        return logits.astype(np.float32)

    def prefill(self, token_ids: list[int]) -> None:
        if len(token_ids) > 32:
            raise ValueError("context capacity exceeded")
        self._tokens = list(token_ids)
        self._last_logits = self._forward(self._tokens)[-1] if self._tokens else None

    def step(self, token_id: int) -> tuple[np.ndarray, int]:
        if len(self._tokens) >= 32:
            raise ValueError("context capacity exceeded")
        self._tokens.append(int(token_id))
        logits = self._forward(self._tokens)[-1]
        self._last_logits = logits
        return logits.copy(), int(np.argmax(logits))

    def generate(self, prompt_ids: list[int], n_tokens: int) -> list[int]:
        if n_tokens < 0:
            raise ValueError("n_tokens must be non-negative")
        if len(prompt_ids) > 32 or len(prompt_ids) + n_tokens > 32:
            raise ValueError("context capacity exceeded")
        self.reset()
        if not prompt_ids or prompt_ids[-1] == 50256 or n_tokens == 0:
            return []
        self.prefill(prompt_ids[:-1])
        _, token = self.step(prompt_ids[-1])
        output = []
        for index in range(n_tokens):
            output.append(token)
            if token == 50256 or index + 1 == n_tokens:
                break
            _, token = self.step(token)
        return output

    def trace(self, prompt_ids: list[int], n_tokens: int) -> dict[str, np.ndarray]:
        self.generate(prompt_ids, n_tokens)
        return {name: value.copy() for name, value in self._last_trace.items()}
