"""Bit-defined Q16/Q24 GPT-Neo reference for the synthesizable datapath."""

from __future__ import annotations

import math
import pathlib

import numpy as np

from .int_reference import IntegerGPTNeo


Q_VALUE = 16
Q_SCALE = 24


def round_shift(values: np.ndarray, shift: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    magnitude = np.abs(values)
    rounded = (magnitude + (1 << (shift - 1))) >> shift
    return np.where(values < 0, -rounded, rounded)


def trunc_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.int64)
    denominator = np.asarray(denominator, dtype=np.int64)
    quotient = np.abs(numerator) // np.abs(denominator)
    return np.where((numerator < 0) != (denominator < 0), -quotient, quotient)


def fixed_layer_norm(values: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
    rows = np.asarray(values, dtype=np.int64).reshape(-1, values.shape[-1])
    outputs = np.empty_like(rows)
    width = rows.shape[1]
    epsilon_q32 = 42950
    for row_index, row in enumerate(rows):
        mean = int(np.sum(row, dtype=np.int64))
        mean = abs(mean) // width * (-1 if mean < 0 else 1)
        deltas = row - mean
        variance = int(np.sum(deltas * deltas, dtype=np.int64)) // width + epsilon_q32
        deviation = math.isqrt(variance)
        normalized = trunc_divide(deltas << Q_VALUE, deviation)
        outputs[row_index] = (normalized * gamma >> Q_VALUE) + beta
    return outputs.reshape(values.shape)


def _gelu_table() -> np.ndarray:
    result = []
    for index in range(8192):
        value = -8.0 + index / 512.0
        gelu = 0.5 * value * (
            1.0 + math.tanh(math.sqrt(2.0 / math.pi) *
                            (value + 0.044715 * value ** 3))
        )
        result.append(max(-32768, min(32767, round(gelu * 4096.0))))
    return np.asarray(result, dtype=np.int64)


GELU_TABLE = _gelu_table()
EXP_TABLE = np.asarray([
    round(math.exp((index - 4096) / 256.0) * (1 << 20))
    for index in range(4096)
], dtype=np.int64)


def fixed_gelu(values_q16: np.ndarray) -> np.ndarray:
    q12 = np.clip(round_shift(values_q16, 4), -32768, 32767)
    biased = q12 + 32768
    index = biased >> 3
    fraction = biased & 7
    upper_index = np.minimum(index + 1, 8191)
    result_q12 = GELU_TABLE[index] + (
        (GELU_TABLE[upper_index] - GELU_TABLE[index]) * fraction >> 3
    )
    return result_q12 << 4


class FixedGPTNeo:
    def __init__(self, package_dir: pathlib.Path):
        self.package = IntegerGPTNeo(package_dir)
        self.manifest = self.package.manifest
        self.codes = self.package.tensor_codes
        self.weight_scales = {
            name: np.rint(values.astype(np.float64) * (1 << Q_SCALE)).astype(np.int64)
            for name, values in self.package.tensor_scales.items()
        }
        self.activation_scales = {
            name: np.rint(values.astype(np.float64) * (1 << Q_SCALE)).astype(np.int64)
            for name, values in self.package.activation_scales.items()
        }
        self.parameters = {
            name: np.rint(value.astype(np.float64) * (1 << Q_VALUE)).astype(np.int64)
            for name, value in self.package.tensors.items()
            if name not in self.codes
        }
        self.reset()

    def reset(self) -> None:
        self.tokens: list[int] = []
        self.last_logits: np.ndarray | None = None

    def _embedding(self, name: str, rows: np.ndarray) -> np.ndarray:
        codes = self.codes[name][rows].astype(np.int64)
        scales = self.weight_scales[name][rows].astype(np.int64)
        return round_shift(codes * scales[:, None], Q_SCALE - Q_VALUE)

    def _activation_codes(self, values: np.ndarray, key: str) -> np.ndarray:
        scale = self.activation_scales[key]
        # Round-to-nearest correction using the discarded remainder.
        numerator = np.asarray(values, dtype=np.int64) << (Q_SCALE-Q_VALUE)
        magnitude = (np.abs(numerator) + scale // 2) // scale
        codes = np.where(numerator < 0, -magnitude, magnitude)
        return np.clip(codes, -128, 127).astype(np.int64)

    def _gemv(self, values: np.ndarray, weight: str, bias: str | None,
              module: str, output_quantized: bool = True) -> np.ndarray:
        input_scale = self.activation_scales[f"{module}.input"]
        input_codes = self._activation_codes(values, f"{module}.input")
        scaled_input = input_codes * input_scale
        accum = scaled_input @ self.codes[weight].astype(np.int64).T
        real_q16 = round_shift(accum * self.weight_scales[weight], 2*Q_SCALE-Q_VALUE)
        if bias is not None:
            real_q16 = real_q16 + self.parameters[bias]
        if not output_quantized:
            return real_q16
        key = f"{module}.output"
        output_codes = self._activation_codes(real_q16, key)
        return round_shift(output_codes * self.activation_scales[key], Q_SCALE-Q_VALUE)

    @staticmethod
    def _attention(query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
        length = query.shape[0]
        output = np.empty_like(query)
        for head in range(16):
            columns = slice(head*4, head*4+4)
            q = query[:, columns]
            k = key[:, columns]
            v = value[:, columns]
            for position in range(length):
                scores = (q[position][None, :] * k[:position+1]).sum(axis=1) >> 24
                delta = np.clip(scores - scores.max(), -4096, 0)
                table_index = np.minimum(4096 + delta, 4095)
                probabilities = np.where(delta == 0, 1 << 20, EXP_TABLE[table_index])
                denominator = int(probabilities.sum())
                numerator = (probabilities[:, None] * v[:position+1]).sum(axis=0)
                output[position, columns] = trunc_divide(
                    numerator + np.where(numerator < 0, -(denominator//2), denominator//2),
                    denominator,
                )
        return output

    def forward(self, token_ids: list[int]) -> np.ndarray:
        ids = np.asarray(token_ids, dtype=np.int64)
        positions = np.arange(len(ids), dtype=np.int64)
        x = self._embedding("token_embedding.weight", ids)
        x += self._embedding("position_embedding.weight", positions)
        for layer in range(8):
            block = f"blocks.{layer}"
            source = f"transformer.h.{layer}"
            normalized = fixed_layer_norm(
                x, self.parameters[f"{block}.ln1.weight"],
                self.parameters[f"{block}.ln1.bias"],
            )
            query = self._gemv(normalized, f"{block}.attn.q.weight", None,
                               f"{source}.attn.attention.q_proj")
            key = self._gemv(normalized, f"{block}.attn.k.weight", None,
                             f"{source}.attn.attention.k_proj")
            value = self._gemv(normalized, f"{block}.attn.v.weight", None,
                               f"{source}.attn.attention.v_proj")
            context = self._attention(query, key, value)
            attention = self._gemv(context, f"{block}.attn.out.weight",
                                   f"{block}.attn.out.bias",
                                   f"{source}.attn.attention.out_proj")
            x = x + attention
            normalized = fixed_layer_norm(
                x, self.parameters[f"{block}.ln2.weight"],
                self.parameters[f"{block}.ln2.bias"],
            )
            hidden = self._gemv(normalized, f"{block}.mlp.fc.weight",
                                f"{block}.mlp.fc.bias", f"{source}.mlp.c_fc")
            hidden = fixed_gelu(hidden)
            projected = self._gemv(hidden, f"{block}.mlp.proj.weight",
                                   f"{block}.mlp.proj.bias", f"{source}.mlp.c_proj")
            x = x + projected
        normalized = fixed_layer_norm(
            x, self.parameters["final_ln.weight"], self.parameters["final_ln.bias"]
        )
        input_codes = self._activation_codes(normalized, "lm_head.input")
        scaled_input = input_codes * self.activation_scales["lm_head.input"]
        accum = scaled_input @ self.codes["token_embedding.weight"].astype(np.int64).T
        logits = round_shift(
            accum * self.weight_scales["token_embedding.weight"],
            2*Q_SCALE-Q_VALUE,
        )
        self.last_logits = logits[-1]
        return logits

    def generate(self, prompt_ids: list[int], count: int) -> list[int]:
        if len(prompt_ids) + count > 32:
            raise ValueError("context capacity exceeded")
        tokens = list(prompt_ids)
        output = []
        for _ in range(count):
            token = int(np.argmax(self.forward(tokens)[-1]))
            output.append(token)
            if token == 50256:
                break
            tokens.append(token)
        return output
