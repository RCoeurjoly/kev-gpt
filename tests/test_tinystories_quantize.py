import unittest
import pathlib
import tempfile
import math
import json

import numpy as np

from tinystories.import_gptneo import ImportedGPTNeo
from tinystories.package_io import write_package
from tinystories.quantize import (
    pack_int4,
    pack_signed,
    quantize_model,
    quantize_symmetric,
    measure_logits_quality,
    fake_quantize_state_dict,
    fake_quantize_activation,
    select_quality_candidate,
)


class TinyStoriesQuantizeTest(unittest.TestCase):
    def synthetic_model(self):
        embedding = np.arange(24, dtype=np.float32).reshape(6, 4) / 8
        return ImportedGPTNeo(
            tensors={
                "token_embedding.weight": embedding,
                "lm_head.weight": embedding,
                "position_embedding.weight": np.ones((4, 4), dtype=np.float32),
                "blocks.0.attn.q.weight": np.eye(4, dtype=np.float32),
                "blocks.0.ln1.bias": np.zeros(4, dtype=np.float32),
            },
            manifest={"schema_version": 1, "max_context": 4},
            source_config={},
            tokenizer_files=(),
        )

    def test_symmetric_int4_per_channel_round_trip_bounds(self):
        weight = np.array([[-9.0, -1.0, 3.0], [0.0, 4.0, 8.0]], dtype=np.float32)
        quantized, scale = quantize_symmetric(weight, bits=4, axis=0)
        self.assertGreaterEqual(int(quantized.min()), -8)
        self.assertLessEqual(int(quantized.max()), 7)
        self.assertEqual(scale.shape, (weight.shape[0],))
        restored = quantized.astype(np.float32) * scale[:, None]
        self.assertLessEqual(float(np.max(np.abs(restored - weight))), float(scale.max()) / 2 + 1e-6)

    def test_int4_packing_uses_low_logical_index_in_low_nibble(self):
        values = np.array([-8, -1, 0, 7], dtype=np.int8)
        self.assertEqual(pack_int4(values).tolist(), [0xF8, 0x70])

    def test_int6_packing_is_little_endian_and_dense(self):
        values = np.array([-32, -1, 0, 31], dtype=np.int8)
        self.assertEqual(pack_signed(values, bits=6).tolist(), [0xE0, 0x0F, 0x7C])

    def test_zero_channel_has_finite_nonzero_scale(self):
        quantized, scale = quantize_symmetric(np.zeros((2, 3), dtype=np.float32), bits=4, axis=0)
        self.assertTrue(np.all(np.isfinite(scale)))
        self.assertTrue(np.all(scale > 0))
        self.assertEqual(quantized.tolist(), [[0, 0, 0], [0, 0, 0]])

    def test_tied_matrix_is_packed_once(self):
        package = quantize_model(self.synthetic_model(), np.array([1, 2, 3]), max_context=4)
        role = package.manifest["tensor_roles"]["lm_head.weight"]
        self.assertEqual(role["alias_of"], "token_embedding.weight")
        self.assertNotIn("lm_head.weight", package.tensors)

    def test_mixed_precision_override_changes_emitted_storage(self):
        model = self.synthetic_model()
        default = quantize_model(model, np.array([1, 2, 3]), max_context=4)
        mixed = quantize_model(
            model,
            np.array([1, 2, 3]),
            max_context=4,
            bits_by_tensor={"token_embedding.weight": 8},
        )
        self.assertEqual(default.tensors["token_embedding.weight"].bits, 4)
        self.assertEqual(mixed.tensors["token_embedding.weight"].bits, 8)
        self.assertEqual(
            len(mixed.tensors["token_embedding.weight"].data),
            2 * len(default.tensors["token_embedding.weight"].data),
        )

    def test_activation_calibration_scales_are_recorded(self):
        package = quantize_model(
            self.synthetic_model(),
            np.array([1, 2, 3]),
            max_context=4,
            activation_scales={"blocks.0.mlp.fc.input": 0.125},
        )
        self.assertEqual(
            package.manifest["activation_scales"]["blocks.0.mlp.fc.input"],
            0.125,
        )

    def test_package_writer_is_byte_deterministic_and_receipted(self):
        package = quantize_model(self.synthetic_model(), np.array([1, 2, 3]), max_context=4)
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_manifest = write_package(package, pathlib.Path(first))
            second_manifest = write_package(package, pathlib.Path(second))
            self.assertEqual(first_manifest, second_manifest)
            self.assertEqual(
                (pathlib.Path(first) / "weights.bin").read_bytes(),
                (pathlib.Path(second) / "weights.bin").read_bytes(),
            )
            self.assertEqual(len(first_manifest["files"]["weights.bin"]["sha256"]), 64)
            receipt = json.loads((pathlib.Path(first) / "receipt.json").read_text())
            self.assertEqual(
                receipt["files"]["manifest.json"]["sha256"],
                receipt["manifest_sha256"],
            )

    def test_quality_selection_rejects_failures_then_minimizes_storage(self):
        candidates = [
            {"name": "small_bad", "perplexity_ratio": 1.11, "top1_agreement": 0.99,
             "packed_bytes": 100, "non_int4_tensors": 0},
            {"name": "large", "perplexity_ratio": 1.05, "top1_agreement": 0.95,
             "packed_bytes": 140, "non_int4_tensors": 1},
            {"name": "small_more_mixed", "perplexity_ratio": 1.09, "top1_agreement": 0.90,
             "packed_bytes": 120, "non_int4_tensors": 3},
            {"name": "small_less_mixed", "perplexity_ratio": 1.10, "top1_agreement": 0.91,
             "packed_bytes": 120, "non_int4_tensors": 2},
        ]
        self.assertEqual(select_quality_candidate(candidates)["name"], "small_less_mixed")

    def test_quality_selection_fails_with_best_observed_evidence(self):
        candidates = [
            {"name": "bad", "perplexity_ratio": 1.2, "top1_agreement": 0.8,
             "packed_bytes": 10, "non_int4_tensors": 0},
        ]
        with self.assertRaisesRegex(ValueError, "bad.*1.2.*0.8"):
            select_quality_candidate(candidates)

    def test_quality_metrics_use_identical_labels_and_report_top1_counts(self):
        fp32 = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float64)
        quantized = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        metrics = measure_logits_quality(fp32, quantized, np.array([0, 1]))
        self.assertAlmostEqual(metrics["fp32_perplexity"], 1.0 + math.exp(-2.0))
        expected_quantized = math.exp(
            (math.log1p(math.exp(-1.0)) + math.log1p(math.exp(1.0))) / 2.0
        )
        self.assertAlmostEqual(metrics["quantized_perplexity"], expected_quantized)
        self.assertEqual(metrics["top1_matches"], 1)
        self.assertEqual(metrics["top1_total"], 2)
        self.assertEqual(metrics["top1_agreement"], 0.5)

    def test_fake_quantization_uses_conv1d_output_columns(self):
        import torch

        weight = torch.tensor([[1.0, 100.0], [2.0, 1.0]])
        state = {"transformer.h.0.mlp.c_fc.weight": weight}
        result = fake_quantize_state_dict(state, {"transformer.h.0.mlp.c_fc.weight": 4})
        # Independent per-output-column scales preserve 2 in column 0 and 100 in column 1.
        self.assertAlmostEqual(float(result["transformer.h.0.mlp.c_fc.weight"][1, 0]), 2.0)
        self.assertAlmostEqual(float(result["transformer.h.0.mlp.c_fc.weight"][0, 1]), 100.0)

    def test_a8_activation_rounds_and_saturates_with_fixed_scale(self):
        values = np.array([-100.0, -0.26, 0.24, 100.0], dtype=np.float32)
        restored = fake_quantize_activation(values, scale=0.5)
        np.testing.assert_array_equal(restored, np.array([-64.0, -0.5, 0.0, 63.5]))

    def test_a8_activation_supports_per_channel_scales(self):
        values = np.array([[1.4, 1.4], [2.6, 2.6]], dtype=np.float32)
        restored = fake_quantize_activation(values, scale=np.array([1.0, 0.5]))
        np.testing.assert_array_equal(restored, np.array([[1.0, 1.5], [3.0, 2.5]]))


if __name__ == "__main__":
    unittest.main()
