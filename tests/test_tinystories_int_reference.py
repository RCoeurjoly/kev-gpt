import json
import pathlib
import shutil
import tempfile
import unittest

import numpy as np

from tinystories.int_reference import (
    IntegerGPTNeo,
    KVCache,
    causal_softmax,
    gelu_new,
    layer_norm,
    merge_heads,
    round_shift_signed,
    saturate_int8,
    split_heads,
    trace_sha256,
)


PACKAGE = pathlib.Path(__file__).resolve().parents[1] / "model_packages" / "tinystories-1m"


class IntegerPrimitiveTest(unittest.TestCase):
    def test_signed_rounding_and_saturation(self):
        values = np.array([-9, -7, -1, 1, 7, 9], dtype=np.int64)
        self.assertEqual(round_shift_signed(values, 2).tolist(), [-2, -2, 0, 0, 2, 2])
        self.assertEqual(saturate_int8(np.array([-200, -128, 127, 300])).tolist(), [-128, -128, 127, 127])

    def test_layernorm_retains_gamma_and_beta(self):
        result = layer_norm(
            np.array([1.0, 3.0]),
            gamma=np.array([2.0, 4.0]),
            beta=np.array([0.5, -0.5]),
            epsilon=0.0,
        )
        np.testing.assert_allclose(result, np.array([-1.5, 3.5]), rtol=0, atol=1e-6)

    def test_gelu_new_endpoints_and_origin(self):
        values = gelu_new(np.array([-8.0, 0.0, 8.0]))
        self.assertLess(abs(float(values[0])), 1e-5)
        self.assertEqual(float(values[1]), 0.0)
        self.assertAlmostEqual(float(values[2]), 8.0, places=5)

    def test_causal_softmax_masks_future_and_normalizes(self):
        probabilities = causal_softmax(np.array([[0.0, 100.0], [0.0, 0.0]]))
        np.testing.assert_allclose(probabilities.sum(axis=-1), 1.0)
        self.assertEqual(float(probabilities[0, 1]), 0.0)
        np.testing.assert_allclose(probabilities[1], [0.5, 0.5])

    def test_sixteen_head_split_merge_round_trip(self):
        value = np.arange(3 * 64).reshape(3, 64)
        heads = split_heads(value, n_head=16)
        self.assertEqual(heads.shape, (16, 3, 4))
        np.testing.assert_array_equal(merge_heads(heads), value)

    def test_kv_cache_append_reset_and_overflow(self):
        cache = KVCache(n_layer=2, max_context=2, n_head=16, head_dim=4)
        cache.append(0, np.ones((16, 4)), np.full((16, 4), 2))
        cache.append(0, np.full((16, 4), 3), np.full((16, 4), 4))
        self.assertEqual(cache.read(0)[0].shape, (16, 2, 4))
        with self.assertRaisesRegex(ValueError, "context"):
            cache.append(0, np.zeros((16, 4)), np.zeros((16, 4)))
        cache.reset()
        self.assertEqual(cache.read(0)[0].shape, (16, 0, 4))

    def test_corrupt_package_is_rejected_before_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            copied = pathlib.Path(temporary) / "package"
            shutil.copytree(PACKAGE, copied)
            (copied / "weights.bin").chmod(0o600)
            with (copied / "weights.bin").open("r+b") as stream:
                stream.seek(0)
                stream.write(b"X")
            with self.assertRaisesRegex(ValueError, "sha256"):
                IntegerGPTNeo(copied)


class IntegerFullReferenceTest(unittest.TestCase):
    def setUp(self):
        self.model = IntegerGPTNeo(PACKAGE)

    def test_three_committed_greedy_streams_and_trace_hashes(self):
        document = json.loads((PACKAGE / "regressions.json").read_text())
        self.assertEqual(len(document["regressions"]), 3)
        for case in document["regressions"]:
            prompt = case["prompt_ids"]
            count = case["requested_tokens"]
            expected = case["output_ids"]
            with self.subTest(prompt=prompt):
                self.assertEqual(self.model.generate(prompt, count), expected)
                trace = self.model.trace(prompt, count)
                self.assertEqual(trace_sha256(trace), case["trace_sha256"])
                self.model.reset()
                self.assertEqual(self.model.generate(prompt, count), expected)
                self.model.reset()

    def test_eos_and_context_limits(self):
        self.assertEqual(self.model.generate([50256], 4), [])
        with self.assertRaisesRegex(ValueError, "context"):
            self.model.generate([1] * 33, 1)

    def test_prefill_step_and_trace_are_deterministic(self):
        self.model.prefill([7454, 2402, 257])
        logits, token = self.model.step(640)
        self.assertEqual(logits.shape, (50257,))
        self.assertEqual(token, 11)
        first = self.model.trace([7454, 2402, 257, 640], 1)
        self.model.reset()
        second = self.model.trace([7454, 2402, 257, 640], 1)
        self.assertEqual(first.keys(), second.keys())
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])


if __name__ == "__main__":
    unittest.main()
