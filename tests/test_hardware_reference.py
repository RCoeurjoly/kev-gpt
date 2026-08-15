import json
import pathlib
import unittest

import numpy as np

from tinystories.hardware_reference import (
    FixedGPTNeo, fixed_gelu, fixed_layer_norm, round_shift, trunc_divide,
)


PACKAGE = pathlib.Path(__file__).resolve().parents[1] / "model_packages/tinystories-1m"


class FixedPrimitiveTest(unittest.TestCase):
    def test_signed_rounding_and_truncating_division(self):
        np.testing.assert_array_equal(round_shift(np.array([-7, 7]), 2), [-2, 2])
        np.testing.assert_array_equal(trunc_divide(np.array([-7, 7]), 2), [-3, 3])

    def test_layernorm_beta_and_gelu(self):
        values = np.full((1, 4), 5 << 16, dtype=np.int64)
        result = fixed_layer_norm(values, np.ones(4, dtype=np.int64) << 16,
                                  np.array([10, 20, 30, 40]))
        np.testing.assert_array_equal(result[0], [10, 20, 30, 40])
        self.assertEqual(int(fixed_gelu(np.array([0]))[0]), 0)


class FixedModelTest(unittest.TestCase):
    def test_fixed_reference_is_deterministic_for_committed_prompts(self):
        model = FixedGPTNeo(PACKAGE)
        cases = json.loads((PACKAGE / "regressions.json").read_text())["regressions"]
        expected = [
            [11, 612, 373, 257, 1310, 2576, 3706, 20037],
            [11, 257, 1263, 11, 4077, 15061, 13, 1375, 2227, 284, 2298, 340],
            [366, 5812, 836, 470, 340, 373],
        ]
        for case, expected_tokens in zip(cases, expected):
            first = model.generate(case["prompt_ids"], case["requested_tokens"])
            second = model.generate(case["prompt_ids"], case["requested_tokens"])
            self.assertEqual(first, second)
            self.assertEqual(first, expected_tokens)


if __name__ == "__main__":
    unittest.main()
