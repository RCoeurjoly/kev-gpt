import json
import pathlib
import tempfile
import unittest

import numpy as np
import torch

from tinystories.import_gptneo import import_model


class GPTNeoImportTest(unittest.TestCase):
    def make_source(self, *, tied=True, omit=None):
        tmp = tempfile.TemporaryDirectory()
        root = pathlib.Path(tmp.name)
        config = {
            "model_type": "gpt_neo",
            "num_layers": 8,
            "hidden_size": 64,
            "num_heads": 16,
            "vocab_size": 50257,
            "window_size": 256,
            "max_position_embeddings": 2048,
            "activation_function": "gelu_new",
        }
        (root / "config.json").write_text(json.dumps(config))
        (root / "tokenizer_config.json").write_text("{}")
        embedding = torch.arange(50257 * 64, dtype=torch.float32).reshape(50257, 64)
        state = {
            "transformer.wte.weight": embedding,
            "transformer.wpe.weight": torch.arange(2048 * 64, dtype=torch.float32).reshape(2048, 64),
            "transformer.ln_f.weight": torch.ones(64),
            "transformer.ln_f.bias": torch.full((64,), 0.25),
            "lm_head.weight": embedding if tied else embedding.clone().add_(1),
        }
        for layer in range(8):
            prefix = f"transformer.h.{layer}"
            state.update({
                f"{prefix}.ln_1.weight": torch.full((64,), layer + 1.0),
                f"{prefix}.ln_1.bias": torch.full((64,), layer + 0.1),
                f"{prefix}.attn.attention.q_proj.weight": torch.full((64, 64), layer + 10.0),
                f"{prefix}.attn.attention.k_proj.weight": torch.full((64, 64), layer + 20.0),
                f"{prefix}.attn.attention.v_proj.weight": torch.full((64, 64), layer + 30.0),
                f"{prefix}.attn.attention.out_proj.weight": torch.full((64, 64), layer + 40.0),
                f"{prefix}.attn.attention.out_proj.bias": torch.zeros(64),
                f"{prefix}.ln_2.weight": torch.full((64,), layer + 2.0),
                f"{prefix}.ln_2.bias": torch.full((64,), layer + 0.2),
                f"{prefix}.mlp.c_fc.weight": torch.full((256, 64), layer + 50.0),
                f"{prefix}.mlp.c_fc.bias": torch.zeros(256),
                f"{prefix}.mlp.c_proj.weight": torch.full((64, 256), layer + 60.0),
                f"{prefix}.mlp.c_proj.bias": torch.zeros(64),
            })
        if omit:
            del state[omit]
        torch.save(state, root / "pytorch_model.bin")
        return tmp, root

    def test_imports_canonical_qkv_layernorm_and_position_slice(self):
        tmp, root = self.make_source()
        with tmp:
            model = import_model(root)
        self.assertEqual(model.tensors["position_embedding.weight"].shape, (32, 64))
        self.assertEqual(float(model.tensors["blocks.0.attn.q.weight"][0, 0]), 10.0)
        self.assertEqual(float(model.tensors["blocks.0.attn.k.weight"][0, 0]), 20.0)
        self.assertEqual(float(model.tensors["blocks.0.attn.v.weight"][0, 0]), 30.0)
        np.testing.assert_allclose(model.tensors["blocks.0.ln1.bias"], 0.1)
        self.assertIs(model.tensors["lm_head.weight"], model.tensors["token_embedding.weight"])

    def test_rejects_untied_head(self):
        tmp, root = self.make_source(tied=False)
        with tmp, self.assertRaisesRegex(ValueError, "tied"):
            import_model(root)

    def test_omitted_head_uses_huggingface_tied_weight_encoding(self):
        tmp, root = self.make_source(omit="lm_head.weight")
        with tmp:
            model = import_model(root)
        self.assertIs(model.tensors["lm_head.weight"], model.tensors["token_embedding.weight"])

    def test_rejects_missing_tensor_with_source_name(self):
        missing = "transformer.h.3.attn.attention.q_proj.weight"
        tmp, root = self.make_source(omit=missing)
        with tmp, self.assertRaisesRegex(ValueError, missing):
            import_model(root)


if __name__ == "__main__":
    unittest.main()
