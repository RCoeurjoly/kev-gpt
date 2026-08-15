import unittest

from tinystories.gptneo_schema import validate_manifest


class GPTNeoSchemaTest(unittest.TestCase):
    def valid_manifest(self):
        return {
            "schema_version": 1,
            "model_type": "gpt_neo",
            "n_layer": 8,
            "hidden_size": 64,
            "n_head": 16,
            "vocab_size": 50257,
            "source_revision": "ac533fb8b4f69c71894bf96badfe11e6294d9fcf",
            "max_context": 32,
            "local_window": 256,
            "tie_word_embeddings": True,
            "activation_function": "gelu_new",
        }

    def test_accepts_pinned_shape(self):
        manifest = validate_manifest(self.valid_manifest())
        self.assertEqual(manifest["max_context"], 32)
        self.assertEqual(manifest["head_dim"], 4)

    def test_rejects_context_larger_than_local_window(self):
        value = self.valid_manifest()
        value["max_context"] = 512
        with self.assertRaisesRegex(ValueError, "max_context <= local_window"):
            validate_manifest(value)

    def test_rejects_unsupported_architecture(self):
        value = self.valid_manifest()
        value["model_type"] = "llama"
        with self.assertRaisesRegex(ValueError, "model_type"):
            validate_manifest(value)

    def test_rejects_untied_embeddings(self):
        value = self.valid_manifest()
        value["tie_word_embeddings"] = False
        with self.assertRaisesRegex(ValueError, "tie_word_embeddings"):
            validate_manifest(value)


if __name__ == "__main__":
    unittest.main()
