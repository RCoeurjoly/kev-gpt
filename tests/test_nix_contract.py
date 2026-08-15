import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
EXPECTED_NIXPKGS = "6fd329b2adfecb86ae49c1cba89689bd0f229e04"


class NixContractTest(unittest.TestCase):
    def test_locked_nixpkgs_revision(self):
        lock = json.loads((ROOT / "flake.lock").read_text(encoding="utf-8"))
        self.assertEqual(
            lock["nodes"]["nixpkgs"]["locked"]["rev"], EXPECTED_NIXPKGS
        )

    def test_eda_adapter_is_narrow_and_external(self):
        source = (ROOT / "nix" / "eda-toolchain.nix").read_text(encoding="utf-8")
        self.assertIn("task3-main", source)
        self.assertIn("builtins.getFlake", source)
        self.assertNotIn("circt", source.lower())
        self.assertNotIn("torchMlir", source)


if __name__ == "__main__":
    unittest.main()
