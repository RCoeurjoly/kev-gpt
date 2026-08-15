import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_kintex_selftest.sh"


class KintexSelftestBuildScriptTest(unittest.TestCase):
    def test_locked_direct_rtl_build_and_provenance_contract(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("nix/kintex-selftest.nix", source)
        self.assertIn("fpga/rtl/kintex_selftest_top.sv", source)
        self.assertIn("--no-update-lock-file", source)
        self.assertIn("flake.lock", source)
        self.assertIn("task3-main/flake.lock", source)
        self.assertIn("sha256sum", source)
        self.assertIn("git -C", source)
        self.assertIn("provenance.json", source)
        self.assertIn(
            "github:NixOS/nixpkgs/6fd329b2adfecb86ae49c1cba89689bd0f229e04#python3",
            source,
        )
        self.assertNotIn("nix flake update", source)
        self.assertNotIn("nix build .#", source)
        self.assertNotIn("matmul-selftest-bitstream", source)


if __name__ == "__main__":
    unittest.main()
