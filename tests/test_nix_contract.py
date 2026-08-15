import json
import pathlib
import subprocess
import tempfile
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

    def test_ypcb_constraints_match_top_ports_and_50mhz_clock(self):
        constraints = (ROOT / "fpga/constraints/kintex_selftest.xdc").read_text()
        self.assertIn("[get_ports {LED[0]}]", constraints)
        self.assertIn("[get_ports {LED[1]}]", constraints)
        self.assertIn("[get_ports {LED[2]}]", constraints)
        self.assertNotIn("led_3bits_tri_o", constraints)
        derivation = (ROOT / "nix/kintex-tinystories.nix").read_text()
        self.assertIn("--freq 50", derivation)

    def test_run_logged_preserves_failure_diagnostics_and_status(self):
        with tempfile.TemporaryDirectory() as directory:
            log = pathlib.Path(directory) / "tool.log"
            result = subprocess.run(
                [
                    "bash",
                    str(ROOT / "scripts" / "run-logged.sh"),
                    str(log),
                    "bash",
                    "-c",
                    "printf 'route failed: bad constraint\\n' >&2; exit 23",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(result.returncode, 23)
            self.assertEqual(result.stdout, "route failed: bad constraint\n")
            self.assertEqual(log.read_text(), "route failed: bad constraint\n")


if __name__ == "__main__":
    unittest.main()
