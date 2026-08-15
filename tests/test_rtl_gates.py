import pathlib
import subprocess
import tempfile
import unittest

from tinystories.rtl_memories import write_exp_lut, write_gelu_lut


ROOT = pathlib.Path(__file__).resolve().parents[1]


class RTLPrimitiveGateTest(unittest.TestCase):
    CASES = {
        "layernorm": "GPTNEO_LAYERNORM_PASS",
        "gelu": "GPTNEO_GELU_PASS",
        "attention": "GPTNEO_ATTN_PASS",
        "gemv": "GPTNEO_GEMV_PASS",
    }


    def test_primitive_simulators_emit_named_verdicts(self):
        for name, verdict in self.CASES.items():
            rtl = ROOT / "fpga" / "rtl" / f"gptneo_{name if name != 'gemv' else 'resident_gemv'}.sv"
            testbench = ROOT / "fpga" / "tb" / f"tb_gptneo_{name}.sv"
            with self.subTest(primitive=name), tempfile.TemporaryDirectory() as temporary:
                if name == "gelu":
                    write_gelu_lut(pathlib.Path(temporary))
                if name == "attention":
                    write_exp_lut(pathlib.Path(temporary))
                executable = pathlib.Path(temporary) / name
                compile_result = subprocess.run(
                    ["iverilog", "-g2012", "-s", f"tb_gptneo_{name}", "-o", executable, rtl, testbench],
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
                simulation = subprocess.run(
                    ["vvp", executable], text=True, capture_output=True, cwd=temporary
                )
                self.assertEqual(
                    simulation.returncode, 0, simulation.stdout + simulation.stderr
                )
                self.assertIn(verdict, simulation.stdout)


if __name__ == "__main__":
    unittest.main()
