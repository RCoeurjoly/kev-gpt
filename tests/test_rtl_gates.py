import pathlib
import json
import subprocess
import tempfile
import unittest

from tinystories.rtl_memories import write_exp_lut, write_gelu_lut
from tinystories.write_rtl_fixture import write_rtl_fixture


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


class RTLSequencerGateTest(unittest.TestCase):
    def test_three_streams_and_corrupt_package_verdict(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            fixture = directory / "fixture"
            write_rtl_fixture(ROOT / "model_packages" / "tinystories-1m", fixture)
            write_gelu_lut(fixture)
            write_exp_lut(fixture)
            object_directory = directory / "obj"
            sources = [
                ROOT / "fpga/rtl/gptneo_sequencer.sv",
                ROOT / "fpga/rtl/gptneo_layernorm.sv",
                ROOT / "fpga/rtl/gptneo_gelu.sv",
                ROOT / "fpga/rtl/gptneo_attention.sv",
                ROOT / "fpga/rtl/gptneo_resident_gemv.sv",
                ROOT / "fpga/tb/tb_gptneo_sequencer.sv",
            ]
            compilation = subprocess.run([
                "verilator", "--binary", "--timing", "-Wno-fatal",
                f"-I{fixture}", "--top-module", "tb_gptneo_sequencer",
                "--Mdir", object_directory, "-o", "sequencer_sim", *sources,
            ], text=True, capture_output=True)
            self.assertEqual(compilation.returncode, 0, compilation.stderr)
            simulation = subprocess.run(
                [object_directory / "sequencer_sim"], cwd=fixture,
                text=True, capture_output=True
            )
            self.assertEqual(
                simulation.returncode, 0, simulation.stdout + simulation.stderr
            )
            regressions = json.loads((fixture / "fixture.json").read_text())["regressions"]
            for index, case in enumerate(regressions):
                self.assertIn(
                    f"GPTNEO_SEQ_PASS case={index} tokens={len(case['output_ids'])}/{len(case['output_ids'])}",
                    simulation.stdout,
                )
            self.assertIn(
                "GPTNEO_SEQ_NEGATIVE_PASS error=PACKAGE_HASH", simulation.stdout
            )


if __name__ == "__main__":
    unittest.main()
