import pathlib
import json
import subprocess
import tempfile
import unittest

from tinystories.rtl_memories import write_exp_lut, write_gelu_lut
from tinystories.write_rtl_fixture import write_rtl_fixture


ROOT = pathlib.Path(__file__).resolve().parents[1]


class RTLPrimitiveGateTest(unittest.TestCase):
    def test_production_top_elides_redundant_physical_package_tag_compare(self):
        sequencer = (ROOT / "fpga/rtl/gptneo_sequencer.sv").read_text()
        top = (ROOT / "fpga/rtl/tinystories_interactive_top.sv").read_text()
        self.assertIn("parameter CHECK_PACKAGE_TAG=1", sequencer)
        self.assertIn("CHECK_PACKAGE_TAG&&package_tag!=GPTNEO_PACKAGE_TAG", sequencer)
        self.assertIn(".CHECK_PACKAGE_TAG(1'b0)", top)

    def test_production_top_exposes_read_only_in_band_debug_snapshot(self):
        top = (ROOT / "fpga/rtl/tinystories_interactive_top.sv").read_text()
        controller = (ROOT / "fpga/rtl/tinystories_packet_controller.sv").read_text()
        gelu = (ROOT / "fpga/rtl/gptneo_gelu.sv").read_text()
        self.assertIn(".seq_debug_status(seq_debug)", top)
        self.assertIn("8'h3f", controller)
        self.assertIn("8'h44", controller)
        self.assertIn("output wire [1:0] debug_state", gelu)
        self.assertIn("gelu_out_valid,gelu_in_ready,gelu_debug_state", (ROOT / "fpga/rtl/gptneo_sequencer.sv").read_text())
        self.assertNotIn("bscan_debug_snapshot", top)

    def test_layernorm_uses_the_iterative_divider(self):
        layernorm = (ROOT / "fpga/rtl/gptneo_layernorm.sv").read_text()
        self.assertIn("gptneo_iterative_divider norm_divider", layernorm)
        self.assertNotIn("norm_numerator /", layernorm)

    def test_kintex_pnr_flattens_cached_synthesis_before_nextpnr(self):
        nix = (ROOT / "nix/kintex-tinystories.nix").read_text()
        self.assertIn('pnrNetlist = pkgs.runCommand', nix)
        self.assertIn('read_json ${synthesis}/design.json; flatten; write_json design.json', nix)
        self.assertIn('--json ${pnrNetlist}/design.json', nix)

    CASES = {
        "iterative_divider": "GPTNEO_DIVIDER_PASS",
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
                sources = [rtl]
                if name in {"attention", "gemv", "layernorm"}:
                    sources.append(ROOT / "fpga/rtl/gptneo_iterative_divider.sv")
                compile_result = subprocess.run(
                    ["iverilog", "-g2012", "-s", f"tb_gptneo_{name}", "-o", executable, *sources, testbench],
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

    def test_gemv_external_memory_handles_delayed_unaligned_reads(self):
        with tempfile.TemporaryDirectory() as temporary:
            executable = pathlib.Path(temporary) / "gemv_external_memory"
            compile_result = subprocess.run(
                [
                    "iverilog", "-g2012", "-s", "tb_gptneo_external_memory",
                    "-o", executable,
                    ROOT / "fpga/rtl/gptneo_resident_gemv.sv",
                    ROOT / "fpga/rtl/gptneo_iterative_divider.sv",
                    ROOT / "fpga/tb/tb_gptneo_external_memory.sv",
                ],
                text=True,
                capture_output=True,
            )
            self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
            simulation = subprocess.run(
                ["vvp", executable], text=True, capture_output=True
            )
            self.assertEqual(
                simulation.returncode, 0, simulation.stdout + simulation.stderr
            )
            self.assertIn("GPTNEO_EXTERNAL_MEMORY_PASS requests=2", simulation.stdout)


class RTLSequencerGateTest(unittest.TestCase):
    def test_sequencer_forwards_model_reads_to_external_memory(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            fixture = directory / "fixture"
            write_rtl_fixture(ROOT / "model_packages" / "tinystories-1m", fixture)
            write_gelu_lut(fixture)
            write_exp_lut(fixture)
            executable = directory / "sequencer_external"
            sources = [
                ROOT / "fpga/rtl/gptneo_sequencer.sv",
                ROOT / "fpga/rtl/gptneo_layernorm.sv",
                ROOT / "fpga/rtl/gptneo_gelu.sv",
                ROOT / "fpga/rtl/gptneo_attention.sv",
                ROOT / "fpga/rtl/gptneo_iterative_divider.sv",
                ROOT / "fpga/rtl/gptneo_resident_gemv.sv",
                ROOT / "fpga/tb/tb_gptneo_sequencer_external_smoke.sv",
            ]
            compilation = subprocess.run(
                [
                    "iverilog", "-g2012", "-s", "tb_gptneo_sequencer_external_smoke",
                    f"-I{fixture}", "-o", executable, *sources,
                ],
                text=True,
                capture_output=True,
            )
            self.assertEqual(compilation.returncode, 0, compilation.stderr)
            simulation = subprocess.run(
                ["vvp", executable], cwd=fixture, text=True, capture_output=True
            )
            self.assertEqual(
                simulation.returncode, 0, simulation.stdout + simulation.stderr
            )
            self.assertIn("GPTNEO_SEQUENCER_EXTERNAL_PASS requests=3", simulation.stdout)

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
                ROOT / "fpga/rtl/gptneo_iterative_divider.sv",
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
