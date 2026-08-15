import pathlib
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class JTAGRTLTest(unittest.TestCase):
    def test_async_fifo_crosses_unrelated_clocks_without_loss(self):
        with tempfile.TemporaryDirectory() as temporary:
            executable = pathlib.Path(temporary) / "fifo"
            compile_result = subprocess.run(
                [
                    "iverilog", "-g2012", "-s", "tb_async_fifo", "-o", executable,
                    ROOT / "fpga/rtl/async_fifo.sv",
                    ROOT / "fpga/tb/tb_async_fifo.sv",
                ],
                text=True, capture_output=True,
            )
            self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
            simulation = subprocess.run(
                ["vvp", executable], text=True, capture_output=True, timeout=10
            )
            self.assertEqual(simulation.returncode, 0, simulation.stdout + simulation.stderr)
            self.assertIn("ASYNC_FIFO_PASS", simulation.stdout)

    def test_bscan_bridge_moves_bytes_in_both_directions(self):
        with tempfile.TemporaryDirectory() as temporary:
            executable = pathlib.Path(temporary) / "bscan"
            compile_result = subprocess.run(
                [
                    "iverilog", "-g2012", "-DBSCAN_SIM", "-s",
                    "tb_bscan_packet_endpoint", "-o", executable,
                    ROOT / "fpga/rtl/async_fifo.sv",
                    ROOT / "fpga/rtl/bscan_packet_endpoint.sv",
                    ROOT / "fpga/tb/tb_bscan_packet_endpoint.sv",
                ], text=True, capture_output=True,
            )
            self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
            simulation = subprocess.run(
                ["vvp", executable], text=True, capture_output=True, timeout=10
            )
            self.assertEqual(simulation.returncode, 0, simulation.stdout + simulation.stderr)
            self.assertIn("BSCAN_ENDPOINT_PASS bytes=64", simulation.stdout)

    def test_packet_controller_drives_accelerator_and_frames_reply(self):
        with tempfile.TemporaryDirectory() as temporary:
            executable = pathlib.Path(temporary) / "controller"
            compile_result = subprocess.run(
                ["iverilog", "-g2012", "-s", "tb_tinystories_packet_controller",
                 "-o", executable,
                 ROOT / "fpga/rtl/tinystories_packet_controller.sv",
                 ROOT / "fpga/tb/tb_tinystories_packet_controller.sv"],
                text=True, capture_output=True,
            )
            self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
            simulation = subprocess.run(
                ["vvp", executable], text=True, capture_output=True, timeout=10
            )
            self.assertEqual(simulation.returncode, 0, simulation.stdout + simulation.stderr)
            self.assertIn("PACKET_CONTROLLER_PASS tokens=2", simulation.stdout)


if __name__ == "__main__":
    unittest.main()
