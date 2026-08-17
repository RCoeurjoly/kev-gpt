import pathlib
import hashlib
import subprocess
import sys
import ctypes
import tempfile
import unittest
from types import SimpleNamespace

from host import kevin_jtag_cli


ROOT = pathlib.Path(__file__).resolve().parents[1]


class HostCLITest(unittest.TestCase):
    def test_inference_receipt_is_provenance_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            package = root / "package"
            package.mkdir()
            manifest = package / "manifest.json"
            manifest.write_bytes(b'{"model":"tinystories-1m"}\n')
            bitstream = root / "tinystories-interactive.bit"
            bitstream.write_bytes(b"verified-bitstream")
            args = SimpleNamespace(
                package=str(package),
                program=str(bitstream),
                prompt="Once upon a time",
                max_new_tokens=16,
            )

            receipt = kevin_jtag_cli.build_inference_receipt(
                args,
                prompt_ids=[7454, 2402, 257, 640],
                output_ids=list(range(16)),
                cycles=1234,
                wall_seconds=1.25,
            )

            self.assertEqual(receipt["schema"], "kev-gpt-kintex-inference-v2")
            self.assertEqual(receipt["prompt"], "Once upon a time")
            self.assertEqual(receipt["prompt_ids"], [7454, 2402, 257, 640])
            self.assertEqual(receipt["requested_tokens"], 16)
            self.assertEqual(receipt["output_ids"], list(range(16)))
            self.assertEqual(receipt["cycles"], 1234)
            self.assertEqual(receipt["wall_seconds"], 1.25)
            self.assertEqual(receipt["timing_boundary"], "request-submit-to-reply")
            self.assertEqual(receipt["transport"], "jtag-debug-baseline")
            self.assertEqual(receipt["board"], "YPCB-00338-1P1")
            self.assertEqual(receipt["fpga"], "xc7k480tffg1156-1")
            self.assertEqual(
                receipt["package_manifest_sha256"],
                hashlib.sha256(manifest.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                receipt["bitstream_sha256"],
                hashlib.sha256(bitstream.read_bytes()).hexdigest(),
            )

    def test_client_contains_no_host_model_inference(self):
        source = pathlib.Path(kevin_jtag_cli.__file__).read_text()
        for forbidden in ["tinystories.int_reference", "import torch", "AutoModel", "transformers."]:
            self.assertNotIn(forbidden, source)

    def test_packet_selftest_command(self):
        result = subprocess.run(
            [sys.executable, "-m", "host.kevin_jtag_cli", "packet-selftest"],
            cwd=ROOT, text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("JTAG_PACKET_SELFTEST_PASS", result.stdout)

    def test_tokenizer_uses_packaged_assets(self):
        tokenizer = kevin_jtag_cli.load_tokenizer(
            ROOT / "model_packages" / "tinystories-1m"
        )
        tokens = tokenizer.encode("Once upon a time", add_special_tokens=False)
        self.assertEqual(tokens, [7454, 2402, 257, 640])

    def test_native_transport_exports_checked_api(self):
        with tempfile.TemporaryDirectory() as temporary:
            library_path = pathlib.Path(temporary) / "libkevin_jtag.so"
            build = subprocess.run(
                ["cc", "-shared", "-fPIC", ROOT / "host/jtag_transport.c",
                 "-o", library_path, "-lftdi1"],
                text=True, capture_output=True,
            )
            self.assertEqual(build.returncode, 0, build.stderr)
            library = ctypes.CDLL(str(library_path))
            library.kj_user1_exchange.restype = ctypes.c_int
            self.assertLess(library.kj_user1_exchange(None, 0, None, 0, 1), 0)
            library.kj_close()

    def test_native_transport_enables_digilent_hs3_buffers(self):
        source = (ROOT / "host/jtag_transport.c").read_text()
        self.assertIn("MPSSE_SET_HIGH", source)
        self.assertIn("MPSSE_SET_LOW, 0x88, 0x8b", source)
        self.assertIn("MPSSE_SET_HIGH, 0x20, 0x30", source)

    def test_infer_tokens_uses_transport_reply_only(self):
        reply = kevin_jtag_cli.encode_reply(0, [11, 12], 123)

        class FakeTransport:
            def __init__(self):
                self.responses = [b"\0" * 8, b"\0R", reply[1:]]
                self.requests = []

            def exchange(self, data, receive_length, timeout_ms):
                self.requests.append(bytes(data))
                return self.responses.pop(0)

        transport = FakeTransport()
        result = kevin_jtag_cli.infer_tokens(transport, [1, 2], 2, timeout_s=1)
        self.assertEqual(result, (0, [11, 12], 123))
        self.assertEqual(kevin_jtag_cli.decode_request(transport.requests[0]), (1, [1, 2], 2))

    def test_debug_snapshot_is_read_over_existing_user1_transport(self):
        payload = (
            b"DBG1" + (123).to_bytes(4, "little") + bytes([18]) + bytes(11)
            + (-123456).to_bytes(4, "little", signed=True)
            + (654321).to_bytes(4, "little", signed=True)
            + (0x123456).to_bytes(3, "little")
            + (-7).to_bytes(1, "little", signed=True)
            + (-5555).to_bytes(4, "little", signed=True)
            + (-7777).to_bytes(4, "little", signed=True)
            + (8888).to_bytes(4, "little", signed=True)
        )

        class FakeTransport:
            def __init__(self):
                self.responses = [b"\0", b"\0\0DB", b"G1" + payload[4:]]
                self.requests = []

            def exchange(self, data, receive_length, timeout_ms):
                self.requests.append(bytes(data))
                return self.responses.pop(0)

        transport = FakeTransport()
        snapshot = kevin_jtag_cli.read_debug_snapshot(transport, timeout_s=1)
        self.assertEqual(snapshot["cycles"], 123)
        self.assertEqual(snapshot["sequencer_state"], 18)
        self.assertEqual(snapshot["embedding_x_q16"], -123456)
        self.assertEqual(snapshot["embedding_token_component_q16"], 654321)
        self.assertEqual(snapshot["embedding_position_scale_q24"], 0x123456)
        self.assertEqual(snapshot["embedding_position_code"], -7)
        self.assertEqual(snapshot["layernorm_y0_q16"], -5555)
        self.assertEqual(snapshot["layernorm_normalized0_q16"], -7777)
        self.assertEqual(snapshot["layernorm_affine0_q16"], 8888)
        self.assertEqual(transport.requests[0], b"?")


if __name__ == "__main__":
    unittest.main()
