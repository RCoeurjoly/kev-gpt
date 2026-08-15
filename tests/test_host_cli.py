import pathlib
import subprocess
import sys
import ctypes
import tempfile
import unittest

from host import kevin_jtag_cli


ROOT = pathlib.Path(__file__).resolve().parents[1]


class HostCLITest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
