import json
import pathlib
import shutil
import tempfile
import unittest

from tinystories.write_rtl_fixture import write_rtl_fixture


ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "model_packages" / "tinystories-1m"


class RTLFixtureTest(unittest.TestCase):
    def test_fixture_is_deterministic_and_contains_hardware_images(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_path = pathlib.Path(first)
            second_path = pathlib.Path(second)
            first_document = write_rtl_fixture(PACKAGE, first_path)
            second_document = write_rtl_fixture(PACKAGE, second_path)
            self.assertEqual(first_document, second_document)
            self.assertEqual(
                sorted(path.name for path in first_path.iterdir()),
                ["expected_tokens.mem", "fixture.json", "gptneo_package.svh",
                 "model_image.mem", "prompt_tokens.mem"],
            )
            for name in first_document["outputs"]:
                self.assertEqual(
                    (first_path / name).read_bytes(), (second_path / name).read_bytes()
                )
            self.assertEqual(first_document["numeric_formats"]["parameter"], "q16.16")
            self.assertEqual(first_document["numeric_formats"]["scale"], "q8.24")
            self.assertEqual(first_document["numeric_formats"]["scale_storage_bits"], 24)
            self.assertEqual(len(first_document["regressions"]), 3)
            header = (first_path / "gptneo_package.svh").read_text()
            self.assertIn("GPTNEO_PACKAGE_TAG", header)
            self.assertIn("GPTNEO_TENSOR_TOKEN_EMBEDDING_WEIGHT_OFFSET", header)
            self.assertIn("GPTNEO_ACT_LM_HEAD_INPUT_OFFSET", header)

    def test_corrupt_package_is_rejected_before_fixture_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            copied = pathlib.Path(temporary) / "package"
            output = pathlib.Path(temporary) / "output"
            shutil.copytree(PACKAGE, copied)
            (copied / "scales.bin").chmod(0o600)
            with (copied / "scales.bin").open("r+b") as stream:
                stream.seek(0)
                stream.write(b"X")
            with self.assertRaisesRegex(ValueError, "sha256"):
                write_rtl_fixture(copied, output)
            self.assertFalse(output.exists())

    def test_fixture_manifest_hashes_every_input_and_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary)
            document = write_rtl_fixture(PACKAGE, output)
            on_disk = json.loads((output / "fixture.json").read_text())
            self.assertEqual(document, on_disk)
            self.assertIn("manifest.json", document["inputs"])
            self.assertIn("weights.bin", document["inputs"])
            self.assertIn("model_image.mem", document["outputs"])
            self.assertEqual(len(document["outputs"]["model_image.mem"]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
