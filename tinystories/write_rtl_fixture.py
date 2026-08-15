"""Translate an authenticated TinyStories package into fixed-point RTL images."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import struct

import numpy as np


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_record(path: pathlib.Path) -> dict[str, int | str]:
    data = path.read_bytes()
    return {"size": len(data), "sha256": _sha256(data)}


def _validate_package(package: pathlib.Path) -> tuple[dict, dict]:
    receipt = json.loads((package / "receipt.json").read_text())
    for name, expected in receipt["files"].items():
        actual = _file_record(package / name)
        if actual != expected:
            raise ValueError(f"sha256 mismatch for {name}: {actual['sha256']}")
    manifest = json.loads((package / "manifest.json").read_text())
    if _sha256((package / "manifest.json").read_bytes()) != receipt["manifest_sha256"]:
        raise ValueError("sha256 mismatch for manifest.json")
    return manifest, receipt


def _fixed(values: np.ndarray, fractional_bits: int) -> np.ndarray:
    scaled = np.rint(np.asarray(values, dtype=np.float64) * (1 << fractional_bits))
    return np.clip(scaled, -(1 << 31), (1 << 31) - 1).astype("<i4")


def _identifier(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").upper()


def _write_words(path: pathlib.Path, image: bytes) -> None:
    padded = image + b"\0" * (-len(image) % 4)
    with path.open("w") as stream:
        for offset in range(0, len(padded), 4):
            stream.write(f"{struct.unpack_from('<I', padded, offset)[0]:08x}\n")


def write_rtl_fixture(package_dir: pathlib.Path, output_dir: pathlib.Path) -> dict:
    package = pathlib.Path(package_dir)
    output = pathlib.Path(output_dir)
    manifest, receipt = _validate_package(package)

    weight_image = bytearray((package / "weights.bin").read_bytes())
    scale_image = bytearray((package / "scales.bin").read_bytes())
    for descriptor in manifest["tensors"].values():
        if descriptor["format"] != "float32":
            continue
        offset = int(descriptor["offset"])
        count = int(np.prod(descriptor["logical_shape"]))
        values = np.frombuffer(weight_image, dtype="<f4", count=count, offset=offset).copy()
        converted = _fixed(values, 16).tobytes()
        weight_image[offset:offset + len(converted)] = converted
    if scale_image:
        scale_values = np.frombuffer(scale_image, dtype="<f4").copy()
        scale_image[:] = _fixed(scale_values, 30).tobytes()

    activation_image = bytearray()
    activation_descriptors = {}
    for name, raw_values in sorted(manifest["activation_scales"].items()):
        values = np.asarray(raw_values, dtype=np.float32).reshape(-1)
        activation_descriptors[name] = {
            "offset": len(activation_image), "count": int(values.size)
        }
        activation_image.extend(_fixed(values, 30).tobytes())

    scale_base = len(weight_image)
    activation_base = scale_base + len(scale_image)
    resident_image = bytes(weight_image + scale_image + activation_image)

    regressions_document = json.loads((package / "regressions.json").read_text())
    prompt_tokens = []
    expected_tokens = []
    regressions = []
    for case in regressions_document["regressions"]:
        record = dict(case)
        record["prompt_offset"] = len(prompt_tokens)
        record["expected_offset"] = len(expected_tokens)
        prompt_tokens.extend(case["prompt_ids"])
        expected_tokens.extend(case["output_ids"])
        regressions.append(record)

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"fixture output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "model_image.mem"
    _write_words(model_path, resident_image)
    (output / "prompt_tokens.mem").write_text(
        "".join(f"{token:04x}\n" for token in prompt_tokens)
    )
    (output / "expected_tokens.mem").write_text(
        "".join(f"{token:04x}\n" for token in expected_tokens)
    )

    header = [
        "`ifndef GPTNEO_PACKAGE_SVH", "`define GPTNEO_PACKAGE_SVH",
        f"localparam integer GPTNEO_MODEL_IMAGE_BYTES = {len(resident_image)};",
        f"localparam integer GPTNEO_MODEL_IMAGE_WORDS = {(len(resident_image)+3)//4};",
        f"localparam integer GPTNEO_SCALE_BASE = {scale_base};",
        f"localparam integer GPTNEO_ACTIVATION_SCALE_BASE = {activation_base};",
        f"localparam integer GPTNEO_REGRESSION_COUNT = {len(regressions)};",
    ]
    for name, descriptor in sorted(manifest["tensors"].items()):
        prefix = f"GPTNEO_TENSOR_{_identifier(name)}"
        header.extend([
            f"localparam integer {prefix}_OFFSET = {descriptor['offset']};",
            f"localparam integer {prefix}_SCALE_OFFSET = GPTNEO_SCALE_BASE + {descriptor['scale_offset']};",
            f"localparam integer {prefix}_BITS = {descriptor['bits']};",
        ])
    for name, descriptor in sorted(activation_descriptors.items()):
        prefix = f"GPTNEO_ACT_{_identifier(name)}"
        header.extend([
            f"localparam integer {prefix}_OFFSET = GPTNEO_ACTIVATION_SCALE_BASE + {descriptor['offset']};",
            f"localparam integer {prefix}_COUNT = {descriptor['count']};",
        ])
    for index, case in enumerate(regressions):
        header.extend([
            f"localparam integer GPTNEO_CASE_{index}_PROMPT_OFFSET = {case['prompt_offset']};",
            f"localparam integer GPTNEO_CASE_{index}_PROMPT_LENGTH = {len(case['prompt_ids'])};",
            f"localparam integer GPTNEO_CASE_{index}_EXPECTED_OFFSET = {case['expected_offset']};",
            f"localparam integer GPTNEO_CASE_{index}_EXPECTED_LENGTH = {len(case['output_ids'])};",
        ])
    header.extend(["`endif", ""])
    (output / "gptneo_package.svh").write_text("\n".join(header))

    inputs = {name: dict(value) for name, value in sorted(receipt["files"].items())}
    outputs = {
        name: _file_record(output / name)
        for name in ["model_image.mem", "prompt_tokens.mem", "expected_tokens.mem",
                     "gptneo_package.svh"]
    }
    document = {
        "schema_version": 1,
        "numeric_formats": {"parameter": "q16.16", "scale": "q2.30"},
        "image_layout": {
            "weights_offset": 0, "weights_bytes": len(weight_image),
            "scales_offset": scale_base, "scales_bytes": len(scale_image),
            "activation_scales_offset": activation_base,
            "activation_scales_bytes": len(activation_image),
            "total_bytes": len(resident_image),
        },
        "inputs": inputs,
        "outputs": outputs,
        "activation_scales": activation_descriptors,
        "regressions": regressions,
    }
    (output / "fixture.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    write_rtl_fixture(args.package, args.output)


if __name__ == "__main__":
    main()
