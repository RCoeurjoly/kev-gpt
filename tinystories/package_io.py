"""Deterministic serialization for FPGA-consumable model packages."""

from __future__ import annotations

import hashlib
import json
import pathlib

from .quantize import QuantizedPackage


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def write_receipt(out_dir: pathlib.Path) -> dict[str, object]:
    """Hash every completed package file except the receipt itself."""

    out_dir = pathlib.Path(out_dir)
    files = {}
    for path in sorted(out_dir.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.name == "receipt.json":
            continue
        data = path.read_bytes()
        files[path.name] = {"size": len(data), "sha256": _digest(data)}
    manifest_sha256 = files["manifest.json"]["sha256"]
    receipt = {
        "schema_version": 1,
        "manifest_sha256": manifest_sha256,
        "files": files,
    }
    (out_dir / "receipt.json").write_bytes(_json_bytes(receipt))
    return receipt


def write_package(pkg: QuantizedPackage, out_dir: pathlib.Path) -> dict[str, object]:
    """Write stable binary images, offsets, hashes, manifest, and receipt."""

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weight_image = bytearray()
    scale_image = bytearray()
    tensor_manifest: dict[str, dict[str, object]] = {}
    for name in sorted(pkg.tensors):
        tensor = pkg.tensors[name]
        weight_offset = len(weight_image)
        scale_offset = len(scale_image)
        weight_image.extend(tensor.data)
        scale_image.extend(tensor.scales)
        tensor_manifest[name] = {
            "offset": weight_offset,
            "nbytes": len(tensor.data),
            "scale_offset": scale_offset,
            "scale_nbytes": len(tensor.scales),
            "logical_shape": list(tensor.logical_shape),
            "packed_shape": list(tensor.packed_shape),
            "bits": tensor.bits,
            "signed": tensor.signed,
            "format": tensor.format,
            "sha256": _digest(tensor.data),
            "scale_sha256": _digest(tensor.scales),
        }
    images = {
        "weights.bin": bytes(weight_image),
        "scales.bin": bytes(scale_image),
        "calibration_ids.bin": pkg.calibration_ids.astype("<i4", copy=False).tobytes(),
    }
    files = {
        name: {"size": len(data), "sha256": _digest(data)}
        for name, data in sorted(images.items())
    }
    manifest = dict(pkg.manifest)
    manifest["tensors"] = tensor_manifest
    manifest["files"] = files
    for name, data in images.items():
        (out_dir / name).write_bytes(data)
    manifest_data = _json_bytes(manifest)
    (out_dir / "manifest.json").write_bytes(manifest_data)
    write_receipt(out_dir)
    return manifest
