"""Measure and gate the fixed-point hardware reference against pinned FP32."""

import argparse
import json
import pathlib

import numpy as np
import torch
from transformers import GPTNeoForCausalLM

from .hardware_reference import FixedGPTNeo
from .quantize import measure_logits_quality


def measure(package: pathlib.Path, source: pathlib.Path) -> dict:
    ids = np.frombuffer((package / "calibration_ids.bin").read_bytes(), dtype="<i4")
    fixed = FixedGPTNeo(package)
    fp32 = GPTNeoForCausalLM.from_pretrained(source, local_files_only=True).eval()
    fixed_logits, fp32_logits, labels = [], [], []
    with torch.no_grad():
        for start in range(0, len(ids)-1, 32):
            window = ids[start:start+32]
            fixed_logits.append(
                fixed.forward(window.tolist())[:-1].astype(np.float64) / 65536.0
            )
            tensor = torch.from_numpy(window.astype(np.int64))[None, :]
            fp32_logits.append(fp32(tensor).logits[0, :-1].cpu().numpy())
            labels.append(window[1:])
    metrics = measure_logits_quality(
        np.concatenate(fp32_logits), np.concatenate(fixed_logits), np.concatenate(labels)
    )
    metrics["perplexity_ratio_limit"] = 1.10
    metrics["top1_agreement_minimum"] = 0.90
    metrics["passed"] = (
        metrics["perplexity_ratio"] <= 1.10 and metrics["top1_agreement"] >= 0.90
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True, type=pathlib.Path)
    parser.add_argument("--source", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    result = measure(args.package, args.source)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not result["passed"]:
        raise SystemExit("fixed-point hardware quality gate failed")


if __name__ == "__main__":
    main()
