#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
compiler_lab="${1:-/home/roland/compiler-lab-llm2fpga}"
artifact_dir="$repo_root/artifacts/kintex-selftest"
locked_python="github:NixOS/nixpkgs/6fd329b2adfecb86ae49c1cba89689bd0f229e04#python3"

test -f "$compiler_lab/flake.lock"
test -f "$compiler_lab/task3-main/flake.lock"
test -f "$repo_root/nix/kintex-selftest.nix"
test -f "$repo_root/fpga/rtl/kintex_selftest_top.sv"
test -f "$repo_root/fpga/constraints/kintex_selftest.xdc"

root_lock_before="$(sha256sum "$compiler_lab/flake.lock" | cut -d' ' -f1)"
task_lock_before="$(sha256sum "$compiler_lab/task3-main/flake.lock" | cut -d' ' -f1)"
source_rev="$(git -C "$compiler_lab" rev-parse HEAD)"

mkdir -p "$artifact_dir"
store_path="$(nix build --impure \
  --file "$repo_root/nix/kintex-selftest.nix" \
  --argstr compilerLab "$compiler_lab" \
  --no-update-lock-file \
  --no-link \
  --print-out-paths \
  -L)"

test -f "$store_path"
test -s "$store_path"

root_lock_after="$(sha256sum "$compiler_lab/flake.lock" | cut -d' ' -f1)"
task_lock_after="$(sha256sum "$compiler_lab/task3-main/flake.lock" | cut -d' ' -f1)"
test "$root_lock_before" = "$root_lock_after"
test "$task_lock_before" = "$task_lock_after"

ln -sfn "$store_path" "$artifact_dir/kintex-selftest.bit"
bitstream_sha256="$(sha256sum "$store_path" | cut -d' ' -f1)"
rtl_sha256="$(sha256sum "$repo_root/fpga/rtl/kintex_selftest_top.sv" | cut -d' ' -f1)"
nix_expr_sha256="$(sha256sum "$repo_root/nix/kintex-selftest.nix" | cut -d' ' -f1)"

nix shell "$locked_python" --command python - \
  "$artifact_dir/provenance.json" "$compiler_lab" "$source_rev" \
  "$root_lock_after" "$task_lock_after" "$store_path" \
  "$bitstream_sha256" "$rtl_sha256" "$nix_expr_sha256" <<'PY'
import json
import pathlib
import sys

(output, checkout, revision, root_lock, task_lock, store_path, bitstream,
 rtl_sha256, nix_expr_sha256) = sys.argv[1:]
record = {
    "schema_version": 1,
    "board": "YPCB-00338-1P1",
    "fpga_part": "xc7k480tffg1156-1",
    "source_checkout": checkout,
    "source_revision": revision,
    "root_flake_lock_sha256": root_lock,
    "task3_flake_lock_sha256": task_lock,
    "nix_store_path": store_path,
    "bitstream_sha256": bitstream,
    "rtl_sha256": rtl_sha256,
    "nix_expression_sha256": nix_expr_sha256,
    "build_target": "nix/kintex-selftest.nix",
}
pathlib.Path(output).write_text(
    json.dumps(record, indent=2) + "\n", encoding="utf-8"
)
PY

printf '%s\n' "$artifact_dir/kintex-selftest.bit"
