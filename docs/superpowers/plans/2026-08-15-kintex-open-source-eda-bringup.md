# Kintex Open-Source EDA Bring-Up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, preserve, and physically verify the existing locked matrix-multiply self-test bitstream on the YPCB-00338-1P1 Kintex-7 board.

**Architecture:** A small repository-local shell command validates the external compiler checkout, snapshots both lock-file hashes, and invokes only `task3-main#matmul-selftest-bitstream` with lock updates disabled. It places a stable symlink and machine-readable provenance beside the result; a separate board receipt records the later physical observation without pretending that a host build proves hardware behavior.

**Tech Stack:** Bash, Nix flakes, Yosys, nextpnr-Xilinx, Project X-Ray/FASM, Python `unittest`, SHA-256, Git.

## Global Constraints

- Target board: YPCB-00338-1P1.
- Target FPGA: `xc7k480tffg1156-1`.
- Authoritative EDA source checkout: `/home/roland/compiler-lab-llm2fpga`.
- Build target: `task3-main#matmul-selftest-bitstream`.
- Pass `--no-update-lock-file`; neither external `flake.lock` may change.
- Do not request unrelated CIRCT, LLVM, or full-model outputs.
- Preserve pins `AA28` (clock), `R28` (active-low reset), `P30` (heartbeat), `M30` (pass), and `N30` (failure), all LVCMOS18.
- Do not claim hardware qualification until the exact recorded artifact is configured and observed on the board.
- Phase 2 accelerator and DDR3 integration are outside this plan.

## Approved Execution Amendment

The original `task3-main#matmul-selftest-bitstream` target was dry-run and found
to require 34 uncached derivations, including custom LLVM and CIRCT. With user
approval, Task 2 instead builds `fpga/rtl/kintex_selftest_top.sv` through
`nix/kintex-selftest.nix`. That expression obtains only Yosys, nextpnr-Xilinx,
the Kintex-7 chip database, FASM, and Project X-Ray from the exact
`task3-main/flake.lock` graph. The revised dry run contains six derivations and
no LLVM, MLIR, or CIRCT build.

---

## File Structure

- `scripts/build_kintex_selftest.sh`: validate the source checkout, run the narrow locked Nix build, and emit artifact provenance.
- `tests/test_build_kintex_selftest.py`: static regression tests for the safety and reproducibility contract of the build command.
- `.gitignore`: ignore the local `artifacts/kintex-selftest/` symlinks and receipts.
- `docs/kintex-selftest.md`: exact build, programming handoff, LED interpretation, and receipt procedure.
- `artifacts/kintex-selftest/`: generated local output containing `matmul-selftest.bit` and `provenance.json`; never committed.

### Task 1: Add the Locked Self-Test Build Command

**Files:**
- Create: `scripts/build_kintex_selftest.sh`
- Create: `tests/test_build_kintex_selftest.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: optional first argument `COMPILER_LAB`, defaulting to `/home/roland/compiler-lab-llm2fpga`.
- Produces: `artifacts/kintex-selftest/matmul-selftest.bit` as a symlink to the Nix store output and `artifacts/kintex-selftest/provenance.json` as a regular file.
- Returns: exit status zero only when lock hashes are unchanged and the bitstream is nonempty.

- [ ] **Step 1: Write the failing static contract test**

Create `tests/test_build_kintex_selftest.py`:

```python
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_kintex_selftest.sh"


class KintexSelftestBuildScriptTest(unittest.TestCase):
    def test_locked_narrow_build_and_provenance_contract(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("task3-main#matmul-selftest-bitstream", source)
        self.assertIn("--no-update-lock-file", source)
        self.assertIn("flake.lock", source)
        self.assertIn("task3-main/flake.lock", source)
        self.assertIn("sha256sum", source)
        self.assertIn("git -C", source)
        self.assertIn("provenance.json", source)
        self.assertNotIn("nix flake update", source)
        self.assertNotIn("nix build .#", source)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify the script is absent**

Run: `python3 -m unittest tests.test_build_kintex_selftest -v`

Expected: ERROR with `FileNotFoundError` for `scripts/build_kintex_selftest.sh`.

- [ ] **Step 3: Implement the minimal locked build command**

Create `scripts/build_kintex_selftest.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
compiler_lab="${1:-/home/roland/compiler-lab-llm2fpga}"
artifact_dir="$repo_root/artifacts/kintex-selftest"

test -f "$compiler_lab/flake.lock"
test -f "$compiler_lab/task3-main/flake.lock"
test -f "$compiler_lab/task3-main/fpga/constraints/matmul_selftest.xdc"

root_lock_before="$(sha256sum "$compiler_lab/flake.lock" | cut -d' ' -f1)"
task_lock_before="$(sha256sum "$compiler_lab/task3-main/flake.lock" | cut -d' ' -f1)"
source_rev="$(git -C "$compiler_lab" rev-parse HEAD)"

mkdir -p "$artifact_dir"
store_path="$(nix build \
  "$compiler_lab/task3-main#matmul-selftest-bitstream" \
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

ln -sfn "$store_path" "$artifact_dir/matmul-selftest.bit"
bitstream_sha256="$(sha256sum "$store_path" | cut -d' ' -f1)"

python3 - "$artifact_dir/provenance.json" "$compiler_lab" "$source_rev" \
  "$root_lock_after" "$task_lock_after" "$store_path" "$bitstream_sha256" <<'PY'
import json
import pathlib
import sys

output, checkout, revision, root_lock, task_lock, store_path, bitstream = sys.argv[1:]
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
    "build_target": "task3-main#matmul-selftest-bitstream",
}
pathlib.Path(output).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
PY

printf '%s\n' "$artifact_dir/matmul-selftest.bit"
```

Make it executable: `chmod +x scripts/build_kintex_selftest.sh`.

Append to `.gitignore`:

```gitignore
# Local Kintex bitstreams and physical-test receipts
/artifacts/kintex-selftest/
```

- [ ] **Step 4: Run the focused tests**

Run: `python3 -m unittest tests.test_build_kintex_selftest -v`

Expected: one test passes.

Run: `bash -n scripts/build_kintex_selftest.sh`

Expected: exit status zero with no output.

- [ ] **Step 5: Commit the build command**

```bash
git add .gitignore scripts/build_kintex_selftest.sh tests/test_build_kintex_selftest.py
git commit -m "build: add locked Kintex self-test bitstream command"
```

### Task 2: Build and Verify the Exact Bitstream Artifact

**Files:**
- Generate: `artifacts/kintex-selftest/matmul-selftest.bit`
- Generate: `artifacts/kintex-selftest/provenance.json`

**Interfaces:**
- Consumes: `scripts/build_kintex_selftest.sh` from Task 1 and the external locked checkout.
- Produces: a nonempty `.bit` store artifact and provenance fields defined in Task 1.

- [ ] **Step 1: Record the pre-build lock state**

Run:

```bash
sha256sum /home/roland/compiler-lab-llm2fpga/flake.lock \
  /home/roland/compiler-lab-llm2fpga/task3-main/flake.lock
git -C /home/roland/compiler-lab-llm2fpga status --short
```

Expected: two hashes are printed. Preserve any pre-existing status entries; do not alter or clean them.

- [ ] **Step 2: Run only the locked bitstream build**

Run: `scripts/build_kintex_selftest.sh /home/roland/compiler-lab-llm2fpga`

Expected: Nix builds or substitutes the narrow dependency closure, nextpnr routes successfully, FASM conversion succeeds, and the last line is the local bitstream symlink path.

- [ ] **Step 3: Verify the generated artifact and provenance**

Run:

```bash
test -L artifacts/kintex-selftest/matmul-selftest.bit
test -s artifacts/kintex-selftest/matmul-selftest.bit
python3 -m json.tool artifacts/kintex-selftest/provenance.json
sha256sum -c <(python3 - <<'PY'
import json
from pathlib import Path

record = json.loads(Path("artifacts/kintex-selftest/provenance.json").read_text())
print(record["bitstream_sha256"], " artifacts/kintex-selftest/matmul-selftest.bit")
PY
)
```

Expected: JSON parses and `matmul-selftest.bit: OK` is printed.

- [ ] **Step 4: Confirm both external lock files are unchanged**

Run:

```bash
sha256sum /home/roland/compiler-lab-llm2fpga/flake.lock \
  /home/roland/compiler-lab-llm2fpga/task3-main/flake.lock
git -C /home/roland/compiler-lab-llm2fpga status --short
```

Expected: hashes match Step 1 and no new lock-file changes appear.

### Task 3: Document and Perform the Physical Board Check

**Files:**
- Create: `docs/kintex-selftest.md`
- Generate after board access: `artifacts/kintex-selftest/board-receipt.json`

**Interfaces:**
- Consumes: exact bitstream and `provenance.json` from Task 2, plus the board's established JTAG programming command.
- Produces: human instructions and a receipt that distinguishes host-built from hardware-passed status.

- [ ] **Step 1: Write the board handoff documentation**

Create `docs/kintex-selftest.md` with:

```markdown
# Kintex self-test

Build the exact locked image:

```bash
scripts/build_kintex_selftest.sh /home/roland/compiler-lab-llm2fpga
```

The generated path is `artifacts/kintex-selftest/matmul-selftest.bit`.
Program that exact file using the board's established JTAG procedure. Do not
substitute a manually rebuilt image.

Expected outputs after configuration and reset release:

- P30 toggles: configured clocked logic is alive.
- M30 high: matrix-multiply result matched the compiled expected value.
- N30 high: incorrect result or 50,000,000-cycle timeout.

A host build is only RTL-qualified. Record `hardware_passed: true` only after
P30 toggles, M30 is high, and N30 remains low on the physical board.
```

Do not invent a programmer command until the connected JTAG adapter and existing board procedure are identified.

- [ ] **Step 2: Commit the handoff documentation**

Because `docs/` is ignored, force-add only this file:

```bash
git add -f docs/kintex-selftest.md
git commit -m "docs: add Kintex self-test board procedure"
```

- [ ] **Step 3: Program the exact artifact with the established board tool**

Run the board owner's established JTAG programming command with the absolute path resolved by:

```bash
readlink -f artifacts/kintex-selftest/matmul-selftest.bit
```

Expected: FPGA configuration completes successfully. If no established command is available, stop here and request the adapter/tool details; do not guess cable voltage, chain position, or programmer flags.

- [ ] **Step 4: Observe and record physical status**

After releasing `SYS_RSTN`, observe P30, M30, and N30. If P30 toggles, M30 is
high, and N30 remains low, create the passing receipt directly from the recorded
provenance:

```bash
python3 - <<'PY'
import json
from pathlib import Path

artifact_dir = Path("artifacts/kintex-selftest")
provenance = json.loads((artifact_dir / "provenance.json").read_text())
receipt = {
    "schema_version": 1,
    "board": "YPCB-00338-1P1",
    "fpga_part": "xc7k480tffg1156-1",
    "bitstream_sha256": provenance["bitstream_sha256"],
    "heartbeat_p30": True,
    "pass_m30": True,
    "failure_n30": False,
    "hardware_passed": True,
}
(artifact_dir / "board-receipt.json").write_text(
    json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
)
PY
```

Expected: `hardware_passed` is true only when the three physical observations
match the recorded values. For any other observation, record the actual three
booleans and set `hardware_passed` to false before diagnosing the board.

- [ ] **Step 5: Run final host-side verification**

Run:

```bash
python3 -m unittest tests.test_build_kintex_selftest -v
bash -n scripts/build_kintex_selftest.sh
python3 -m json.tool artifacts/kintex-selftest/provenance.json
python3 -m json.tool artifacts/kintex-selftest/board-receipt.json
```

Expected: the unit test passes, shell syntax passes, and both JSON documents parse. Report the Nix store path, bitstream SHA-256, lock preservation, programming result, and three LED observations separately.
