# Kintex Open-Source EDA Bring-Up

## Goal

Produce and test a reproducible self-test bitstream for the YPCB-00338-1P1
Kintex-7 board (`xc7k480tffg1156-1`) using the already pinned open-source EDA
stack in `/home/roland/compiler-lab-llm2fpga`. Preserve a clean path from this
board proof to a complete Kevin accelerator port.

## Phase 1: Board and Toolchain Proof

Build the existing `task3-main#matmul-selftest-bitstream` package directly
from `/home/roland/compiler-lab-llm2fpga`. Invoke Nix with
`--no-update-lock-file` so the checked-in lock graph remains authoritative.
Building this narrow package exercises the pinned Yosys, nextpnr-Xilinx,
Project X-Ray FASM, and bitstream-generation path without requesting unrelated
CIRCT, LLVM, or full-model outputs.

The self-test retains the existing board interface:

- `SYS_CLK`: package pin `AA28`, LVCMOS18
- `SYS_RSTN`: package pin `R28`, LVCMOS18, active low
- heartbeat LED: package pin `P30`, LVCMOS18
- pass LED: package pin `M30`, LVCMOS18
- failure/timeout LED: package pin `N30`, LVCMOS18

After configuration, the heartbeat must toggle. The self-test initializes two
internal vectors, runs the generated matrix-multiply core, and compares its
result in hardware. Pass or failure is latched on its corresponding LED.

The deliverable in this repository is a link or copy of the exact Nix-produced
`.bit` file plus a small provenance record containing the source checkout
revision, lock-file hash, Nix store path, and bitstream hash. No manually
rebuilt or edited bitstream counts as evidence.

## Phase 2: Kevin Accelerator Port

Keep the phase-1 clock, reset, programming, and observable-status boundary.
First replace the matrix-multiply core with a bounded Kevin accelerator
self-test whose inputs and expected outputs are compiled into the design. This
separates accelerator correctness from external-memory and host-interface
bring-up.

Once the bounded accelerator test passes on the board, add the YPCB DDR3
controller, board-level constraints, and a host-control/data interface. The
KV260 Zynq PS and AXI block design are not portable to this standalone Kintex
target and will not be carried over as implicit dependencies.

## Failure Handling and Evidence

Host builds fail if synthesis, placement, routing, FASM conversion, or
bitstream generation fails. Board testing distinguishes configuration failure,
missing heartbeat, self-test timeout/failure, and pass. Timing or performance
claims require measurements from the exact content-addressed artifact recorded
in the provenance file.

## Verification

Phase 1 is complete only when:

1. Nix evaluates and builds the locked self-test package without changing either
   lock file.
2. The output is a nonempty `.bit` file with recorded hashes and store path.
3. The board configures with that exact file.
4. The heartbeat toggles and the pass LED latches without the failure LED.

Phase 2 will receive a separate implementation design after phase 1 establishes
the board and toolchain baseline.
