# Kintex Complete Accelerator Self-Test

## Goal

Build and physically validate the complete four-layer Kevin transformer on the
YPCB-00338-1P1 Kintex-7 board (`xc7k480tffg1156-1`) using only the open-source,
Nix-locked EDA flow proven by the arithmetic board self-test. The design must
run the existing canonical prompt `"once upo"`, greedily generate eight tokens,
and accept only the established `"n time t"` token stream.

This phase proves the complete accelerator datapath and sequencer independently
of external DDR3 and a runtime host interface. It does not make a throughput
claim or attempt to turn the standalone Kintex board into the earlier KV260
Zynq system.

## Selected Approach

Use a self-contained, correctness-first build with `LANES=16`. All model data
needed by the four-layer transformer is compiled into initialized Kintex-7 BRAM
by the bitstream. A small board wrapper starts the accelerator automatically,
checks every generated token, and reports the result through the already proven
LED pins.

This approach was selected over a wider 128-lane image because the first
complete openXC7 port should minimize placement and routing risk. It was also
selected over runtime JTAG, UART, or DDR3 loading because those paths would mix
new board-interface failures with accelerator-correctness failures.

## Reproducible Build Boundary

The authoritative EDA dependency graph remains
`/home/roland/compiler-lab-llm2fpga/task3-main/flake.lock`. Nix supplies every
build-time program, including Python used to generate model memories, RTL
simulation tools, Yosys, nextpnr-Xilinx, the Kintex chip database, Project X-Ray
FASM tools, and openFPGALoader. The build must pass `--no-update-lock-file` and
must prove that neither compiler-lab lock file changed.

No ambient Python interpreter, pip environment, proprietary synthesis tool, or
manually edited generated memory file is part of the build. Generated model
images and the final bitstream are content-addressed outputs with recorded
SHA-256 hashes.

## Accelerator Architecture

The accelerator is the existing `sequencer_fast` four-layer, single-stream
transformer dataflow:

1. Read token and positional embeddings for the canonical prompt.
2. For each of four transformer blocks, execute LayerNorm, QKV GEMV, KV-cached
   attention and softmax, projection, residual addition, second LayerNorm, MLP
   GEMV, GELU, projection, and residual addition.
3. Execute final LayerNorm and the vocabulary head.
4. Select the greedy token, append it to the token stream, feed it back, and
   continue until eight tokens have been emitted.

The port uses `LANES=16` and preserves the bit-true fixed-point formats and the
canonical golden stream already established by the simulator and prior KV260
silicon tests.

The UltraScale+-oriented runtime weight store cannot be carried over unchanged:
the standalone board has no Zynq processing system to stream weights after
configuration, and the target is a Kintex-7 rather than an UltraScale+ device.
For this bounded test, the model weights are represented as a single-read,
bitstream-initialized BRAM image in the resident GEMV engine. Embeddings,
position data, gamma values, dequantization tables, activation scales, GELU LUT,
and the prompt are likewise generated as deterministic initialization files.
The synthesis gate must demonstrate that these memories remain present rather
than being replaced by zeros or optimized away.

## Board Wrapper and Observability

The top-level wrapper retains the proven board boundary:

- `SYS_CLK`: pin `AA28`, LVCMOS18, constrained to 12 MHz
- `SYS_RSTN`: pin `R28`, LVCMOS18, active low
- heartbeat LED: pin `P30`, LVCMOS18
- pass LED: pin `M30`, LVCMOS18
- failure LED: pin `N30`, LVCMOS18

After reset is released, the wrapper waits a fixed settling interval and issues
one `go` pulse. It reads the generated token stream and compares all eight token
IDs in order against the canonical expected IDs. The pass LED latches only when
the accelerator reports exactly eight matching tokens. The failure LED latches
on the first token mismatch, an invalid token count, or a conservative execution
timeout. Pass and failure are mutually exclusive and remain visible until reset.
The heartbeat continues independently so configuration and clock activity can
be distinguished from accelerator progress.

## Verification Strategy

Verification proceeds through independent gates:

1. **Memory-generation gate:** Nix generates all images twice-equivalently by
   content hash, validates their dimensions and ranges, and derives the expected
   token IDs from the checked-in model export rather than hand-editing them.
2. **RTL simulation gate:** the board wrapper and complete accelerator run from
   reset through completion and must latch pass for the canonical eight-token
   stream. A negative test corrupts one expected token or memory value and must
   latch failure.
3. **Synthesis gate:** Yosys completes for `xc7`, reports the complete hierarchy,
   and retains nonzero initialized memory resources. The build fails on missing
   memory files, pruned model state, multiple drivers, or unsupported constructs.
4. **Implementation gate:** nextpnr-Xilinx places and routes for
   `xc7k480tffg1156-1` at the conservative 12 MHz board constraint; FASM and
   bitstream conversion complete using the same locked graph.
5. **Hardware gate:** openFPGALoader detects the expected `xc7k480t`, loads the
   exact hashed SRAM bitstream, reports configuration `DONE=1`, and the physical
   board shows heartbeat plus pass without failure.

Host build success is not hardware success. Each gate records its own evidence,
and no performance figure is inferred from the conservative correctness build.

## Failure Handling

Generation errors identify the malformed or missing model artifact. Simulation
reports the first mismatching token and index. Synthesis and implementation
failures preserve their full Nix build logs. On hardware, the LED contract
separates missing clock/configuration (no heartbeat), accelerator failure or
timeout (failure LED), and complete token-stream success (pass LED).

If initialized wide BRAM cannot be mapped reliably by the locked open-source
stack, the fallback is not to weaken the oracle. The design will instead use a
small synthesizable boot loader that copies an initialized narrow BRAM ROM into
the resident GEMV memory before asserting `go`; DDR3 and external host loading
remain out of scope for this phase.

## Deliverables

- Deterministic Nix model-memory generator and validation tests.
- Kintex-compatible resident-weight accelerator RTL and board wrapper.
- Full positive and negative RTL simulation gates.
- Locked Nix bitstream target and build script with provenance.
- Exact `.bit` artifact, hashes, implementation evidence, and board result.
- Updated board documentation describing reset, runtime, LED meanings, and the
  later boundary for DDR3/runtime-control work.

## Explicitly Deferred Work

- DDR3 controller and board DDR3 pin constraints.
- UART, Ethernet, PCIe, or JTAG data/control protocol.
- Runtime prompt or weight replacement.
- Nonvolatile flash programming.
- Width or clock optimization and throughput characterization.
- Multi-stream or batched serving.
