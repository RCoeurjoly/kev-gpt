# YPCB Power-On Reset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the YPCB interactive TinyStories accelerator enter the same deterministic reset state as the exact RTL simulation and test whether this restores bit-exact hardware inference.

**Architecture:** An initialized saturating counter in `tinystories_interactive_top` asserts the existing active-high reset net for the first 16 `SYS_CLK` edges after configuration. External active-low `SYS_RSTN` remains authoritative and can reassert reset at any time; no datapath, model image, clock, or packet interface changes.

**Tech Stack:** SystemVerilog, Verilator, Python `unittest`, locked Nix flake, Yosys, nextpnr-xilinx, Project X-Ray, openFPGALoader, Digilent HS3/JTAG.

## Global Constraints

- Use only tools and Python supplied by the locked Nix environment; never use global Python or Vivado.
- Target YPCB-00338-1P1 `xc7k480tffg1156-1` at its 50 MHz clock on AA28.
- Preserve the complete resident TinyStories-1M W8A8 model and existing JTAG protocol.
- Change only reset behavior for this hypothesis; do not change model data or arithmetic.
- Hardware success requires three consecutive status-zero runs of prompt `[7454, 2402, 257, 640]`, one generated token, returning token `11`.

---

### Task 1: Test and implement deterministic startup reset

**Files:**
- Create: `fpga/tb/tb_tinystories_interactive_reset.sv`
- Modify: `tests/test_rtl_gates.py`
- Modify: `fpga/rtl/tinystories_interactive_top.sv`

**Interfaces:**
- Consumes: top-level `SYS_CLK` and active-low `SYS_RSTN`.
- Produces: internal active-high `reset` asserted for at least 16 startup edges and whenever `SYS_RSTN == 0`.

- [ ] **Step 1: Write the failing reset test**

Create a testbench that supplies inert stubs for the top's four child modules,
starts `SYS_RSTN=1`, counts rising edges for which `dut.reset` is high,
requires exactly 16, then drives `SYS_RSTN=0` and requires internal reset
assertion. Add a focused `unittest` method that generates the package header
in a temporary fixture, compiles the top and testbench with `iverilog`, runs
`vvp`, and expects `TINY_RESET_PASS startup_cycles=16 external=1`.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
nix develop --command python -m unittest tests.test_rtl_gates.RTLPrimitiveGateTest.test_interactive_top_generates_power_on_reset -v
```

Expected: FAIL because the current top defines `reset` only as `!SYS_RSTN` and releases it immediately when `SYS_RSTN` starts high.

- [ ] **Step 3: Implement the minimal reset counter**

In `tinystories_interactive_top.sv`, replace the direct reset expression with an initialized five-bit saturating counter. Increment it while `SYS_RSTN` is high until bit 4 becomes one; clear it whenever `SYS_RSTN` is low. Define `reset = !SYS_RSTN || !startup_count[4]`. Keep every existing module connected to this reset net.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
nix develop --command python -m unittest \
  tests.test_rtl_gates.RTLPrimitiveGateTest.test_interactive_top_generates_power_on_reset \
  tests.test_rtl_gates.RTLPrimitiveGateTest.test_production_top_exposes_read_only_in_band_debug_snapshot \
  tests.test_host_cli.HostCLITest.test_debug_snapshot_is_read_over_existing_user1_transport -v
```

Expected: all PASS, including `TINY_RESET_PASS startup_cycles=16 external=1`.

- [ ] **Step 5: Commit the isolated reset change**

```bash
git add fpga/tb/tb_tinystories_interactive_reset.sv tests/test_rtl_gates.py fpga/rtl/tinystories_interactive_top.sv
git commit -m "fix: reset YPCB accelerator after configuration"
```

### Task 2: Verify exact RTL and build the locked bitstream

**Files:**
- Verify: `fpga/rtl/*.sv`
- Verify: `model_packages/tinystories-1m/regressions.json`
- Build: `nix/kintex-tinystories.nix`

**Interfaces:**
- Consumes: Task 1's reset top and the committed package fixture.
- Produces: routed `tinystories-interactive.bit` with post-route timing evidence.

- [ ] **Step 1: Run the full exact sequencer regression**

```bash
nix develop --command python -m unittest tests.test_rtl_gates.RTLSequencerGateTest.test_three_streams_and_corrupt_package_verdict -v
```

Expected: PASS for all three package streams and the corrupt-package negative case.

- [ ] **Step 2: Run the focused software/RTL suite**

```bash
nix develop --command python -m unittest tests.test_host_cli tests.test_jtag_packet tests.test_rtl_gates -v
```

Expected: PASS with no test failure.

- [ ] **Step 3: Build the interactive image from the locked derivation**

```bash
nix build --impure --expr 'import ./nix/kintex-tinystories.nix { interactive = true; }' -L
```

Expected: successful synthesis, flattening, routing, FASM/frame conversion, and bitstream assembly.

- [ ] **Step 4: Audit implementation evidence**

Resolve the store path with `nix path-info`, then require `overused=0`, `archfail=0`, and post-route `controller.clk` maximum frequency at or above 50 MHz in `nextpnr.log`. Require a non-empty `tinystories-interactive.bit` and record its store path.

### Task 3: Test the reset hypothesis on YPCB hardware

**Files:**
- Consume: bitstream from Task 2
- Consume: `host/kevin_jtag_cli.py`
- Consume: `model_packages/tinystories-1m/regressions.json`

**Interfaces:**
- Consumes: Digilent HS3-connected YPCB and locked Nix host transport.
- Produces: three-run hardware token evidence accepting or rejecting the reset hypothesis.

- [ ] **Step 1: Program the exact routed bitstream**

```bash
bitstream_store="$(nix path-info --impure --expr \
  'import ./nix/kintex-tinystories.nix { interactive = true; }')"
nix develop --command openFPGALoader --cable digilent_hs3 \
  "$bitstream_store/tinystories-interactive.bit"
```

Expected: `INIT=1` and `DONE=1`.

- [ ] **Step 2: Run the canonical prompt three times**

Use `NativeTransport` and `infer_tokens` from inside `nix develop` with prompt `[7454, 2402, 257, 640]`, generation count `1`, and a 10-second timeout. Keep one transport open and issue three complete transactions, printing status, tokens, and FPGA cycles for each.

Expected acceptance result: all three replies are status `0`, tokens `[11]`, with completed cycle counts. Any other token, timeout, status, or run-to-run difference rejects the hypothesis.

- [ ] **Step 3: Choose the evidence-driven next action**

If all three runs match, retain the reset fix and proceed to the remaining multi-token hardware validation. If any run fails, revert only the reset implementation commit while retaining the test/diagnostic evidence, then instrument the earliest embedding/package-memory boundary before proposing another fix.

- [ ] **Step 4: Update the active implementation plan**

Record the measured result and set the next in-progress task to either multi-token hardware qualification or earliest-divergence instrumentation. Do not claim the full accelerator complete from this one-token experiment alone.
