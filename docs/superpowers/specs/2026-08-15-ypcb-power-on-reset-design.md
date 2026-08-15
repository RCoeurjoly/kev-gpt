# YPCB deterministic power-on reset experiment

## Context

The locked open-source flow produces an XC7K480T bitstream that routes with
zero overuse and passes post-route timing at 86.81 MHz for the board's 50 MHz
clock. JTAG requests complete with status zero and an invariant cycle count,
but identical canonical prompts produce different, incorrect token IDs. The
exact RTL simulation pulses reset and returns the package reference tokens;
the board top currently relies only on external `SYS_RSTN` and does not assert
reset after FPGA configuration.

## Hypothesis

One or more datapath registers begin from an unspecified configuration state
because the external reset is not guaranteed to pulse after programming. A
short synchronous power-on reset will make the hardware state match the reset
state used by the exact RTL simulation.

## Design

Add a small initialized counter to `tinystories_interactive_top`. While the
counter has not saturated, assert the existing internal active-high reset.
Also assert that reset whenever external `SYS_RSTN` is low. All existing
controller, transport, sequencer, and sub-engine reset connections remain on
this single reset net; packet and inference interfaces do not change.

The counter will hold reset for at least 16 rising edges of the 50 MHz board
clock. This is long enough for synchronous logic and asynchronous FIFO reset
synchronizers while adding negligible startup latency. No clock generation,
model data, arithmetic, or host protocol changes are part of this experiment.

## Verification

Follow test-driven development:

1. Add an RTL testbench that starts with external `SYS_RSTN` high and proves
   internal reset remains asserted for the specified startup interval, then
   releases. It must also prove an external low level reasserts reset.
2. Run the new test before implementation and observe the expected failure.
3. Implement only the power-on reset counter and make the test pass.
4. Run focused controller/sequencer/host tests and exact RTL regression.
5. Rebuild through the locked Nix derivation, confirm zero-overuse routing and
   50 MHz post-route timing, and program the connected YPCB.
6. Run the canonical prompt `[7454, 2402, 257, 640]` for one generated token at
   least three times. The experiment succeeds only if all runs return status
   zero and token `11`, matching `regressions.json`.

If results remain incorrect or nondeterministic, reject the reset hypothesis
and instrument the earliest package-memory/embedding boundary; do not layer
unrelated fixes onto this experiment.
