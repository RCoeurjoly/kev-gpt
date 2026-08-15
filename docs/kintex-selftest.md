# Kintex self-test

Build the exact locked image entirely through Nix:

```bash
scripts/build_kintex_selftest.sh /home/roland/compiler-lab-llm2fpga
```

The generated path is
`artifacts/kintex-selftest/kintex-selftest.bit`. It is a symlink to the exact
Nix store artifact recorded in `artifacts/kintex-selftest/provenance.json`.
Program that file using the board's established JTAG procedure; do not
substitute a manually rebuilt image.

Expected outputs after configuration and reset release:

- P30 toggles: configured clocked logic is alive.
- M30 high: the sequential sum of 1 through 16 matched 136.
- N30 high: the arithmetic self-test produced an incorrect result.

A host build is only RTL-qualified. Record `hardware_passed: true` only after
P30 toggles, M30 is high, and N30 remains low on the physical board.

The build intentionally does not prescribe a programmer command. Confirm the
connected JTAG adapter, chain position, and established board procedure before
programming.
