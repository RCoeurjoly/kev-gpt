# TinyStories-1M Inference on YPCB Kintex-7 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Nix-reproducible, Vivado-free interactive `roneneldan/TinyStories-1M` inference system for the YPCB-00338-1P1 XC7K480T board.

**Architecture:** A strict GPT-Neo importer converts a pinned Hugging Face checkpoint into a committed canonical W4A8/mixed-precision package. A package-driven integer reference and parameterized Stage-3-derived RTL implement exact 32-token greedy decode in initialized Kintex-7 BRAM; a BSCANE2 packet endpoint and Nix-built host CLI provide interactive prompts through the Digilent HS3. The existing compiler-lab checkout supplies the locked Yosys, nextpnr-Xilinx, chip database, openXC7, and programming stack without pulling its LLVM/CIRCT closure.

**Tech Stack:** Nix flakes, pinned nixpkgs revision `6fd329b2adfecb86ae49c1cba89689bd0f229e04`, PyTorch/Transformers, NumPy, Python `unittest`, SystemVerilog, Icarus Verilog, Yosys, nextpnr-Xilinx, Project X-Ray/openXC7, BSCANE2, libftdi/libusb, openFPGALoader.

## Global Constraints

- Primary repository: `kev-gpt`; `/home/roland/compiler-lab-llm2fpga` is a read-only EDA dependency.
- Target board/device: YPCB-00338-1P1, `xc7k480tffg1156-1`.
- Pretrained source: `roneneldan/TinyStories-1M` revision `ac533fb8b4f69c71894bf96badfe11e6294d9fcf`.
- Do not train, fine-tune, or run QAT without new approval.
- Do not invoke ambient Python, pip, Vivado, Vitis, MIG, or proprietary debug IP.
- Every Python/test command runs through the repository flake or the exact locked nixpkgs revision.
- Do not modify either compiler-lab lock file; pass `--no-update-lock-file` to external-flake operations.
- Hardware context is 32 total tokens; greedy completion is at most 16 tokens.
- Quality gates: quantized perplexity regression <= 10 percent and next-token top-1 agreement >= 90 percent on the pinned validation fixture.
- Correctness gate: integer reference, RTL, and hardware token streams are exactly identical.
- Initial response-time gate: a 16-token hardware response completes within five seconds.
- Model state remains on-chip. DDR3, PCIe, stochastic sampling, flash deployment, and arbitrary Hugging Face architectures are deferred.

---

## File Structure

### Nix and source pinning

- `flake.nix`: development shell, checks, host application, model regeneration, simulation, bitstream, and qualification outputs.
- `flake.lock`: pins the software package graph; nixpkgs must resolve to the exact revision above.
- `nix/model-source.nix`: fixed-output Hugging Face files and immutable source manifest.
- `nix/eda-toolchain.nix`: narrow adapter to compiler-lab's locked task3 toolchain.
- `nix/kintex-tinystories.nix`: Yosys through bitstream derivation.

### Canonical model package

- `tinystories/gptneo_schema.py`: typed manifest validation and package reader.
- `tinystories/import_gptneo.py`: strict Hugging Face state-dict/config/tokenizer adapter.
- `tinystories/quantize.py`: calibration, W4A8/mixed-precision selection, packing, and quality metrics.
- `tinystories/int_reference.py`: package-driven integer GPT-Neo greedy reference.
- `tinystories/package_io.py`: deterministic binary/hex writers and SHA-256 receipt generation.
- `model_packages/tinystories-1m/`: committed manifest, tokenizer assets, packed images, regression vectors, and receipts.
- `tests/test_gptneo_schema.py`, `tests/test_gptneo_import.py`, `tests/test_tinystories_quantize.py`, `tests/test_tinystories_int_reference.py`: host gates.

### RTL and simulation

- `fpga/rtl/gptneo_layernorm.sv`: gamma/beta LayerNorm.
- `fpga/rtl/gptneo_gelu.sv`: package-generated `gelu_new` fixed-point lookup/interpolation.
- `fpga/rtl/gptneo_attention.sv`: 16-head causal attention for context <= 32.
- `fpga/rtl/gptneo_resident_gemv.sv`: BRAM-initialized tied/resident matrix engine.
- `fpga/rtl/gptneo_sequencer.sv`: eight-layer package-parameterized autoregressive controller.
- `fpga/rtl/tinystories_selftest_top.sv`: compiled-prompt board proof and LED oracle.
- `fpga/rtl/bscan_packet_endpoint.sv`: BSCANE2 framing, CRC, and command/reply queues.
- `fpga/rtl/async_fifo.sv`: dual-clock Gray-pointer FIFO.
- `fpga/rtl/tinystories_interactive_top.sv`: final board top.
- `fpga/tb/`: focused block, sequencer, self-test, FIFO, packet, and interactive testbenches.
- `fpga/constraints/tinystories_ypcb.xdc`: 50 MHz input, reset, LEDs, and generated-clock constraints.
- `tests/test_rtl_gates.py`: Nix/Icarus gate orchestration and exact output comparison.

### Host and qualification

- `host/jtag_transport.c`, `host/jtag_transport.h`: HS3/libftdi JTAG TAP and USER-chain byte transport.
- `host/kevin_jtag_cli.py`: tokenizer, packet codec, program/send/receive command, JSON receipt.
- `tests/test_jtag_packet.py`, `tests/test_host_cli.py`: protocol and no-host-inference contracts.
- `scripts/qualify_ypcb.py`: exact artifact programming and repeated physical regression runner.
- `docs/ypcb-tinystories.md`: build, use, recovery, LED, and evidence guide.

---

### Task 1: Establish the Nix-Only Project Environment

**Files:**
- Create: `flake.nix`
- Create: `flake.lock`
- Create: `nix/eda-toolchain.nix`
- Create: `tests/test_nix_contract.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `devShells.x86_64-linux.default`, `checks.x86_64-linux.nix-contract`, and an EDA adapter function invoked as `import ./nix/eda-toolchain.nix { compilerLab = "/home/roland/compiler-lab-llm2fpga"; }`.
- The shell exposes Python with `numpy`, `torch`, `transformers`, and `tokenizers`, plus Icarus, Verilator, libftdi, libusb, and openFPGALoader.

- [ ] **Step 1: Write the failing lock/toolchain contract test.**

```python
import json
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXPECTED_NIXPKGS = "6fd329b2adfecb86ae49c1cba89689bd0f229e04"

class NixContractTest(unittest.TestCase):
    def test_locked_nixpkgs_and_external_lock_safety(self):
        lock = json.loads((ROOT / "flake.lock").read_text())
        self.assertEqual(lock["nodes"]["nixpkgs"]["locked"]["rev"], EXPECTED_NIXPKGS)
        source = (ROOT / "nix" / "eda-toolchain.nix").read_text()
        self.assertIn("task3-main", source)
        self.assertNotIn("circt", source.lower())
        self.assertNotIn("torchMlir", source)
```

- [ ] **Step 2: Run the bootstrap test and verify it fails because the flake is absent.**

Run:

```bash
nix shell github:NixOS/nixpkgs/6fd329b2adfecb86ae49c1cba89689bd0f229e04#python3 --command python -m unittest -v tests/test_nix_contract.py
```

Expected: failure opening `flake.lock`.

- [ ] **Step 3: Add the minimal project flake and narrow EDA adapter.**

`flake.nix` must define `nixpkgs.url = "github:NixOS/nixpkgs/6fd329b2adfecb86ae49c1cba89689bd0f229e04"`; use `pkgs.python312.withPackages` for all Python dependencies. `nix/eda-toolchain.nix` must use `builtins.getFlake "path:${compilerLab}/task3-main"` and export only `yosys`, `nextpnr`, `chipdb`, `fasm`, `prjxray`, `familyDb`, and `partFile`. Add `/result*`, `/artifacts/tinystories/`, and `/data/huggingface/` to `.gitignore`.

- [ ] **Step 4: Lock and verify without touching compiler-lab.**

Run:

```bash
sha256sum /home/roland/compiler-lab-llm2fpga/flake.lock /home/roland/compiler-lab-llm2fpga/task3-main/flake.lock
nix flake lock
nix develop --command python -m unittest -v tests/test_nix_contract.py
nix flake show --no-update-lock-file
sha256sum /home/roland/compiler-lab-llm2fpga/flake.lock /home/roland/compiler-lab-llm2fpga/task3-main/flake.lock
```

Expected: test passes and before/after external hashes match exactly.

- [ ] **Step 5: Commit the Nix boundary.**

```bash
git add flake.nix flake.lock nix/eda-toolchain.nix tests/test_nix_contract.py .gitignore
git commit -m "build: add locked TinyStories Kintex environment"
```

### Task 2: Pin and Validate the Hugging Face Source

**Files:**
- Create: `nix/model-source.nix`
- Create: `tinystories/gptneo_schema.py`
- Create: `tests/test_gptneo_schema.py`
- Modify: `flake.nix`

**Interfaces:**
- `validate_manifest(value: dict) -> dict` returns a normalized manifest or raises `ValueError` with a field path.
- Nix package `.#tinystories-1m-source` contains `config.json`, checkpoint, tokenizer files, and `source-manifest.json` with URL, revision, size, and SHA-256 for each file.

- [ ] **Step 1: Write schema tests for the exact accepted architecture and rejection paths.**

```python
class GPTNeoSchemaTest(unittest.TestCase):
    def valid_manifest(self):
        return {
        "schema_version": 1, "model_type": "gpt_neo", "n_layer": 8,
        "hidden_size": 64, "n_head": 16, "vocab_size": 50257,
        "source_revision": "ac533fb8b4f69c71894bf96badfe11e6294d9fcf",
        "max_context": 32, "local_window": 256, "tie_word_embeddings": True,
        "activation_function": "gelu_new",
        }

    def test_accepts_pinned_shape(self):
        m = validate_manifest(self.valid_manifest())
        self.assertEqual(m["max_context"], 32)

    def test_rejects_semantic_drift(self):
        value = self.valid_manifest()
        value["max_context"] = 512
        with self.assertRaisesRegex(ValueError, "max_context <= local_window"):
            validate_manifest(value)
```

- [ ] **Step 2: Run the focused test and verify import failure.**

Run: `nix develop --command python -m unittest -v tests/test_gptneo_schema.py`

Expected: `ModuleNotFoundError: tinystories.gptneo_schema`.

- [ ] **Step 3: Implement strict manifest validation.**

The validator must require GPT-Neo, eight layers, width 64, 16 heads, vocabulary 50,257, `gelu_new`, tied embeddings, context 32, and local window >= 32. Unknown fields may be retained under `source_config`, but missing semantic fields fail.

- [ ] **Step 4: Add fixed-output model downloads.**

Use `pkgs.fetchurl` URLs containing the exact revision. Start each new file with `hash = pkgs.lib.fakeHash`, run `nix build .#tinystories-1m-source -L`, replace each reported hash, then rebuild. Include only the checkpoint/config and tokenizer files required by `AutoTokenizer` and `GPTNeoForCausalLM`; do not fetch a mutable branch URL.

- [ ] **Step 5: Verify immutable source contents and commit.**

Run:

```bash
nix build .#tinystories-1m-source --no-update-lock-file -L
nix develop --command python -m unittest -v tests/test_gptneo_schema.py
nix develop --command python -m json.tool result/source-manifest.json
```

Expected: build succeeds, every source has a concrete SRI hash, and schema tests pass.

```bash
git add flake.nix nix/model-source.nix tinystories/gptneo_schema.py tests/test_gptneo_schema.py
git commit -m "build: pin TinyStories-1M source model"
```

### Task 3: Import and Quantize the Canonical Model Package

**Files:**
- Create: `tinystories/package_io.py`
- Create: `tinystories/import_gptneo.py`
- Create: `tinystories/quantize.py`
- Create: `tests/test_gptneo_import.py`
- Create: `tests/test_tinystories_quantize.py`
- Create/generated: `model_packages/tinystories-1m/**`
- Modify: `flake.nix`

**Interfaces:**
- `import_model(source_dir: pathlib.Path) -> ImportedGPTNeo` returns named NumPy tensors plus validated config/tokenizer metadata.
- `quantize_model(model: ImportedGPTNeo, calibration_ids: np.ndarray, max_context: int = 32) -> QuantizedPackage`.
- `write_package(pkg: QuantizedPackage, out_dir: pathlib.Path) -> dict` writes deterministic files and returns the final manifest.
- Nix package `.#tinystories-1m-package-regenerated` regenerates the package and compares it byte-for-byte with `model_packages/tinystories-1m`.

- [ ] **Step 1: Write importer tests using a tiny synthetic GPT-Neo state dictionary.**

Test exact Q/K/V ordering, LayerNorm gamma/beta retention, tied embedding/head identity, first-32 positional-row slicing, and rejection of untied or missing tensors. Assert canonical tensor names such as `blocks.0.attn.q.weight`, `blocks.0.ln1.bias`, and `token_embedding.weight`.

- [ ] **Step 2: Write quantizer tests before implementation.**

```python
def test_symmetric_int4_per_channel_round_trip_bounds(self):
    q, scale = quantize_symmetric(w, bits=4, axis=0)
    self.assertGreaterEqual(int(q.min()), -8)
    self.assertLessEqual(int(q.max()), 7)
    self.assertEqual(scale.shape, (w.shape[0],))

def test_tied_matrix_is_packed_once(self):
    package = quantize_model(synthetic_model(), fixture_ids(), max_context=32)
    roles = package.manifest["tensor_roles"]
    self.assertEqual(roles["lm_head.weight"]["alias_of"], "token_embedding.weight")
```

- [ ] **Step 3: Run the tests and verify they fail on missing modules.**

Run: `nix develop --command python -m unittest -v tests/test_gptneo_import.py tests/test_tinystories_quantize.py`

- [ ] **Step 4: Implement deterministic import, PTQ, and package I/O.**

Use `torch.no_grad()` only to read the pinned checkpoint and collect calibration activations. Quantization uses deterministic NumPy rounding/saturation; record every tensor's offset, logical shape, packed shape, bit width, signedness, scale offset, and SHA-256. Pack two INT4 values per byte, low logical index in the low nibble. Store only 32 positional rows. Never duplicate the tied head matrix.

- [ ] **Step 5: Add objective quality measurement and mixed-precision selection.**

Evaluate FP32 and quantized-reference negative log likelihood over the same committed token-ID fixture. Emit `quality.json` with FP32/quantized perplexity, ratio, top-1 matches/total, agreement, and chosen formats. Fail unless ratio <= 1.10 and agreement >= 0.90. Candidate selection is lexicographic: passing quality, then lowest packed bytes, then fewer non-INT4 tensors.

- [ ] **Step 6: Generate, inspect, and commit the compact package.**

Run:

```bash
nix build .#tinystories-1m-package-regenerated --no-update-lock-file -L
nix develop --command python -m unittest -v tests/test_gptneo_import.py tests/test_tinystories_quantize.py
nix develop --command python -m json.tool result/quality.json
du -sh result
```

Expected: both quality thresholds pass, manifest reports one tied matrix allocation, total package fits the XC7K480T BRAM bit budget with explicit scratch/KV reserve, and regeneration comparison is byte-identical.

```bash
git add tinystories/package_io.py tinystories/import_gptneo.py tinystories/quantize.py tests/test_gptneo_import.py tests/test_tinystories_quantize.py model_packages/tinystories-1m flake.nix
git commit -m "feat: add quantized TinyStories-1M model package"
```

### Task 4: Build the Package-Driven Integer Reference

**Files:**
- Create: `tinystories/int_reference.py`
- Create: `tests/test_tinystories_int_reference.py`
- Modify/generated: `model_packages/tinystories-1m/regressions.json`

**Interfaces:**
- `IntegerGPTNeo(package_dir: pathlib.Path)` validates all package hashes at construction.
- `prefill(token_ids: list[int]) -> None`, `step(token_id: int) -> tuple[np.ndarray, int]`, and `generate(prompt_ids: list[int], n_tokens: int) -> list[int]`.
- `trace(prompt_ids, n_tokens) -> dict[str, np.ndarray]` returns named intermediate vectors for RTL localization.

- [ ] **Step 1: Write unit tests for fixed-point primitives and state behavior.**

Cover signed rounding, saturation, gamma/beta LayerNorm, `gelu_new` LUT endpoints/interpolation, causal softmax normalization, 16-head reshape/merge, KV append/read, EOS termination, context overflow, and deterministic reset.

- [ ] **Step 2: Write a full-reference test against the quantized PyTorch oracle.**

For three committed prompts, compare logits after every input position within the documented fixed-point tolerance, then require exact greedy token streams. Re-run after `reset()` and require identical traces.

- [ ] **Step 3: Run and verify the new tests fail.**

Run: `nix develop --command python -m unittest -v tests/test_tinystories_int_reference.py`

- [ ] **Step 4: Implement the minimal package-only reference.**

The module may import NumPy and package helpers, but must not import Torch or Transformers. All arithmetic paths read formats/scales from the manifest. Reject a corrupt package before inference.

- [ ] **Step 5: Freeze regression prompts and commit.**

Include a short story prefix, a prompt whose tokenized input reaches the supported limit with requested output, and a punctuation/unknown-input host case. Store prompt text, prompt IDs, requested count, output IDs, decoded output, and reference trace hash.

Run: `nix develop --command python -m unittest -v tests/test_tinystories_int_reference.py`

Expected: all primitive and three full-stream tests pass.

```bash
git add tinystories/int_reference.py tests/test_tinystories_int_reference.py model_packages/tinystories-1m/regressions.json
git commit -m "feat: add integer TinyStories reference"
```

### Task 5: Implement GPT-Neo RTL Primitives

**Files:**
- Create: `fpga/rtl/gptneo_layernorm.sv`
- Create: `fpga/rtl/gptneo_gelu.sv`
- Create: `fpga/rtl/gptneo_attention.sv`
- Create: `fpga/rtl/gptneo_resident_gemv.sv`
- Create: `fpga/tb/tb_gptneo_layernorm.sv`
- Create: `fpga/tb/tb_gptneo_gelu.sv`
- Create: `fpga/tb/tb_gptneo_attention.sv`
- Create: `fpga/tb/tb_gptneo_gemv.sv`
- Create: `tests/test_rtl_gates.py`
- Modify: `flake.nix`

**Interfaces:**
- Each block uses ready/valid request/response interfaces and manifest-generated widths.
- `gptneo_resident_gemv` accepts matrix ID, M/K, input stream, and returns indexed accumulator outputs; tied matrix ID is shared for embedding/head access.
- Nix check `.#checks.x86_64-linux.rtl-primitives` generates memories, builds all four Icarus simulations, and compares dumps with `IntegerGPTNeo.trace`.

- [ ] **Step 1: Add failing test orchestration for four named verdicts.**

Require `GPTNEO_LAYERNORM_PASS`, `GPTNEO_GELU_PASS`, `GPTNEO_ATTN_PASS`, and `GPTNEO_GEMV_PASS`. The Python test must fail if a simulator exits zero without its verdict.

- [ ] **Step 2: Run the gate and verify missing RTL failures.**

Run: `nix build .#checks.x86_64-linux.rtl-primitives -L`

- [ ] **Step 3: Port LayerNorm and GELU with GPT-Neo semantics.**

Start from Stage-3 arithmetic, add LayerNorm beta, bind epsilon `1e-5` through the package format, and generate the `gelu_new` LUT from the same oracle used by the integer reference. Tests cover zeros, extrema, real calibration vectors, and backpressure.

- [ ] **Step 4: Implement bounded causal attention.**

Parameterize `D=64`, `NHEAD=16`, `HEAD_DIM=4`, and `TMAX=32`. Store per-layer K/V in BRAM-compatible single-write/registered-read memories. The test drives positions 0, 1, 15, and 31 and compares scores, probabilities, context, and cache contents.

- [ ] **Step 5: Implement initialized resident GEMV and tied lookup/head storage.**

Infer Kintex-7 BRAM with registered reads and deterministic `.mem` initialization. Support INT4/mixed tensor formats from generated localparams. The test checks signed nibble extraction, padded rows, matrix offsets, and the tied embedding/head alias.

- [ ] **Step 6: Run gates, inspect Yosys memory inference, and commit.**

Run:

```bash
nix build .#checks.x86_64-linux.rtl-primitives -L
nix build .#gptneo-rtl-primitives-yosys-report -L
nix develop --command python -m unittest -v tests/test_rtl_gates.py
```

Expected: all verdicts pass; Yosys reports nonzero RAMB18/RAMB36 cells and no inferred latches or multiple drivers.

```bash
git add fpga/rtl/gptneo_*.sv fpga/tb/tb_gptneo_*.sv tests/test_rtl_gates.py flake.nix
git commit -m "feat: add package-driven GPT-Neo RTL blocks"
```

### Task 6: Integrate Exact Multi-Token RTL Inference

**Files:**
- Create: `fpga/rtl/gptneo_sequencer.sv`
- Create: `fpga/tb/tb_gptneo_sequencer.sv`
- Create: `tinystories/write_rtl_fixture.py`
- Modify: `tests/test_rtl_gates.py`
- Modify: `flake.nix`

**Interfaces:**
- `gptneo_sequencer` accepts `start`, prompt length/data write port, requested generation count, and exposes `busy`, `done`, error code, cycle count, output length, and token read port.
- Nix check `.#checks.x86_64-linux.rtl-sequencer` runs all three committed prompts and one corrupted-package negative case.

- [ ] **Step 1: Add the failing sequencer verdict contract.**

Require one line per prompt: `GPTNEO_SEQ_PASS case=<name> tokens=<n>/<n>`, plus `GPTNEO_SEQ_NEGATIVE_PASS error=PACKAGE_HASH` for the corruption case.

- [ ] **Step 2: Generate simulation fixtures from the canonical package.**

`write_rtl_fixture.py` writes prompt, expected tokens, tensor images, localparams, and trace slices into a Nix build directory. It must verify package hashes before writing and emit `fixture.json` containing every input hash.

- [ ] **Step 3: Implement the eight-layer sequencer.**

Schedule embedding, LN1, Q/K/V GEMVs, attention, output projection, residual, LN2, MLP up, GELU, MLP down, residual, final LN, tied head, argmax, KV append, and token feedback. End on EOS, requested count, context limit, or explicit error. Preserve a watchdog counter and expose first-error stage/index.

- [ ] **Step 4: Run the full simulation gate.**

Run: `nix build .#checks.x86_64-linux.rtl-sequencer -L`

Expected: three exact streams pass, negative corruption fails deterministically, and simulated cycles predict <= five seconds at 12.5 MHz for 16 output tokens.

- [ ] **Step 5: Commit the complete simulated accelerator.**

```bash
git add fpga/rtl/gptneo_sequencer.sv fpga/tb/tb_gptneo_sequencer.sv tinystories/write_rtl_fixture.py tests/test_rtl_gates.py flake.nix
git commit -m "feat: integrate TinyStories GPT-Neo sequencer"
```

### Task 7: Produce and Qualify the Compiled-Prompt Kintex Bitstream

**Files:**
- Create: `fpga/rtl/tinystories_selftest_top.sv`
- Create: `fpga/tb/tb_tinystories_selftest_top.sv`
- Create: `fpga/constraints/tinystories_ypcb.xdc`
- Create: `nix/kintex-tinystories.nix`
- Create: `scripts/build_tinystories_bitstream.sh`
- Modify: `flake.nix`
- Modify: `tests/test_rtl_gates.py`

**Interfaces:**
- Top ports are `SYS_CLK`, `SYS_RSTN`, and `led_3bits_tri_o[2:0]` on the proven pins.
- `.#tinystories-selftest-bitstream` emits `.bit`, Yosys/nextpnr logs, utilization/timing JSON, model/source/tool hashes, and board command JSON.

- [ ] **Step 1: Write failing wrapper/build contract tests.**

Assert real 50 MHz input constraint, derived 12.5 MHz accelerator timing, pins AA28/R28/P30/M30/N30, no external data ports, exact model-package reference, lock hash checks, and absence of CIRCT/LLVM/Vivado commands.

- [ ] **Step 2: Implement wrapper simulation first.**

After reset settling, auto-run the short committed regression prompt. Latch green/pass only after exact token count and sequence, red/fail on mismatch/timeout, and drive heartbeat independently. A compile-time negative expected token must latch failure.

- [ ] **Step 3: Add narrow locked synthesis/P&R/bitstream derivation.**

Follow the proven `nix/kintex-selftest.nix` Yosys -> nextpnr -> FASM -> frames -> bitstream commands. Read all RTL and generated memories from immutable Nix store paths. Save complete logs and parse resource counts; fail if tied storage is duplicated, BRAM exceeds capacity/reserve, timing misses 12.5 MHz, or any initialized memory is absent.

- [ ] **Step 4: Run host gates and build the exact bitstream.**

Run:

```bash
nix build .#checks.x86_64-linux.tinystories-selftest -L
nix build .#tinystories-selftest-bitstream --no-update-lock-file -L
nix path-info -S .#tinystories-selftest-bitstream
```

Expected: simulation passes, P&R succeeds for `xc7k480tffg1156-1`, timing meets 12.5 MHz, and output receipt records nonzero BRAM/DSP/LUT/FF use.

- [ ] **Step 5: Program and observe the compiled-prompt gate.**

Use the bitstream's `board-command.json` with locked openFPGALoader, verify detected IDCODE `0x23751093`, configuration `DONE=1`, heartbeat activity, M30 asserted, and N30 deasserted. Record human LED observation separately from host configuration evidence.

- [ ] **Step 6: Commit the board-proof path.**

```bash
git add fpga/rtl/tinystories_selftest_top.sv fpga/tb/tb_tinystories_selftest_top.sv fpga/constraints/tinystories_ypcb.xdc nix/kintex-tinystories.nix scripts/build_tinystories_bitstream.sh tests/test_rtl_gates.py flake.nix
git commit -m "build: add TinyStories Kintex self-test image"
```

### Task 8: Build and Prove the BSCAN Packet Transport

**Files:**
- Create: `fpga/rtl/async_fifo.sv`
- Create: `fpga/rtl/bscan_packet_endpoint.sv`
- Create: `fpga/tb/tb_async_fifo.sv`
- Create: `fpga/tb/tb_bscan_packet_endpoint.sv`
- Create: `host/jtag_transport.h`
- Create: `host/jtag_transport.c`
- Create: `host/kevin_jtag_cli.py`
- Create: `tests/test_jtag_packet.py`
- Create: `tests/test_host_cli.py`
- Modify: `flake.nix`

**Interfaces:**
- Packet v1 request: magic `0x4b47`, version, command, prompt count, generation count, little-endian 16-bit token IDs, CRC32.
- Reply: magic `0x4b52`, version, status, output count, 64-bit cycles, token IDs, CRC32.
- C library functions: `kj_open(serial)`, `kj_user1_exchange(tx, tx_len, rx, rx_cap, timeout_ms)`, `kj_close()`.
- CLI subcommands: `packet-selftest`, `program`, `infer`, and `qualify`.

- [ ] **Step 1: Write packet codec and malformed-input tests.**

Test empty/maximum prompt, context overflow, bad magic/version/CRC, truncation, busy, timeout, unknown token, and reply-length mismatch. Assert the CLI module contains no import of `tinystories.int_reference`, Torch, or Transformers model classes.

- [ ] **Step 2: Write asynchronous FIFO tests with unrelated clocks and reset orderings.**

Drive producer/consumer clocks at coprime periods, random backpressure, FIFO full/empty wraparound, and reset asserted from either domain. Require ordered lossless output and no X values after reset.

- [ ] **Step 3: Implement packet endpoint and BSCANE2 wrapper.**

Use Gray-coded pointers with two-flop synchronizers. Keep packet assembly in the accelerator domain after bytes cross the FIFO. Instantiate `BSCANE2` with one fixed user chain, shift bytes LSB-first, and return explicit status packets. Simulation replaces the primitive with a behavioral scan driver through a compile define.

- [ ] **Step 4: Implement the libftdi host transport and tokenizer CLI.**

The C layer owns TAP state transitions and USER1 scans; it selects HS3 serial `210299BF3824` unless overridden. Python calls the shared library through `ctypes`, loads tokenizer assets from the canonical package, enforces 32 total tokens, and emits a receipt. It must never calculate model outputs.

- [ ] **Step 5: Run simulation and host loopback gates.**

Run:

```bash
nix build .#checks.x86_64-linux.jtag-transport -L
nix develop --command python -m unittest -v tests/test_jtag_packet.py tests/test_host_cli.py
nix run .#kevin-jtag -- packet-selftest
```

Expected: FIFO/packet tests pass and host codec round-trips all boundary cases without hardware.

- [ ] **Step 6: Build a transport-only bitstream and prove physical loopback.**

Route BSCAN packets to an echo/status engine, build with the locked EDA stack, program SRAM, and exchange at least 1,000 packets of varied lengths. Require zero CRC, ordering, timeout, or data errors before accelerator integration.

- [ ] **Step 7: Commit the independently qualified transport.**

```bash
git add fpga/rtl/async_fifo.sv fpga/rtl/bscan_packet_endpoint.sv fpga/tb/tb_async_fifo.sv fpga/tb/tb_bscan_packet_endpoint.sv host tests/test_jtag_packet.py tests/test_host_cli.py flake.nix
git commit -m "feat: add open-source JTAG prompt transport"
```

### Task 9: Integrate Interactive Hardware Inference

**Files:**
- Create: `fpga/rtl/tinystories_interactive_top.sv`
- Create: `fpga/tb/tb_tinystories_interactive_top.sv`
- Create: `scripts/qualify_ypcb.py`
- Modify: `nix/kintex-tinystories.nix`
- Modify: `host/kevin_jtag_cli.py`
- Modify: `flake.nix`

**Interfaces:**
- `.#tinystories-interactive-bitstream` produces the final bitstream and qualification bundle.
- `nix run .#kevin-jtag -- infer --program --prompt TEXT --max-new-tokens N --json-receipt PATH` is the user-facing command.
- `qualify_ypcb.py --bundle PATH --repeat 3` runs the committed hardware cases and compares replies with `regressions.json`.

- [ ] **Step 1: Write the failing end-to-end simulation test.**

Drive serialized request packets through the BSCAN behavioral model, wait for the reply, decode it with the production host codec, and compare exact token IDs/cycles/status. Cover success, context rejection, and accelerator watchdog timeout.

- [ ] **Step 2: Connect packet commands to the sequencer.**

Copy accepted prompt IDs into the sequencer prompt RAM, validate `prompt_count + generation_count <= 32`, pulse start once, collect output IDs, and form the reply. Keep the BSCAN and accelerator reset/error state observable in status. LEDs retain heartbeat/activity/failure meanings.

- [ ] **Step 3: Build and verify the final bitstream.**

Run:

```bash
nix build .#checks.x86_64-linux.interactive-sim -L
nix build .#tinystories-interactive-bitstream --no-update-lock-file -L
nix run .#kevin-jtag -- infer --program --prompt "Once upon a time" --max-new-tokens 16 --json-receipt artifacts/tinystories/manual.json
```

Expected: P&R/timing gates pass, hardware reply status is success, completion is decoded, and measured wall time is <= five seconds.

- [ ] **Step 4: Run the repeated physical qualification suite.**

Run: `nix run .#qualify-ypcb -- --bundle result --repeat 3`

Expected: three regression prompts each pass 3/3 with exact integer-reference tokens; one arbitrary prompt completes interactively; receipt hashes match the programmed bundle.

- [ ] **Step 5: Commit the integrated system.**

```bash
git add fpga/rtl/tinystories_interactive_top.sv fpga/tb/tb_tinystories_interactive_top.sv scripts/qualify_ypcb.py nix/kintex-tinystories.nix host/kevin_jtag_cli.py flake.nix
git commit -m "feat: run interactive TinyStories inference on YPCB"
```

### Task 10: Final Reproducibility Audit and Documentation

**Files:**
- Create: `docs/ypcb-tinystories.md`
- Modify: `README.md`
- Generate/ignore: `artifacts/tinystories/qualification.json`

**Interfaces:**
- Documentation gives exact Nix build/use commands, model/tool hashes, recovery actions, LED meanings, limitations, and the distinction between RTL-qualified and hardware-qualified evidence.

- [ ] **Step 1: Run the full fresh verification set.**

```bash
nix flake check --no-update-lock-file -L
nix build .#tinystories-1m-package-regenerated --no-update-lock-file -L
nix build .#tinystories-interactive-bitstream --no-update-lock-file -L
nix run .#qualify-ypcb -- --bundle result --repeat 3
git status --short
```

Expected: every check/build passes, hardware qualification passes 3/3 for all cases, no external lock changed, and only ignored artifact receipts remain untracked.

- [ ] **Step 2: Write the operator guide and truthful status summary.**

Document `nix run .#kevin-jtag -- infer`, HS3 serial override, reset/reprogram recovery, maximum context/completion, greedy-only behavior, quantization quality numbers, actual measured wall time/cycles, nextpnr timing/utilization, and deferred DDR3/general-architecture work. Remove or clearly mark obsolete KV260/Vivado claims from the top-level path without rewriting historical engineering logs.

- [ ] **Step 3: Verify documentation commands and links.**

Run every read-only command copied into the guide and `rg -n "Vivado|required KV260|global python|pip install" README.md docs/ypcb-tinystories.md`. Expected: no active YPCB instructions require those dependencies.

- [ ] **Step 4: Commit documentation.**

```bash
git add README.md docs/ypcb-tinystories.md
git commit -m "docs: document open-source YPCB TinyStories inference"
```

---

## Completion Evidence Checklist

- Quantized package regenerates from the pinned Hugging Face revision and matches committed bytes.
- Perplexity regression is <= 10 percent; top-1 agreement is >= 90 percent.
- Integer reference is Torch-free and package-hash validating.
- Primitive and full-sequencer RTL gates match the integer reference exactly.
- Yosys retains initialized BRAM and a single tied embedding/head allocation.
- nextpnr routes `xc7k480tffg1156-1` and meets the 12.5 MHz accelerator timing gate.
- Compiled-prompt bitstream passes on the physical board before JTAG integration.
- BSCAN transport completes 1,000 physical loopback packets without error.
- Final hardware passes three prompts, three repeats each, with exact token streams.
- A 16-token interactive response completes within five seconds.
- Nix/store, model, source, lock, RTL, bitstream, prompt, output, and timing hashes are recorded.
