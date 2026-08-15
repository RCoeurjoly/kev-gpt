# TinyStories-1M Inference on YPCB Kintex-7

## Goal

Run real, interactive TinyStories inference on the YPCB-00338-1P1 board
(`xc7k480tffg1156-1`) without Vivado, proprietary IP, global Python packages,
or model training. A host command sends a text prompt through the connected
Digilent HS3, the FPGA performs the complete autoregressive transformer
inference, and the command prints the generated text.

The primary implementation repository is `kev-gpt`. The checkout at
`/home/roland/compiler-lab-llm2fpga` is a read-only, pinned source of the proven
Nix/open-source YPCB toolchain: Yosys, nextpnr-Xilinx, the Kintex-7 chip
database, Project X-Ray/openXC7 bitstream tools, and openFPGALoader.

## Product Acceptance

The first release accepts a UTF-8 prompt, supports at most 32 total tokens, and
greedily generates at most 16 tokens. A correct 16-token response must complete
within five seconds. The host performs tokenization, JTAG transport, and text
rendering only; transformer inference and greedy token selection execute in
FPGA logic.

The normal user path is one Nix command that reuses cached artifacts, programs
the volatile SRAM image when requested, sends a prompt, receives tokens, and
prints text plus a machine-readable receipt. Nonvolatile flash programming is
not a release gate.

## Pretrained Model

The model is a content-hash-pinned revision of
`roneneldan/TinyStories-1M`, not a newly trained Kevin checkpoint. Its relevant
configuration is GPT-Neo causal language modeling with eight layers, hidden
width 64, 16 attention heads, vocabulary size 50,257, learned positional
embeddings, `gelu_new`, alternating global/local attention, and a local window
of 256 tokens.

The hardware context is limited to 32 tokens. Therefore local and global causal
attention have identical semantics within the supported domain. The importer
must prove `hardware_context <= local_window`; it rejects configurations that
violate the condition.

No optimizer step or fine-tuning is allowed in the baseline. A pinned
TinyStories validation slice is used only for post-training calibration and
quality measurement. If post-training quantization cannot meet the acceptance
threshold after mixed-precision exploration, the pipeline stops with evidence
and requests approval before any QAT work.

## Model Import and Generality Boundary

A strict Hugging Face `GPTNeoForCausalLM` adapter consumes the pinned checkpoint,
configuration, and tokenizer. It emits a canonical model package rather than
exposing Hugging Face filenames to RTL or host code.

The package contains:

- a versioned manifest with dimensions, tensor roles, shapes, layouts, numeric
  formats, attention mode, context limit, and source hashes;
- the GPT-2-compatible tokenizer assets needed by the host;
- packed weights, embeddings, positional rows 0 through 31, LayerNorm
  parameters, quantization scales, and nonlinear lookup tables;
- fixed software and hardware regression prompts and golden token streams;
- quality, memory-budget, and package-integrity receipts.

The word embedding and language-model head are tied and stored physically once.
The canonical manifest is the future generality boundary: other Hugging Face
architectures require new adapters, but the hardware/package interface does not
depend on their repository layout. This release supports the pinned GPT-Neo
model only and does not claim arbitrary Hugging Face compatibility.

The compact quantized package is committed so a fresh clone can build the FPGA
image without downloading PyTorch or rerunning conversion. A separate Nix target
regenerates and byte-validates the package from the pinned upstream model.

## Quantization and Numeric Contract

The initial target is per-channel or grouped INT4 weights and INT8 activations,
with wider formats only for operations proven sensitive. Embeddings and the tied
head are included in the explicit mixed-precision search because they dominate
storage. Accumulators, residuals, LayerNorm, softmax, GELU, and scaling use
fixed-point formats selected by calibration and recorded in the manifest.

The quantized package is accepted only when:

1. validation perplexity is no more than 10 percent worse than the pinned FP32
   checkpoint on the pinned validation slice;
2. next-token top-1 agreement is at least 90 percent on that slice; and
3. the integer reference, RTL simulation, and FPGA emit identical token IDs for
   every hardware regression prompt.

The first two gates protect model quality; the third protects implementation
correctness. Quantized output is not required to be token-identical to FP32 for
every prompt.

## Accelerator Architecture

The existing Stage-3 `kev-gpt` accelerator is the implementation foundation.
It is adapted rather than replaced by the compiler-lab generated-RTL route,
whose full TinyStories path currently stops before synthesizable SystemVerilog.

The parameterized accelerator implements:

1. token and learned positional embedding lookup;
2. eight GPT-Neo blocks containing LayerNorm with weight and bias, Q/K/V
   projections, causal KV-cached attention, softmax, output projection,
   residuals, second LayerNorm, MLP, exact `gelu_new` approximation, and the
   second residual;
3. final LayerNorm;
4. tied vocabulary-head GEMV and greedy argmax; and
5. autoregressive feedback until the requested count, EOS, or context limit.

The reference implementation consumes exactly the same canonical model package
as the RTL. Tensor layout, rounding, saturation, scale application, lookup-table
addressing, and tie semantics have one definition in the manifest/export code.

Kintex-7 has BRAM rather than UltraScale+ URAM, so the resident memories are
restructured as single/true-dual-port initialized BRAM with access schedules
supported by Yosys and nextpnr-Xilinx. The pretrained model, the first 32
positional rows, KV state, scratch state, and tables remain on-chip. No DDR3
traffic occurs in the inference loop.

## Resource-Fit Policy

Constrained Yosys and nextpnr evidence decides whether the design fits; source
parameter counts or simulation do not. If placement fails, changes are tried in
this order:

1. prove the tied embedding/head is stored only once;
2. prove unused positional rows are absent;
3. improve BRAM width/depth packing and reduce lane parallelism without changing
   arithmetic;
4. revise mixed-precision allocation while retaining the quality thresholds;
5. stop and request approval before starting an external-DDR3 phase.

Layers, vocabulary, hidden width, and checkpoint weights are never silently
reduced. Such a change would be a different model.

## Interactive JTAG Transport

The YPCB exposes no established general-purpose UART connector. The interactive
interface therefore uses the already connected Digilent HS3 and the 7-series
`BSCANE2` user chain. The locked Yosys and nextpnr-Xilinx stack contains support
for the primitive and its Kintex-7 BSCAN resources.

A versioned binary packet contains a command, prompt-token count, generation
count, token IDs, and CRC. The reply contains status, accelerator cycle count,
output-token count, token IDs, and CRC. Malformed packets, unsupported versions,
unknown tokenizer input, excess context, busy submission, timeout, and CRC
failure return distinct errors.

JTAG DRCK and the 50 MHz board clock are asynchronous. Small dual-clock FIFOs
and synchronized control flags isolate the domains. No unsynchronized payload or
single-cycle pulse crosses the boundary.

The Nix-built host CLI uses an open-source FTDI/JTAG library, selects the HS3 by
serial number, tokenizes from package assets, transfers packets, decodes the
reply, and prints text and JSON evidence. It contains no inference fallback; a
hardware failure cannot be hidden by computing tokens on the host.

## Board Boundary

The established board interface is retained:

- `SYS_CLK`: package pin `AA28`, LVCMOS18, physical 50 MHz source;
- `SYS_RSTN`: package pin `R28`, LVCMOS18, active low;
- heartbeat LED: package pin `P30`, LVCMOS18;
- pass/activity LED: package pin `M30`, LVCMOS18;
- failure LED: package pin `N30`, LVCMOS18.

The first implementation may use a divided clock or clock enable equivalent to
12.5 MHz for accelerator correctness. Timing constraints must describe the real
50 MHz input and every generated/derived clock. The heartbeat distinguishes a
configured, clocked design; activity shows an accepted request or completed
regression; failure latches protocol, timeout, or self-test failure until reset.

## Nix and Open-Source Build

`kev-gpt` gains its own flake and lock. It follows the exact nixpkgs/tool inputs
already locked by compiler-lab rather than introducing a second EDA universe.
Additional packages needed for model conversion and JTAG transport are pinned by
source revision and content hash. No command invokes ambient Python, pip,
Vivado, Vitis, Xilinx MIG, or proprietary debug IP.

Separate Nix outputs provide:

- upstream model fetch and integrity check;
- calibration and quantized-package regeneration;
- integer-reference tests and quality receipts;
- RTL lint/simulation and negative tests;
- Yosys synthesis reports;
- nextpnr-Xilinx placement, routing, timing, and utilization reports;
- openXC7 FASM/frame/bitstream conversion;
- host JTAG CLI and one-command runner; and
- an aggregate qualification bundle with hashes and logs.

Normal builds use the committed package and do not pull the custom LLVM/CIRCT
closure. Lock files are never updated as a side effect of building.

## Verification Gates

Work proceeds in fail-fast order:

1. **Import and quality:** fetch the pinned model, produce the package, pass
   integrity, perplexity, top-1, and memory-budget gates.
2. **Integer and RTL equivalence:** implement GPT-Neo semantics in the integer
   reference and RTL; run exact multi-token tests, including negative corruption
   and timeout cases.
3. **Compiled-prompt silicon proof:** synthesize and route the complete model
   without JTAG transport, program the exact hashed image, and pass a built-in
   golden prompt through the LEDs. This separates accelerator/BRAM correctness
   from transport correctness.
4. **JTAG transport:** independently simulate packet framing, CRC, CDC/FIFOs, and
   recovery, then integrate BSCAN and prove a transport loopback on hardware.
5. **Interactive inference:** run three committed prompts covering a short
   request, maximum supported context, and punctuation/unknown-input handling.
   Each prompt runs three times on hardware and must match the integer reference.
   Then run one arbitrary interactive prompt through the CLI.

Every physical receipt records model-package hash, bitstream and Nix store path,
source revision, both relevant lock hashes, tool versions, prompt tokens, output
tokens, cycles, wall time, and configuration status. Host simulation, successful
configuration, or an asserted `DONE` pin alone is not an inference pass.

## Explicitly Deferred Work

- Model training, fine-tuning, or QAT without separate approval.
- Contexts longer than 32 tokens or completions longer than 16 tokens.
- Temperature, top-k, nucleus, or stochastic sampling.
- Arbitrary Hugging Face architecture support.
- DDR3, PCIe, Ethernet, or external UART interfaces.
- Nonvolatile flash as a required deployment path.
- Multi-stream serving and throughput optimization beyond the five-second gate.
