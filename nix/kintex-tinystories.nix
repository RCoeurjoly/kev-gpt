{ compilerLab ? "/home/roland/compiler-lab-llm2fpga", source ? ../.
, interactive ? false }:
let
  system = builtins.currentSystem;
  task3 = builtins.getFlake "path:${compilerLab}/task3-main";
  pkgs = task3.inputs.nixpkgs.legacyPackages.${system};
  openXC7 = task3.inputs.openXC7.packages.${system};
  toolchain = task3.lib.${system}.task3Toolchain;
  yosys = task3.inputs.nix-eda.packages.${system}.yosys;
  fasm = openXC7.fasm;
  prjxray = openXC7.prjxray;
  python = pkgs.python312.withPackages (ps: [
    ps.numpy ps.intervaltree ps.json5 ps.progressbar2 ps.pyyaml ps.simplejson
  ]);
  familyDb = "${toolchain.nextpnr}/share/nextpnr/external/prjxray-db/kintex7";
  part = "xc7k480tffg1156-1";
  partFile = "${familyDb}/${part}/part.yaml";
  modelPackage = source + "/model_packages/tinystories-1m";
  top = if interactive then "tinystories_interactive_top" else "tinystories_selftest_top";
  topSource = if interactive then "${source}/fpga/rtl/tinystories_interactive_top.sv ${source}/fpga/rtl/bscan_packet_endpoint.sv ${source}/fpga/rtl/async_fifo.sv ${source}/fpga/rtl/tinystories_packet_controller.sv" else "${source}/fpga/rtl/tinystories_selftest_top.sv";
  bitName = if interactive then "tinystories-interactive.bit" else "tinystories-selftest.bit";
in pkgs.runCommand "tinystories-ypcb-${if interactive then "interactive" else "selftest"}-bitstream" {
  nativeBuildInputs = [ fasm prjxray python ];
} ''
  set -euo pipefail
  mkdir work "$out"
  export PYTHONPATH=${source}
  ${python}/bin/python -m tinystories.write_rtl_fixture --package ${modelPackage} --output work/fixture
  ${python}/bin/python ${source}/tinystories/rtl_memories.py --output work/fixture
  cd work/fixture
  ${yosys}/bin/yosys -l "$out/yosys.log" -p 'read_verilog -sv -I. ${topSource} ${source}/fpga/rtl/gptneo_sequencer.sv ${source}/fpga/rtl/gptneo_layernorm.sv ${source}/fpga/rtl/gptneo_gelu.sv ${source}/fpga/rtl/gptneo_attention.sv ${source}/fpga/rtl/gptneo_iterative_divider.sv ${source}/fpga/rtl/gptneo_resident_gemv.sv; synth_xilinx -family xc7 -top ${top}; write_json design.json'
  bash ${source}/scripts/run-logged.sh "$out/nextpnr.log" \
    ${toolchain.nextpnr}/bin/nextpnr-xilinx \
    --chipdb ${toolchain.chipdb} \
    --freq 50 \
    --xdc ${source}/fpga/constraints/kintex_selftest.xdc \
    --json design.json --fasm design.fasm
  export PYTHONPATH="${fasm}/lib/python3.12/site-packages:${python}/${pkgs.python312.sitePackages}:${prjxray}/usr/share/python3''${PYTHONPATH:+:$PYTHONPATH}"
  export PRJXRAY_DB_DIR="${familyDb}"
  export PRJXRAY_PYTHON_DIR="${prjxray}/usr/share/python3"
  fasm2frames --db-root "${familyDb}" --part ${part} design.fasm design.frm
  xc7frames2bit --part_file "${partFile}" --frm_file design.frm --output_file "$out/${bitName}"
  cp gptneo_package.svh "$out/"
  printf '%s\n' '{"cable":"digilent_hs3","idcode":"0x23751093","part":"${part}","bitstream":"${bitName}"}' > "$out/board-command.json"
''
