{ pkgs, compilerLab, source ? ../., interactive ? false }:
let
  eda = import ./eda-toolchain.nix {
    inherit compilerLab;
    system = pkgs.system;
  };
  inherit (eda) yosys nextpnr chipdb fasm prjxray familyDb part partFile;
  python = pkgs.python312.withPackages (ps: [
    ps.numpy ps.intervaltree ps.json5 ps.progressbar2 ps.pyyaml ps.simplejson
  ]);
  sourcePath = builtins.path {
    path = source;
    name = "kev-gpt-source";
  };
  buildSource = sourcePath;
  modelPackage = buildSource + "/model_packages/tinystories-1m";
  top = if interactive then "tinystories_interactive_top" else "tinystories_selftest_top";
  topSource = if interactive then "${buildSource}/fpga/rtl/tinystories_interactive_top.sv ${buildSource}/fpga/rtl/bscan_packet_endpoint.sv ${buildSource}/fpga/rtl/async_fifo.sv ${buildSource}/fpga/rtl/tinystories_packet_controller.sv" else "${buildSource}/fpga/rtl/tinystories_selftest_top.sv";
  bitName = if interactive then "tinystories-interactive.bit" else "tinystories-selftest.bit";
  name = "tinystories-ypcb-${if interactive then "interactive" else "selftest"}";
  synthesis = pkgs.runCommand "${name}-synthesis" { } ''
    set -euo pipefail
    mkdir work "$out"
    export PYTHONPATH=${buildSource}
    ${python}/bin/python -m tinystories.write_rtl_fixture --package ${modelPackage} --output work/fixture
    ${python}/bin/python ${buildSource}/tinystories/rtl_memories.py --output work/fixture
    cd work/fixture
    ${yosys}/bin/yosys -l "$out/yosys.log" -p 'read_verilog -sv -I. ${topSource} ${buildSource}/fpga/rtl/gptneo_sequencer.sv ${buildSource}/fpga/rtl/gptneo_layernorm.sv ${buildSource}/fpga/rtl/gptneo_gelu.sv ${buildSource}/fpga/rtl/gptneo_attention.sv ${buildSource}/fpga/rtl/gptneo_iterative_divider.sv ${buildSource}/fpga/rtl/gptneo_resident_gemv.sv; synth_xilinx -family xc7 -top ${top}; write_json design.json'
    cp design.json "$out/"
    cp gptneo_package.svh "$out/"
  '';
  # The locked nextpnr hierarchical frontend has a latent merge_nets bug when
  # two submodule ports alias one net.  Normalize the cached synthesis result
  # to a flat JSON netlist so P&R never enters that importer path.
  pnrNetlist = pkgs.runCommand "${name}-pnr-netlist" { } ''
    mkdir "$out"
    ${yosys}/bin/yosys -l "$out/flatten.log" -p \
      'read_json ${synthesis}/design.json; flatten; write_json design.json'
    cp design.json "$out/"
  '';
in pkgs.runCommand "${name}-bitstream" {
  nativeBuildInputs = [ fasm prjxray python ];
  passthru = {
    inherit synthesis pnrNetlist modelPackage part;
    buildMode = if interactive then "interactive" else "selftest";
  };
} ''
  set -euo pipefail
  mkdir work "$out"
  cd work
  bash ${buildSource}/scripts/run-logged.sh "$out/nextpnr.log" \
    ${nextpnr}/bin/nextpnr-xilinx \
    --chipdb ${chipdb} \
    --freq 50 \
    --xdc ${buildSource}/fpga/constraints/kintex_selftest.xdc \
    --json ${pnrNetlist}/design.json --fasm design.fasm
  export PYTHONPATH="${fasm}/lib/python3.12/site-packages:${python}/${pkgs.python312.sitePackages}:${prjxray}/usr/share/python3''${PYTHONPATH:+:$PYTHONPATH}"
  export PRJXRAY_DB_DIR="${familyDb}"
  export PRJXRAY_PYTHON_DIR="${prjxray}/usr/share/python3"
  fasm2frames --db-root "${familyDb}" --part ${part} design.fasm design.frm
  xc7frames2bit --part_file "${partFile}" --frm_file design.frm --output_file "$out/${bitName}"
  cp ${synthesis}/gptneo_package.svh "$out/"
  printf '%s\n' '{"cable":"digilent_hs3","idcode":"0x23751093","part":"${part}","bitstream":"${bitName}"}' > "$out/board-command.json"
''
