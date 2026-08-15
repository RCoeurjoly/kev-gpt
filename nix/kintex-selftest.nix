{ compilerLab ? "/home/roland/compiler-lab-llm2fpga" }:

let
  system = builtins.currentSystem;
  task3 = builtins.getFlake "path:${compilerLab}/task3-main";
  pkgs = task3.inputs.nixpkgs.legacyPackages.${system};
  openXC7 = task3.inputs.openXC7.packages.${system};
  toolchain = task3.lib.${system}.task3Toolchain;
  yosys = task3.inputs.nix-eda.packages.${system}.yosys;
  fasm = openXC7.fasm;
  prjxray = openXC7.prjxray;
  pythonDeps = pkgs.python312.withPackages (ps: [
    ps.intervaltree
    ps.json5
    ps.progressbar2
    ps.pyyaml
    ps.simplejson
  ]);
  familyDb = "${toolchain.nextpnr}/share/nextpnr/external/prjxray-db/kintex7";
  part = "xc7k480tffg1156-1";
  partFile = "${familyDb}/${part}/part.yaml";
  rtl = ../fpga/rtl/kintex_selftest_top.sv;
  xdc = ../fpga/constraints/kintex_selftest.xdc;
in
pkgs.runCommand "kintex-selftest.bit" {
  nativeBuildInputs = [ fasm prjxray pythonDeps ];
} ''
  set -euo pipefail

  ${yosys}/bin/yosys -p \
    "read_verilog -sv ${rtl}; synth_xilinx -family xc7 -top kintex_selftest_top; write_json design.json"

  ${toolchain.nextpnr}/bin/nextpnr-xilinx \
    --chipdb ${toolchain.chipdb} \
    --xdc ${xdc} \
    --json design.json \
    --fasm design.fasm

  export PYTHONPATH="${fasm}/lib/python3.12/site-packages:${pythonDeps}/${pkgs.python312.sitePackages}:${prjxray}/usr/share/python3''${PYTHONPATH:+:$PYTHONPATH}"
  export PRJXRAY_DB_DIR="${familyDb}"
  export PRJXRAY_PYTHON_DIR="${prjxray}/usr/share/python3"

  fasm2frames \
    --db-root "${familyDb}" \
    --part ${part} \
    design.fasm design.frm
  xc7frames2bit \
    --part_file "${partFile}" \
    --frm_file design.frm \
    --output_file "$out"
''
