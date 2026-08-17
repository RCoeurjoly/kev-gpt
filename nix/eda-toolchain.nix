{ compilerLab, system }:

let
  task3 = compilerLab;
  toolchain = task3.lib.${system}.task3Toolchain;
  openXC7 = task3.inputs.openXC7.packages.${system};
  nextpnr = toolchain.nextpnr or openXC7.nextpnr-xilinx;
  chipdb = toolchain.chipdb or openXC7.nextpnr-xilinx-chipdb.kintex7;
  familyDb = "${nextpnr}/share/nextpnr/external/prjxray-db/kintex7";
  part = "xc7k480tffg1156-1";
in
{
  inherit toolchain part;
  yosys = task3.inputs.nix-eda.packages.${system}.yosys;
  inherit nextpnr chipdb;
  fasm = openXC7.fasm;
  prjxray = openXC7.prjxray;
  inherit familyDb;
  partFile = "${familyDb}/${part}/part.yaml";
}
