{ compilerLab }:

let
  system = builtins.currentSystem;
  task3 = builtins.getFlake "path:${compilerLab}/task3-main";
  toolchain = task3.lib.${system}.task3Toolchain;
  openXC7 = task3.inputs.openXC7.packages.${system};
  familyDb = "${toolchain.nextpnr}/share/nextpnr/external/prjxray-db/kintex7";
  part = "xc7k480tffg1156-1";
in
{
  yosys = task3.inputs.nix-eda.packages.${system}.yosys;
  nextpnr = toolchain.nextpnr;
  chipdb = toolchain.chipdb;
  fasm = openXC7.fasm;
  prjxray = openXC7.prjxray;
  inherit familyDb;
  partFile = "${familyDb}/${part}/part.yaml";
}
