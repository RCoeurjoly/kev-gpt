{ compilerLab, system }:

let
  task3 = compilerLab;
  toolchain = task3.lib.${system}.task3Toolchain;
  openXC7 = task3.inputs.openXC7.packages.${system};
  pinnedNextpnr = openXC7.nextpnr-xilinx.overrideAttrs (_: {
    src = task3.inputs.nextpnrXilinxFork;
  });
  nextpnr = toolchain.nextpnr or pinnedNextpnr;
  pinnedChipdb = openXC7.nextpnr-xilinx-chipdb.kintex7.override {
    chipdbFootprints = [ "xc7k480tffg1156" ];
    "nextpnr-xilinx" = nextpnr;
  };
  chipdb = toolchain.chipdb or "${pinnedChipdb}/xc7k480tffg1156.bin";
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
