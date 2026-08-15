{
  description = "Open-source TinyStories inference for the YPCB Kintex-7 board";

  inputs.nixpkgs.url =
    "github:NixOS/nixpkgs/6fd329b2adfecb86ae49c1cba89689bd0f229e04";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      python = pkgs.python312.withPackages (ps: [
        ps.numpy
        ps.torch
        ps.transformers
        ps.tokenizers
      ]);
      modelSource = import ./nix/model-source.nix { inherit pkgs; };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          python
          pkgs.iverilog
          pkgs.verilator
          pkgs.libusb1
          pkgs.libftdi1
          pkgs.openfpgaloader
        ];
      };

      checks.${system} = {
        nix-contract = pkgs.runCommand "nix-contract" {
          nativeBuildInputs = [ python ];
        } ''
          cd ${self}
          python -m unittest -v tests/test_nix_contract.py
          touch $out
        '';

        gptneo-schema = pkgs.runCommand "gptneo-schema" {
          nativeBuildInputs = [ python ];
        } ''
          cd ${self}
          python -m unittest -v tests/test_gptneo_schema.py
          touch $out
        '';
      };

      lib.edaToolchain = compilerLab:
        import ./nix/eda-toolchain.nix { inherit compilerLab; };

      packages.${system} = {
        tinystories-1m-source = modelSource;
        default = modelSource;
      };
    };
}
