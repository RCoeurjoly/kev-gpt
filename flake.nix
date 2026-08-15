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
      regeneratedModelPackage = pkgs.runCommand "tinystories-1m-package-regenerated" {
        nativeBuildInputs = [ python ];
      } ''
        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1
        cd ${self}
        python -m tinystories.build_package \
          --source ${modelSource} \
          --output "$out"
      '';
      rtlFixture = pkgs.runCommand "tinystories-rtl-fixture" {
        nativeBuildInputs = [ python ];
      } ''
        export PYTHONPATH=${self}
        python -m tinystories.write_rtl_fixture \
          --package ${self}/model_packages/tinystories-1m \
          --output "$out"
      '';
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

        model-package-unit = pkgs.runCommand "model-package-unit" {
          nativeBuildInputs = [ python ];
        } ''
          cd ${self}
          python -m unittest -v \
            tests/test_gptneo_import.py \
            tests/test_tinystories_quantize.py \
            tests/test_tinystories_int_reference.py
          touch $out
        '';

        model-package-reproducible = pkgs.runCommand "model-package-reproducible" {
          nativeBuildInputs = [ pkgs.diffutils ];
        } ''
          diff -r \
            ${self}/model_packages/tinystories-1m \
            ${regeneratedModelPackage}
          touch $out
        '';

        rtl-primitives = pkgs.runCommand "rtl-primitives" {
          nativeBuildInputs = [ python pkgs.iverilog ];
        } ''
          cd ${self}
          python -m unittest -v tests/test_rtl_gates.py
          touch $out
        '';

        rtl-fixture = pkgs.runCommand "rtl-fixture-check" {
          nativeBuildInputs = [ python ];
        } ''
          cd ${self}
          python -m unittest -v tests/test_rtl_fixture.py
          test -f ${rtlFixture}/fixture.json
          test -f ${rtlFixture}/model_image.mem
          touch $out
        '';
      };

      lib.edaToolchain = compilerLab:
        import ./nix/eda-toolchain.nix { inherit compilerLab; };

      packages.${system} = {
        tinystories-1m-source = modelSource;
        tinystories-1m-package-regenerated = regeneratedModelPackage;
        tinystories-rtl-fixture = rtlFixture;
        gptneo-rtl-primitives-yosys-report = pkgs.runCommand "gptneo-rtl-primitives-yosys-report" {
          nativeBuildInputs = [ python pkgs.yosys ];
        } ''
          mkdir work
          cd work
          python ${self}/tinystories/rtl_memories.py --output .
          mkdir -p "$out"
          yosys -p 'read_verilog -sv ${self}/fpga/rtl/gptneo_gelu.sv; synth_xilinx -family xc7 -top gptneo_gelu; stat' > "$out/gelu.log"
          yosys -p 'read_verilog -sv ${self}/fpga/rtl/gptneo_layernorm.sv; synth_xilinx -family xc7 -top gptneo_layernorm; stat' > "$out/layernorm.log"
          yosys -p 'read_verilog -sv ${self}/fpga/rtl/gptneo_attention.sv; synth_xilinx -family xc7 -top gptneo_attention; stat' > "$out/attention.log"
          yosys -p 'read_verilog -sv ${self}/fpga/rtl/gptneo_resident_gemv.sv; synth_xilinx -family xc7 -top gptneo_resident_gemv; stat' > "$out/gemv.log"
        '';
        default = modelSource;
      };
    };
}
