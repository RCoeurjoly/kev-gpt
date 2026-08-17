{
  description = "Open-source TinyStories inference for the YPCB Kintex-7 board";

  inputs.nixpkgs.url =
    "github:NixOS/nixpkgs/6fd329b2adfecb86ae49c1cba89689bd0f229e04";
  inputs.compilerLab.url = "github:RCoeurjoly/compiler-lab-llm2fpga";

  outputs = { self, nixpkgs, compilerLab }:
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
      hardwareQuality = pkgs.runCommand "tinystories-hardware-quality.json" {
        nativeBuildInputs = [ python ];
      } ''
        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1
        export PYTHONPATH=${self}
        python -m tinystories.measure_hardware_quality \
          --package ${self}/model_packages/tinystories-1m \
          --source ${modelSource} \
          --output "$out"
      '';
      jtagTransport = pkgs.stdenv.mkDerivation {
        pname = "kevin-jtag-transport";
        version = "1";
        src = self;
        buildInputs = [ pkgs.libftdi1 ];
        dontConfigure = true;
        buildPhase = ''
          cc -O2 -Wall -Wextra -Werror -fPIC -shared \
            host/jtag_transport.c -o libkevin_jtag.so -lftdi1
        '';
        installPhase = ''
          mkdir -p "$out/lib" "$out/include"
          cp libkevin_jtag.so "$out/lib/"
          cp host/jtag_transport.h "$out/include/"
        '';
      };
      kevinJtag = pkgs.writeShellApplication {
        name = "kevin-jtag";
        runtimeInputs = [ python pkgs.openfpgaloader ];
        text = ''
          export PYTHONPATH=${self}
          export KEVIN_JTAG_LIBRARY=${jtagTransport}/lib/libkevin_jtag.so
          export KEVIN_MODEL_PACKAGE=${self}/model_packages/tinystories-1m
          exec python -m host.kevin_jtag_cli "$@"
        '';
      };
      kintexInteractive = import ./nix/kintex-tinystories.nix {
        inherit pkgs compilerLab;
        source = self;
        interactive = true;
      };
      kintexSelftest = import ./nix/kintex-tinystories.nix {
        inherit pkgs compilerLab;
        source = self;
        interactive = false;
      };
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
          nativeBuildInputs = [ python pkgs.iverilog pkgs.verilator pkgs.stdenv.cc ];
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

        hardware-quality = pkgs.runCommand "hardware-quality-check" {} ''
          test -s ${hardwareQuality}
          grep -q '"passed": true' ${hardwareQuality}
          touch $out
        '';

        jtag-transport = pkgs.runCommand "jtag-transport-check" {
          nativeBuildInputs = [ python pkgs.iverilog pkgs.stdenv.cc pkgs.libftdi1 ];
        } ''
          cd ${self}
          python -m unittest -v \
            tests/test_jtag_packet.py \
            tests/test_host_cli.py \
            tests/test_jtag_rtl.py
          ${kevinJtag}/bin/kevin-jtag packet-selftest
          touch $out
        '';
      };

      apps.${system}.kevin-jtag = {
        type = "app";
        program = "${kevinJtag}/bin/kevin-jtag";
      };

      lib.edaToolchain = compilerLab:
        import ./nix/eda-toolchain.nix { inherit compilerLab system; };

      packages.${system} = {
        kintex-tinystories-interactive = kintexInteractive;
        kintex-tinystories-selftest = kintexSelftest;
        kintex-tinystories-interactive-synthesis =
          kintexInteractive.passthru.synthesis;
        tinystories-1m-source = modelSource;
        tinystories-1m-package-regenerated = regeneratedModelPackage;
        tinystories-rtl-fixture = rtlFixture;
        tinystories-hardware-quality = hardwareQuality;
        kevin-jtag-transport = jtagTransport;
        kevin-jtag = kevinJtag;
        gptneo-rtl-primitives-yosys-report = pkgs.runCommand "gptneo-rtl-primitives-yosys-report" {
          nativeBuildInputs = [ python pkgs.yosys ];
        } ''
          mkdir work
          cd work
          python ${self}/tinystories/rtl_memories.py --output .
          mkdir -p "$out"
          yosys -p 'read_verilog -sv ${self}/fpga/rtl/gptneo_gelu.sv; synth_xilinx -family xc7 -top gptneo_gelu; stat' > "$out/gelu.log"
          yosys -p 'read_verilog -sv ${self}/fpga/rtl/gptneo_layernorm.sv ${self}/fpga/rtl/gptneo_iterative_divider.sv; hierarchy -top gptneo_layernorm; proc; opt; select -assert-none t:$div; synth_xilinx -family xc7 -top gptneo_layernorm; stat' > "$out/layernorm.log"
          layernorm_dsp_count="$(awk '$2 == "DSP48E1" { count = $1 } END { print count + 0 }' "$out/layernorm.log")"
          if [ "$layernorm_dsp_count" -gt 8 ]; then
            echo "LayerNorm synthesized $layernorm_dsp_count DSP48E1 cells; expected at most 8" >&2
            exit 1
          fi
          yosys -p 'read_verilog -sv ${self}/fpga/rtl/gptneo_attention.sv ${self}/fpga/rtl/gptneo_iterative_divider.sv; synth_xilinx -family xc7 -top gptneo_attention; stat' > "$out/attention.log"
          yosys -p 'read_verilog -sv ${self}/fpga/rtl/gptneo_resident_gemv.sv ${self}/fpga/rtl/gptneo_iterative_divider.sv; hierarchy -top gptneo_resident_gemv; proc; opt; select -assert-none t:$div; synth_xilinx -family xc7 -top gptneo_resident_gemv; stat' > "$out/gemv.log"
        '';
        gptneo-sequencer-yosys-report = pkgs.runCommand "gptneo-sequencer-yosys-report" {
          nativeBuildInputs = [ python pkgs.yosys ];
        } ''
          mkdir work
          cp ${rtlFixture}/model_image.mem ${rtlFixture}/gptneo_package.svh work/
          python ${self}/tinystories/rtl_memories.py --output work
          cd work
          mkdir -p "$out"
          yosys -p 'read_verilog -sv -I. ${self}/fpga/rtl/gptneo_sequencer.sv ${self}/fpga/rtl/gptneo_layernorm.sv ${self}/fpga/rtl/gptneo_gelu.sv ${self}/fpga/rtl/gptneo_attention.sv ${self}/fpga/rtl/gptneo_iterative_divider.sv ${self}/fpga/rtl/gptneo_resident_gemv.sv; synth_xilinx -family xc7 -top gptneo_sequencer; stat' > "$out/sequencer.log"
        '';
        default = modelSource;
      };
    };
}
