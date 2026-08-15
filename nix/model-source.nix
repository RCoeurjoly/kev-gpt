{ pkgs }:

let
  owner = "roneneldan";
  repository = "TinyStories-1M";
  revision = "ac533fb8b4f69c71894bf96badfe11e6294d9fcf";
  baseUrl = "https://huggingface.co/${owner}/${repository}/resolve/${revision}";
  fileSpecs = [
    {
      name = "config.json";
      size = 1021;
      hash = "sha256-/3TDDV67WrHaDy6kea33GXxQS0K1UiqFjDNKuR7UlYw=";
    }
    {
      name = "merges.txt";
      size = 456318;
      hash = "sha256-HOFmR3PFDz4MyIQmGak+3EYkUltyixiKngvjO3cmrcU=";
    }
    {
      name = "pytorch_model.bin";
      size = 48578813;
      hash = "sha256-B/lgnqiCuBY/87I9QOK4LLcV1AljG+sVyEsWTzh32uc=";
    }
    {
      name = "special_tokens_map.json";
      size = 438;
      hash = "sha256-mEEhN65Dx3+K9S61GxnDU20yQstVM5Fn2EEAX6lKI7c=";
    }
    {
      name = "tokenizer.json";
      size = 2107652;
      hash = "sha256-9u09MHAQwkTCKu/73gX0Gc8nfCPmTPmLZzysVEnP7/U=";
    }
    {
      name = "tokenizer_config.json";
      size = 722;
      hash = "sha256-PXbaD9N0k/v80/D6l1d1PTH5Lhd569kTCAm0VUamAmE=";
    }
    {
      name = "vocab.json";
      size = 798156;
      hash = "sha256-O6PDEJ/zOXbEvZZlicEe4U/KofTJ5eFUwu1/mdgHCec=";
    }
  ];
  fetchedFiles = map
    (spec: spec // {
      url = "${baseUrl}/${spec.name}";
      source = pkgs.fetchurl {
        inherit (spec) hash;
        url = "${baseUrl}/${spec.name}";
      };
    })
    fileSpecs;
  manifest = pkgs.writeText "source-manifest.json" (builtins.toJSON {
    schema_version = 1;
    inherit owner repository revision;
    files = map
      (file: {
        inherit (file) name size hash url;
      })
      fetchedFiles;
  });
in
pkgs.runCommand "tinystories-1m-source-${builtins.substring 0 12 revision}" { } (
  ''
    mkdir -p "$out"
    cp ${manifest} "$out/source-manifest.json"
  ''
  + builtins.concatStringsSep "\n" (map
    (file: ''cp ${file.source} "$out/${file.name}"'')
    fetchedFiles)
)
