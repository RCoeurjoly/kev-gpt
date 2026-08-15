"""Generate deterministic nonlinear-function memories for GPT-Neo RTL."""

import argparse
import math
import pathlib


def write_gelu_lut(directory: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(directory) / "gptneo_gelu.mem"
    with path.open("w") as stream:
        for index in range(8192):
            value = -8.0 + index / 512.0
            result = 0.5 * value * (
                1.0 + math.tanh(math.sqrt(2.0 / math.pi) *
                                (value + 0.044715 * value ** 3))
            )
            code = max(-32768, min(32767, round(result * 4096.0)))
            stream.write(f"{code & 0xffff:04x}\n")
    return path


def write_exp_lut(directory: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(directory) / "gptneo_exp.mem"
    with path.open("w") as stream:
        for index in range(4096):
            value = math.exp((index - 4096) / 256.0)
            stream.write(f"{round(value * (1 << 20)):06x}\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    write_gelu_lut(args.output)
    write_exp_lut(args.output)


if __name__ == "__main__":
    main()
