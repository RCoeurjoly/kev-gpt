"""Packet codec and command-line client for the YPCB JTAG endpoint.

This module deliberately contains transport and tokenizer logic only.  Model
inference is performed by the FPGA and is never emulated by the host client.
"""

from __future__ import annotations

import struct
import zlib
import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


REQUEST_MAGIC = 0x4B47
REPLY_MAGIC = 0x4B52
PROTOCOL_VERSION = 1
COMMAND_INFER = 1
MAX_CONTEXT = 32
_REQUEST_HEADER = struct.Struct("<HBBBB")
_REPLY_HEADER = struct.Struct("<HBBBQ")
_CRC = struct.Struct("<I")


class PacketError(ValueError):
    """A packet violates the version-one wire contract."""


def _byte(value: int, name: str) -> int:
    if not isinstance(value, int) or not 0 <= value <= 0xFF:
        raise PacketError(f"{name} must fit in one byte")
    return value


def _token_list(tokens) -> list[int]:
    result = list(tokens)
    if any(not isinstance(token, int) or not 0 <= token <= 0xFFFF for token in result):
        raise PacketError("token IDs must fit in 16 bits")
    return result


def _with_crc(payload: bytes) -> bytes:
    return payload + _CRC.pack(zlib.crc32(payload) & 0xFFFFFFFF)


def _checked_payload(packet: bytes) -> bytes:
    if len(packet) < _CRC.size:
        raise PacketError("truncated packet")
    payload, encoded_crc = packet[:-4], packet[-4:]
    if _CRC.unpack(encoded_crc)[0] != (zlib.crc32(payload) & 0xFFFFFFFF):
        raise PacketError("CRC mismatch")
    return payload


def encode_request(tokens, generated: int, command: int = COMMAND_INFER) -> bytes:
    token_ids = _token_list(tokens)
    generated = _byte(generated, "generation count")
    command = _byte(command, "command")
    if not token_ids:
        raise PacketError("prompt must contain at least one token")
    if len(token_ids) > MAX_CONTEXT or len(token_ids) + generated > MAX_CONTEXT:
        raise PacketError("prompt and completion exceed the 32-token context")
    header = _REQUEST_HEADER.pack(
        REQUEST_MAGIC, PROTOCOL_VERSION, command, len(token_ids), generated
    )
    body = struct.pack(f"<{len(token_ids)}H", *token_ids)
    return _with_crc(header + body)


def decode_request(packet: bytes) -> tuple[int, list[int], int]:
    payload = _checked_payload(bytes(packet))
    if len(payload) < _REQUEST_HEADER.size:
        raise PacketError("truncated request header")
    magic, version, command, prompt_count, generated = _REQUEST_HEADER.unpack_from(payload)
    expected = _REQUEST_HEADER.size + 2 * prompt_count
    if len(payload) != expected:
        raise PacketError("request length does not match prompt count")
    if magic != REQUEST_MAGIC or version != PROTOCOL_VERSION:
        raise PacketError("request magic or version mismatch")
    if command != COMMAND_INFER:
        raise PacketError("unknown command")
    if prompt_count == 0 or prompt_count + generated > MAX_CONTEXT:
        raise PacketError("invalid context length")
    tokens = list(struct.unpack_from(f"<{prompt_count}H", payload, _REQUEST_HEADER.size))
    return command, tokens, generated


def encode_reply(status: int, tokens, cycles: int) -> bytes:
    status = _byte(status, "status")
    token_ids = _token_list(tokens)
    if len(token_ids) > MAX_CONTEXT:
        raise PacketError("too many reply tokens")
    if not isinstance(cycles, int) or not 0 <= cycles <= 0xFFFFFFFFFFFFFFFF:
        raise PacketError("cycle count must fit in 64 bits")
    header = _REPLY_HEADER.pack(
        REPLY_MAGIC, PROTOCOL_VERSION, status, len(token_ids), cycles
    )
    body = struct.pack(f"<{len(token_ids)}H", *token_ids)
    return _with_crc(header + body)


def decode_reply(packet: bytes) -> tuple[int, list[int], int]:
    payload = _checked_payload(bytes(packet))
    if len(payload) < _REPLY_HEADER.size:
        raise PacketError("truncated reply header")
    magic, version, status, count, cycles = _REPLY_HEADER.unpack_from(payload)
    if magic != REPLY_MAGIC or version != PROTOCOL_VERSION:
        raise PacketError("reply magic or version mismatch")
    if count > MAX_CONTEXT or len(payload) != _REPLY_HEADER.size + 2 * count:
        raise PacketError("reply length does not match token count")
    tokens = list(struct.unpack_from(f"<{count}H", payload, _REPLY_HEADER.size))
    return status, tokens, cycles


class ReplyStreamDecoder:
    """Recover one framed reply from padded or fragmented JTAG scan bytes."""

    def __init__(self):
        self.buffer = bytearray()

    def feed(self, data: bytes):
        self.buffer.extend(data)
        magic = struct.pack("<H", REPLY_MAGIC)
        while True:
            offset = self.buffer.find(magic)
            if offset < 0:
                if self.buffer[-1:] == magic[:1]:
                    self.buffer[:] = self.buffer[-1:]
                else:
                    self.buffer.clear()
                return None
            if offset:
                del self.buffer[:offset]
            if len(self.buffer) < 5:
                return None
            total = _REPLY_HEADER.size + 2 * self.buffer[4] + _CRC.size
            if len(self.buffer) < total:
                return None
            candidate = bytes(self.buffer[:total])
            try:
                reply = decode_reply(candidate)
            except PacketError:
                del self.buffer[0]
                continue
            del self.buffer[:total]
            return reply


class NativeTransport:
    def __init__(self, library: str | Path | None = None, serial: str | None = None):
        library = library or os.environ.get("KEVIN_JTAG_LIBRARY")
        if not library:
            raise RuntimeError("KEVIN_JTAG_LIBRARY is not set")
        self.library = ctypes.CDLL(str(library))
        self.library.kj_open.argtypes = [ctypes.c_char_p]
        self.library.kj_open.restype = ctypes.c_int
        self.library.kj_user1_exchange.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_uint,
        ]
        self.library.kj_user1_exchange.restype = ctypes.c_int
        self.library.kj_last_error.restype = ctypes.c_char_p
        encoded_serial = serial.encode() if serial else None
        if self.library.kj_open(encoded_serial) < 0:
            message = self.library.kj_last_error().decode(errors="replace")
            raise RuntimeError(f"cannot open JTAG transport: {message}")
        self.opened = True

    def exchange(self, data: bytes, receive_length: int, timeout_ms: int) -> bytes:
        outgoing = bytes(data)
        tx = (ctypes.c_ubyte * len(outgoing)).from_buffer_copy(outgoing)
        rx = (ctypes.c_ubyte * receive_length)()
        result = self.library.kj_user1_exchange(
            tx, len(outgoing), rx, receive_length, timeout_ms
        )
        if result < 0:
            message = self.library.kj_last_error().decode(errors="replace")
            raise RuntimeError(f"JTAG exchange failed: {message}")
        return bytes(rx[:result])

    def close(self):
        if getattr(self, "opened", False):
            self.library.kj_close()
            self.opened = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def infer_tokens(transport, tokens, generated: int, timeout_s: float = 30.0):
    request = encode_request(tokens, generated)
    decoder = ReplyStreamDecoder()
    reply = decoder.feed(transport.exchange(request, len(request), 1000))
    deadline = time.monotonic() + timeout_s
    while reply is None:
        if time.monotonic() >= deadline:
            raise TimeoutError("FPGA inference reply timed out")
        reply = decoder.feed(transport.exchange(bytes(64), 64, 1000))
    return reply


def read_debug_snapshot(transport, timeout_s: float = 3.0):
    data = bytearray(transport.exchange(b"?", 1, 1000))
    deadline = time.monotonic() + timeout_s
    while True:
        marker = data.find(b"DBG1")
        if marker >= 0 and len(data) >= marker + 44:
            payload = data[marker + 4:marker + 44]
            return {
                "cycles": int.from_bytes(payload[0:4], "little"),
                "sequencer_state": payload[4] & 0x7f,
                "layer": payload[5] & 0x7,
                "position": payload[6] & 0x3f,
                "index": payload[7] | ((payload[8] & 1) << 8),
                "gemv_operation": payload[9] & 0x7,
                "gemv_output_index": payload[10] | (payload[11] << 8),
                "gemv_busy": bool(payload[12] & 1),
                "layernorm_busy": bool(payload[12] & 2),
                "attention_busy": bool(payload[12] & 4),
                "token_count": payload[13] & 0x3f,
                "generated_count": payload[14] & 0x3f,
                "gelu_state": payload[15] & 0x3,
                "gelu_ready": bool(payload[15] & 0x4),
                "gelu_valid": bool(payload[15] & 0x8),
                "sequencer_error": bool(payload[15] & 0x10),
                "embedding_x_q16": int.from_bytes(payload[16:20], "little", signed=True),
                "embedding_token_component_q16": int.from_bytes(
                    payload[20:24], "little", signed=True
                ),
                "embedding_position_scale_q24": int.from_bytes(payload[24:27], "little"),
                "embedding_position_code": int.from_bytes(payload[27:28], "little", signed=True),
                "layernorm_y0_q16": int.from_bytes(payload[28:32], "little", signed=True),
                "layernorm_normalized0_q16": int.from_bytes(
                    payload[32:36], "little", signed=True
                ),
                "layernorm_affine0_q16": int.from_bytes(
                    payload[36:40], "little", signed=True
                ),
            }
        if time.monotonic() >= deadline:
            raise TimeoutError("FPGA debug snapshot timed out")
        data.extend(transport.exchange(bytes(64), 64, 1000))
        if len(data) > 256:
            del data[:-256]


def load_tokenizer(package: str | Path):
    """Load only the pinned tokenizer assets from a canonical model package."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(Path(package)), local_files_only=True, trust_remote_code=False
    )


def _packet_selftest() -> None:
    cases = [([1], 0), ([7454, 2402, 257, 640], 8), (list(range(16)), 16)]
    for tokens, generated in cases:
        command, decoded, decoded_generated = decode_request(
            encode_request(tokens, generated)
        )
        if (command, decoded, decoded_generated) != (
            COMMAND_INFER, tokens, generated
        ):
            raise PacketError("request self-test mismatch")
    reply = (0, [11, 12, 13], 123456789)
    if decode_reply(encode_reply(*reply)) != reply:
        raise PacketError("reply self-test mismatch")
    print("JTAG_PACKET_SELFTEST_PASS cases=3")


def program_bitstream(path: str | Path, cable: str = "digilent_hs3") -> None:
    bitstream = Path(path).resolve(strict=True)
    subprocess.run(
        ["openFPGALoader", "--cable", cable, str(bitstream)], check=True
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_inference_receipt(
    args, *, prompt_ids, output_ids, cycles, wall_seconds
) -> dict[str, object]:
    package = Path(args.package).resolve(strict=True)
    bitstream = Path(args.program).resolve(strict=True) if args.program else None
    return {
        "schema": "kev-gpt-kintex-inference-v2",
        "prompt": args.prompt,
        "prompt_ids": list(prompt_ids),
        "requested_tokens": args.max_new_tokens,
        "output_ids": list(output_ids),
        "cycles": int(cycles),
        "wall_seconds": float(wall_seconds),
        "timing_boundary": "request-submit-to-reply",
        "transport": "jtag-debug-baseline",
        "package_manifest_sha256": sha256_file(package / "manifest.json"),
        "bitstream_sha256": sha256_file(bitstream) if bitstream else None,
        "board": "YPCB-00338-1P1",
        "fpga": "xc7k480tffg1156-1",
    }


def _infer_command(args) -> None:
    package = Path(args.package).resolve(strict=True)
    if args.program:
        program_bitstream(args.program, args.cable)
    tokenizer = load_tokenizer(package)
    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    if not prompt_ids:
        raise PacketError("prompt tokenized to an empty sequence")
    if len(prompt_ids) + args.max_new_tokens > MAX_CONTEXT:
        raise PacketError("prompt and completion exceed the 32-token context")
    with NativeTransport(serial=args.serial) as transport:
        started = time.monotonic()
        status, output_ids, cycles = infer_tokens(
            transport, prompt_ids, args.max_new_tokens, args.timeout
        )
    elapsed = time.monotonic() - started
    if status:
        raise RuntimeError(f"FPGA returned status {status}")
    completion = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(completion)
    if args.json_receipt:
        receipt = build_inference_receipt(
            args,
            prompt_ids=prompt_ids,
            output_ids=output_ids,
            cycles=cycles,
            wall_seconds=elapsed,
        )
        receipt_path = Path(args.json_receipt)
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def _debug_command(args) -> None:
    with NativeTransport(serial=args.serial) as transport:
        snapshot = read_debug_snapshot(transport, args.timeout)
    print(json.dumps(snapshot, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("packet-selftest", help="exercise the packet codec")
    program = subcommands.add_parser("program", help="program FPGA SRAM")
    program.add_argument("bitstream")
    program.add_argument("--cable", default="digilent_hs3")
    infer = subcommands.add_parser("infer", help="run greedy inference on the FPGA")
    infer.add_argument("--prompt", required=True)
    infer.add_argument("--max-new-tokens", type=int, default=8)
    infer.add_argument(
        "--package", default=os.environ.get("KEVIN_MODEL_PACKAGE", "model_packages/tinystories-1m")
    )
    infer.add_argument("--serial")
    infer.add_argument("--timeout", type=float, default=300.0)
    infer.add_argument("--program", metavar="BITSTREAM")
    infer.add_argument("--cable", default="digilent_hs3")
    infer.add_argument("--json-receipt")
    debug = subcommands.add_parser("debug", help="read an in-flight FPGA state snapshot")
    debug.add_argument("--serial")
    debug.add_argument("--timeout", type=float, default=3.0)
    args = parser.parse_args(argv)
    if args.command == "packet-selftest":
        _packet_selftest()
    elif args.command == "program":
        program_bitstream(args.bitstream, args.cable)
    elif args.command == "infer":
        _infer_command(args)
    elif args.command == "debug":
        _debug_command(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
