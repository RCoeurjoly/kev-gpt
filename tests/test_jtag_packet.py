import unittest

from host.kevin_jtag_cli import (
    PacketError,
    decode_reply,
    decode_request,
    encode_reply,
    encode_request,
    ReplyStreamDecoder,
)


class JTAGPacketCodecTest(unittest.TestCase):
    def test_request_round_trip_at_boundaries(self):
        for tokens, generated in [([1], 0), (list(range(32)), 0), ([1], 31)]:
            with self.subTest(tokens=len(tokens), generated=generated):
                packet = encode_request(tokens, generated)
                self.assertEqual(decode_request(packet), (1, tokens, generated))

    def test_request_rejects_invalid_context(self):
        for tokens, generated in [([], 1), ([1], 32), (list(range(33)), 0)]:
            with self.subTest(tokens=len(tokens), generated=generated):
                with self.assertRaises(PacketError):
                    encode_request(tokens, generated)

    def test_request_rejects_bad_fields_crc_and_truncation(self):
        good = bytearray(encode_request([7454, 2402, 257, 640], 2))
        variants = []
        for index, value in [(0, 0), (2, 2), (3, 0xff)]:
            bad = bytearray(good)
            bad[index] = value
            variants.append(bytes(bad))
        bad_crc = bytearray(good)
        bad_crc[6] ^= 1
        variants.extend([bytes(bad_crc), bytes(good[:-1]), bytes(good + b"\0")])
        for packet in variants:
            with self.subTest(packet=packet.hex()):
                with self.assertRaises(PacketError):
                    decode_request(packet)

    def test_reply_round_trip_and_rejects_malformed_length(self):
        packet = encode_reply(0, [11, 12], 0x0102030405060708)
        self.assertEqual(decode_reply(packet), (0, [11, 12], 0x0102030405060708))
        for malformed in [packet[:-1], packet + b"\0", b"", bytes([0]) * 16]:
            with self.assertRaises(PacketError):
                decode_reply(malformed)

    def test_token_and_field_ranges_are_checked(self):
        for tokens in [[-1], [65536]]:
            with self.assertRaises(PacketError):
                encode_request(tokens, 1)
        with self.assertRaises(PacketError):
            encode_reply(256, [], 0)
        with self.assertRaises(PacketError):
            encode_reply(0, [], 1 << 64)

    def test_reply_stream_decoder_skips_padding_and_handles_fragments(self):
        packet = encode_reply(0, [11, 12], 99)
        decoder = ReplyStreamDecoder()
        self.assertIsNone(decoder.feed(b"\0\xffR"))
        self.assertIsNone(decoder.feed(packet[1:8]))
        self.assertEqual(decoder.feed(packet[8:] + b"garbage"), (0, [11, 12], 99))


if __name__ == "__main__":
    unittest.main()
