"""Command-line interface for base62."""

import argparse
import sys

from base62 import decode, encode, decode_bytes, encode_bytes


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="base62",
        description="Encode/decode integers or bytes using base62.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- encode int ---
    enc = subparsers.add_parser("encode", help="Encode a non-negative integer.")
    enc.add_argument("number", type=int, help="Non-negative integer to encode.")

    # --- decode str ---
    dec = subparsers.add_parser("decode", help="Decode a base62 string to an integer.")
    dec.add_argument("string", help="Base62 string to decode.")

    # --- encode bytes (hex input) ---
    encb = subparsers.add_parser("encode-bytes", help="Encode hex bytes to base62.")
    encb.add_argument("hex", help="Hex string representing the bytes (e.g. deadbeef).")

    # --- decode bytes (hex output) ---
    decb = subparsers.add_parser("decode-bytes", help="Decode base62 to hex bytes.")
    decb.add_argument("string", help="Base62 string to decode.")
    decb.add_argument(
        "--length", type=int, default=None, help="Expected byte length (optional)."
    )

    args = parser.parse_args()

    match args.command:
        case "encode":
            print(encode(args.number))
        case "decode":
            print(decode(args.string))
        case "encode-bytes":
            try:
                data = bytes.fromhex(args.hex)
            except ValueError as exc:
                print(f"error: {exc}", file=sys.stderr)
                sys.exit(1)
            print(encode_bytes(data))
        case "decode-bytes":
            result = decode_bytes(args.string, length=args.length)
            print(result.hex())


if __name__ == "__main__":
    main()
