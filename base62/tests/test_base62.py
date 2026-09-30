"""Tests for base62 encoder/decoder."""

import pytest
from base62 import decode, decode_bytes, encode, encode_bytes


# ── Integer round-trips ──────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [0, 1, 61, 62, 123456789, 2**64, 2**128 - 1])
def test_encode_decode_roundtrip(n: int) -> None:
    assert decode(encode(n)) == n


def test_encode_zero() -> None:
    assert encode(0) == "0"


def test_encode_61() -> None:
    assert encode(61) == "z"


def test_encode_62() -> None:
    assert encode(62) == "10"


def test_decode_known() -> None:
    assert decode("10") == 62
    assert decode("z") == 61


def test_encode_negative_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        encode(-1)


def test_encode_wrong_type_raises() -> None:
    with pytest.raises(TypeError):
        encode("hello")  # type: ignore[arg-type]


def test_decode_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        decode("")


def test_decode_invalid_char_raises() -> None:
    with pytest.raises(ValueError, match="invalid base62 character"):
        decode("hello!")


# ── Bytes round-trips ────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"\x00",
        b"\x00\x00",
        b"hello",
        b"\xff\xfe\xfd",
        bytes(range(256)),
    ],
)
def test_encode_decode_bytes_roundtrip(data: bytes) -> None:
    encoded = encode_bytes(data)
    # Provide the original length so leading zero bytes are restored.
    assert decode_bytes(encoded, length=len(data)) == data


def test_encode_bytes_wrong_type_raises() -> None:
    with pytest.raises(TypeError):
        encode_bytes("not bytes")  # type: ignore[arg-type]


def test_decode_bytes_empty() -> None:
    assert decode_bytes("") == b""


def test_decode_bytes_wrong_type_raises() -> None:
    with pytest.raises(TypeError):
        decode_bytes(42)  # type: ignore[arg-type]
