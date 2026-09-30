"""Base62 encoder/decoder.

Alphabet: 0-9, A-Z, a-z  (62 characters, index 0–61)
"""

__all__ = ["encode", "decode", "encode_bytes", "decode_bytes"]

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
BASE = len(ALPHABET)  # 62
_CHAR_TO_INDEX: dict[str, int] = {ch: i for i, ch in enumerate(ALPHABET)}


def encode(n: int) -> str:
    """Encode a non-negative integer to a base62 string.

    Args:
        n: A non-negative integer.

    Returns:
        The base62 representation as a string.

    Raises:
        ValueError: If *n* is negative.
        TypeError: If *n* is not an integer.
    """
    if not isinstance(n, int):
        raise TypeError(f"expected int, got {type(n).__name__!r}")
    if n < 0:
        raise ValueError(f"base62 encoding requires a non-negative integer, got {n}")
    if n == 0:
        return ALPHABET[0]

    digits: list[str] = []
    while n:
        n, remainder = divmod(n, BASE)
        digits.append(ALPHABET[remainder])
    return "".join(reversed(digits))


def decode(s: str) -> int:
    """Decode a base62 string to a non-negative integer.

    Args:
        s: A non-empty base62 string.

    Returns:
        The decoded non-negative integer.

    Raises:
        ValueError: If *s* is empty or contains characters outside the alphabet.
        TypeError: If *s* is not a string.
    """
    if not isinstance(s, str):
        raise TypeError(f"expected str, got {type(s).__name__!r}")
    if not s:
        raise ValueError("base62 string must not be empty")

    result = 0
    for char in s:
        if char not in _CHAR_TO_INDEX:
            raise ValueError(
                f"invalid base62 character {char!r}; "
                f"alphabet is {ALPHABET!r}"
            )
        result = result * BASE + _CHAR_TO_INDEX[char]
    return result


def encode_bytes(data: bytes) -> str:
    """Encode arbitrary bytes to a base62 string via big-endian integer conversion.

    Args:
        data: The bytes to encode.

    Returns:
        The base62 representation. A leading ``'0'`` is prepended to preserve
        any leading zero bytes so that the original length can be recovered.

    Raises:
        TypeError: If *data* is not :class:`bytes`.
    """
    if not isinstance(data, bytes):
        raise TypeError(f"expected bytes, got {type(data).__name__!r}")
    if not data:
        return ""

    # Count leading zero bytes so we can restore them on decode.
    leading_zeros = len(data) - len(data.lstrip(b"\x00"))
    n = int.from_bytes(data, byteorder="big")
    encoded = encode(n) if n else ""
    return ALPHABET[0] * leading_zeros + encoded


def decode_bytes(s: str, length: int | None = None) -> bytes:
    """Decode a base62 string produced by :func:`encode_bytes` back to bytes.

    Args:
        s: The base62 string to decode.
        length: Expected number of bytes.  When supplied the result is
            left-padded with zero bytes to reach *length*.  Useful when the
            caller already knows the original payload size (e.g. fixed-width
            UUIDs or hashes).

    Returns:
        The decoded :class:`bytes`.

    Raises:
        TypeError: If *s* is not a string.
        ValueError: If *s* contains invalid characters.
    """
    if not isinstance(s, str):
        raise TypeError(f"expected str, got {type(s).__name__!r}")
    if not s:
        return b""

    # Extract leading '0' chars — each represents a leading zero byte.
    leading_zeros = len(s) - len(s.lstrip(ALPHABET[0]))
    tail = s.lstrip(ALPHABET[0])

    n = decode(tail) if tail else 0
    # Determine byte length automatically when not provided.
    byte_length = length if length is not None else (n.bit_length() + 7) // 8
    raw = n.to_bytes(byte_length, byteorder="big") if byte_length else b""
    return b"\x00" * leading_zeros + raw
