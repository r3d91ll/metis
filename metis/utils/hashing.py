"""
Hashing and Key Utilities

Provides utilities for generating content hashes and sanitizing keys for database storage.
"""

import hashlib


def sha1_hex(data: bytes) -> str:
    """
    Compute SHA-1 hash of data and return as hexadecimal string.

    Args:
        data: Bytes to hash

    Returns:
        Hexadecimal SHA-1 hash

    Example:
        >>> sha1_hex(b"hello world")
        '2aae6c35c94fcfb415dbe95f408b9ce91ee846ed'
    """
    h = hashlib.sha1()
    h.update(data)
    return h.hexdigest()


def sanitize_key(relpath: str, max_len: int = 254) -> str:
    """
    Sanitize a file path into a valid ArangoDB document key.

    Converts path separators and special characters to underscores,
    ensures the key is not too long, and handles edge cases.

    Args:
        relpath: Relative file path to sanitize
        max_len: Maximum byte length for the key (default: 254)

    Returns:
        Sanitized key suitable for use as ArangoDB _key

    Example:
        >>> sanitize_key("src/utils/helper.py")
        'src_utils_helper.py'
        >>> sanitize_key("file with spaces.txt")
        'file_with_spaces.txt'
    """
    # Replace path separators and spaces with underscores
    s = relpath.replace("/", "_").replace(" ", "_")

    # Keep only alphanumeric, dash, underscore, period, colon, at, parens, plus
    s = "".join(ch if ch.isalnum() or ch in "-_.:@()+" else "_" for ch in s)

    # Strip leading/trailing special chars
    s = s.strip("._-") or "item"

    # Check byte length
    if len(s.encode("utf-8")) <= max_len:
        return s

    # If too long, truncate and add hash suffix
    h = sha1_hex(relpath.encode("utf-8"))[:10]
    base = s[: max_len - 1 - len(h)]
    return f"{base}_{h}"
