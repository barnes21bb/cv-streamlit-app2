import re

from pathlib import Path

# File upload limits in bytes
WARNING_SIZE_BYTES = 200 * 1024 * 1024  # 200 MB
MAX_SIZE_BYTES = 500 * 1024 * 1024      # 500 MB


def validate_email(email: str) -> bool:
    """Basic email validation."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def check_file_size(size_bytes: int) -> str:
    """Return ``'reject'`` if the file is too large,
    ``'warn'`` if it is large but acceptable, otherwise ``'ok'``."""
    if size_bytes > MAX_SIZE_BYTES:
        return "reject"
    if size_bytes > WARNING_SIZE_BYTES:
        return "warn"
    return "ok"
