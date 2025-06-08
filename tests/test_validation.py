import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import validate_email


VALID_EMAILS = [
    "user@example.com",
    "user.name+tag@sub.domain.co.uk",
    "user_name@example.co",
]

INVALID_EMAILS = [
    "plainaddress",
    "user@domain",
    "user@domain.c",
    "user@sub_domain.com",
    "user name@example.com",
]


import pytest


@pytest.mark.parametrize("email", VALID_EMAILS)
def test_validate_email_valid(email):
    assert validate_email(email)


@pytest.mark.parametrize("email", INVALID_EMAILS)
def test_validate_email_invalid(email):
    assert not validate_email(email)
