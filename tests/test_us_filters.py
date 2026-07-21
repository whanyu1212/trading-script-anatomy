"""Tests for US-market eligibility rules."""

import pytest

from trading_script_anatomy.data.models import SecurityInfo
from trading_script_anatomy.strategy.us_filters import (
    is_derivative_ticker,
    us_eligibility,
)


def info(symbol: str = "ACME", **overrides: object) -> SecurityInfo:
    """Build listing metadata for an ordinary NASDAQ common stock."""
    fields: dict = {
        "name": "Acme Micro Corp",
        "exchange": "NASDAQ",
        "is_etf": False,
        "is_adr": False,
        "is_actively_trading": True,
    }
    fields.update(overrides)
    return SecurityInfo(symbol=symbol, **fields)


def test_ordinary_common_stock_is_eligible() -> None:
    """Accept an actively traded common stock on an allowed exchange."""
    assert us_eligibility("ACME", info()) is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"is_etf": True},
        {"is_adr": True},
        {"is_actively_trading": False},
        {"exchange": "OTC"},
    ],
)
def test_disqualifying_metadata_excludes(overrides: dict) -> None:
    """Exclude ETFs, ADRs, inactive listings, and off-exchange venues."""
    assert us_eligibility("ACME", info(**overrides)) is False


def test_missing_metadata_fails_open() -> None:
    """Pass securities whose profile fields FMP does not populate."""
    unknown = info(exchange=None, is_etf=None, is_adr=None, is_actively_trading=None)

    assert us_eligibility("ACME", unknown) is True


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [
        ("ABC-WT", True),
        ("ABC-U", True),
        ("ABC-RT", True),
        ("BRK-B", False),
        ("ABCDW", True),
        ("ABCDU", True),
        ("F", False),
        ("ACME", False),
    ],
)
def test_derivative_ticker_heuristic(symbol: str, expected: bool) -> None:
    """Flag warrant, unit, and right patterns while keeping share classes."""
    assert is_derivative_ticker(symbol) is expected
    assert us_eligibility(symbol, info(symbol)) is (not expected)
