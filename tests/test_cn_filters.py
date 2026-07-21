"""Tests for mainland-China security eligibility rules."""

import pytest

from trading_script_anatomy.strategy.cn_filters import is_excluded_board


@pytest.mark.parametrize(
    "symbol",
    [
        "300001.SZ",
        "688001.SS",
        "830001.BJ",
        "430001.BJ",
        "920008.BJ",
    ],
)
def test_excluded_board_prefixes(symbol: str) -> None:
    """Exclude ChiNext, STAR Market, and Beijing Exchange symbols."""
    assert is_excluded_board(symbol) is True


@pytest.mark.parametrize("symbol", ["000001.SZ", "600000.SS"])
def test_main_board_prefixes_remain_eligible(symbol: str) -> None:
    """Keep ordinary Shenzhen and Shanghai main-board symbols."""
    assert is_excluded_board(symbol) is False


@pytest.mark.parametrize("symbol", ["", "4", "8"])
def test_short_invalid_symbols_do_not_match_board_prefixes(symbol: str) -> None:
    """Require enough characters to identify an exchange board."""
    assert is_excluded_board(symbol) is False
