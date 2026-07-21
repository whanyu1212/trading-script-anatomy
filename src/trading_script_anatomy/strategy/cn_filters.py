"""Mainland-China (A-share) eligibility rules for the stock selector."""

from trading_script_anatomy.data.models import SecurityInfo


def is_st_stock(name: str) -> bool:
    """Return whether a security name indicates special treatment status.

    Args:
        name: Security display name.

    Returns:
        True if the name contains an ST marker.
    """
    return "ST" in name.upper()


def is_delisting_stock(name: str) -> bool:
    """Return whether a security name indicates a pending delisting.

    Args:
        name: Security display name.

    Returns:
        True if the name contains the Chinese delisting marker.
    """
    return "退" in name


def is_excluded_board(symbol: str) -> bool:
    """Return whether a ticker belongs to an excluded mainland-China board.

    Args:
        symbol: Ticker whose first digits identify its exchange board.

    Returns:
        True for ChiNext, STAR Market, and Beijing Exchange prefixes.
    """
    return len(symbol) >= 2 and symbol[:2] in {"30", "68", "8", "4"}


def a_share_eligibility(symbol: str, info: SecurityInfo) -> bool:
    """Return whether a mainland-China security passes name and board rules.

    Args:
        symbol: Ticker whose prefix identifies its exchange board.
        info: Security metadata supplied by the data provider.

    Returns:
        True when the security is neither ST, delisting, nor on an excluded
        board.
    """
    return not (
        is_st_stock(info.name)
        or is_delisting_stock(info.name)
        or is_excluded_board(symbol)
    )
