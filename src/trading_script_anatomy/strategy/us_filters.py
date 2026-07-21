"""US-market eligibility rules for the stock selector."""

from trading_script_anatomy.data.models import SecurityInfo

ALLOWED_EXCHANGES = frozenset({"NYSE", "NASDAQ", "AMEX"})

DERIVATIVE_SUFFIXES = frozenset({"W", "WS", "WT", "U", "UN", "R", "RT"})


def us_eligibility(symbol: str, info: SecurityInfo) -> bool:
    """Return whether a US security is eligible for selection.

    Replaces the A-share ST/delisting/board rules. The FMP screener applies
    server-side filters first (exchange, market-cap band, actively trading,
    not a fund); this hook re-checks profile-level facts as defense in depth
    and adds rules the screener cannot express.

    Policy on missing metadata: unknown values pass (fail open). The screener
    has already vetted the universe once, so a missing profile field more
    likely reflects an FMP data gap than a disqualifying security; only an
    explicit disqualifying value excludes. SPACs need no name rule here — the
    fundamental filter's revenue and profit floors already exclude shell
    companies structurally.

    Args:
        symbol: FMP ticker symbol.
        info: Security metadata from the FMP company profile.

    Returns:
        True when the security may enter the candidate pool.
    """
    if info.is_etf is True or info.is_adr is True:
        return False
    if info.is_actively_trading is False:
        return False
    if info.exchange is not None and info.exchange.upper() not in ALLOWED_EXCHANGES:
        return False
    return not is_derivative_ticker(symbol)


def is_derivative_ticker(symbol: str) -> bool:
    """Return whether a ticker looks like a warrant, unit, or right.

    Heuristic based on US ticker conventions: dash suffixes such as ``-WT``
    or ``-U`` denote derivatives (single letters like ``-B`` are share
    classes and remain eligible), and five-letter NASDAQ tickers ending in
    W, R, or U are warrants, rights, or units.

    Args:
        symbol: FMP ticker symbol.

    Returns:
        True when the ticker matches a derivative naming pattern.
    """
    root, _, suffix = symbol.partition("-")
    if suffix:
        return suffix.upper() in DERIVATIVE_SUFFIXES
    return len(root) == 5 and root[-1].upper() in {"W", "R", "U"}
