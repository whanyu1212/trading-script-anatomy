"""Deterministic data-source fakes for strategy tests."""

from collections.abc import Mapping, Sequence
from datetime import date

import pandas as pd

from trading_script_anatomy.data.models import (
    FinancialSnapshot,
    RankedSecurity,
    SecurityInfo,
)


class FakeMarketData:
    """Return caller-provided metadata, financials, and daily bars.

    Args:
        infos: Security metadata keyed by symbol.
        financials: Financial snapshots keyed by symbol.
        bars: Daily-bars frames keyed by symbol.
    """

    def __init__(
        self,
        infos: Mapping[str, SecurityInfo] | None = None,
        financials: Mapping[str, FinancialSnapshot] | None = None,
        bars: Mapping[str, pd.DataFrame] | None = None,
    ) -> None:
        self.infos = dict(infos or {})
        self.financials = dict(financials or {})
        self.bars = {
            symbol: frame.copy() for symbol, frame in (bars or {}).items()
        }
        self.info_calls: list[str] = []
        self.financial_calls: list[str] = []

    def daily_bars(self, symbol: str, periods: int, as_of: date) -> pd.DataFrame:
        """Return supplied bars for a symbol.

        Args:
            symbol: Ticker requested by the strategy.
            periods: Maximum number of bars to return.
            as_of: Retained for protocol compatibility.

        Returns:
            Trailing requested rows from the configured frame.
        """
        del as_of
        return self.bars.get(symbol, pd.DataFrame()).tail(periods).copy()

    def security_info(self, symbol: str) -> SecurityInfo | None:
        """Return supplied security metadata.

        Args:
            symbol: Ticker requested by the strategy.

        Returns:
            Configured metadata, if any.
        """
        self.info_calls.append(symbol)
        return self.infos.get(symbol)

    def financial_snapshot(
        self, symbol: str, as_of: date
    ) -> FinancialSnapshot | None:
        """Return supplied financial data.

        Args:
            symbol: Ticker requested by the strategy.
            as_of: Retained for protocol compatibility.

        Returns:
            Configured financial snapshot, if any.
        """
        del as_of
        self.financial_calls.append(symbol)
        return self.financials.get(symbol)

    def profitability(
        self, symbol: str, as_of: date
    ) -> FinancialSnapshot | None:
        """Return supplied financial data for the profitability-only path.

        The configured ``FinancialSnapshot`` is returned directly; it carries
        the profitability fields structurally.

        Args:
            symbol: Ticker requested by the strategy.
            as_of: Retained for protocol compatibility.

        Returns:
            Configured financial snapshot, if any.
        """
        del as_of
        self.financial_calls.append(symbol)
        return self.financials.get(symbol)


class FakeUniverse:
    """Return a fixed ordered index universe.

    Args:
        symbols: Constituent symbols returned for every index request.
    """

    def __init__(self, symbols: Sequence[str]) -> None:
        self.symbols = tuple(symbols)

    def constituents(self, index_symbol: str, as_of: date) -> tuple[str, ...]:
        """Return configured symbols.

        Args:
            index_symbol: Retained for protocol compatibility.
            as_of: Retained for protocol compatibility.

        Returns:
            Fixed ticker symbols.
        """
        del index_symbol, as_of
        return self.symbols


class FakeRankedUniverse:
    """Return a fixed universe pre-ranked by ascending market value.

    Args:
        entries: Ranked securities returned for every index request.
    """

    def __init__(self, entries: Sequence[RankedSecurity]) -> None:
        self.entries = tuple(entries)

    def constituents(self, index_symbol: str, as_of: date) -> tuple[str, ...]:
        """Return configured symbols in ranked order.

        Args:
            index_symbol: Retained for protocol compatibility.
            as_of: Retained for protocol compatibility.

        Returns:
            Fixed ticker symbols, smallest market value first.
        """
        del index_symbol, as_of
        return tuple(entry.symbol for entry in self.entries)

    def ranked_constituents(
        self, index_symbol: str, as_of: date
    ) -> tuple[RankedSecurity, ...]:
        """Return configured ranked securities.

        Args:
            index_symbol: Retained for protocol compatibility.
            as_of: Retained for protocol compatibility.

        Returns:
            Fixed ranked securities, smallest market value first.
        """
        del index_symbol, as_of
        return self.entries


class FakeHTTPResponse:
    """Provide the response surface used by the HTTP adapters."""

    def __init__(self, payload: object, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self) -> object:
        """Return the configured payload."""
        return self._payload


class FakeHTTPSession:
    """Serve scripted JSON payloads keyed by method and URL fragment.

    Each key maps to a list of payloads consumed in order; the final payload
    repeats for later calls, which models polling the same endpoint. Requests
    are recorded as ``(method, url, body)`` tuples where the body is the JSON
    payload when present and the query parameters otherwise.

    Args:
        script: Mapping from ``(method, url fragment)`` to payloads.
    """

    def __init__(self, script: Mapping[tuple[str, str], Sequence[object]]) -> None:
        self._script = {key: list(payloads) for key, payloads in script.items()}
        self.calls: list[tuple[str, str, dict | None]] = []

    def request(
        self,
        method: str,
        url: str,
        headers: dict | None = None,
        params: dict | None = None,
        json: dict | None = None,
        timeout: int = 0,
    ) -> FakeHTTPResponse:
        """Return the next scripted payload matching the request."""
        del headers, timeout
        self.calls.append((method, url, json if json is not None else params))
        for (wanted_method, fragment), payloads in self._script.items():
            if method == wanted_method and fragment in url:
                payload = payloads.pop(0) if len(payloads) > 1 else payloads[0]
                return FakeHTTPResponse(payload)
        return FakeHTTPResponse({"message": "not scripted"}, status_code=404)

    def get(
        self, url: str, params: dict | None = None, timeout: int = 0
    ) -> FakeHTTPResponse:
        """Serve a GET request through the same scripted routing."""
        return self.request("GET", url, params=params, timeout=timeout)


def bars(
    closes: Sequence[float],
    opens: Sequence[float] | None = None,
    high_limits: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Build a normalized daily-bars frame.

    Args:
        closes: Closing prices.
        opens: Opening prices. Defaults to closing prices.
        high_limits: Optional upper price limits.

    Returns:
        Data frame compatible with the market-data protocol.
    """
    frame = pd.DataFrame({"close": closes, "open": opens or closes})
    if high_limits is not None:
        frame["high_limit"] = high_limits
    return frame
