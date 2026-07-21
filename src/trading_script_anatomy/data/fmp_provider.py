"""Financial Modeling Prep-backed market-data and universe adapters."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, timedelta
import os
from typing import Any

import pandas as pd
import requests

from trading_script_anatomy.data.models import (
    FinancialSnapshot,
    ProfitabilitySnapshot,
    RankedSecurity,
    SecurityInfo,
)
from trading_script_anatomy.values import to_finite_float

DEFAULT_BASE_URL = "https://financialmodelingprep.com/stable"


class FMPError(RuntimeError):
    """Raised when Financial Modeling Prep rejects or fails a request."""


class FMPClient:
    """Minimal authenticated JSON client for Financial Modeling Prep.

    Args:
        api_key: FMP API key. Defaults to the ``FMP_API_KEY`` environment
            variable; library code never reads ``.env`` files itself.
        session: Object providing ``requests.Session``-style ``get``. This
            injection point permits network-free tests.
        base_url: API root without a trailing slash.

    Raises:
        FMPError: If no API key is available.
    """

    def __init__(
        self,
        api_key: str | None = None,
        session: Any = None,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        key = api_key or os.environ.get("FMP_API_KEY", "")
        if not key:
            raise FMPError(
                "An FMP API key is required; set FMP_API_KEY or pass api_key"
            )
        self._api_key = key
        self._session = session or requests.Session()
        self._base_url = base_url.rstrip("/")

    def get_json(self, path: str, params: Mapping[str, object] | None = None) -> Any:
        """Return decoded JSON for an authenticated GET request.

        Args:
            path: Endpoint path relative to the API root.
            params: Query parameters. ``None`` values are omitted.

        Returns:
            The decoded JSON payload.

        Raises:
            FMPError: On a non-200 response or an FMP error payload, including
                the plan-restriction errors paid endpoints return.
        """
        query = {key: value for key, value in (params or {}).items() if value is not None}
        query["apikey"] = self._api_key
        response = self._session.get(
            f"{self._base_url}/{path}", params=query, timeout=30
        )
        if response.status_code != 200:
            raise FMPError(
                f"FMP request {path!r} failed with HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )
        payload = response.json()
        if isinstance(payload, dict) and "Error Message" in payload:
            raise FMPError(f"FMP request {path!r} rejected: {payload['Error Message']}")
        return payload


class FMPMarketDataProvider:
    """Fetch market data supported by Financial Modeling Prep.

    ``financial_snapshot`` combines the latest quarterly statement filed on or
    before the requested date with the *current* market capitalization, so
    market value is not point-in-time for historical dates. Bars contain no
    ``high_limit``/``low_limit`` columns because US equities have no daily
    price limits; limit-based filters are skipped by design.

    Args:
        client: Authenticated FMP API client.
        statement_limit: Quarterly statements requested per symbol. The free
            plan rejects values above 5; raise this on paid plans when
            backtesting far enough back that five quarters cannot cover the
            filing-date look-back.
    """

    def __init__(self, client: FMPClient, statement_limit: int = 5) -> None:
        self._client = client
        self._statement_limit = statement_limit
        self._profile_cache: dict[str, dict[str, Any] | None] = {}

    def daily_bars(self, symbol: str, periods: int, as_of: date) -> pd.DataFrame:
        """Return normalized daily bars ending on or before ``as_of``.

        Args:
            symbol: FMP ticker, such as ``IWM``.
            periods: Maximum number of daily bars to return.
            as_of: Inclusive history end date.

        Returns:
            A date-indexed frame with lower-case columns ordered oldest to
            newest. An empty frame is returned if FMP has no data.

        Raises:
            ValueError: If ``periods`` is less than one.
        """
        if periods < 1:
            raise ValueError("periods must be at least 1")

        start = as_of - timedelta(days=max(periods * 3, 30))
        payload = self._client.get_json(
            "historical-price-eod/full",
            {"symbol": symbol, "from": start.isoformat(), "to": as_of.isoformat()},
        )
        rows = payload.get("historical") if isinstance(payload, dict) else payload
        if not rows:
            return pd.DataFrame()

        bars = pd.DataFrame(rows).rename(columns=str.lower)
        if "date" not in bars.columns:
            return pd.DataFrame()
        bars["date"] = pd.to_datetime(bars["date"])
        return bars.set_index("date").sort_index().tail(periods)

    def security_info(self, symbol: str) -> SecurityInfo | None:
        """Return FMP company-profile metadata when available.

        Args:
            symbol: FMP ticker.

        Returns:
            Normalized security metadata, or ``None`` if FMP has no profile.
        """
        profile = self._profile(symbol)
        if not profile:
            return None
        return SecurityInfo(
            symbol=symbol,
            name=str(profile.get("companyName") or ""),
            listed_on=_to_date(profile.get("ipoDate")),
            exchange=_to_text(
                profile.get("exchangeShortName") or profile.get("exchange")
            ),
            is_etf=_to_bool(profile.get("isEtf")),
            is_adr=_to_bool(profile.get("isAdr")),
            is_actively_trading=_to_bool(profile.get("isActivelyTrading")),
        )

    def profitability(
        self, symbol: str, as_of: date
    ) -> ProfitabilitySnapshot | None:
        """Return the latest filed quarterly income-statement values.

        The statement is selected by filing date to avoid look-ahead: only
        reports filed on or before ``as_of`` are considered. US consolidated
        statements do not break out a parent-owner share, so
        ``net_profit_parent`` equals ``net_profit``.

        Args:
            symbol: FMP ticker.
            as_of: Date at which the values are requested.

        Returns:
            Profitability values, or ``None`` when a value is unavailable.
        """
        statements = self._client.get_json(
            "income-statement",
            {"symbol": symbol, "period": "quarter", "limit": self._statement_limit},
        )
        statement = _latest_filed_on_or_before(statements, as_of)
        if statement is None:
            return None

        operating_revenue = to_finite_float(statement.get("revenue"))
        net_profit = to_finite_float(statement.get("netIncome"))
        if operating_revenue is None or net_profit is None:
            return None
        return ProfitabilitySnapshot(
            operating_revenue=operating_revenue,
            net_profit=net_profit,
            net_profit_parent=net_profit,
        )

    def financial_snapshot(
        self, symbol: str, as_of: date
    ) -> FinancialSnapshot | None:
        """Return current market value with the latest filed quarterly results.

        Market value comes from the company profile and is current, not
        point-in-time; the income-statement values honor the filing-date
        contract via ``profitability``.

        Args:
            symbol: FMP ticker.
            as_of: Date at which the snapshot is requested.

        Returns:
            Required financial values, or ``None`` when a value is unavailable.
        """
        profit = self.profitability(symbol, as_of)
        profile = self._profile(symbol) or {}
        market_value = to_finite_float(
            profile.get("marketCap") or profile.get("mktCap")
        )
        if profit is None or market_value is None:
            return None
        return FinancialSnapshot(
            market_value=market_value,
            operating_revenue=profit.operating_revenue,
            net_profit=profit.net_profit,
            net_profit_parent=profit.net_profit_parent,
        )

    def _profile(self, symbol: str) -> dict[str, Any] | None:
        """Return the cached company profile for a symbol."""
        if symbol not in self._profile_cache:
            payload = self._client.get_json("profile", {"symbol": symbol})
            rows = payload if isinstance(payload, list) else [payload]
            first = rows[0] if rows else None
            self._profile_cache[symbol] = first if isinstance(first, dict) else None
        return self._profile_cache[symbol]


@dataclass(frozen=True, slots=True)
class ScreenerQuery:
    """Server-side screening parameters that define a tradable universe.

    Attributes:
        market_cap_more_than: Minimum market capitalization in USD.
        market_cap_lower_than: Maximum market capitalization in USD.
        exchanges: Exchanges searched by the screener.
        price_more_than: Optional minimum share price.
        price_lower_than: Optional maximum share price.
        volume_more_than: Optional minimum daily share volume.
        limit: Maximum number of rows requested from the screener.
    """

    market_cap_more_than: float
    market_cap_lower_than: float
    exchanges: tuple[str, ...] = ("NYSE", "NASDAQ", "AMEX")
    price_more_than: float | None = None
    price_lower_than: float | None = None
    volume_more_than: float | None = None
    limit: int = 3000


class FMPScreenerUniverseProvider:
    """Build index universes from the FMP company screener.

    The engine passes ``config.benchmark_symbol`` to ``constituents``, so
    register the screen under that symbol, for example
    ``{"IWM": ScreenerQuery(...)}``. The screener reflects current listings
    only; ``as_of`` serves as a per-day cache key, and backtests built on this
    provider therefore carry survivorship bias.

    Args:
        client: Authenticated FMP API client.
        queries: Mapping from index symbol to its screening parameters.
    """

    def __init__(
        self, client: FMPClient, queries: Mapping[str, ScreenerQuery]
    ) -> None:
        self._client = client
        self._queries = dict(queries)
        self._cache: dict[tuple[str, date], tuple[RankedSecurity, ...]] = {}

    def constituents(self, index_symbol: str, as_of: date) -> tuple[str, ...]:
        """Return screened symbols ordered by ascending market value.

        Args:
            index_symbol: Symbol whose configured screen defines the universe.
            as_of: Requested date, used only to cache one result per day.

        Returns:
            Screened ticker symbols, smallest market value first.

        Raises:
            KeyError: If no screen was configured for ``index_symbol``.
            FMPError: If the screener request fails.
        """
        return tuple(
            entry.symbol for entry in self.ranked_constituents(index_symbol, as_of)
        )

    def ranked_constituents(
        self, index_symbol: str, as_of: date
    ) -> tuple[RankedSecurity, ...]:
        """Return screened securities ordered by ascending market value.

        Args:
            index_symbol: Symbol whose configured screen defines the universe.
            as_of: Requested date, used only to cache one result per day.

        Returns:
            Screened securities with market values, smallest first.

        Raises:
            KeyError: If no screen was configured for ``index_symbol``.
            FMPError: If the screener request fails.
        """
        if index_symbol not in self._queries:
            raise KeyError(f"No screener query configured for {index_symbol!r}")
        cached = self._cache.get((index_symbol, as_of))
        if cached is not None:
            return cached

        query = self._queries[index_symbol]
        rows = self._client.get_json(
            "company-screener",
            {
                "marketCapMoreThan": query.market_cap_more_than,
                "marketCapLowerThan": query.market_cap_lower_than,
                "exchange": ",".join(query.exchanges),
                "priceMoreThan": query.price_more_than,
                "priceLowerThan": query.price_lower_than,
                "volumeMoreThan": query.volume_more_than,
                "isActivelyTrading": "true",
                "isEtf": "false",
                "isFund": "false",
                "limit": query.limit,
            },
        )

        ranked: list[tuple[float, str]] = []
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            symbol = _to_text(row.get("symbol"))
            market_cap = to_finite_float(row.get("marketCap"))
            if symbol and market_cap is not None and market_cap > 0:
                ranked.append((market_cap, symbol))
        ranked.sort()

        entries = tuple(
            RankedSecurity(symbol=symbol, market_value=market_cap)
            for market_cap, symbol in ranked
        )
        self._cache[(index_symbol, as_of)] = entries
        return entries


def _latest_filed_on_or_before(
    statements: object, as_of: date
) -> dict[str, Any] | None:
    """Return the newest statement whose filing date is on or before a date."""
    best: tuple[date, dict[str, Any]] | None = None
    for statement in statements if isinstance(statements, list) else []:
        if not isinstance(statement, dict):
            continue
        filed = _to_date(
            statement.get("filingDate")
            or statement.get("fillingDate")
            or statement.get("date")
        )
        if filed is None or filed > as_of:
            continue
        if best is None or filed > best[0]:
            best = (filed, statement)
    return None if best is None else best[1]


def _to_date(value: object) -> date | None:
    """Parse the date portion of an FMP date or timestamp string."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _to_text(value: object) -> str | None:
    """Return a non-empty string value when possible."""
    return value if isinstance(value, str) and value else None


def _to_bool(value: object) -> bool | None:
    """Return a boolean provider value when possible."""
    return value if isinstance(value, bool) else None
