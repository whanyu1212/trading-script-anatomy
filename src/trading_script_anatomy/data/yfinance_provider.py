"""Yahoo Finance-backed market-data adapter."""

from datetime import UTC, date, datetime, timedelta
import logging
from typing import Any

import pandas as pd
import yfinance as yf

from trading_script_anatomy.data.models import (
    FinancialSnapshot,
    ProfitabilitySnapshot,
    SecurityInfo,
)
from trading_script_anatomy.values import to_finite_float

_logger = logging.getLogger(__name__)


class YFinanceMarketDataProvider:
    """Fetch market data supported by Yahoo Finance through ``yfinance``.

    Yahoo Finance does not provide historical Chinese-index constituents or
    exchange price-limit fields. Pair this adapter with a separate universe
    provider; price-limit filters will be skipped when limit columns are absent.

    Args:
        ticker_factory: Factory used to construct yfinance ticker objects. This
            injection point permits network-free tests.
    """

    def __init__(self, ticker_factory: Any = yf.Ticker) -> None:
        self._ticker_factory = ticker_factory
        self._pit_warning_emitted = False

    def daily_bars(self, symbol: str, periods: int, as_of: date) -> pd.DataFrame:
        """Return normalized daily bars ending on or before ``as_of``.

        Args:
            symbol: Yahoo Finance ticker, such as ``399106.SZ``.
            periods: Maximum number of daily bars to return.
            as_of: Inclusive history end date.

        Returns:
            A date-indexed frame with lower-case columns. An empty frame is
            returned if Yahoo Finance has no data.

        Raises:
            ValueError: If ``periods`` is less than one.
        """
        if periods < 1:
            raise ValueError("periods must be at least 1")

        start = as_of - timedelta(days=max(periods * 3, 30))
        history = self._ticker_factory(symbol).history(
            start=start,
            end=as_of + timedelta(days=1),
            interval="1d",
            auto_adjust=False,
            actions=False,
        )
        if history.empty:
            return pd.DataFrame()

        bars = history.rename(columns=str.lower).copy()
        if isinstance(bars.index, pd.DatetimeIndex) and bars.index.tz is not None:
            bars.index = bars.index.tz_convert(None)
        return bars.tail(periods)

    def security_info(self, symbol: str) -> SecurityInfo | None:
        """Return Yahoo Finance security metadata when available.

        Args:
            symbol: Yahoo Finance ticker.

        Returns:
            Normalized security metadata, or ``None`` if Yahoo Finance returns
            no metadata.
        """
        info = self._ticker_factory(symbol).get_info() or {}
        if not info:
            return None
        return SecurityInfo(
            symbol=symbol,
            name=str(info.get("shortName") or info.get("longName") or ""),
            listed_on=self._to_date(info.get("firstTradeDateEpochUtc")),
        )

    def profitability(
        self, symbol: str, as_of: date
    ) -> ProfitabilitySnapshot | None:
        """Return the most recent Yahoo Finance income-statement values.

        Yahoo Finance reports current statements without filing dates, so
        values are NOT point-in-time; asking for a historical date emits a
        look-ahead warning once per provider instance.

        Args:
            symbol: Yahoo Finance ticker.
            as_of: Requested evaluation date.

        Returns:
            Profitability values, or ``None`` if a value is unavailable.
        """
        self._warn_if_not_point_in_time(as_of)
        financials = self._ticker_factory(symbol).financials
        if financials is None or financials.empty:
            return None

        operating_revenue = self._statement_value(financials, "Total Revenue")
        net_profit = self._statement_value(financials, "Net Income")
        net_profit_parent = (
            self._statement_value(financials, "Net Income Common Stockholders")
            or net_profit
        )
        if any(
            value is None
            for value in (operating_revenue, net_profit, net_profit_parent)
        ):
            return None
        return ProfitabilitySnapshot(
            operating_revenue=operating_revenue,
            net_profit=net_profit,
            net_profit_parent=net_profit_parent,
        )

    def financial_snapshot(
        self, symbol: str, as_of: date
    ) -> FinancialSnapshot | None:
        """Return current Yahoo Finance fundamentals with market value.

        Both halves are current-only; see ``profitability`` for the
        point-in-time caveat.

        Args:
            symbol: Yahoo Finance ticker.
            as_of: Requested evaluation date.

        Returns:
            Required financial values, or ``None`` if a value is unavailable.
        """
        profit = self.profitability(symbol, as_of)
        if profit is None:
            return None
        info = self._ticker_factory(symbol).get_info() or {}
        market_value = to_finite_float(info.get("marketCap"))
        if market_value is None:
            return None
        return FinancialSnapshot(
            market_value=market_value,
            operating_revenue=profit.operating_revenue,
            net_profit=profit.net_profit,
            net_profit_parent=profit.net_profit_parent,
        )

    def _warn_if_not_point_in_time(self, as_of: date) -> None:
        """Warn once when historical fundamentals are requested."""
        if as_of >= date.today() or self._pit_warning_emitted:
            return
        self._pit_warning_emitted = True
        _logger.warning(
            "Yahoo Finance fundamentals are not point-in-time: values "
            "requested for %s reflect currently published reports, which is "
            "look-ahead in a backtest",
            as_of,
        )

    @staticmethod
    def _statement_value(frame: pd.DataFrame, row: str) -> float | None:
        """Return the newest numeric value from a financial-statement row."""
        if row not in frame.index:
            return None
        values = pd.to_numeric(frame.loc[row], errors="coerce").dropna()
        return None if values.empty else float(values.iloc[0])

    @staticmethod
    def _to_date(value: object) -> date | None:
        """Convert a Unix timestamp from Yahoo Finance to a UTC date."""
        if value is None:
            return None
        try:
            return datetime.fromtimestamp(int(value), tz=UTC).date()
        except (OSError, TypeError, ValueError):
            return None
