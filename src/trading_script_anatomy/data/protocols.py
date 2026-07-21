"""Protocols for strategy data sources."""

from datetime import date
from typing import Protocol, Sequence, runtime_checkable

import pandas as pd

from trading_script_anatomy.data.models import (
    FinancialSnapshot,
    ProfitabilitySnapshot,
    RankedSecurity,
    SecurityInfo,
)


class IndexUniverseProvider(Protocol):
    """Supply the constituent universe for an index."""

    def constituents(self, index_symbol: str, as_of: date) -> Sequence[str]:
        """Return constituent symbols for an index on a date.

        Args:
            index_symbol: Provider-specific index ticker.
            as_of: Date for which constituents are requested.

        Returns:
            Ordered constituent ticker symbols.
        """


@runtime_checkable
class RankedUniverseProvider(Protocol):
    """Supply a universe pre-ranked by ascending market value.

    Providers that can rank cheaply — one screener request instead of one
    fundamental request per symbol — advertise it through this protocol. The
    stock selector detects the capability with ``isinstance`` and walks the
    ranking lazily, stopping as soon as enough candidates qualify.
    """

    def ranked_constituents(
        self, index_symbol: str, as_of: date
    ) -> Sequence[RankedSecurity]:
        """Return constituents ordered by ascending market value.

        Args:
            index_symbol: Provider-specific index ticker.
            as_of: Date for which constituents are requested.

        Returns:
            Constituents with their ranking market values, smallest first.
        """


class BarProvider(Protocol):
    """Supply daily price bars.

    The narrow interface for consumers that need prices only, such as the
    risk manager; ``MarketDataProvider`` extends it with security metadata
    and fundamentals.
    """

    def daily_bars(self, symbol: str, periods: int, as_of: date) -> pd.DataFrame:
        """Return recent daily bars ending on or before a date.

        Returned frames must include lower-case ``open`` and ``close`` columns.
        Optional ``high_limit`` and ``low_limit`` columns enable exchange-limit
        filters.

        Args:
            symbol: Provider-specific ticker symbol.
            periods: Maximum number of daily bars to return.
            as_of: Inclusive end date for the requested history.

        Returns:
            A date-indexed frame ordered from oldest to newest.
        """


class MarketDataProvider(BarProvider, Protocol):
    """Supply market, security, and fundamental data to the strategy."""

    def security_info(self, symbol: str) -> SecurityInfo | None:
        """Return security metadata when it is available.

        Args:
            symbol: Provider-specific ticker symbol.

        Returns:
            Security metadata, or ``None`` when unavailable.
        """

    def financial_snapshot(
        self, symbol: str, as_of: date
    ) -> FinancialSnapshot | None:
        """Return market value with the fields of the profitability filter.

        This is a point-in-time contract: implementations must report only
        values published on or before ``as_of``. An adapter that cannot honor
        this must document the deviation and emit a warning when asked for
        historical dates, because silently substituting current values
        introduces look-ahead into backtests.

        Args:
            symbol: Provider-specific ticker symbol.
            as_of: Date at which the snapshot is requested.

        Returns:
            Financial values, or ``None`` when unavailable.
        """

    def profitability(
        self, symbol: str, as_of: date
    ) -> ProfitabilitySnapshot | None:
        """Return income-statement values without requiring market value.

        Consumers that already know a security's market value — such as the
        ranked selection walk — use this method so that missing market-value
        data cannot reject an otherwise-qualified security. The same
        point-in-time contract as ``financial_snapshot`` applies.

        Args:
            symbol: Provider-specific ticker symbol.
            as_of: Date at which the values are requested.

        Returns:
            Profitability values, or ``None`` when unavailable.
        """
