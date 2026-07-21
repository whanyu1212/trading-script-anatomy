"""Domain models owned by the market-data layer."""

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class SecurityInfo:
    """Static metadata for a tradable security.

    Attributes:
        symbol: Provider-specific ticker symbol.
        name: Display name supplied by the data provider.
        listed_on: Exchange listing date when available.
        delisted_on: Delisting date when available.
        exchange: Short exchange name when available, such as ``NASDAQ``.
        is_etf: Whether the security is an ETF, when the provider reports it.
        is_adr: Whether the security is an ADR, when the provider reports it.
        is_actively_trading: Whether the security currently trades, when the
            provider reports it.
    """

    symbol: str
    name: str = ""
    listed_on: date | None = None
    delisted_on: date | None = None
    exchange: str | None = None
    is_etf: bool | None = None
    is_adr: bool | None = None
    is_actively_trading: bool | None = None


@dataclass(frozen=True, slots=True)
class RankedSecurity:
    """A universe constituent with the market value that ranked it.

    Attributes:
        symbol: Provider-specific ticker symbol.
        market_value: Market value used for ascending ranking.
    """

    symbol: str
    market_value: float


@dataclass(frozen=True, slots=True)
class ProfitabilitySnapshot:
    """Income-statement fields used by the profitability filter.

    Kept separate from ``FinancialSnapshot`` because these values change
    quarterly with filings while market value changes daily with prices;
    consumers that need only profitability must not fail on missing
    market-value data.

    Attributes:
        operating_revenue: Operating revenue for the latest period.
        net_profit: Net profit for the latest period.
        net_profit_parent: Profit attributable to parent-company owners.
    """

    operating_revenue: float
    net_profit: float
    net_profit_parent: float


@dataclass(frozen=True, slots=True)
class FinancialSnapshot:
    """Financial fields used by the fundamental filter.

    Attributes:
        market_value: Total market value for the security.
        operating_revenue: Operating revenue for the latest period.
        net_profit: Net profit for the latest period.
        net_profit_parent: Profit attributable to parent-company owners.
    """

    market_value: float
    operating_revenue: float
    net_profit: float
    net_profit_parent: float
