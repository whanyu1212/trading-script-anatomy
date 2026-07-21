"""Configuration for the small-cap rotation strategy."""

from dataclasses import dataclass, field
from typing import Final

DEFAULT_POSITION_MAPPING: Final[tuple[tuple[float, float, int], ...]] = (
    (float("inf"), 500, 3),
    (500, 200, 3),
    (200, -200, 4),
    (-200, -500, 5),
    (-500, float("-inf"), 6),
)

US_POSITION_MAPPING: Final[tuple[tuple[float, float, int], ...]] = (
    (float("inf"), 725, 3),
    (725, 290, 3),
    (290, -290, 4),
    (-290, -725, 5),
    (-725, float("-inf"), 6),
)


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    """Immutable parameters used by the trading strategy.

    Attributes:
        benchmark_symbol: Yahoo Finance ticker used to calculate market trend.
        safe_etf_symbol: Instrument used to hold cash during defensive periods.
        initial_stock_count: Starting number of equity positions.
        highest_price: Maximum purchase price for a newly selected security.
        min_market_value: Minimum eligible market value in provider currency.
        max_market_value: Maximum eligible market value in provider currency.
        rebalance_weekday: Python weekday for weekly rebalancing, Monday is 0.
        empty_months: Months in which equities are replaced by the safe ETF.
        min_operating_revenue: Revenue floor, exclusive, in provider currency.
        stop_profit_ratio: Gain ratio at which a position is sold.
        stop_loss_ratio: Loss ratio at which a position is sold.
        market_stop_loss_threshold: Absolute benchmark intraday return that
            triggers liquidation.
        min_listing_days: Minimum listing age for eligibility.
        finance_candidate_multiplier: Multiplier used before the price filter.
        position_mapping: Benchmark-difference to desired-position-count mapping.
    """

    benchmark_symbol: str = "399106.SZ"
    safe_etf_symbol: str = "511880.SS"
    initial_stock_count: int = 4
    highest_price: float = 50.0
    min_market_value: float = 1e9
    max_market_value: float = 1e10
    min_operating_revenue: float = 1e8
    rebalance_weekday: int = 1
    empty_months: tuple[int, ...] = (1, 4)
    stop_profit_ratio: float = 2.0
    stop_loss_ratio: float = 0.09
    market_stop_loss_threshold: float = 0.05
    min_listing_days: int = 375
    finance_candidate_multiplier: int = 3
    position_mapping: tuple[tuple[float, float, int], ...] = field(
        default=DEFAULT_POSITION_MAPPING
    )


def us_strategy_config() -> StrategyConfig:
    """Return strategy parameters adapted for US micro-cap trading.

    A-share-specific behavior is removed: empty months encoded mainland-China
    disclosure-season seasonality, and the benchmark-constituent trend rules
    now read the Russell 2000 ETF. Values marked TODO are provisional
    placeholders awaiting a strategy decision, not tuned parameters.

    Returns:
        US-market strategy parameters.
    """
    return StrategyConfig(
        # The Russell 2000 index itself: FMP's free tier serves index bars but
        # gates most ETF symbols (IWM included), and an index avoids tracking
        # noise in the trend rule anyway.
        benchmark_symbol="^RUT",
        # Only ordered through the broker; the data layer never requests its
        # bars, so FMP's ETF-symbol gating does not affect it.
        safe_etf_symbol="SGOV",
        # Deliberate carry-overs from the A-share strategy, pinned explicitly
        # so a change to the class defaults cannot silently alter this preset.
        initial_stock_count=4,
        highest_price=50.0,
        rebalance_weekday=1,
        stop_profit_ratio=2.0,
        stop_loss_ratio=0.09,
        min_listing_days=375,
        finance_candidate_multiplier=3,
        # Preserves the original's market role (smallest investable decile)
        # rather than its literal currency conversion (~$140M-$1.4B), which
        # would land in small-cap territory and dilute the size premise.
        min_market_value=50e6,
        max_market_value=500e6,
        # Per single quarter, unlike the original's CNY 100M against Chinese
        # year-to-date statements, whose strictness varied by report season
        # (equivalent to roughly $3.5M-$14M per quarter). $5M holds the
        # middle of that range constant year-round.
        min_operating_revenue=5e6,
        empty_months=(),
        # The original 5% assumed daily moves capped at +/-10%; without price
        # limits the same mean move implies an even rarer tail event, so 4%
        # keeps the trigger frequency near the original's intent.
        market_stop_loss_threshold=0.04,
        # DEFAULT_POSITION_MAPPING bands scaled by ^RUT's ~2900 level versus
        # the SZSE Composite's ~2000, preserving the original percentage
        # thresholds. Point-based bands decay as the index level drifts; a
        # percentage-based trend rule would be the durable fix.
        position_mapping=US_POSITION_MAPPING,
    )
