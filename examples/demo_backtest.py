"""Run a one-month backtest against live FMP data using FMP_API_KEY.

Run with ``uv run python examples/demo_backtest.py``. Uses roughly 100 API
calls, comfortably inside the free tier's daily budget.

Free-tier compromises, so this is a machinery check rather than a strategy
result: the screener is unavailable, so a static large-cap universe stands in
for the micro-cap screen (with the cap band widened to match), and a static
universe means the exhaustive selection path is exercised rather than the
ranked walk.
"""

from dataclasses import replace
from datetime import date
from pathlib import Path

from trading_script_anatomy.backtest.simulator import Backtester, US_MICROCAP_COSTS
from trading_script_anatomy.config import us_strategy_config
from trading_script_anatomy.data.fmp_provider import FMPClient, FMPMarketDataProvider
from trading_script_anatomy.data.universe import StaticIndexUniverseProvider
from trading_script_anatomy.env import load_dotenv
from trading_script_anatomy.strategy.us_filters import us_eligibility

DEMO_UNIVERSE = ("F", "T", "VZ", "PFE", "GM")


def main() -> None:
    """Backtest one month of the strategy over a small static universe."""
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    config = replace(
        us_strategy_config(),
        # Large caps stand in for the gated screener; widen the band so the
        # demo universe passes the market-value filter.
        min_market_value=1e9,
        max_market_value=5e12,
        # Demo-only substitution: SGOV bars are gated on the free tier, so
        # SPY stands in to exercise the safe-asset leg. SPY is NOT risk-free
        # and stays exempt from all risk controls while parked — do not copy
        # this into a live configuration.
        safe_etf_symbol="SPY",
    )
    market_data = FMPMarketDataProvider(FMPClient())
    universe = StaticIndexUniverseProvider(
        {config.benchmark_symbol: DEMO_UNIVERSE}
    )
    backtester = Backtester(
        config,
        market_data,
        universe,
        costs=US_MICROCAP_COSTS,
        eligibility=us_eligibility,
    )

    result = backtester.run(date(2026, 6, 15), date(2026, 7, 17))

    print(result.summary())
    print("\nFills:")
    for order in result.orders:
        print(f"  {order.side.value:4s} {order.symbol:5s} "
              f"{order.quantity:12.4f} @ {order.price:8.2f}  ({order.reason})")


if __name__ == "__main__":
    main()
