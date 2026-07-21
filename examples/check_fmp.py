"""Check the FMP adapters against the live API using FMP_API_KEY.

Run with ``uv run python examples/check_fmp.py``. Uses roughly five API calls:
one screener request, one profile, one income statement, and one bar history.
"""

from datetime import date
from pathlib import Path

from trading_script_anatomy.config import us_strategy_config
from trading_script_anatomy.data.fmp_provider import (
    FMPClient,
    FMPError,
    FMPMarketDataProvider,
    FMPScreenerUniverseProvider,
    ScreenerQuery,
)
from trading_script_anatomy.env import load_dotenv


def main() -> None:
    """Exercise each adapter method once and print the results."""
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    config = us_strategy_config()
    client = FMPClient()
    market_data = FMPMarketDataProvider(client)
    universe = FMPScreenerUniverseProvider(
        client,
        {
            config.benchmark_symbol: ScreenerQuery(
                market_cap_more_than=config.min_market_value,
                market_cap_lower_than=config.max_market_value,
                # $2 floor replaces the A-share ST filter's economic role:
                # exchange delisting notices start below $1, and sub-$2 names
                # sit in that risk zone. Ceiling mirrors filter_price.
                price_more_than=2.0,
                price_lower_than=config.highest_price,
            )
        },
    )
    today = date.today()

    bars = market_data.daily_bars(config.benchmark_symbol, periods=3, as_of=today)
    print(f"Last 3 {config.benchmark_symbol} bars:")
    print(bars[["open", "close"]] if not bars.empty else "  (no data)")

    demo_symbol = "F"
    try:
        symbols = universe.constituents(config.benchmark_symbol, today)
    except FMPError as error:
        print(f"\nScreener unavailable on this plan (needs Starter): {error}")
    else:
        print(f"\nScreened universe: {len(symbols)} symbols "
              f"(cap band ${config.min_market_value:,.0f}-"
              f"${config.max_market_value:,.0f})")
        print(f"Five smallest: {symbols[:5]}")
        demo_symbol = symbols[0]

    info = market_data.security_info(demo_symbol)
    print(f"\nsecurity_info({demo_symbol}): {info}")

    snapshot = market_data.financial_snapshot(demo_symbol, today)
    print(f"financial_snapshot({demo_symbol}): {snapshot}")


if __name__ == "__main__":
    main()
