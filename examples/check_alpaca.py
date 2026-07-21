"""Check the Alpaca paper-trading adapter using keys from .env.

Run with ``uv run python examples/check_alpaca.py``. Read-only by default:
fetches the account snapshot, positions, and market clock. When the market
is open, additionally round-trips a $2 notional SPY order to validate the
full order path.
"""

from pathlib import Path

from trading_script_anatomy.broker.alpaca import AlpacaBroker
from trading_script_anatomy.env import load_dotenv


def main() -> None:
    """Exercise the read paths and, during market hours, one order."""
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    broker = AlpacaBroker()

    clock = broker.clock()
    print(f"Market open: {clock.get('is_open')} "
          f"(next open {clock.get('next_open')}, next close {clock.get('next_close')})")

    portfolio = broker.portfolio
    print(f"Paper cash: ${portfolio.cash:,.2f}")
    print(f"Positions: {len(portfolio.positions)}")
    for position in portfolio.positions.values():
        print(f"  {position.symbol}: {position.quantity} @ "
              f"cost {position.cost_basis:.2f}, last {position.last_price:.2f}")

    if clock.get("is_open"):
        print("\nMarket is open — validating the order path with $2 of SPY...")
        broker.order_value("SPY", 2.0, "order_path_check")
        broker.refresh()
        spy = broker.portfolio.positions.get("SPY")
        print(f"SPY position after buy: {spy}")
    else:
        print("\nMarket is closed — skipping the live order test. "
              "Re-run during US market hours to validate the order path.")


if __name__ == "__main__":
    main()
