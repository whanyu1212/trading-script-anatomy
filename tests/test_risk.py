"""Tests for position and market-wide risk controls."""

from datetime import date

import pytest

from trading_script_anatomy.config import StrategyConfig
from trading_script_anatomy.portfolio import Position
from trading_script_anatomy.strategy.risk import RiskManager

from tests.fakes import FakeMarketData, bars


AS_OF = date(2025, 6, 3)

BENCHMARK = StrategyConfig().benchmark_symbol


def test_stop_loss_generates_full_liquidation_order() -> None:
    """Generate a stop-loss sale when the current price declines sufficiently."""
    manager = RiskManager(StrategyConfig(), FakeMarketData())

    orders = manager.position_exit_orders(
        [Position("000001.SZ", quantity=100, cost_basis=10, last_price=9)],
        AS_OF,
    )

    assert [(order.symbol, order.quantity, order.reason) for order in orders] == [
        ("000001.SZ", 100, "stop_loss")
    ]


@pytest.mark.parametrize(
    ("open_price", "close_price", "expected"),
    [
        (100.0, 94.9, True),
        (100.0, 106.0, True),
        (100.0, 96.0, False),
    ],
)
def test_market_stop_loss_uses_benchmark_intraday_return(
    open_price: float, close_price: float, expected: bool
) -> None:
    """Trigger liquidation on a large benchmark move in either direction."""
    data = FakeMarketData(bars={BENCHMARK: bars([close_price], [open_price])})
    manager = RiskManager(StrategyConfig(market_stop_loss_threshold=0.05), data)

    assert manager.market_stop_loss_triggered(AS_OF) is expected


def test_market_stop_loss_is_safe_without_benchmark_data() -> None:
    """Never liquidate on missing data."""
    manager = RiskManager(StrategyConfig(), FakeMarketData())

    assert manager.market_stop_loss_triggered(AS_OF) is False


