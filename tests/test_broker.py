"""Tests for the deterministic broker implementation."""

import pytest

from trading_script_anatomy.broker.memory import InMemoryBroker
from trading_script_anatomy.broker.models import CostModel, OrderSide
from trading_script_anatomy.portfolio import Portfolio, Position


def test_in_memory_broker_updates_cash_positions_and_order_history() -> None:
    """Fill orders at supplied prices and preserve resulting state."""
    broker = InMemoryBroker(Portfolio(cash=100.0), {"000001.SZ": 10.0})

    broker.order_value("000001.SZ", 50.0, "entry")
    broker.order_quantity("000001.SZ", -2.0, "trim")

    position = broker.portfolio.positions["000001.SZ"]
    assert broker.portfolio.cash == 70.0
    assert position.quantity == 3.0
    assert [order.side for order in broker.orders] == [
        OrderSide.BUY,
        OrderSide.SELL,
    ]


def test_cost_model_prices_commissions_and_investable_amounts() -> None:
    """Apply slippage, proportional and floored commission, and netting."""
    costs = CostModel(commission_rate=0.001, min_commission=5.0, slippage_rate=0.01)

    assert costs.buy_price(100.0) == pytest.approx(101.0)
    assert costs.sell_price(100.0) == pytest.approx(99.0)
    assert costs.commission(10_000.0) == pytest.approx(10.0)
    assert costs.commission(1_000.0) == pytest.approx(5.0)
    assert costs.investable(10_000.0) == pytest.approx(10_000.0 / 1.001)
    assert costs.investable(1_000.0) == pytest.approx(995.0)


def test_buy_applies_slippage_to_fill_price_and_cash() -> None:
    """Charge the slippage-adjusted price on buys."""
    broker = InMemoryBroker(
        Portfolio(cash=10_000.0),
        {"ACME": 100.0},
        costs=CostModel(slippage_rate=0.01),
    )

    broker.order_quantity("ACME", 50.0, "entry")

    position = broker.portfolio.positions["ACME"]
    assert position.cost_basis == pytest.approx(101.0)
    assert broker.portfolio.cash == pytest.approx(10_000.0 - 50 * 101.0)


def test_order_value_spends_exactly_value_including_minimum_commission() -> None:
    """Net the commission out of the requested order value."""
    broker = InMemoryBroker(
        Portfolio(cash=1_000.0),
        {"ACME": 10.0},
        costs=CostModel(min_commission=5.0),
    )

    broker.order_value("ACME", 1_000.0, "entry")

    assert broker.portfolio.positions["ACME"].quantity == pytest.approx(99.5)
    assert broker.portfolio.cash == pytest.approx(0.0)


def test_order_value_smaller_than_minimum_commission_raises() -> None:
    """Refuse orders whose value cannot cover execution costs."""
    broker = InMemoryBroker(
        Portfolio(cash=1_000.0),
        {"ACME": 10.0},
        costs=CostModel(min_commission=5.0),
    )

    with pytest.raises(ValueError):
        broker.order_value("ACME", 4.0, "entry")


def test_sell_nets_out_sale_tax() -> None:
    """Deduct the sale-only tax from sale proceeds."""
    broker = InMemoryBroker(
        Portfolio(cash=0.0, positions={"S": Position("S", 100.0, 10.0, 10.0)}),
        {"S": 10.0},
        costs=CostModel(sell_tax_rate=0.1),
    )

    broker.order_quantity("S", -100.0, "exit")

    assert broker.portfolio.cash == pytest.approx(900.0)
    assert "S" not in broker.portfolio.positions


def test_price_resolver_supplies_missing_prices() -> None:
    """Fall back to the resolver for symbols without a configured price."""
    broker = InMemoryBroker(
        Portfolio(cash=100.0), {}, price_resolver=lambda symbol: 10.0
    )

    broker.order_value("NEW", 50.0, "entry")

    assert broker.portfolio.positions["NEW"].quantity == pytest.approx(5.0)


def test_set_price_marks_held_positions_to_market() -> None:
    """Keep position last prices in sync with the configured price."""
    broker = InMemoryBroker(
        Portfolio(cash=0.0, positions={"S": Position("S", 10.0, 10.0, 10.0)}),
        {},
    )

    broker.set_price("S", 12.5)

    assert broker.portfolio.positions["S"].last_price == 12.5


def test_execution_price_changes_fill_without_marking_position() -> None:
    """Keep a fill price separate from the strategy-visible market mark."""
    broker = InMemoryBroker(
        Portfolio(cash=0.0, positions={"S": Position("S", 10.0, 10.0, 12.0)}),
        {"S": 12.0},
    )

    broker.set_execution_price("S", 9.0)
    assert broker.portfolio.positions["S"].last_price == 12.0

    broker.order_quantity("S", -1.0, "exit")

    assert broker.portfolio.positions["S"].last_price == 9.0
    assert broker.orders[-1].price == 9.0


def test_clearing_execution_prices_prevents_stale_fills() -> None:
    """Refuse an order rather than reusing a price from an earlier cycle."""
    broker = InMemoryBroker(
        Portfolio(cash=0.0, positions={"S": Position("S", 10.0, 10.0, 12.0)}),
        {"S": 12.0},
    )

    broker.clear_execution_prices()

    with pytest.raises(ValueError, match="missing positive execution price"):
        broker.order_quantity("S", -1.0, "exit")
    assert broker.portfolio.positions["S"].last_price == 12.0


@pytest.mark.parametrize(
    "price", [True, 0.0, -1.0, float("nan"), float("inf"), float("-inf")]
)
def test_execution_price_requires_a_positive_finite_number(price: float) -> None:
    """Reject invalid execution prices before they reach an order."""
    broker = InMemoryBroker(Portfolio(cash=100.0), {})

    with pytest.raises(ValueError, match="positive and finite"):
        broker.set_execution_price("S", price)
