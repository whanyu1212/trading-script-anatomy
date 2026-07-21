"""Tests for the deterministic broker implementation."""

import pytest

from trading_script_anatomy.broker.memory import InMemoryBroker
from trading_script_anatomy.broker.models import (
    BrokerExecutionError,
    CostModel,
    Order,
    OrderOutcome,
    OrderOutcomeStatus,
    OrderSide,
)
from trading_script_anatomy.portfolio import Portfolio, Position


def test_in_memory_broker_updates_cash_positions_and_order_history() -> None:
    """Fill orders at supplied prices and preserve resulting state."""
    broker = InMemoryBroker(Portfolio(cash=100.0), {"000001.SZ": 10.0})

    buy = broker.order_value("000001.SZ", 50.0, "entry")
    sell = broker.order_quantity("000001.SZ", -2.0, "trim")

    position = broker.portfolio.positions["000001.SZ"]
    assert broker.portfolio.cash == 70.0
    assert position.quantity == 3.0
    assert [order.side for order in broker.orders] == [
        OrderSide.BUY,
        OrderSide.SELL,
    ]
    assert buy.status is OrderOutcomeStatus.FILLED
    assert buy.fill is broker.orders[0]
    assert sell.fill is broker.orders[1]


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

    with pytest.raises(BrokerExecutionError):
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

    with pytest.raises(
        BrokerExecutionError, match="missing positive execution price"
    ):
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


def test_zero_quantity_returns_an_explicit_skipped_outcome() -> None:
    """Distinguish a deliberate no-op from a submitted or filled order."""
    broker = InMemoryBroker(Portfolio(cash=100.0), {})

    outcome = broker.order_quantity("S", 0.0, "no_op")

    assert outcome == OrderOutcome(OrderOutcomeStatus.SKIPPED)
    assert broker.orders == []


@pytest.mark.parametrize("quantity", [True, float("nan"), float("inf")])
def test_order_quantity_rejects_non_finite_or_boolean_values(
    quantity: float,
) -> None:
    """Prevent malformed quantities from corrupting portfolio arithmetic."""
    broker = InMemoryBroker(Portfolio(cash=100.0), {"S": 10.0})

    with pytest.raises(ValueError, match="quantity must be finite"):
        broker.order_quantity("S", quantity, "invalid")


@pytest.mark.parametrize("value", [True, float("nan"), float("inf")])
def test_order_value_rejects_non_finite_or_boolean_values(value: float) -> None:
    """Prevent malformed notionals from reaching execution arithmetic."""
    broker = InMemoryBroker(Portfolio(cash=100.0), {"S": 10.0})

    with pytest.raises(ValueError, match="positive and finite"):
        broker.order_value("S", value, "invalid")


def test_order_outcome_rejects_contradictory_states() -> None:
    """Keep the shared broker result model internally coherent."""
    fill = Order("S", 1.0, OrderSide.BUY, 10.0, "entry")

    with pytest.raises(ValueError, match="requires a confirmed fill"):
        OrderOutcome(OrderOutcomeStatus.FILLED)
    with pytest.raises(ValueError, match="requires a broker order id"):
        OrderOutcome(OrderOutcomeStatus.WORKING)
    with pytest.raises(ValueError, match="requires an order reference"):
        OrderOutcome(OrderOutcomeStatus.UNKNOWN)
    with pytest.raises(ValueError, match="cannot contain"):
        OrderOutcome(OrderOutcomeStatus.SKIPPED, fill=fill)
