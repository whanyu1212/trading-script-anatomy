"""Tests for strategy-engine orchestration."""

from datetime import date, datetime

import pytest

from trading_script_anatomy.broker.memory import InMemoryBroker
from trading_script_anatomy.broker.models import (
    BrokerExecutionError,
    Order,
    OrderOutcome,
    OrderOutcomeStatus,
    OrderSide,
)
from trading_script_anatomy.config import StrategyConfig
from trading_script_anatomy.engine import StrategyEngine
from trading_script_anatomy.data.models import FinancialSnapshot, SecurityInfo
from trading_script_anatomy.portfolio import Portfolio, Position
from trading_script_anatomy.strategy.state import StrategyState

from tests.fakes import FakeMarketData, FakeUniverse, bars


class ScriptedBroker:
    """Return caller-supplied outcomes without mutating its portfolio."""

    def __init__(
        self,
        portfolio: Portfolio,
        *,
        quantity_results: tuple[OrderOutcome | BaseException, ...] = (),
        value_results: tuple[OrderOutcome | BaseException, ...] = (),
        reconcile_results: tuple[OrderOutcome | BaseException, ...] = (),
    ) -> None:
        self.portfolio = portfolio
        self._quantity_results = list(quantity_results)
        self._value_results = list(value_results)
        self._reconcile_results = list(reconcile_results)
        self.quantity_calls: list[tuple[str, float, str]] = []
        self.value_calls: list[tuple[str, float, str]] = []
        self.reconcile_calls: list[str] = []

    def positions(self) -> tuple[Position, ...]:
        """Return the configured portfolio positions."""
        return tuple(self.portfolio.positions.values())

    def order_quantity(
        self, symbol: str, quantity: float, reason: str
    ) -> OrderOutcome:
        """Return the next scripted quantity-order response."""
        self.quantity_calls.append((symbol, quantity, reason))
        return self._next(self._quantity_results)

    def order_value(self, symbol: str, value: float, reason: str) -> OrderOutcome:
        """Return the next scripted value-order response."""
        self.value_calls.append((symbol, value, reason))
        return self._next(self._value_results)

    def reconcile_order(self, reference: str) -> OrderOutcome:
        """Return the next scripted reconciliation response."""
        self.reconcile_calls.append(reference)
        return self._next(self._reconcile_results)

    @staticmethod
    def _next(results: list[OrderOutcome | BaseException]) -> OrderOutcome:
        if not results:
            raise AssertionError("unexpected broker order")
        result = results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def test_weekly_rebalance_buys_selected_positions_equally() -> None:
    """Invest available cash equally across the selected target holdings."""
    as_of = date(2025, 6, 3)
    symbols = ("000001.SZ", "000002.SZ")
    data = FakeMarketData(
        infos={
            symbol: SecurityInfo(symbol, symbol, date(2023, 1, 1))
            for symbol in symbols
        },
        financials={
            "000001.SZ": FinancialSnapshot(1.5e9, 2e8, 1e7, 1e7),
            "000002.SZ": FinancialSnapshot(2e9, 2e8, 1e7, 1e7),
        },
        bars={
            "399106.SZ": bars([100.0] * 10),
            "000001.SZ": bars([10.0]),
            "000002.SZ": bars([10.0]),
        },
    )
    broker = InMemoryBroker(
        Portfolio(cash=1_000.0),
        {"000001.SZ": 10.0, "000002.SZ": 10.0, "511880.SS": 1.0},
    )
    engine = StrategyEngine(
        StrategyConfig(initial_stock_count=2, finance_candidate_multiplier=1),
        data,
        FakeUniverse(symbols),
        broker,
    )

    engine.weekly_rebalance(as_of)

    assert broker.portfolio.cash == 0.0
    assert {
        symbol: position.quantity
        for symbol, position in broker.portfolio.positions.items()
    } == {"000001.SZ": 50.0, "000002.SZ": 50.0}
    assert engine.state.last_rebalance_date == as_of


def test_engine_uses_supplied_persisted_state() -> None:
    """Resume with the exact state object supplied by the caller."""
    state = StrategyState(
        stock_count=6,
        candidates=["000001.SZ"],
        last_rebalance_date=date(2025, 5, 27),
    )

    engine = StrategyEngine(
        StrategyConfig(initial_stock_count=2),
        FakeMarketData(),
        FakeUniverse(()),
        InMemoryBroker(Portfolio(cash=100.0), {}),
        state=state,
    )

    assert engine.state is state
    assert engine.state.stock_count == 6
    assert engine.state.candidates == ["000001.SZ"]


def test_stop_loss_funds_move_to_safe_etf_after_two_pm() -> None:
    """Invest defensive cash once after a strategy risk event."""
    broker = InMemoryBroker(Portfolio(cash=100.0), {"511880.SS": 1.0})
    engine = StrategyEngine(
        StrategyConfig(),
        FakeMarketData(),
        FakeUniverse(()),
        broker,
    )
    engine.state.stopped_out = True

    engine.handle_stop_loss_funds(datetime(2025, 6, 3, 14, 0))

    assert broker.portfolio.cash == 0.0
    assert broker.portfolio.positions["511880.SS"].quantity == 100.0
    assert engine.state.stop_loss_etf_bought is True
    assert engine.state.stop_loss_etf_order_reference is None
    assert engine.state.stopped_out is False


def test_empty_month_rebalance_sells_equities_before_buying_safe_etf() -> None:
    """Move invested capital directly into the defensive asset."""
    config = StrategyConfig(empty_months=(4,))
    broker = InMemoryBroker(
        Portfolio(
            cash=20.0,
            positions={
                "000001.SZ": Position("000001.SZ", 10.0, 10.0, 10.0),
                config.safe_etf_symbol: Position(
                    config.safe_etf_symbol, 5.0, 1.0, 1.0
                ),
            },
        ),
        {"000001.SZ": 10.0, config.safe_etf_symbol: 1.0},
    )
    engine = StrategyEngine(
        config,
        FakeMarketData(),
        FakeUniverse(()),
        broker,
    )

    engine.weekly_rebalance(date(2025, 4, 1))

    assert [order.reason for order in broker.orders] == [
        "empty_month_clear",
        "buy_safe_etf",
    ]
    assert broker.portfolio.cash == 0.0
    assert set(broker.portfolio.positions) == {config.safe_etf_symbol}
    assert broker.portfolio.positions[config.safe_etf_symbol].quantity == 125.0


@pytest.mark.parametrize(
    "outcome",
    [
        OrderOutcome(OrderOutcomeStatus.WORKING, order_id="oid-1"),
        OrderOutcome(
            OrderOutcomeStatus.UNKNOWN,
            client_order_id="client-1",
        ),
    ],
)
def test_pending_stop_loss_etf_order_is_not_submitted_twice(
    outcome: OrderOutcome,
) -> None:
    """Track unresolved defensive orders separately from confirmed fills."""
    broker = ScriptedBroker(
        Portfolio(cash=100.0),
        value_results=(outcome,),
        reconcile_results=(outcome,),
    )
    engine = StrategyEngine(
        StrategyConfig(), FakeMarketData(), FakeUniverse(()), broker
    )
    engine.state.stopped_out = True
    now = datetime(2025, 6, 3, 14, 0)

    engine.handle_stop_loss_funds(now)
    engine.handle_stop_loss_funds(now)

    assert len(broker.value_calls) == 1
    assert engine.state.stop_loss_etf_bought is False
    assert engine.state.stop_loss_etf_order_reference == outcome.reference
    assert engine.state.stopped_out is False


def test_failed_stop_loss_etf_order_leaves_state_retryable() -> None:
    """Do not claim that defensive cash moved when execution failed."""
    broker = ScriptedBroker(
        Portfolio(cash=100.0),
        value_results=(BrokerExecutionError("rejected"),),
    )
    engine = StrategyEngine(
        StrategyConfig(), FakeMarketData(), FakeUniverse(()), broker
    )
    engine.state.stopped_out = True

    engine.handle_stop_loss_funds(datetime(2025, 6, 3, 14, 0))

    assert engine.state.stopped_out is True
    assert engine.state.stop_loss_etf_bought is False
    assert engine.state.stop_loss_etf_order_reference is None


def test_engine_does_not_swallow_invalid_broker_calls() -> None:
    """Allow programming and validation defects to escape orchestration."""
    broker = ScriptedBroker(
        Portfolio(cash=100.0),
        value_results=(ValueError("invalid request"),),
    )
    engine = StrategyEngine(
        StrategyConfig(), FakeMarketData(), FakeUniverse(()), broker
    )
    engine.state.stopped_out = True

    with pytest.raises(ValueError, match="invalid request"):
        engine.handle_stop_loss_funds(datetime(2025, 6, 3, 14, 0))


def test_empty_month_waits_for_working_equity_sale_before_safe_etf() -> None:
    """Do not sequence the defensive purchase before liquidation completes."""
    config = StrategyConfig(empty_months=(4,))
    broker = ScriptedBroker(
        Portfolio(
            cash=20.0,
            positions={"000001.SZ": Position("000001.SZ", 10.0, 10.0, 10.0)},
        ),
        quantity_results=(
            OrderOutcome(OrderOutcomeStatus.WORKING, order_id="sell-1"),
        ),
    )
    engine = StrategyEngine(
        config, FakeMarketData(), FakeUniverse(()), broker
    )

    engine.weekly_rebalance(date(2025, 4, 1))

    assert len(broker.quantity_calls) == 1
    assert broker.value_calls == []


def test_working_rebalance_purchase_does_not_mark_rebalance_complete() -> None:
    """Stop dependent buys when the first purchase remains unresolved."""
    as_of = date(2025, 6, 3)
    symbols = ("000001.SZ", "000002.SZ")
    data = FakeMarketData(
        infos={
            symbol: SecurityInfo(symbol, symbol, date(2023, 1, 1))
            for symbol in symbols
        },
        financials={
            symbol: FinancialSnapshot(1.5e9, 2e8, 1e7, 1e7)
            for symbol in symbols
        },
        bars={
            "399106.SZ": bars([100.0] * 10),
            **{symbol: bars([10.0]) for symbol in symbols},
        },
    )
    broker = ScriptedBroker(
        Portfolio(cash=100.0),
        value_results=(
            OrderOutcome(OrderOutcomeStatus.WORKING, order_id="buy-1"),
        ),
    )
    engine = StrategyEngine(
        StrategyConfig(initial_stock_count=2, finance_candidate_multiplier=1),
        data,
        FakeUniverse(symbols),
        broker,
    )

    engine.weekly_rebalance(as_of)

    assert engine.state.target_positions == list(symbols)
    assert len(broker.value_calls) == 1
    assert engine.state.last_rebalance_date is None


def test_working_safe_etf_sale_blocks_a_repurchase() -> None:
    """Do not buy the safe ETF while its prerequisite sale remains open."""
    config = StrategyConfig()
    broker = ScriptedBroker(
        Portfolio(
            cash=20.0,
            positions={
                config.safe_etf_symbol: Position(
                    config.safe_etf_symbol, 5.0, 1.0, 1.0
                )
            },
        ),
        quantity_results=(
            OrderOutcome(OrderOutcomeStatus.WORKING, order_id="sell-safe-1"),
        ),
    )
    engine = StrategyEngine(
        config, FakeMarketData(), FakeUniverse(()), broker
    )

    engine.weekly_rebalance(date(2025, 6, 3))

    assert broker.quantity_calls == [
        (config.safe_etf_symbol, -5.0, "sell_safe_etf")
    ]
    assert broker.value_calls == []


def test_risk_check_does_not_duplicate_a_working_exit() -> None:
    """Submit at most one live sell per symbol across repeated daily checks."""
    symbol = "000001.SZ"
    config = StrategyConfig(market_stop_loss_threshold=0.05)
    broker = ScriptedBroker(
        Portfolio(
            cash=0.0,
            positions={symbol: Position(symbol, 10.0, 10.0, 8.0)},
        ),
        quantity_results=(
            OrderOutcome(OrderOutcomeStatus.WORKING, order_id="sell-1"),
        ),
        reconcile_results=(
            OrderOutcome(OrderOutcomeStatus.WORKING, order_id="sell-1"),
        ),
    )
    data = FakeMarketData(
        bars={config.benchmark_symbol: bars([90.0], [100.0])}
    )
    engine = StrategyEngine(config, data, FakeUniverse(()), broker)
    as_of = date(2025, 6, 3)

    engine.risk_check(as_of)
    engine.risk_check(as_of)

    assert broker.quantity_calls == [(symbol, -10.0, "stop_loss")]
    assert engine.state.pending_exit_orders == {symbol: "sell-1"}
    assert engine.state.stopped_out is False


def test_terminal_pending_exit_is_released_and_retried() -> None:
    """Resume risk protection after an earlier unresolved order expires."""
    symbol = "000001.SZ"
    broker = ScriptedBroker(
        Portfolio(
            cash=0.0,
            positions={symbol: Position(symbol, 10.0, 10.0, 8.0)},
        ),
        quantity_results=(
            OrderOutcome(OrderOutcomeStatus.WORKING, order_id="sell-2"),
        ),
        reconcile_results=(
            OrderOutcome(OrderOutcomeStatus.FAILED, order_id="sell-1"),
        ),
    )
    state = StrategyState(
        stock_count=1,
        pending_exit_orders={symbol: "sell-1"},
    )
    engine = StrategyEngine(
        StrategyConfig(),
        FakeMarketData(),
        FakeUniverse(()),
        broker,
        state=state,
    )

    engine.risk_check(date(2025, 6, 3))

    assert broker.reconcile_calls == ["sell-1"]
    assert broker.quantity_calls == [(symbol, -10.0, "stop_loss")]
    assert state.pending_exit_orders == {symbol: "sell-2"}


def test_defensive_buy_waits_for_every_pending_exit() -> None:
    """Sweep proceeds only after all unresolved risk sales have settled."""
    first_fill = Order("S1", 10.0, OrderSide.SELL, 9.0, "stop_loss")
    second_fill = Order("S2", 10.0, OrderSide.SELL, 9.0, "stop_loss")
    etf_fill = Order("511880.SS", 180.0, OrderSide.BUY, 1.0, "buy_safe_etf")
    broker = ScriptedBroker(
        Portfolio(cash=180.0),
        value_results=(
            OrderOutcome(OrderOutcomeStatus.FILLED, fill=etf_fill),
        ),
        reconcile_results=(
            OrderOutcome(
                OrderOutcomeStatus.FILLED,
                order_id="sell-1",
                fill=first_fill,
            ),
            OrderOutcome(OrderOutcomeStatus.WORKING, order_id="sell-2"),
            OrderOutcome(
                OrderOutcomeStatus.FILLED,
                order_id="sell-2",
                fill=second_fill,
            ),
        ),
    )
    state = StrategyState(
        stock_count=2,
        stopped_out=True,
        pending_exit_orders={"S1": "sell-1", "S2": "sell-2"},
    )
    engine = StrategyEngine(
        StrategyConfig(),
        FakeMarketData(),
        FakeUniverse(()),
        broker,
        state=state,
    )
    now = datetime(2025, 6, 3, 14, 0)

    engine.handle_stop_loss_funds(now)

    assert broker.value_calls == []
    assert state.pending_exit_orders == {"S2": "sell-2"}

    engine.handle_stop_loss_funds(now)

    assert len(broker.value_calls) == 1
    assert state.pending_exit_orders == {}
    assert state.stop_loss_etf_bought is True
    assert state.stopped_out is False


def test_daily_reset_preserves_unresolved_order_references() -> None:
    """Require reconciliation instead of permitting next-day duplicates."""
    state = StrategyState(
        stock_count=1,
        stop_loss_etf_order_reference="safe-1",
        pending_exit_orders={"000001.SZ": "sell-1"},
    )
    engine = StrategyEngine(
        StrategyConfig(),
        FakeMarketData(),
        FakeUniverse(()),
        InMemoryBroker(Portfolio(cash=0.0), {}),
        state=state,
    )

    engine.before_trading_start(date(2025, 6, 4))

    assert state.stop_loss_etf_order_reference == "safe-1"
    assert state.pending_exit_orders == {"000001.SZ": "sell-1"}
