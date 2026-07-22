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


def test_sub_cent_safe_etf_cash_is_skipped_without_broker_call() -> None:
    """Treat defensive cash dust as retryable instead of raising validation."""
    broker = ScriptedBroker(Portfolio(cash=0.001))
    engine = StrategyEngine(
        StrategyConfig(), FakeMarketData(), FakeUniverse(()), broker
    )
    engine.state.stopped_out = True

    engine.handle_stop_loss_funds(datetime(2025, 6, 3, 14, 0))

    assert broker.value_calls == []
    assert engine.state.stopped_out is True
    assert engine.state.stop_loss_etf_bought is False


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
        reconcile_results=(
            OrderOutcome(OrderOutcomeStatus.WORKING, order_id="sell-1"),
        ),
    )
    engine = StrategyEngine(
        config, FakeMarketData(), FakeUniverse(()), broker
    )

    engine.weekly_rebalance(date(2025, 4, 1))
    engine.handle_empty_month_clear(date(2025, 4, 1))

    assert len(broker.quantity_calls) == 1
    assert broker.reconcile_calls == ["sell-1"]
    assert broker.value_calls == []


def test_empty_month_etf_does_not_duplicate_pending_weekly_buy() -> None:
    """Reconcile a defensive weekly purchase before considering a retry."""
    config = StrategyConfig(empty_months=(4,))
    working = OrderOutcome(OrderOutcomeStatus.WORKING, order_id="safe-buy-1")
    broker = ScriptedBroker(
        Portfolio(cash=100.0),
        value_results=(working,),
        reconcile_results=(working,),
    )
    engine = StrategyEngine(
        config, FakeMarketData(), FakeUniverse(()), broker
    )
    as_of = date(2025, 4, 1)

    engine.handle_empty_month_etf(as_of)
    engine.handle_empty_month_etf(as_of)

    assert broker.value_calls == [
        (config.safe_etf_symbol, 100.0, "buy_safe_etf")
    ]
    assert broker.reconcile_calls == ["safe-buy-1"]
    assert engine.state.pending_weekly_buy_orders == {
        config.safe_etf_symbol: "safe-buy-1"
    }


@pytest.mark.parametrize(
    "outcome",
    [
        OrderOutcome(OrderOutcomeStatus.WORKING, order_id="buy-1"),
        OrderOutcome(OrderOutcomeStatus.UNKNOWN, client_order_id="buy-1"),
        OrderOutcome(
            OrderOutcomeStatus.PARTIAL,
            order_id="buy-1",
            fill=Order(
                "000001.SZ", 2.0, OrderSide.BUY, 10.0, "rebalance_buy"
            ),
        ),
    ],
)
def test_uncertain_rebalance_purchase_stops_later_targets(
    outcome: OrderOutcome,
) -> None:
    """Stop dependent buys when the first purchase may have consumed cash."""
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
        value_results=(outcome,),
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


def test_terminal_partial_rebalance_buy_tops_up_unfilled_value() -> None:
    """Finish a terminal partial buy before allocating the next target."""
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
    partial_fill = Order(
        symbols[0], 2.0, OrderSide.BUY, 10.0, "rebalance_buy"
    )
    top_up_fill = Order(
        symbols[0], 3.0, OrderSide.BUY, 10.0, "rebalance_buy"
    )
    second_fill = Order(
        symbols[1], 5.0, OrderSide.BUY, 10.0, "rebalance_buy"
    )
    portfolio = Portfolio(cash=100.0)
    broker = ScriptedBroker(
        portfolio,
        value_results=(
            OrderOutcome(
                OrderOutcomeStatus.PARTIAL,
                order_id="buy-1",
                fill=partial_fill,
            ),
            OrderOutcome(OrderOutcomeStatus.FILLED, fill=top_up_fill),
            OrderOutcome(OrderOutcomeStatus.FILLED, fill=second_fill),
        ),
    )
    engine = StrategyEngine(
        StrategyConfig(initial_stock_count=2, finance_candidate_multiplier=1),
        data,
        FakeUniverse(symbols),
        broker,
    )

    engine.weekly_rebalance(as_of)

    assert broker.value_calls == [(symbols[0], 50.0, "rebalance_buy")]
    assert engine.state.weekly_buy_remaining_values == {symbols[0]: 30.0}
    assert engine.state.last_rebalance_date is None

    portfolio.cash = 80.0
    portfolio.positions[symbols[0]] = Position(
        symbols[0], 2.0, 10.0, 10.0
    )
    engine.weekly_rebalance(as_of)

    assert broker.value_calls == [
        (symbols[0], 50.0, "rebalance_buy"),
        (symbols[0], 30.0, "rebalance_buy"),
        (symbols[1], 50.0, "rebalance_buy"),
    ]
    assert engine.state.weekly_buy_remaining_values == {}
    assert engine.state.last_rebalance_date == as_of


def test_sub_cent_rebalance_allocation_is_retained_without_broker_call() -> None:
    """Leave an unaffordable target incomplete without aborting rebalance."""
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
    broker = ScriptedBroker(Portfolio(cash=0.01))
    engine = StrategyEngine(
        StrategyConfig(initial_stock_count=2, finance_candidate_multiplier=1),
        data,
        FakeUniverse(symbols),
        broker,
    )

    engine.weekly_rebalance(as_of)

    assert broker.value_calls == []
    assert engine.state.weekly_buy_remaining_values == {symbols[0]: 0.005}
    assert engine.state.last_rebalance_date is None


def test_failed_rebalance_purchase_does_not_block_later_targets() -> None:
    """Continue independent purchases after a confirmed zero-fill failure."""
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
    fill = Order(symbols[1], 5.0, OrderSide.BUY, 10.0, "rebalance_buy")
    broker = ScriptedBroker(
        Portfolio(cash=100.0),
        value_results=(
            OrderOutcome(OrderOutcomeStatus.FAILED, order_id="buy-1"),
            OrderOutcome(OrderOutcomeStatus.FILLED, fill=fill),
        ),
    )
    engine = StrategyEngine(
        StrategyConfig(initial_stock_count=2, finance_candidate_multiplier=1),
        data,
        FakeUniverse(symbols),
        broker,
    )

    engine.weekly_rebalance(as_of)

    assert broker.value_calls == [
        (symbols[0], 50.0, "rebalance_buy"),
        (symbols[1], 50.0, "rebalance_buy"),
    ]
    assert engine.state.last_rebalance_date is None


def test_weekly_rebalance_reconciles_pending_buy_before_later_targets() -> None:
    """Do not submit another buy while an earlier purchase may consume cash."""
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
    partial_fill = Order(
        symbols[0], 2.0, OrderSide.BUY, 10.0, "rebalance_buy"
    )
    completed_fill = Order(
        symbols[0], 5.0, OrderSide.BUY, 10.0, "rebalance_buy"
    )
    second_fill = Order(
        symbols[1], 5.0, OrderSide.BUY, 10.0, "rebalance_buy"
    )
    working = OrderOutcome(
        OrderOutcomeStatus.WORKING,
        order_id="buy-1",
        fill=partial_fill,
    )
    portfolio = Portfolio(cash=100.0)
    broker = ScriptedBroker(
        portfolio,
        value_results=(
            working,
            OrderOutcome(OrderOutcomeStatus.FILLED, fill=second_fill),
        ),
        reconcile_results=(
            working,
            OrderOutcome(
                OrderOutcomeStatus.FILLED,
                order_id="buy-1",
                fill=completed_fill,
            ),
        ),
    )
    engine = StrategyEngine(
        StrategyConfig(initial_stock_count=2, finance_candidate_multiplier=1),
        data,
        FakeUniverse(symbols),
        broker,
    )

    engine.weekly_rebalance(as_of)
    engine.weekly_rebalance(as_of)

    assert broker.value_calls == [(symbols[0], 50.0, "rebalance_buy")]
    assert broker.reconcile_calls == ["buy-1"]
    assert engine.state.pending_weekly_buy_orders == {symbols[0]: "buy-1"}

    portfolio.cash = 50.0
    portfolio.positions[symbols[0]] = Position(
        symbols[0], 5.0, 10.0, 10.0
    )
    engine.weekly_rebalance(as_of)

    assert broker.value_calls == [
        (symbols[0], 50.0, "rebalance_buy"),
        (symbols[1], 50.0, "rebalance_buy"),
    ]
    assert broker.reconcile_calls == ["buy-1", "buy-1"]
    assert engine.state.pending_weekly_buy_orders == {}
    assert engine.state.last_rebalance_date == as_of


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


def test_weekly_rebalance_reconciles_safe_etf_sale_before_buying() -> None:
    """Resume a weekly rebalance without duplicating an unresolved ETF sale."""
    as_of = date(2025, 6, 3)
    symbol = "000001.SZ"
    config = StrategyConfig(
        initial_stock_count=1, finance_candidate_multiplier=1
    )
    data = FakeMarketData(
        infos={symbol: SecurityInfo(symbol, symbol, date(2023, 1, 1))},
        financials={symbol: FinancialSnapshot(1.5e9, 2e8, 1e7, 1e7)},
        bars={
            config.benchmark_symbol: bars([100.0] * 10),
            symbol: bars([10.0]),
        },
    )
    sale_fill = Order(
        config.safe_etf_symbol, 5.0, OrderSide.SELL, 1.0, "sell_safe_etf"
    )
    buy_fill = Order(symbol, 10.0, OrderSide.BUY, 10.0, "rebalance_buy")
    portfolio = Portfolio(
        cash=20.0,
        positions={
            config.safe_etf_symbol: Position(
                config.safe_etf_symbol, 5.0, 1.0, 1.0
            )
        },
    )
    broker = ScriptedBroker(
        portfolio,
        quantity_results=(
            OrderOutcome(OrderOutcomeStatus.WORKING, order_id="sell-safe-1"),
        ),
        value_results=(
            OrderOutcome(OrderOutcomeStatus.FILLED, fill=buy_fill),
        ),
        reconcile_results=(
            OrderOutcome(
                OrderOutcomeStatus.FILLED,
                order_id="sell-safe-1",
                fill=sale_fill,
            ),
        ),
    )
    engine = StrategyEngine(
        config, data, FakeUniverse((symbol,)), broker
    )

    engine.weekly_rebalance(as_of)

    assert engine.state.pending_weekly_sale_orders == {
        config.safe_etf_symbol: "sell-safe-1"
    }
    assert broker.value_calls == []

    portfolio.positions.clear()
    portfolio.cash = 100.0
    engine.weekly_rebalance(as_of)

    assert broker.quantity_calls == [
        (config.safe_etf_symbol, -5.0, "sell_safe_etf")
    ]
    assert broker.reconcile_calls == ["sell-safe-1"]
    assert broker.value_calls == [(symbol, 100.0, "rebalance_buy")]
    assert engine.state.pending_weekly_sale_orders == {}
    assert engine.state.last_rebalance_date == as_of


def test_weekly_rebalance_reconciles_non_target_sale_before_buying() -> None:
    """Do not duplicate a non-target sale while its outcome remains open."""
    as_of = date(2025, 6, 3)
    old_symbol = "000001.SZ"
    target_symbol = "000002.SZ"
    config = StrategyConfig(
        initial_stock_count=1, finance_candidate_multiplier=1
    )
    data = FakeMarketData(
        infos={
            target_symbol: SecurityInfo(
                target_symbol, target_symbol, date(2023, 1, 1)
            )
        },
        financials={
            target_symbol: FinancialSnapshot(1.5e9, 2e8, 1e7, 1e7)
        },
        bars={
            config.benchmark_symbol: bars([100.0] * 10),
            target_symbol: bars([10.0]),
        },
    )
    portfolio = Portfolio(
        cash=0.0,
        positions={
            old_symbol: Position(old_symbol, 10.0, 10.0, 10.0),
        },
    )
    broker = ScriptedBroker(
        portfolio,
        quantity_results=(
            OrderOutcome(OrderOutcomeStatus.WORKING, order_id="sell-old-1"),
        ),
        value_results=(
            OrderOutcome(
                OrderOutcomeStatus.FILLED,
                fill=Order(
                    target_symbol,
                    10.0,
                    OrderSide.BUY,
                    10.0,
                    "rebalance_buy",
                ),
            ),
        ),
        reconcile_results=(
            OrderOutcome(OrderOutcomeStatus.WORKING, order_id="sell-old-1"),
            OrderOutcome(
                OrderOutcomeStatus.FILLED,
                order_id="sell-old-1",
                fill=Order(
                    old_symbol,
                    10.0,
                    OrderSide.SELL,
                    10.0,
                    "rebalance_sell",
                ),
            ),
        ),
    )
    engine = StrategyEngine(
        config, data, FakeUniverse((target_symbol,)), broker
    )

    engine.weekly_rebalance(as_of)
    engine.weekly_rebalance(as_of)

    assert broker.quantity_calls == [
        (old_symbol, -10.0, "rebalance_sell")
    ]
    assert broker.value_calls == []
    assert engine.state.pending_weekly_sale_orders == {
        old_symbol: "sell-old-1"
    }

    portfolio.positions.clear()
    portfolio.cash = 100.0
    engine.weekly_rebalance(as_of)

    assert broker.quantity_calls == [
        (old_symbol, -10.0, "rebalance_sell")
    ]
    assert broker.reconcile_calls == ["sell-old-1", "sell-old-1"]
    assert broker.value_calls == [(target_symbol, 100.0, "rebalance_buy")]
    assert engine.state.pending_weekly_sale_orders == {}
    assert engine.state.last_rebalance_date == as_of


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
    assert engine.state.incomplete_exit_reasons == {symbol: "stop_loss"}
    assert engine.state.stopped_out is False


def test_risk_check_excludes_symbol_with_pending_weekly_sale() -> None:
    """Do not submit a risk exit against an unresolved weekly liquidation."""
    symbol = "000001.SZ"
    broker = ScriptedBroker(
        Portfolio(
            cash=0.0,
            positions={symbol: Position(symbol, 10.0, 10.0, 8.0)},
        ),
        reconcile_results=(
            OrderOutcome(OrderOutcomeStatus.WORKING, order_id="weekly-sell-1"),
        ),
    )
    state = StrategyState(
        stock_count=1,
        pending_weekly_sale_orders={symbol: "weekly-sell-1"},
    )
    engine = StrategyEngine(
        StrategyConfig(),
        FakeMarketData(),
        FakeUniverse(()),
        broker,
        state=state,
    )

    engine.risk_check(date(2025, 6, 3))

    assert broker.reconcile_calls == ["weekly-sell-1"]
    assert broker.quantity_calls == []
    assert state.pending_weekly_sale_orders == {symbol: "weekly-sell-1"}


def test_risk_check_excludes_symbol_with_pending_weekly_buy() -> None:
    """Do not sell a partial position while its weekly buy remains live."""
    symbol = "000001.SZ"
    partial_fill = Order(
        symbol, 2.0, OrderSide.BUY, 10.0, "rebalance_buy"
    )
    broker = ScriptedBroker(
        Portfolio(
            cash=80.0,
            positions={symbol: Position(symbol, 2.0, 10.0, 8.0)},
        ),
        reconcile_results=(
            OrderOutcome(
                OrderOutcomeStatus.WORKING,
                order_id="weekly-buy-1",
                fill=partial_fill,
            ),
        ),
    )
    state = StrategyState(
        stock_count=1,
        pending_weekly_buy_orders={symbol: "weekly-buy-1"},
    )
    engine = StrategyEngine(
        StrategyConfig(),
        FakeMarketData(),
        FakeUniverse(()),
        broker,
        state=state,
    )

    engine.risk_check(date(2025, 6, 3))

    assert broker.reconcile_calls == ["weekly-buy-1"]
    assert broker.quantity_calls == []
    assert state.pending_weekly_buy_orders == {symbol: "weekly-buy-1"}


def test_weekly_rebalance_waits_for_pending_risk_exit() -> None:
    """Do not start weekly liquidation while a risk sale remains unresolved."""
    symbol = "000001.SZ"
    broker = ScriptedBroker(
        Portfolio(
            cash=0.0,
            positions={symbol: Position(symbol, 10.0, 10.0, 8.0)},
        ),
        reconcile_results=(
            OrderOutcome(OrderOutcomeStatus.WORKING, order_id="risk-sell-1"),
        ),
    )
    state = StrategyState(
        stock_count=1,
        pending_exit_orders={symbol: "risk-sell-1"},
        incomplete_exit_reasons={symbol: "stop_loss"},
    )
    engine = StrategyEngine(
        StrategyConfig(),
        FakeMarketData(),
        FakeUniverse(()),
        broker,
        state=state,
    )

    engine.weekly_rebalance(date(2025, 6, 3))

    assert broker.reconcile_calls == ["risk-sell-1"]
    assert broker.quantity_calls == []
    assert broker.value_calls == []
    assert state.pending_exit_orders == {symbol: "risk-sell-1"}


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
    assert broker.quantity_calls == [(symbol, -10.0, "risk_exit_retry")]
    assert state.pending_exit_orders == {symbol: "sell-2"}
    assert state.incomplete_exit_reasons == {symbol: "risk_exit_retry"}


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


@pytest.mark.parametrize(
    "terminal_outcome",
    [
        OrderOutcome(OrderOutcomeStatus.FAILED, order_id="sell-1"),
        OrderOutcome(
            OrderOutcomeStatus.PARTIAL,
            order_id="sell-1",
            fill=Order(
                "000001.SZ", 5.0, OrderSide.SELL, 8.0, "stop_loss"
            ),
        ),
    ],
)
def test_incomplete_pending_exit_blocks_defensive_buy_until_risk_retry(
    terminal_outcome: OrderOutcome,
) -> None:
    """Keep held equity as a barrier after its pending sale terminates."""
    symbol = "000001.SZ"
    broker = ScriptedBroker(
        Portfolio(
            cash=90.0,
            positions={symbol: Position(symbol, 10.0, 10.0, 8.0)},
        ),
        quantity_results=(
            OrderOutcome(OrderOutcomeStatus.WORKING, order_id="sell-2"),
        ),
        reconcile_results=(terminal_outcome,),
    )
    state = StrategyState(
        stock_count=1,
        stopped_out=True,
        pending_exit_orders={symbol: "sell-1"},
        incomplete_exit_reasons={symbol: "stop_loss"},
    )
    engine = StrategyEngine(
        StrategyConfig(),
        FakeMarketData(),
        FakeUniverse(()),
        broker,
        state=state,
    )

    engine.handle_stop_loss_funds(datetime(2025, 6, 3, 14, 0))

    assert broker.value_calls == []
    assert state.pending_exit_orders == {}
    assert state.incomplete_exit_reasons == {symbol: "stop_loss"}

    engine.risk_check(date(2025, 6, 3))

    assert broker.quantity_calls == [(symbol, -10.0, "stop_loss")]
    assert state.pending_exit_orders == {symbol: "sell-2"}
    assert state.incomplete_exit_reasons == {symbol: "stop_loss"}


def test_daily_reset_preserves_unresolved_order_references() -> None:
    """Require reconciliation instead of permitting next-day duplicates."""
    state = StrategyState(
        stock_count=1,
        stop_loss_etf_order_reference="safe-1",
        pending_exit_orders={"000001.SZ": "sell-1"},
        incomplete_exit_reasons={"000001.SZ": "stop_loss"},
        pending_weekly_sale_orders={"511880.SS": "weekly-1"},
        pending_weekly_buy_orders={"000002.SZ": "weekly-buy-1"},
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
    assert state.incomplete_exit_reasons == {"000001.SZ": "stop_loss"}
    assert state.pending_weekly_sale_orders == {"511880.SS": "weekly-1"}
    assert state.pending_weekly_buy_orders == {"000002.SZ": "weekly-buy-1"}
