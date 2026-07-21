"""Tests for strategy-engine orchestration."""

from datetime import date, datetime

from trading_script_anatomy.broker.memory import InMemoryBroker
from trading_script_anatomy.config import StrategyConfig
from trading_script_anatomy.engine import StrategyEngine
from trading_script_anatomy.data.models import FinancialSnapshot, SecurityInfo
from trading_script_anatomy.portfolio import Portfolio
from trading_script_anatomy.strategy.state import StrategyState

from tests.fakes import FakeMarketData, FakeUniverse, bars


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
    assert engine.state.stopped_out is False
