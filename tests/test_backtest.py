"""Tests for the backtest driver and its data-visibility rules."""

from datetime import date

import pandas as pd
import pytest

from trading_script_anatomy.backtest.simulator import (
    Backtester,
    DelayedMarketData,
)
from trading_script_anatomy.config import StrategyConfig
from trading_script_anatomy.data.models import (
    FinancialSnapshot,
    RankedSecurity,
    SecurityInfo,
)

from tests.fakes import FakeMarketData, FakeRankedUniverse


class WindowedFakeMarketData(FakeMarketData):
    """A market-data fake whose bars honor the ``as_of`` visibility date."""

    def daily_bars(self, symbol: str, periods: int, as_of: date) -> pd.DataFrame:
        """Return configured bars visible on or before ``as_of``."""
        frame = self.bars.get(symbol)
        if frame is None or frame.empty:
            return pd.DataFrame()
        visible = frame[frame.index <= pd.Timestamp(as_of)]
        return visible.tail(periods).copy()


def flat_bars(dates: pd.DatetimeIndex, price: float) -> pd.DataFrame:
    """Build a constant-price daily-bars frame."""
    return pd.DataFrame({"open": price, "close": price}, index=dates)


def single_symbol_backtester(
    dates: pd.DatetimeIndex,
    opens: list[float],
    closes: list[float],
    *,
    empty_months: tuple[int, ...] = (),
) -> Backtester:
    """Build a deterministic one-stock backtest for lifecycle scenarios."""
    data = WindowedFakeMarketData(
        infos={"S1": SecurityInfo("S1", "One Corp", date(2020, 1, 1))},
        financials={"S1": FinancialSnapshot(200, 100, 5, 5)},
        bars={
            "BENCH": flat_bars(dates, 100.0),
            "SAFE": flat_bars(dates, 1.0),
            "S1": pd.DataFrame({"open": opens, "close": closes}, index=dates),
        },
    )
    config = StrategyConfig(
        benchmark_symbol="BENCH",
        safe_etf_symbol="SAFE",
        initial_stock_count=1,
        min_market_value=100,
        max_market_value=1_000,
        min_operating_revenue=10,
        highest_price=1_000,
        empty_months=empty_months,
        market_stop_loss_threshold=1.0,
    )
    return Backtester(
        config,
        data,
        FakeRankedUniverse([RankedSecurity("S1", 200.0)]),
        initial_cash=100.0,
        eligibility=lambda symbol, info: True,
    )


def test_delayed_market_data_hides_the_current_day() -> None:
    """Shift strategy bar requests to end strictly before the request day."""

    class RecordingProvider:
        def __init__(self) -> None:
            self.as_of_seen: list[date] = []

        def daily_bars(self, symbol: str, periods: int, as_of: date) -> pd.DataFrame:
            self.as_of_seen.append(as_of)
            return pd.DataFrame()

    inner = RecordingProvider()
    delayed = DelayedMarketData(inner)  # type: ignore[arg-type]

    delayed.daily_bars("BENCH", 1, date(2026, 7, 7))

    assert inner.as_of_seen == [date(2026, 7, 6)]


def test_backtester_runs_flat_market_end_to_end() -> None:
    """Rebalance on Tuesdays and preserve equity in a frictionless flat market."""
    dates = pd.bdate_range("2026-07-06", "2026-07-17")
    listing_date = date(2020, 1, 1)
    data = WindowedFakeMarketData(
        infos={
            "S1": SecurityInfo("S1", "One Corp", listing_date),
            "S2": SecurityInfo("S2", "Two Corp", listing_date),
        },
        financials={
            "S1": FinancialSnapshot(200, 100, 5, 5),
            "S2": FinancialSnapshot(300, 100, 5, 5),
        },
        bars={
            "BENCH": flat_bars(dates, 100.0),
            "S1": flat_bars(dates, 10.0),
            "S2": flat_bars(dates, 10.0),
        },
    )
    universe = FakeRankedUniverse(
        [RankedSecurity("S1", 200.0), RankedSecurity("S2", 300.0)]
    )
    config = StrategyConfig(
        benchmark_symbol="BENCH",
        safe_etf_symbol="SAFE",
        initial_stock_count=2,
        min_market_value=100,
        max_market_value=1_000,
        min_operating_revenue=10,
        empty_months=(),
    )
    backtester = Backtester(
        config,
        data,
        universe,
        eligibility=lambda symbol, info: True,
    )

    result = backtester.run(date(2026, 7, 6), date(2026, 7, 17))

    assert len(result.equity_curve) == len(dates)
    assert result.equity_curve.tolist() == [100_000.0] * len(dates)
    assert result.total_return == 0.0
    assert result.max_drawdown == 0.0
    assert [order.reason for order in result.orders] == [
        "rebalance_buy",
        "rebalance_buy",
    ]
    first_tuesday = pd.Timestamp("2026-07-07")
    assert result.benchmark_curve.loc[first_tuesday] == 100.0
    assert "Orders executed:   2" in result.summary()


def test_backtester_rejects_empty_windows() -> None:
    """Fail loudly when the benchmark has no bars in the window."""
    data = WindowedFakeMarketData()
    backtester = Backtester(
        StrategyConfig(benchmark_symbol="BENCH"),
        data,
        FakeRankedUniverse([]),
    )

    try:
        backtester.run(date(2026, 7, 6), date(2026, 7, 17))
    except ValueError as error:
        assert "no benchmark trading days" in str(error)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for an empty window")


@pytest.mark.parametrize(
    ("opens", "closes", "reason", "expected_fill"),
    [
        ([100.0, 100.0, 80.0, 70.0], [100.0, 100.0, 80.0, 70.0], "stop_loss", 70.0),
        (
            [100.0, 100.0, 200.0, 190.0],
            [100.0, 100.0, 210.0, 190.0],
            "stop_profit",
            190.0,
        ),
    ],
)
def test_position_exits_use_previous_close_and_fill_at_next_open(
    opens: list[float],
    closes: list[float],
    reason: str,
    expected_fill: float,
) -> None:
    """Keep day-D opens hidden from stop decisions while using them for fills."""
    dates = pd.bdate_range("2026-07-06", "2026-07-09")
    backtester = single_symbol_backtester(dates, opens, closes)

    result = backtester.run(date(2026, 7, 6), date(2026, 7, 9))

    exit_orders = [order for order in result.orders if order.reason == reason]
    assert len(exit_orders) == 1
    assert exit_orders[0].price == expected_fill


def test_missing_open_does_not_reuse_a_previous_execution_price() -> None:
    """Delay an exit when the current session has no valid opening price."""
    dates = pd.bdate_range("2026-07-06", "2026-07-10")
    backtester = single_symbol_backtester(
        dates,
        [100.0, 100.0, float("nan"), 70.0, 70.0],
        [100.0, 80.0, 80.0, 70.0, 70.0],
    )

    result = backtester.run(date(2026, 7, 6), date(2026, 7, 10))

    stop_orders = [order for order in result.orders if order.reason == "stop_loss"]
    assert len(stop_orders) == 1
    assert stop_orders[0].price == 70.0


def test_empty_month_rebalance_parks_sale_proceeds_immediately() -> None:
    """Sell equities before buying the safe ETF on an empty-month Tuesday."""
    dates = pd.bdate_range("2025-03-24", "2025-04-02")
    backtester = single_symbol_backtester(
        dates,
        [10.0] * len(dates),
        [10.0] * len(dates),
        empty_months=(4,),
    )

    result = backtester.run(date(2025, 3, 24), date(2025, 4, 2))

    assert [order.reason for order in result.orders] == [
        "rebalance_buy",
        "empty_month_clear",
        "buy_safe_etf",
    ]
    assert result.equity_curve.iloc[-1] == 100.0
