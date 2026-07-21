"""Tests for the backtest driver and its data-visibility rules."""

from datetime import date

import pandas as pd

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
