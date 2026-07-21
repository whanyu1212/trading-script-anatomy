"""Daily bar-level backtest driver for the strategy engine.

Simulation conventions:

- Strategy decisions on day D see data through D-1 only, matching how the
  live system behaves against end-of-day data feeds (``DelayedMarketData``).
- Orders fill at day D's open plus configured slippage, approximating the
  strategy's mid-morning execution schedule.
- Equity is marked at day D's close.

Known data limitations that this driver cannot fix: a current-only universe
(such as the FMP screener) introduces survivorship bias, and current-only
market caps put look-ahead into the size ranking. Treat results built on
such sources as plausibility checks, not performance claims.
"""

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
import logging
import math

import pandas as pd

from trading_script_anatomy.broker.memory import InMemoryBroker
from trading_script_anatomy.broker.models import CostModel, Order
from trading_script_anatomy.config import StrategyConfig
from trading_script_anatomy.data.models import (
    FinancialSnapshot,
    ProfitabilitySnapshot,
    SecurityInfo,
)
from trading_script_anatomy.data.protocols import (
    IndexUniverseProvider,
    MarketDataProvider,
)
from trading_script_anatomy.engine import StrategyEngine
from trading_script_anatomy.portfolio import Portfolio
from trading_script_anatomy.strategy.selection import EligibilityFilter
from trading_script_anatomy.values import to_finite_float

TRADING_DAYS_PER_YEAR = 252

US_MICROCAP_COSTS = CostModel(slippage_rate=0.005)
"""Zero-commission execution with a 0.5% half-spread, a deliberately harsh
but defensible friction level for US micro-caps."""

A_SHARE_COSTS = CostModel(
    commission_rate=0.00025,
    min_commission=5.0,
    slippage_rate=0.0003,
    sell_tax_rate=0.001,
)
"""The cost constants of the archived PTrade strategy."""

_AFTERNOON_CHECK = time(14, 30)


class DelayedMarketData:
    """Restrict daily-bar visibility to strictly before the requested day.

    Live operation against an end-of-day feed cannot see day D's bar during
    day D; a backtest querying historical data can. This adapter removes that
    look-ahead by shifting bar requests to end one day earlier. Metadata and
    fundamentals pass through unchanged — fundamentals carry their own
    point-in-time contract.

    Args:
        inner: The provider whose bars are being delayed.
    """

    def __init__(self, inner: MarketDataProvider) -> None:
        self._inner = inner

    def daily_bars(self, symbol: str, periods: int, as_of: date) -> pd.DataFrame:
        """Return bars ending no later than the day before ``as_of``."""
        return self._inner.daily_bars(symbol, periods, as_of - timedelta(days=1))

    def security_info(self, symbol: str) -> SecurityInfo | None:
        """Return the inner provider's security metadata."""
        return self._inner.security_info(symbol)

    def financial_snapshot(
        self, symbol: str, as_of: date
    ) -> FinancialSnapshot | None:
        """Return the inner provider's financial snapshot."""
        return self._inner.financial_snapshot(symbol, as_of)

    def profitability(
        self, symbol: str, as_of: date
    ) -> ProfitabilitySnapshot | None:
        """Return the inner provider's profitability values."""
        return self._inner.profitability(symbol, as_of)


def _total_return(curve: pd.Series) -> float:
    """Return the fractional change from a curve's first to last value."""
    if len(curve) < 2:
        return 0.0
    first, last = curve.iloc[0], curve.iloc[-1]
    if pd.isna(first) or pd.isna(last) or first == 0:
        return 0.0
    return float(last / first - 1)


@dataclass(frozen=True)
class BacktestResult:
    """Outcome of a backtest run.

    Attributes:
        equity_curve: Daily portfolio equity, indexed by bar timestamp.
        benchmark_curve: Daily benchmark closes over the same calendar.
        orders: Every simulated fill, in execution order.
    """

    equity_curve: pd.Series
    benchmark_curve: pd.Series
    orders: tuple[Order, ...] = field(default_factory=tuple)

    @property
    def total_return(self) -> float:
        """Return the portfolio's total return over the run."""
        return _total_return(self.equity_curve)

    @property
    def benchmark_total_return(self) -> float:
        """Return the benchmark's total return over the run."""
        return _total_return(self.benchmark_curve)

    @property
    def cagr(self) -> float:
        """Return the compound annual growth rate of the equity curve."""
        if len(self.equity_curve) < 2 or self.equity_curve.iloc[0] <= 0:
            return 0.0
        days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
        if days <= 0:
            return 0.0
        growth = float(self.equity_curve.iloc[-1] / self.equity_curve.iloc[0])
        return growth ** (365.25 / days) - 1 if growth > 0 else -1.0

    @property
    def max_drawdown(self) -> float:
        """Return the deepest peak-to-trough loss as a negative fraction."""
        if self.equity_curve.empty:
            return 0.0
        drawdowns = self.equity_curve / self.equity_curve.cummax() - 1
        return float(drawdowns.min())

    @property
    def annualized_volatility(self) -> float:
        """Return annualized daily-return volatility."""
        returns = self.equity_curve.pct_change().dropna()
        if len(returns) < 2:
            return 0.0
        return float(returns.std() * math.sqrt(TRADING_DAYS_PER_YEAR))

    @property
    def sharpe_ratio(self) -> float:
        """Return the annualized Sharpe ratio with a zero risk-free rate."""
        returns = self.equity_curve.pct_change().dropna()
        if len(returns) < 2 or float(returns.std()) == 0.0:
            return 0.0
        return float(
            returns.mean() / returns.std() * math.sqrt(TRADING_DAYS_PER_YEAR)
        )

    def summary(self) -> str:
        """Return a human-readable metrics summary."""
        return (
            f"Period: {self.equity_curve.index[0].date()} to "
            f"{self.equity_curve.index[-1].date()} "
            f"({len(self.equity_curve)} trading days)\n"
            f"Total return:      {self.total_return:+.2%} "
            f"(benchmark {self.benchmark_total_return:+.2%})\n"
            f"CAGR:              {self.cagr:+.2%}\n"
            f"Max drawdown:      {self.max_drawdown:.2%}\n"
            f"Annualized vol:    {self.annualized_volatility:.2%}\n"
            f"Sharpe (rf=0):     {self.sharpe_ratio:.2f}\n"
            f"Orders executed:   {len(self.orders)}"
        )


class Backtester:
    """Drive the strategy engine over historical daily bars.

    Args:
        config: Strategy parameters for the simulated market.
        market_data: Historical data source. Bars are delayed for the
            strategy's view; the driver itself reads same-day bars to fill
            orders and mark equity.
        universe: Universe provider for the simulated market.
        costs: Execution-cost model. Defaults to frictionless.
        initial_cash: Starting account equity.
        eligibility: Market-specific security filter for the selector.
        logger: Optional logger for simulation diagnostics.
    """

    def __init__(
        self,
        config: StrategyConfig,
        market_data: MarketDataProvider,
        universe: IndexUniverseProvider,
        costs: CostModel | None = None,
        initial_cash: float = 100_000.0,
        eligibility: EligibilityFilter | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._config = config
        self._market_data = market_data
        self._universe = universe
        self._costs = costs
        self._initial_cash = initial_cash
        self._eligibility = eligibility
        self._logger = logger or logging.getLogger(__name__)
        self._bar_lookup: dict[str, dict[date, tuple[float | None, float | None]]] = {}
        self._window_periods = 0
        self._window_end = date.min
        self._today = date.min

    def run(self, start: date, end: date) -> BacktestResult:
        """Simulate the strategy over a date window.

        Args:
            start: First candidate trading day, inclusive.
            end: Last candidate trading day, inclusive.

        Returns:
            The simulated equity curve, benchmark closes, and fills.

        Raises:
            ValueError: If the window contains no benchmark trading days.
        """
        if end < start:
            raise ValueError("end must not precede start")
        calendar_days = (end - start).days + 10
        benchmark_bars = self._market_data.daily_bars(
            self._config.benchmark_symbol, calendar_days, end
        )
        if not benchmark_bars.empty:
            benchmark_bars = benchmark_bars[
                benchmark_bars.index >= pd.Timestamp(start)
            ]
        if benchmark_bars.empty:
            raise ValueError("no benchmark trading days in the requested window")
        self._window_periods = len(benchmark_bars) + 10
        self._window_end = end
        self._bar_lookup = {}

        broker = InMemoryBroker(
            Portfolio(cash=self._initial_cash),
            {},
            costs=self._costs,
            price_resolver=self._resolve_fill_price,
        )
        engine = StrategyEngine(
            self._config,
            DelayedMarketData(self._market_data),
            self._universe,
            broker,
            self._logger,
            eligibility=self._eligibility,
        )

        equity: dict[pd.Timestamp, float] = {}
        benchmark: dict[pd.Timestamp, float] = {}
        for timestamp in benchmark_bars.index:
            day = timestamp.date()
            self._today = day
            self._set_execution_prices(broker, day)
            engine.before_trading_start(day)
            engine.weekly_rebalance(day)
            engine.risk_check(day)
            engine.handle_data(datetime.combine(day, _AFTERNOON_CHECK))
            engine.handle_empty_month_clear(day)
            self._mark_positions(broker, day, "close")
            equity[timestamp] = broker.portfolio.cash + sum(
                position.quantity * position.last_price
                for position in broker.portfolio.positions.values()
            )
            close = to_finite_float(benchmark_bars.loc[timestamp].get("close"))
            benchmark[timestamp] = close if close is not None else float("nan")

        return BacktestResult(
            equity_curve=pd.Series(equity),
            benchmark_curve=pd.Series(benchmark),
            orders=tuple(broker.orders),
        )

    def _set_execution_prices(self, broker: InMemoryBroker, day: date) -> None:
        """Set day-open fill prices without exposing them to strategy decisions."""
        broker.clear_execution_prices()
        for symbol in list(broker.portfolio.positions):
            price = self._bar_value(symbol, day, "open")
            if price is not None and price > 0:
                broker.set_execution_price(symbol, price)

    def _mark_positions(
        self, broker: InMemoryBroker, day: date, price_field: str
    ) -> None:
        """Refresh broker prices for held positions from a day's bar field."""
        for symbol in list(broker.portfolio.positions):
            price = self._bar_value(symbol, day, price_field)
            if price is not None and price > 0:
                broker.set_price(symbol, price)

    def _resolve_fill_price(self, symbol: str) -> float | None:
        """Supply the current day's open for a symbol the broker cannot price."""
        return self._bar_value(symbol, self._today, "open")

    def _bar_value(self, symbol: str, day: date, price_field: str) -> float | None:
        """Return a day's open or close for a symbol from the cached window."""
        if symbol not in self._bar_lookup:
            bars = self._market_data.daily_bars(
                symbol, self._window_periods, self._window_end
            )
            lookup: dict[date, tuple[float | None, float | None]] = {}
            if not bars.empty and isinstance(bars.index, pd.DatetimeIndex):
                for timestamp, row in bars.iterrows():
                    lookup[timestamp.date()] = (
                        to_finite_float(row.get("open")),
                        to_finite_float(row.get("close")),
                    )
            self._bar_lookup[symbol] = lookup
        values = self._bar_lookup[symbol].get(day)
        if values is None:
            return None
        return values[0] if price_field == "open" else values[1]
