"""Orchestration for the modular small-cap rotation strategy."""

from collections.abc import Collection
from datetime import date, datetime
import logging

from trading_script_anatomy.broker.protocols import Broker
from trading_script_anatomy.config import StrategyConfig
from trading_script_anatomy.data.protocols import IndexUniverseProvider, MarketDataProvider
from trading_script_anatomy.portfolio import Position
from trading_script_anatomy.strategy.state import StrategyState
from trading_script_anatomy.strategy.risk import RiskManager
from trading_script_anatomy.strategy.selection import EligibilityFilter, StockSelector


class StrategyEngine:
    """Coordinate selection, risk checks, portfolio orders, and defensive cash.

    This class replaces the PTrade callback lifecycle. A scheduler invokes the
    public methods at its intended times, while data access and order execution
    are supplied through explicit adapters. The original's limit-up tail-check
    lifecycle is intentionally absent: it managed daily price-limit mechanics
    that exist on mainland-China exchanges but not in US markets.

    Args:
        config: Immutable strategy parameters.
        market_data: Source for prices, metadata, and fundamentals.
        universe: Source for point-in-time benchmark constituents.
        broker: Order-execution and portfolio adapter.
        logger: Optional logger used for execution diagnostics.
        state: Optional persisted strategy state to resume.
        eligibility: Market-specific security filter forwarded to the stock
            selector. Defaults to the A-share rules.
    """

    def __init__(
        self,
        config: StrategyConfig,
        market_data: MarketDataProvider,
        universe: IndexUniverseProvider,
        broker: Broker,
        logger: logging.Logger | None = None,
        state: StrategyState | None = None,
        eligibility: EligibilityFilter | None = None,
    ) -> None:
        self.config = config
        self.broker = broker
        self.state = state or StrategyState(stock_count=config.initial_stock_count)
        self._logger = logger or logging.getLogger(__name__)
        self._selector = StockSelector(
            config, market_data, universe, self._logger, eligibility
        )
        self._risk = RiskManager(config, market_data, self._logger)

    def before_trading_start(self, as_of: date) -> None:
        """Reset daily state.

        Args:
            as_of: Current trading date.
        """
        del as_of
        self.state.stopped_out = False
        self.state.stop_loss_etf_bought = False

    def weekly_rebalance(self, as_of: date) -> None:
        """Execute weekly selection and an equal-value rebalance when scheduled.

        Args:
            as_of: Trading date on which the rebalance is evaluated.
        """
        if as_of.weekday() != self.config.rebalance_weekday:
            return
        if self.is_empty_month(as_of):
            self.handle_empty_month_etf(as_of)
            return

        self._sell_safe_etf()
        self.state.candidates = []

        target_symbols = self._selector.select_targets(
            as_of,
            self._equity_symbols(),
            self.state,
        )
        if not target_symbols:
            self._buy_safe_etf()
            self.state.target_positions = []
            return

        self.state.target_positions = target_symbols
        self._sell_positions_not_in(target_symbols)
        self._buy_missing_positions(target_symbols)
        self.state.last_rebalance_date = as_of

    def risk_check(self, as_of: date) -> None:
        """Run position-level exits and the market-wide stop-loss control.

        Args:
            as_of: Trading date on which risk is evaluated.
        """
        if self.is_empty_month(as_of):
            return

        sold_any = False
        for instruction in self._risk.position_exit_orders(
            self._equity_positions(), as_of
        ):
            sold_any |= self._sell(
                instruction.symbol, instruction.quantity, instruction.reason
            )

        if self._risk.market_stop_loss_triggered(as_of):
            for position in self._equity_positions():
                sold_any |= self._sell(
                    position.symbol, position.quantity, "market_stop_loss"
                )

        if sold_any:
            self.state.stopped_out = True

    def handle_empty_month_clear(self, as_of: date) -> None:
        """Liquidate equities during an empty month.

        Args:
            as_of: Trading date used to determine whether the month is empty.
        """
        if not self.is_empty_month(as_of):
            return
        for position in self._equity_positions():
            self._sell(position.symbol, position.quantity, "empty_month_clear")

    def handle_empty_month_etf(self, as_of: date) -> None:
        """Invest available cash in the safe ETF on the rebalance weekday.

        Args:
            as_of: Trading date used to determine whether the purchase is due.
        """
        if (
            self.is_empty_month(as_of)
            and as_of.weekday() == self.config.rebalance_weekday
        ):
            self._buy_safe_etf()

    def handle_stop_loss_funds(self, now: datetime) -> None:
        """Invest post-risk-sale cash in the safe ETF after 14:00.

        Args:
            now: Current timestamp in the trading calendar's timezone.
        """
        if (
            self.state.stopped_out
            and not self.state.stop_loss_etf_bought
            and now.hour >= 14
        ):
            self._buy_safe_etf()
            self.state.stop_loss_etf_bought = True
            self.state.stopped_out = False

    def handle_data(self, now: datetime) -> None:
        """Run the intraday action formerly called by PTrade.

        Args:
            now: Current timestamp in the trading calendar's timezone.
        """
        self.handle_stop_loss_funds(now)

    def is_empty_month(self, as_of: date) -> bool:
        """Return whether equity holdings should be replaced for this month.

        Args:
            as_of: Date whose month is evaluated.

        Returns:
            True when the month is configured as empty.
        """
        return as_of.month in self.config.empty_months

    def _sell(self, symbol: str, quantity: float, reason: str) -> bool:
        """Submit a sale, logging instead of raising on failure.

        Args:
            symbol: Ticker to sell.
            quantity: Positive share quantity to sell.
            reason: Strategy event that triggered the sale.

        Returns:
            Whether the order was submitted successfully.
        """
        try:
            self.broker.order_quantity(symbol, -quantity, reason)
            return True
        except Exception as error:
            self._logger.error(
                "Unable to execute %s for %s: %s", reason, symbol, error
            )
            return False

    def _sell_positions_not_in(self, target_symbols: Collection[str]) -> None:
        """Sell positions that are not part of the target holdings."""
        target_set = set(target_symbols)
        for position in self._equity_positions():
            if position.symbol not in target_set:
                self._sell(position.symbol, position.quantity, "rebalance_sell")

    def _buy_missing_positions(self, target_symbols: Collection[str]) -> None:
        """Buy missing targets with an equal division of available cash."""
        held_symbols = self._equity_symbols()
        to_buy = [
            symbol for symbol in target_symbols if symbol not in held_symbols
        ]
        slots = min(len(target_symbols), self.state.stock_count) - len(held_symbols)
        to_buy = to_buy[:max(slots, 0)]
        if not to_buy or self.broker.portfolio.cash <= 0:
            return

        cash_per_symbol = self.broker.portfolio.cash / len(to_buy)
        for symbol in to_buy:
            try:
                self.broker.order_value(symbol, cash_per_symbol, "rebalance_buy")
            except Exception as error:
                self._logger.error(
                    "Unable to rebalance-buy %s: %s", symbol, error
                )

    def _sell_safe_etf(self) -> None:
        """Sell the configured safe ETF before a normal weekly rebalance."""
        position = self._position(self.config.safe_etf_symbol)
        if position is None or position.quantity <= 0:
            return
        self._sell(self.config.safe_etf_symbol, position.quantity, "sell_safe_etf")

    def _buy_safe_etf(self) -> None:
        """Invest all currently available cash in the configured safe ETF."""
        cash = self.broker.portfolio.cash
        if cash <= 0:
            return
        try:
            self.broker.order_value(
                self.config.safe_etf_symbol,
                cash,
                "buy_safe_etf",
            )
        except Exception as error:
            self._logger.error("Unable to buy safe ETF: %s", error)

    def _equity_positions(self) -> list[Position]:
        """Return positions excluding the configured defensive ETF."""
        return [
            position
            for position in self.broker.positions()
            if (
                position.symbol != self.config.safe_etf_symbol
                and position.quantity > 0
            )
        ]

    def _equity_symbols(self) -> set[str]:
        """Return symbols of current non-ETF positions."""
        return {position.symbol for position in self._equity_positions()}

    def _position(self, symbol: str) -> Position | None:
        """Return a held position matching a symbol, when present."""
        return self.broker.portfolio.positions.get(symbol)
