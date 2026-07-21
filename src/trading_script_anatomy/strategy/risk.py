"""Position and market-wide risk controls."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
import logging

from trading_script_anatomy.config import StrategyConfig
from trading_script_anatomy.data.protocols import BarProvider
from trading_script_anatomy.portfolio import Position
from trading_script_anatomy.values import latest_number


@dataclass(frozen=True, slots=True)
class SellInstruction:
    """A position liquidation requested by a risk control.

    Attributes:
        symbol: Ticker symbol to sell.
        quantity: Positive number of shares to sell.
        reason: Machine-readable reason for the sale.
    """

    symbol: str
    quantity: float
    reason: str


class RiskManager:
    """Evaluate stop-profit, stop-loss, and market-trend rules.

    Args:
        config: Immutable strategy parameters.
        market_data: Source for daily market bars. Only bars are required,
            so any ``BarProvider`` suffices.
        logger: Optional logger used for risk diagnostics.
    """

    def __init__(
        self,
        config: StrategyConfig,
        market_data: BarProvider,
        logger: logging.Logger | None = None,
    ) -> None:
        self._config = config
        self._market_data = market_data
        self._logger = logger or logging.getLogger(__name__)

    def position_exit_orders(
        self, positions: Sequence[Position], as_of: date
    ) -> list[SellInstruction]:
        """Return full-liquidation orders triggered by position-level controls.

        Args:
            positions: Current non-ETF positions.
            as_of: Date used when a position lacks a current price.

        Returns:
            Stop-profit or stop-loss sale instructions.
        """
        orders: list[SellInstruction] = []
        for position in positions:
            if position.quantity <= 0 or position.cost_basis <= 0:
                continue
            current_price = self._current_price(position, as_of)
            if current_price is None or current_price <= 0:
                continue

            profit_ratio = (current_price - position.cost_basis) / position.cost_basis
            if profit_ratio >= self._config.stop_profit_ratio - 1:
                orders.append(
                    SellInstruction(position.symbol, position.quantity, "stop_profit")
                )
            elif profit_ratio <= -self._config.stop_loss_ratio:
                orders.append(
                    SellInstruction(position.symbol, position.quantity, "stop_loss")
                )
        return orders

    def market_stop_loss_triggered(self, as_of: date) -> bool:
        """Return whether the benchmark's intraday move requires liquidation.

        The PTrade original averaged the open-to-close return of every
        benchmark constituent, which costs one provider request per symbol
        per day. The benchmark's own latest bar approximates that breadth
        signal in a single request. A cap-weighted index moves less than the
        equal-weighted constituent mean on extreme days, so a given threshold
        triggers somewhat more rarely than the original; both formulations
        are tail-event guards rather than routine controls.

        Missing or invalid data safely returns ``False``.

        Args:
            as_of: Date used to retrieve the latest benchmark bar.

        Returns:
            Whether the absolute benchmark open-to-close return meets the
            configured threshold.
        """
        try:
            bars = self._market_data.daily_bars(
                self._config.benchmark_symbol, periods=1, as_of=as_of
            )
            open_price = latest_number(bars, "open")
            close_price = latest_number(bars, "close")
        except Exception as error:
            self._logger.warning(
                "Unable to read benchmark bar for market stop: %s", error
            )
            return False

        if open_price is None or close_price is None or open_price <= 0:
            return False
        return (
            abs(close_price / open_price - 1)
            >= self._config.market_stop_loss_threshold
        )

    def _current_price(self, position: Position, as_of: date) -> float | None:
        """Use a position price first and fall back to the latest close."""
        if position.last_price > 0:
            return position.last_price
        try:
            bars = self._market_data.daily_bars(position.symbol, 1, as_of)
        except Exception as error:
            self._logger.warning(
                "Unable to read fallback price for %s: %s", position.symbol, error
            )
            return None
        return latest_number(bars, "close")
