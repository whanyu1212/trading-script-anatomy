"""Protocols for order execution."""

from typing import Protocol, Sequence

from trading_script_anatomy.portfolio import Portfolio, Position


class Broker(Protocol):
    """Execute orders and expose the portfolio owned by the strategy."""

    @property
    def portfolio(self) -> Portfolio:
        """Return current cash and holdings."""

    def positions(self) -> Sequence[Position]:
        """Return all current positions.

        Returns:
            Current position objects.
        """

    def order_quantity(self, symbol: str, quantity: float, reason: str) -> None:
        """Submit a signed quantity order.

        Args:
            symbol: Provider-specific ticker symbol.
            quantity: Positive to buy and negative to sell.
            reason: Strategy event that triggered the order.
        """

    def order_value(self, symbol: str, value: float, reason: str) -> None:
        """Submit a positive cash-value purchase order.

        Args:
            symbol: Provider-specific ticker symbol.
            value: Cash value to invest.
            reason: Strategy event that triggered the order.
        """
