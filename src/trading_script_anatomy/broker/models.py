"""Order and execution-cost models owned by the broker/execution layer."""

from dataclasses import dataclass
from enum import StrEnum


class OrderSide(StrEnum):
    """Direction of an order."""

    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True, slots=True)
class CostModel:
    """Execution costs applied to simulated fills.

    The default instance is frictionless, which preserves the historical
    behavior of the in-memory broker in tests.

    Attributes:
        commission_rate: Commission as a fraction of traded notional.
        min_commission: Commission floor per order in account currency.
        slippage_rate: Half-spread paid on each fill as a price fraction.
        sell_tax_rate: Sale-only tax as a fraction of sale notional, such as
            the mainland-China stamp tax.
    """

    commission_rate: float = 0.0
    min_commission: float = 0.0
    slippage_rate: float = 0.0
    sell_tax_rate: float = 0.0

    def buy_price(self, price: float) -> float:
        """Return the effective per-share price paid when buying."""
        return price * (1 + self.slippage_rate)

    def sell_price(self, price: float) -> float:
        """Return the effective per-share price received when selling."""
        return price * (1 - self.slippage_rate)

    def commission(self, notional: float) -> float:
        """Return the commission charged on a traded notional."""
        if notional <= 0:
            return 0.0
        return max(notional * self.commission_rate, self.min_commission)

    def sell_tax(self, notional: float) -> float:
        """Return the sale tax charged on a sale notional."""
        return max(notional, 0.0) * self.sell_tax_rate

    def investable(self, value: float) -> float:
        """Return the notional investable from a cash amount after commission.

        Args:
            value: Cash committed to the purchase, commission included.

        Returns:
            The purchasable notional; non-positive when ``value`` cannot
            cover the minimum commission.
        """
        proportional = value / (1 + self.commission_rate)
        if proportional * self.commission_rate >= self.min_commission:
            return proportional
        return value - self.min_commission


@dataclass(frozen=True, slots=True)
class Order:
    """An order recorded by a broker.

    Attributes:
        symbol: Provider-specific ticker symbol.
        quantity: Positive share quantity.
        side: Whether the order buys or sells.
        price: Execution price per share.
        reason: Strategy event that generated the order.
    """

    symbol: str
    quantity: float
    side: OrderSide
    price: float
    reason: str
