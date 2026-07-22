"""Order and execution-cost models owned by the broker/execution layer."""

from dataclasses import dataclass
from enum import StrEnum

MIN_NOTIONAL_ORDER_VALUE = 0.01


class OrderSide(StrEnum):
    """Direction of an order."""

    BUY = "buy"
    SELL = "sell"


class OrderOutcomeStatus(StrEnum):
    """State reached by a synchronous broker order attempt."""

    FILLED = "filled"
    PARTIAL = "partial"
    FAILED = "failed"
    WORKING = "working"
    UNKNOWN = "unknown"
    SKIPPED = "skipped"


class BrokerExecutionError(RuntimeError):
    """Raised when a valid order cannot be executed by a broker."""


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


@dataclass(frozen=True, slots=True)
class OrderOutcome:
    """Explicit result of asking a broker to execute an order.

    ``fill`` contains only quantity that the broker has confirmed as executed.
    A working outcome may carry a partial fill while the remainder stays open.
    A partial outcome records execution before the remaining order terminated;
    a failed outcome confirms that the order terminated without a fill.
    An unknown outcome means submission or final state could not be confirmed;
    its identifiers must be reconciled before retrying.
    """

    status: OrderOutcomeStatus
    order_id: str | None = None
    client_order_id: str | None = None
    fill: Order | None = None

    def __post_init__(self) -> None:
        """Reject internally contradictory execution outcomes."""
        if self.status in {
            OrderOutcomeStatus.FILLED,
            OrderOutcomeStatus.PARTIAL,
        } and self.fill is None:
            raise ValueError("a filled or partial outcome requires a confirmed fill")
        if self.status is OrderOutcomeStatus.FAILED and not self.reference:
            raise ValueError("a failed outcome requires an order reference")
        if self.status is OrderOutcomeStatus.FAILED and self.fill is not None:
            raise ValueError("a failed outcome cannot contain a fill")
        if self.status is OrderOutcomeStatus.WORKING and not self.order_id:
            raise ValueError("a working outcome requires a broker order id")
        if self.status is OrderOutcomeStatus.UNKNOWN and not (
            self.order_id or self.client_order_id
        ):
            raise ValueError("an unknown outcome requires an order reference")
        if self.status is OrderOutcomeStatus.SKIPPED and (
            self.order_id is not None
            or self.client_order_id is not None
            or self.fill is not None
        ):
            raise ValueError("a skipped outcome cannot contain an order or fill")

    @property
    def is_filled(self) -> bool:
        """Return whether the requested order completed in full."""
        return self.status is OrderOutcomeStatus.FILLED

    @property
    def is_working(self) -> bool:
        """Return whether the order remains open at the broker."""
        return self.status is OrderOutcomeStatus.WORKING

    @property
    def has_fill(self) -> bool:
        """Return whether the outcome contains confirmed executed quantity."""
        return self.fill is not None

    @property
    def is_pending(self) -> bool:
        """Return whether retrying could duplicate a live broker order."""
        return self.status in {
            OrderOutcomeStatus.WORKING,
            OrderOutcomeStatus.UNKNOWN,
        }

    @property
    def reference(self) -> str | None:
        """Return the best identifier available for reconciliation."""
        return self.order_id or self.client_order_id
