"""Portfolio domain models shared by strategy and execution components."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class Position:
    """A currently held security.

    Attributes:
        symbol: Provider-specific ticker symbol.
        quantity: Number of shares held.
        cost_basis: Average acquisition price per share.
        last_price: Latest observed price per share.
    """

    symbol: str
    quantity: float
    cost_basis: float
    last_price: float = 0.0


@dataclass(slots=True)
class Portfolio:
    """Cash and positions controlled by a broker.

    Attributes:
        cash: Settled cash available for purchases.
        positions: Positions keyed by ticker symbol.
    """

    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
