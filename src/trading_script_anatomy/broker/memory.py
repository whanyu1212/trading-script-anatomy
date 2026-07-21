"""A deterministic in-memory broker for tests, research, and backtests."""

from collections.abc import Callable, Mapping, Sequence
import math

from trading_script_anatomy.broker.models import CostModel, Order, OrderSide
from trading_script_anatomy.portfolio import Portfolio, Position


class InMemoryBroker:
    """Execute immediate fills at caller-supplied prices.

    Args:
        portfolio: Starting cash and positions. This instance mutates it.
        prices: Mapping from ticker symbols to current execution prices.
        costs: Execution costs applied to fills. Defaults to frictionless.
        price_resolver: Optional fallback used to look up a price for symbols
            absent from ``prices``, letting a backtest driver supply fill
            prices lazily without predicting which symbols will be ordered.
    """

    def __init__(
        self,
        portfolio: Portfolio,
        prices: Mapping[str, float],
        costs: CostModel | None = None,
        price_resolver: Callable[[str], float | None] | None = None,
    ) -> None:
        self._portfolio = portfolio
        self._prices = dict(prices)
        self._costs = costs or CostModel()
        self._price_resolver = price_resolver
        self.orders: list[Order] = []

    @property
    def portfolio(self) -> Portfolio:
        """Return the mutable portfolio controlled by this broker."""
        return self._portfolio

    def positions(self) -> Sequence[Position]:
        """Return currently held positions.

        Returns:
            Current position objects.
        """
        return tuple(self._portfolio.positions.values())

    def set_price(self, symbol: str, price: float) -> None:
        """Set a fill price and mark an existing position to that price.

        Args:
            symbol: Provider-specific ticker symbol.
            price: Positive execution price per share.

        Raises:
            ValueError: If ``price`` is not a positive finite number.
        """
        self.set_execution_price(symbol, price)
        position = self._portfolio.positions.get(symbol)
        if position is not None:
            position.last_price = price

    def set_execution_price(self, symbol: str, price: float) -> None:
        """Set a fill price without changing a held position's market mark.

        Args:
            symbol: Provider-specific ticker symbol.
            price: Positive execution price per share.

        Raises:
            ValueError: If ``price`` is not a positive finite number.
        """
        if isinstance(price, bool) or not math.isfinite(price) or price <= 0:
            raise ValueError("price must be positive and finite")
        self._prices[symbol] = price

    def clear_execution_prices(self) -> None:
        """Remove all immediate-fill prices from the previous execution cycle."""
        self._prices.clear()

    def order_quantity(self, symbol: str, quantity: float, reason: str) -> None:
        """Execute a signed quantity order at the configured price.

        Args:
            symbol: Provider-specific ticker symbol.
            quantity: Positive to buy and negative to sell.
            reason: Strategy event that triggered the order.

        Raises:
            ValueError: If the order is invalid or cannot be filled.
        """
        if quantity == 0:
            return
        price = self._price_for(symbol)
        if quantity > 0:
            self._buy(symbol, quantity, price, reason)
        else:
            self._sell(symbol, -quantity, price, reason)

    def order_value(self, symbol: str, value: float, reason: str) -> None:
        """Invest a positive cash amount at the configured price.

        Args:
            symbol: Provider-specific ticker symbol.
            value: Positive cash amount to invest.
            reason: Strategy event that triggered the order.

        Raises:
            ValueError: If ``value`` is non-positive, cannot cover execution
                costs, or exceeds available cash.
        """
        if value <= 0:
            raise ValueError("order value must be positive")
        budget = self._costs.investable(value)
        if budget <= 0:
            raise ValueError("order value does not cover execution costs")
        fill_price = self._costs.buy_price(self._price_for(symbol))
        self.order_quantity(symbol, budget / fill_price, reason)

    def _buy(self, symbol: str, quantity: float, price: float, reason: str) -> None:
        fill_price = self._costs.buy_price(price)
        notional = quantity * fill_price
        total = notional + self._costs.commission(notional)
        if total > self._portfolio.cash + 1e-9:
            raise ValueError(f"insufficient cash to buy {symbol}")
        position = self._portfolio.positions.get(symbol)
        if position is None:
            self._portfolio.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                cost_basis=fill_price,
                last_price=fill_price,
            )
        else:
            total_cost = position.cost_basis * position.quantity + notional
            position.quantity += quantity
            position.cost_basis = total_cost / position.quantity
            position.last_price = fill_price
        self._portfolio.cash -= total
        self.orders.append(Order(symbol, quantity, OrderSide.BUY, fill_price, reason))

    def _sell(self, symbol: str, quantity: float, price: float, reason: str) -> None:
        position = self._portfolio.positions.get(symbol)
        if position is None or quantity > position.quantity + 1e-9:
            raise ValueError(f"insufficient shares to sell {symbol}")
        fill_price = self._costs.sell_price(price)
        notional = quantity * fill_price
        fees = self._costs.commission(notional) + self._costs.sell_tax(notional)
        position.quantity -= quantity
        position.last_price = fill_price
        self._portfolio.cash += notional - fees
        if position.quantity <= 1e-9:
            del self._portfolio.positions[symbol]
        self.orders.append(Order(symbol, quantity, OrderSide.SELL, fill_price, reason))

    def _price_for(self, symbol: str) -> float:
        price = self._prices.get(symbol)
        if price is None and self._price_resolver is not None:
            price = self._price_resolver(symbol)
        if (
            price is None
            or isinstance(price, bool)
            or not math.isfinite(price)
            or price <= 0
        ):
            raise ValueError(f"missing positive execution price for {symbol}")
        return price
