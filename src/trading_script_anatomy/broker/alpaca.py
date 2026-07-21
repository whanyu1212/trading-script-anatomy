"""Alpaca-backed broker adapter for paper and live trading."""

from collections.abc import Callable, Sequence
import logging
import os
import time
from typing import Any
from uuid import uuid4

import requests

from trading_script_anatomy.broker.models import Order, OrderSide
from trading_script_anatomy.portfolio import Portfolio, Position
from trading_script_anatomy.values import to_finite_float

PAPER_BASE_URL = "https://paper-api.alpaca.markets"

_TERMINAL_FAILURE_STATUSES = frozenset(
    {"canceled", "expired", "rejected", "stopped", "suspended"}
)


class AlpacaError(RuntimeError):
    """Raised when Alpaca rejects or fails a request."""


class AlpacaBroker:
    """Execute strategy orders through the Alpaca trading API.

    Orders are submitted as market day orders and awaited until filled, so
    the engine's synchronous assumption — sell proceeds are available for the
    next purchase — holds during market hours. Outside market hours orders
    are accepted but do not fill; the adapter logs a warning after the fill
    timeout and continues, leaving the order working at the exchange open.

    ``portfolio`` is a snapshot fetched from Alpaca and cached until the next
    submitted order, unlike the authoritative in-memory broker.

    Args:
        api_key: Alpaca key ID. Defaults to the ``APCA_API_KEY_ID``
            environment variable; library code never reads ``.env`` files.
        api_secret: Alpaca secret. Defaults to ``APCA_API_SECRET_KEY``.
        session: Object providing ``requests.Session``-style ``request``.
            This injection point permits network-free tests.
        base_url: Trading API root. Defaults to the paper-trading host; pass
            the live host only deliberately.
        fill_timeout: Seconds to wait for a submitted order to fill.
        poll_interval: Seconds between order-status polls.
        sleep: Sleep function, injectable for tests.
        logger: Optional logger used for execution diagnostics.

    Raises:
        AlpacaError: If credentials are not available.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        session: Any = None,
        base_url: str = PAPER_BASE_URL,
        fill_timeout: float = 30.0,
        poll_interval: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
        logger: logging.Logger | None = None,
    ) -> None:
        key = api_key or os.environ.get("APCA_API_KEY_ID", "")
        secret = api_secret or os.environ.get("APCA_API_SECRET_KEY", "")
        if not key or not secret:
            raise AlpacaError(
                "Alpaca credentials are required; set APCA_API_KEY_ID and "
                "APCA_API_SECRET_KEY or pass api_key and api_secret"
            )
        self._headers = {
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": secret,
        }
        self._session = session or requests.Session()
        self._base_url = base_url.rstrip("/")
        self._fill_timeout = fill_timeout
        self._poll_interval = poll_interval
        self._sleep = sleep
        self._logger = logger or logging.getLogger(__name__)
        self._portfolio_cache: Portfolio | None = None
        self.orders: list[Order] = []

    @property
    def portfolio(self) -> Portfolio:
        """Return a snapshot of account cash and holdings.

        The snapshot is cached until the next submitted order. Call
        ``refresh`` to force a new fetch.
        """
        if self._portfolio_cache is None:
            self._portfolio_cache = self._fetch_portfolio()
        return self._portfolio_cache

    def refresh(self) -> None:
        """Discard the cached portfolio snapshot."""
        self._portfolio_cache = None

    def positions(self) -> Sequence[Position]:
        """Return all current positions.

        Returns:
            Current position objects from the portfolio snapshot.
        """
        return tuple(self.portfolio.positions.values())

    def clock(self) -> dict[str, Any]:
        """Return Alpaca's market clock.

        Returns:
            The clock payload with ``is_open``, ``next_open``, and
            ``next_close`` fields.
        """
        payload = self._request("GET", "/v2/clock")
        return payload if isinstance(payload, dict) else {}

    def is_market_open(self) -> bool:
        """Return whether the market is currently open for trading."""
        return bool(self.clock().get("is_open"))

    def order_quantity(self, symbol: str, quantity: float, reason: str) -> None:
        """Submit a signed market quantity order and await its fill.

        Args:
            symbol: Alpaca ticker symbol.
            quantity: Positive to buy and negative to sell.
            reason: Strategy event that triggered the order.

        Raises:
            AlpacaError: If Alpaca rejects the order or it terminally fails.
        """
        if quantity == 0:
            return
        side = OrderSide.BUY if quantity > 0 else OrderSide.SELL
        self._submit_order(
            {
                "symbol": symbol,
                "qty": _format_quantity(abs(quantity)),
                "side": side.value,
                "type": "market",
                "time_in_force": "day",
                "client_order_id": _client_order_id(reason),
            },
            abs(quantity),
            side,
            reason,
        )

    def order_value(self, symbol: str, value: float, reason: str) -> None:
        """Submit a notional market purchase order and await its fill.

        Notional orders require the asset to be fractionable on Alpaca, which
        holds for major ETFs and most listed equities.

        Args:
            symbol: Alpaca ticker symbol.
            value: Positive cash value to invest.
            reason: Strategy event that triggered the order.

        Raises:
            ValueError: If ``value`` is not positive.
            AlpacaError: If Alpaca rejects the order or it terminally fails.
        """
        if value <= 0:
            raise ValueError("order value must be positive")
        self._submit_order(
            {
                "symbol": symbol,
                "notional": f"{value:.2f}",
                "side": OrderSide.BUY.value,
                "type": "market",
                "time_in_force": "day",
                "client_order_id": _client_order_id(reason),
            },
            0.0,
            OrderSide.BUY,
            reason,
        )

    def _submit_order(
        self,
        payload: dict[str, str],
        quantity: float,
        side: OrderSide,
        reason: str,
    ) -> None:
        """Submit an order, await its fill, and record the outcome."""
        submitted = self._request("POST", "/v2/orders", payload)
        self.refresh()
        filled = self._await_fill(str(submitted.get("id", "")))
        fill_price = to_finite_float((filled or {}).get("filled_avg_price")) or 0.0
        fill_quantity = to_finite_float((filled or {}).get("filled_qty")) or quantity
        self.orders.append(Order(payload["symbol"], fill_quantity, side, fill_price, reason))

    def _await_fill(self, order_id: str) -> dict[str, Any] | None:
        """Poll an order until it fills, terminally fails, or times out.

        Returns:
            The filled order payload, or ``None`` on timeout.

        Raises:
            AlpacaError: If the order reaches a terminal failure status.
        """
        if not order_id:
            return None
        deadline = time.monotonic() + self._fill_timeout
        while True:
            order = self._request("GET", f"/v2/orders/{order_id}")
            status = str(order.get("status", ""))
            if status == "filled":
                return order
            if status in _TERMINAL_FAILURE_STATUSES:
                raise AlpacaError(f"Order {order_id} ended with status {status!r}")
            if time.monotonic() >= deadline:
                self._logger.warning(
                    "Order %s not filled after %.0fs (status %r); it remains "
                    "working — likely outside market hours",
                    order_id,
                    self._fill_timeout,
                    status,
                )
                return None
            self._sleep(self._poll_interval)

    def _fetch_portfolio(self) -> Portfolio:
        """Build a portfolio snapshot from the account and positions APIs."""
        account = self._request("GET", "/v2/account")
        rows = self._request("GET", "/v2/positions")
        positions: dict[str, Position] = {}
        for row in rows if isinstance(rows, list) else []:
            symbol = str(row.get("symbol", ""))
            quantity = to_finite_float(row.get("qty"))
            if not symbol or quantity is None or quantity == 0:
                continue
            positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                cost_basis=to_finite_float(row.get("avg_entry_price")) or 0.0,
                last_price=to_finite_float(row.get("current_price")) or 0.0,
            )
        return Portfolio(
            cash=to_finite_float(account.get("cash")) or 0.0,
            positions=positions,
        )

    def _request(
        self, method: str, path: str, payload: dict[str, str] | None = None
    ) -> Any:
        """Return decoded JSON for an authenticated API request.

        Raises:
            AlpacaError: On any non-2xx response.
        """
        response = self._session.request(
            method,
            f"{self._base_url}{path}",
            headers=self._headers,
            json=payload,
            timeout=30,
        )
        if not 200 <= response.status_code < 300:
            raise AlpacaError(
                f"Alpaca {method} {path} failed with HTTP "
                f"{response.status_code}: {response.text[:200]}"
            )
        return response.json()


def _format_quantity(quantity: float) -> str:
    """Format a share quantity without scientific notation."""
    return f"{quantity:.9f}".rstrip("0").rstrip(".")


def _client_order_id(reason: str) -> str:
    """Build a unique order id that keeps the strategy reason auditable."""
    return f"tsa-{reason[:24]}-{uuid4().hex[:8]}"
