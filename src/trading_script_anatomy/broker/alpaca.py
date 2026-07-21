"""Alpaca-backed broker adapter for paper and live trading."""

from collections.abc import Callable, Sequence
import logging
import math
import os
import time
from typing import Any
from uuid import uuid4

import requests

from trading_script_anatomy.broker.models import (
    BrokerExecutionError,
    Order,
    OrderOutcome,
    OrderOutcomeStatus,
    OrderSide,
)
from trading_script_anatomy.portfolio import Portfolio, Position
from trading_script_anatomy.values import to_finite_float

PAPER_BASE_URL = "https://paper-api.alpaca.markets"

_TERMINAL_FAILURE_STATUSES = frozenset(
    {
        "calculated",
        "canceled",
        "done_for_day",
        "expired",
        "rejected",
        "replaced",
    }
)

_WORKING_STATUSES = frozenset(
    {
        "accepted",
        "accepted_for_bidding",
        "held",
        "new",
        "partially_filled",
        "pending_cancel",
        "pending_new",
        "pending_replace",
        "stopped",
        "suspended",
    }
)


class AlpacaError(BrokerExecutionError):
    """Raised when Alpaca rejects or fails a request."""


class AlpacaRequestError(AlpacaError):
    """Raised when transport or decoding leaves a request result uncertain."""


class AlpacaTerminalOrderError(AlpacaError):
    """Raised when Alpaca confirms that an order ended without filling."""


class AlpacaBroker:
    """Execute strategy orders through the Alpaca trading API.

    Orders are submitted as market day orders and awaited until filled, so
    the engine's synchronous assumption — sell proceeds are available for the
    next purchase — holds during market hours. Outside market hours orders
    are accepted but do not fill; the adapter logs a warning after the fill
    timeout and continues, leaving the order working at the exchange open.
    If submission or polling becomes ambiguous, the returned ``UNKNOWN``
    outcome carries the available broker/client identifier and must be
    reconciled before the caller retries.

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

    def order_quantity(
        self, symbol: str, quantity: float, reason: str
    ) -> OrderOutcome:
        """Submit a signed market quantity order and await its fill.

        Args:
            symbol: Alpaca ticker symbol.
            quantity: Positive to buy and negative to sell.
            reason: Strategy event that triggered the order.

        Raises:
            AlpacaError: If Alpaca rejects the order or it terminally fails.

        Returns:
            Filled, working, unknown, or skipped execution outcome.
        """
        if isinstance(quantity, bool) or not math.isfinite(quantity):
            raise ValueError("order quantity must be finite")
        if quantity == 0:
            return OrderOutcome(OrderOutcomeStatus.SKIPPED)
        side = OrderSide.BUY if quantity > 0 else OrderSide.SELL
        return self._submit_order(
            {
                "symbol": symbol,
                "qty": _format_quantity(abs(quantity)),
                "side": side.value,
                "type": "market",
                "time_in_force": "day",
                "client_order_id": _client_order_id(reason),
            },
            side,
            reason,
        )

    def order_value(self, symbol: str, value: float, reason: str) -> OrderOutcome:
        """Submit a notional market purchase order and await its fill.

        Notional orders require the asset to be fractionable on Alpaca, which
        holds for major ETFs and most listed equities.

        Args:
            symbol: Alpaca ticker symbol.
            value: Positive cash value to invest.
            reason: Strategy event that triggered the order.

        Raises:
            ValueError: If ``value`` is non-finite, boolean, or below one cent.
            AlpacaError: If Alpaca rejects the order or it terminally fails.

        Returns:
            Filled, working, or unknown execution outcome.
        """
        if isinstance(value, bool) or not math.isfinite(value) or value < 0.01:
            raise ValueError("order value must be finite and at least 0.01")
        return self._submit_order(
            {
                "symbol": symbol,
                "notional": f"{value:.2f}",
                "side": OrderSide.BUY.value,
                "type": "market",
                "time_in_force": "day",
                "client_order_id": _client_order_id(reason),
            },
            OrderSide.BUY,
            reason,
        )

    def _submit_order(
        self,
        payload: dict[str, str],
        side: OrderSide,
        reason: str,
    ) -> OrderOutcome:
        """Submit an order, await its fill, and record the outcome."""
        client_order_id = payload["client_order_id"]
        try:
            submitted = self._request("POST", "/v2/orders", payload)
        except AlpacaRequestError as error:
            self.refresh()
            return self._unknown_outcome(client_order_id, error)
        self.refresh()
        if not isinstance(submitted, dict):
            return self._unknown_outcome(
                client_order_id,
                AlpacaError("order submission returned a non-object payload"),
            )
        order_id = str(submitted.get("id") or "").strip()
        if not order_id:
            return self._unknown_outcome(
                client_order_id,
                AlpacaError("order submission returned no order id"),
            )

        try:
            status, order_payload = self._await_outcome(order_id)
            fill = self._confirmed_fill(
                order_payload, payload["symbol"], side, reason
            )
            if status is OrderOutcomeStatus.FILLED and fill is None:
                raise AlpacaError(
                    f"Order {order_id} reported filled without a valid fill"
                )
        except AlpacaTerminalOrderError:
            raise
        except AlpacaError as error:
            return self._unknown_outcome(
                client_order_id, error, order_id=order_id
            )
        if fill is not None:
            self.orders.append(fill)
        return OrderOutcome(
            status,
            order_id=order_id,
            client_order_id=client_order_id,
            fill=fill,
        )

    def _unknown_outcome(
        self,
        client_order_id: str,
        error: AlpacaError,
        *,
        order_id: str | None = None,
    ) -> OrderOutcome:
        """Return an outcome that callers must reconcile before retrying."""
        reference = order_id or client_order_id
        self._logger.error(
            "Unable to determine outcome for order %s: %s", reference, error
        )
        return OrderOutcome(
            OrderOutcomeStatus.UNKNOWN,
            order_id=order_id,
            client_order_id=client_order_id,
        )

    def _await_outcome(
        self, order_id: str
    ) -> tuple[OrderOutcomeStatus, dict[str, Any]]:
        """Poll an order until it fills, terminally fails, or times out.

        Returns:
            Normalized outcome status and the latest order payload.

        Raises:
            AlpacaError: If the order reaches a terminal failure status.
        """
        deadline = time.monotonic() + self._fill_timeout
        while True:
            order = self._request("GET", f"/v2/orders/{order_id}")
            if not isinstance(order, dict):
                raise AlpacaError(
                    f"Order {order_id} status returned a non-object payload"
                )
            status = str(order.get("status", ""))
            if not status:
                raise AlpacaError(f"Order {order_id} status was missing")
            if status == "filled":
                return OrderOutcomeStatus.FILLED, order
            if status in _TERMINAL_FAILURE_STATUSES:
                raise AlpacaTerminalOrderError(
                    f"Order {order_id} ended with status {status!r}"
                )
            if status not in _WORKING_STATUSES:
                raise AlpacaError(
                    f"Order {order_id} returned unknown status {status!r}"
                )
            if time.monotonic() >= deadline:
                self._logger.warning(
                    "Order %s not filled after %.0fs (status %r); it remains "
                    "working — likely outside market hours",
                    order_id,
                    self._fill_timeout,
                    status,
                )
                return OrderOutcomeStatus.WORKING, order
            self._sleep(self._poll_interval)

    def _confirmed_fill(
        self,
        payload: dict[str, Any],
        symbol: str,
        side: OrderSide,
        reason: str,
    ) -> Order | None:
        """Build an order record only from a confirmed full or partial fill."""
        quantity = to_finite_float(payload.get("filled_qty"))
        price = to_finite_float(payload.get("filled_avg_price"))
        if quantity is None or quantity == 0:
            if price not in (None, 0.0):
                raise AlpacaError(
                    "Alpaca supplied a fill price without a fill quantity"
                )
            return None
        if quantity < 0 or price is None or price <= 0:
            raise AlpacaError("Alpaca supplied an invalid confirmed fill")
        return Order(symbol, quantity, side, price, reason)

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
        try:
            response = self._session.request(
                method,
                f"{self._base_url}{path}",
                headers=self._headers,
                json=payload,
                timeout=30,
            )
        except requests.RequestException as error:
            raise AlpacaRequestError(
                f"Alpaca {method} {path} request failed: {error}"
            ) from error
        if response.status_code >= 500:
            raise AlpacaRequestError(
                f"Alpaca {method} {path} returned ambiguous HTTP "
                f"{response.status_code}: {response.text[:200]}"
            )
        if not 200 <= response.status_code < 300:
            raise AlpacaError(
                f"Alpaca {method} {path} failed with HTTP "
                f"{response.status_code}: {response.text[:200]}"
            )
        try:
            return response.json()
        except ValueError as error:
            raise AlpacaRequestError(
                f"Alpaca {method} {path} returned invalid JSON"
            ) from error


def _format_quantity(quantity: float) -> str:
    """Format a share quantity without scientific notation."""
    return f"{quantity:.9f}".rstrip("0").rstrip(".")


def _client_order_id(reason: str) -> str:
    """Build a unique order id that keeps the strategy reason auditable."""
    return f"tsa-{reason[:24]}-{uuid4().hex[:8]}"
