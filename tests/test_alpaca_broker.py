"""Tests for the Alpaca broker adapter."""

import pytest
import requests

from trading_script_anatomy.broker.alpaca import AlpacaBroker, AlpacaError
from trading_script_anatomy.broker.models import OrderOutcomeStatus, OrderSide

from tests.fakes import FakeHTTPResponse, FakeHTTPSession

ACCOUNT = {"cash": "1000.50"}
POSITIONS = [
    {"symbol": "ACME", "qty": "3", "avg_entry_price": "10.0", "current_price": "12.5"}
]
FILLED = {"status": "filled", "filled_avg_price": "10.5", "filled_qty": "3"}


def make_broker(
    script: dict[tuple[str, str], list[object]], **kwargs: object
) -> tuple[AlpacaBroker, FakeHTTPSession]:
    """Build a broker backed by a scripted fake session."""
    session = FakeHTTPSession(script)
    broker = AlpacaBroker(
        api_key="key", api_secret="secret", session=session, sleep=lambda _: None, **kwargs
    )
    return broker, session


def test_broker_requires_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail fast when no credentials are configured anywhere."""
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)

    with pytest.raises(AlpacaError):
        AlpacaBroker()


def test_portfolio_parses_account_strings_and_is_cached() -> None:
    """Convert Alpaca's string fields and reuse the snapshot until refresh."""
    broker, session = make_broker(
        {
            ("GET", "/v2/account"): [ACCOUNT],
            ("GET", "/v2/positions"): [POSITIONS],
        }
    )

    portfolio = broker.portfolio
    portfolio_again = broker.portfolio

    assert portfolio.cash == 1000.50
    assert portfolio.positions["ACME"].quantity == 3.0
    assert portfolio.positions["ACME"].cost_basis == 10.0
    assert portfolio.positions["ACME"].last_price == 12.5
    assert portfolio_again is portfolio
    assert len(session.calls) == 2


def test_clock_reports_market_state() -> None:
    """Expose the market clock through a public method."""
    broker, _ = make_broker(
        {("GET", "/v2/clock"): [{"is_open": True, "next_close": "2026-07-21T16:00"}]}
    )

    assert broker.clock()["next_close"] == "2026-07-21T16:00"
    assert broker.is_market_open() is True


def test_order_quantity_submits_market_order_and_awaits_fill() -> None:
    """Send a signed quantity as a sided market order and record the fill."""
    broker, session = make_broker(
        {
            ("POST", "/v2/orders"): [{"id": "oid-1", "status": "accepted"}],
            ("GET", "/v2/orders/oid-1"): [{"status": "new"}, FILLED],
        }
    )

    outcome = broker.order_quantity("ACME", -3, "stop_loss")

    submitted = next(call for call in session.calls if call[0] == "POST")[2]
    assert submitted is not None
    assert submitted["symbol"] == "ACME"
    assert submitted["qty"] == "3"
    assert submitted["side"] == "sell"
    assert submitted["client_order_id"].startswith("tsa-stop_loss-")
    assert [(o.side, o.quantity, o.price) for o in broker.orders] == [
        (OrderSide.SELL, 3.0, 10.5)
    ]
    assert outcome.status is OrderOutcomeStatus.FILLED
    assert outcome.order_id == "oid-1"
    assert outcome.fill is broker.orders[0]


def test_order_invalidates_portfolio_snapshot() -> None:
    """Refetch account state after an order changes it server-side."""
    broker, session = make_broker(
        {
            ("GET", "/v2/account"): [ACCOUNT],
            ("GET", "/v2/positions"): [POSITIONS],
            ("POST", "/v2/orders"): [{"id": "oid-1", "status": "accepted"}],
            ("GET", "/v2/orders/oid-1"): [FILLED],
        }
    )

    broker.portfolio
    broker.order_quantity("ACME", 1, "rebalance_buy")
    broker.portfolio

    account_calls = [call for call in session.calls if "/v2/account" in call[1]]
    assert len(account_calls) == 2


def test_order_value_submits_rounded_notional() -> None:
    """Invest a cash amount through Alpaca's notional order support."""
    broker, session = make_broker(
        {
            ("POST", "/v2/orders"): [{"id": "oid-1", "status": "accepted"}],
            ("GET", "/v2/orders/oid-1"): [FILLED],
        }
    )

    broker.order_value("SGOV", 333.333, "buy_safe_etf")

    submitted = next(call for call in session.calls if call[0] == "POST")[2]
    assert submitted is not None
    assert submitted["notional"] == "333.33"
    assert submitted["side"] == "buy"
    assert "qty" not in submitted


def test_order_value_rejects_non_positive_amounts() -> None:
    """Refuse zero and negative notional orders locally."""
    broker, _ = make_broker({})

    with pytest.raises(ValueError):
        broker.order_value("SGOV", 0, "buy_safe_etf")


def test_zero_quantity_submits_nothing() -> None:
    """Treat a zero-share order as a no-op."""
    broker, session = make_broker({})

    outcome = broker.order_quantity("ACME", 0, "rebalance_sell")

    assert session.calls == []
    assert outcome.status is OrderOutcomeStatus.SKIPPED


@pytest.mark.parametrize(
    "status",
    [
        "calculated",
        "canceled",
        "done_for_day",
        "expired",
        "rejected",
    ],
)
def test_terminally_failed_order_returns_explicit_failure(status: str) -> None:
    """Return a terminal no-fill outcome without leaving an order pending."""
    broker, _ = make_broker(
        {
            ("POST", "/v2/orders"): [{"id": "oid-1", "status": "accepted"}],
            ("GET", "/v2/orders/oid-1"): [{"status": status}],
        }
    )

    outcome = broker.order_quantity("ACME", 1, "rebalance_buy")

    assert outcome.status is OrderOutcomeStatus.FAILED
    assert outcome.fill is None
    assert broker.orders == []


@pytest.mark.parametrize("status", ["canceled", "done_for_day"])
def test_terminal_order_preserves_confirmed_partial_fill(status: str) -> None:
    """Record shares that traded before the remaining order terminated."""
    broker, _ = make_broker(
        {
            ("POST", "/v2/orders"): [{"id": "oid-1", "status": "accepted"}],
            ("GET", "/v2/orders/oid-1"): [
                {
                    "status": status,
                    "filled_avg_price": "10.25",
                    "filled_qty": "1.5",
                }
            ],
        }
    )

    outcome = broker.order_quantity("ACME", 2, "rebalance_buy")

    assert outcome.status is OrderOutcomeStatus.PARTIAL
    assert outcome.fill is broker.orders[0]
    assert outcome.fill.quantity == 1.5
    assert outcome.fill.price == 10.25


def test_replaced_order_tracks_its_live_successor() -> None:
    """Return the replacement ID so callers never retry the original order."""
    broker, _ = make_broker(
        {
            ("POST", "/v2/orders"): [{"id": "oid-1", "status": "accepted"}],
            ("GET", "/v2/orders/oid-1"): [
                {"id": "oid-1", "status": "replaced", "replaced_by": "oid-2"}
            ],
            ("GET", "/v2/orders/oid-2"): [FILLED],
        }
    )

    replaced = broker.order_quantity("ACME", 3, "rebalance_buy")
    filled = broker.reconcile_order(replaced.reference or "")

    assert replaced.status is OrderOutcomeStatus.WORKING
    assert replaced.order_id == "oid-2"
    assert filled.status is OrderOutcomeStatus.FILLED
    assert filled.order_id == "oid-2"


def test_reconciliation_transfers_tracking_to_replacement() -> None:
    """Follow a replacement discovered after an order was already pending."""
    broker, _ = make_broker(
        {
            ("POST", "/v2/orders"): [{"id": "oid-1", "status": "accepted"}],
            ("GET", "/v2/orders/oid-1"): [
                {"id": "oid-1", "status": "new"},
                {"id": "oid-1", "status": "replaced", "replaced_by": "oid-2"},
            ],
        },
        fill_timeout=0.0,
    )

    pending = broker.order_quantity("ACME", 1, "rebalance_buy")
    replaced = broker.reconcile_order(pending.reference or "")

    assert pending.order_id == "oid-1"
    assert replaced.status is OrderOutcomeStatus.WORKING
    assert replaced.order_id == "oid-2"


def test_replacement_preserves_fills_from_both_orders() -> None:
    """Do not overwrite execution that occurred before replacement."""
    broker, _ = make_broker(
        {
            ("POST", "/v2/orders"): [{"id": "oid-1", "status": "accepted"}],
            ("GET", "/v2/orders/oid-1"): [
                {
                    "id": "oid-1",
                    "status": "replaced",
                    "replaced_by": "oid-2",
                    "filled_avg_price": "10.00",
                    "filled_qty": "1",
                }
            ],
            ("GET", "/v2/orders/oid-2"): [
                {
                    "id": "oid-2",
                    "status": "filled",
                    "filled_avg_price": "10.25",
                    "filled_qty": "2",
                }
            ],
        }
    )

    replaced = broker.order_quantity("ACME", 3, "rebalance_buy")
    broker.reconcile_order(replaced.reference or "")

    assert [(order.quantity, order.price) for order in broker.orders] == [
        (1.0, 10.0),
        (2.0, 10.25),
    ]


def test_replaced_order_without_successor_fails_closed() -> None:
    """Keep malformed replacement state ambiguous rather than retrying."""
    broker, _ = make_broker(
        {
            ("POST", "/v2/orders"): [{"id": "oid-1", "status": "accepted"}],
            ("GET", "/v2/orders/oid-1"): [
                {"id": "oid-1", "status": "replaced", "replaced_by": ""}
            ],
        }
    )

    outcome = broker.order_quantity("ACME", 1, "rebalance_buy")

    assert outcome.status is OrderOutcomeStatus.UNKNOWN
    assert outcome.order_id == "oid-1"


def test_unfilled_order_times_out_without_raising() -> None:
    """Leave a working order in place when the market is closed."""
    broker, _ = make_broker(
        {
            ("POST", "/v2/orders"): [{"id": "oid-1", "status": "accepted"}],
            ("GET", "/v2/orders/oid-1"): [{"status": "new"}],
        },
        fill_timeout=0.0,
    )

    outcome = broker.order_quantity("ACME", 2, "rebalance_buy")

    assert outcome.status is OrderOutcomeStatus.WORKING
    assert outcome.order_id == "oid-1"
    assert outcome.fill is None
    assert broker.orders == []


def test_partially_filled_order_reports_only_confirmed_execution() -> None:
    """Keep the open remainder distinct from its confirmed partial fill."""
    broker, _ = make_broker(
        {
            ("POST", "/v2/orders"): [{"id": "oid-1", "status": "accepted"}],
            ("GET", "/v2/orders/oid-1"): [
                {
                    "status": "partially_filled",
                    "filled_avg_price": "10.25",
                    "filled_qty": "1.5",
                }
            ],
        },
        fill_timeout=0.0,
    )

    outcome = broker.order_quantity("ACME", 2, "rebalance_buy")

    assert outcome.status is OrderOutcomeStatus.WORKING
    assert outcome.fill is broker.orders[0]
    assert outcome.fill.quantity == 1.5
    assert outcome.fill.price == 10.25


def test_reconciliation_replaces_cumulative_partial_fill_record() -> None:
    """Update one execution record instead of double-counting cumulative fills."""
    broker, _ = make_broker(
        {
            ("POST", "/v2/orders"): [{"id": "oid-1", "status": "accepted"}],
            ("GET", "/v2/orders/oid-1"): [
                {
                    "status": "partially_filled",
                    "filled_avg_price": "10.00",
                    "filled_qty": "1",
                },
                {
                    "status": "filled",
                    "filled_avg_price": "10.25",
                    "filled_qty": "2",
                },
            ],
        },
        fill_timeout=0.0,
    )

    pending = broker.order_quantity("ACME", 2, "rebalance_buy")
    filled = broker.reconcile_order("oid-1")

    assert pending.status is OrderOutcomeStatus.WORKING
    assert filled.status is OrderOutcomeStatus.FILLED
    assert len(broker.orders) == 1
    assert broker.orders[0].quantity == 2.0
    assert broker.orders[0].price == 10.25


def test_reconciliation_reports_expired_working_order_as_failed() -> None:
    """Release callers to retry after a previously working order terminates."""
    broker, _ = make_broker(
        {
            ("POST", "/v2/orders"): [{"id": "oid-1", "status": "accepted"}],
            ("GET", "/v2/orders/oid-1"): [
                {"status": "new"},
                {"status": "expired", "filled_qty": "0"},
            ],
        },
        fill_timeout=0.0,
    )

    pending = broker.order_quantity("ACME", 1, "rebalance_buy")
    failed = broker.reconcile_order("oid-1")

    assert pending.status is OrderOutcomeStatus.WORKING
    assert failed.status is OrderOutcomeStatus.FAILED
    assert failed.fill is None


def test_submission_without_order_id_is_explicitly_unknown() -> None:
    """Preserve an ambiguous accepted order for later reconciliation."""
    broker, _ = make_broker(
        {("POST", "/v2/orders"): [{"status": "accepted"}]}
    )

    outcome = broker.order_quantity("ACME", 1, "rebalance_buy")

    assert outcome.status is OrderOutcomeStatus.UNKNOWN
    assert outcome.order_id is None
    assert outcome.client_order_id is not None


def test_unknown_submission_can_be_reconciled_by_client_order_id() -> None:
    """Recover a POST response that omitted its broker order identifier."""
    broker, _ = make_broker(
        {
            ("POST", "/v2/orders"): [{"status": "accepted"}],
            ("GET", "/v2/orders:by_client_order_id"): [
                {
                    "id": "oid-1",
                    "client_order_id": "ignored-by-context",
                    "status": "filled",
                    "filled_avg_price": "10.5",
                    "filled_qty": "1",
                }
            ],
        }
    )

    unknown = broker.order_quantity("ACME", 1, "rebalance_buy")
    filled = broker.reconcile_order(unknown.reference or "")

    assert filled.status is OrderOutcomeStatus.FILLED
    assert filled.order_id == "oid-1"
    assert filled.fill is broker.orders[0]


@pytest.mark.parametrize(
    "filled",
    [
        {"status": "filled", "filled_avg_price": "10.5", "filled_qty": "0"},
        {"status": "filled", "filled_avg_price": None, "filled_qty": "1"},
        {"status": "filled", "filled_avg_price": "0", "filled_qty": "1"},
        {"status": "filled", "filled_avg_price": "10.5", "filled_qty": "-1"},
    ],
)
def test_malformed_filled_payload_is_not_recorded(filled: dict[str, object]) -> None:
    """Require positive finite quantity and price for every fill record."""
    broker, _ = make_broker(
        {
            ("POST", "/v2/orders"): [{"id": "oid-1", "status": "accepted"}],
            ("GET", "/v2/orders/oid-1"): [filled],
        }
    )

    outcome = broker.order_quantity("ACME", 1, "rebalance_buy")

    assert outcome.status is OrderOutcomeStatus.UNKNOWN
    assert outcome.order_id == "oid-1"
    assert broker.orders == []


def test_unknown_order_status_fails_closed() -> None:
    """Do not guess whether an unrecognized provider state remains executable."""
    broker, _ = make_broker(
        {
            ("POST", "/v2/orders"): [{"id": "oid-1", "status": "accepted"}],
            ("GET", "/v2/orders/oid-1"): [{"status": "mystery"}],
        }
    )

    outcome = broker.order_quantity("ACME", 1, "rebalance_buy")

    assert outcome.status is OrderOutcomeStatus.UNKNOWN
    assert outcome.order_id == "oid-1"


@pytest.mark.parametrize("status", ["stopped", "suspended"])
def test_non_terminal_order_remains_working(status: str) -> None:
    """Keep non-terminal Alpaca states open until they fill or terminate."""
    broker, _ = make_broker(
        {
            ("POST", "/v2/orders"): [{"id": "oid-1", "status": "accepted"}],
            ("GET", "/v2/orders/oid-1"): [{"status": status}],
        },
        fill_timeout=0.0,
    )

    outcome = broker.order_quantity("ACME", 1, "rebalance_buy")

    assert outcome.status is OrderOutcomeStatus.WORKING


@pytest.mark.parametrize("quantity", [True, float("nan"), float("inf")])
def test_order_quantity_rejects_invalid_numeric_values(quantity: float) -> None:
    """Reject values that cannot form valid Alpaca quantity strings."""
    broker, session = make_broker({})

    with pytest.raises(ValueError, match="quantity must be finite"):
        broker.order_quantity("ACME", quantity, "invalid")
    assert session.calls == []


@pytest.mark.parametrize(
    "value", [True, 0.001, float("nan"), float("inf")]
)
def test_order_value_rejects_invalid_or_sub_cent_values(value: float) -> None:
    """Refuse notionals that would be malformed or rounded down to zero."""
    broker, session = make_broker({})

    with pytest.raises(ValueError, match="finite and at least 0.01"):
        broker.order_value("SGOV", value, "invalid")
    assert session.calls == []


def test_transport_failure_uses_the_common_broker_error_boundary() -> None:
    """Translate request-layer failures into an engine-safe Alpaca error."""

    class FailingSession:
        def request(self, *args: object, **kwargs: object) -> object:
            del args, kwargs
            raise requests.Timeout("timed out")

    broker = AlpacaBroker(
        api_key="key", api_secret="secret", session=FailingSession()
    )

    outcome = broker.order_quantity("ACME", 1, "rebalance_buy")

    assert outcome.status is OrderOutcomeStatus.UNKNOWN
    assert outcome.order_id is None
    assert outcome.client_order_id is not None


def test_server_error_after_submission_attempt_is_unknown() -> None:
    """Avoid retrying a POST that a failing server may already have accepted."""

    class ServerErrorSession:
        def request(self, *args: object, **kwargs: object) -> FakeHTTPResponse:
            del args, kwargs
            return FakeHTTPResponse({"message": "failed"}, status_code=500)

    broker = AlpacaBroker(
        api_key="key", api_secret="secret", session=ServerErrorSession()
    )

    outcome = broker.order_quantity("ACME", 1, "rebalance_buy")

    assert outcome.status is OrderOutcomeStatus.UNKNOWN
    assert outcome.client_order_id is not None


def test_client_rejection_remains_a_definite_execution_error() -> None:
    """Allow safe retries when Alpaca explicitly rejects the HTTP request."""

    class ClientErrorSession:
        def request(self, *args: object, **kwargs: object) -> FakeHTTPResponse:
            del args, kwargs
            return FakeHTTPResponse({"message": "invalid"}, status_code=422)

    broker = AlpacaBroker(
        api_key="key", api_secret="secret", session=ClientErrorSession()
    )

    with pytest.raises(AlpacaError, match="HTTP 422"):
        broker.order_quantity("ACME", 1, "rebalance_buy")
