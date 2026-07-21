"""Tests for the Alpaca broker adapter."""

import pytest

from trading_script_anatomy.broker.alpaca import AlpacaBroker, AlpacaError
from trading_script_anatomy.broker.models import OrderSide

from tests.fakes import FakeHTTPSession

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

    broker.order_quantity("ACME", -3, "stop_loss")

    submitted = next(call for call in session.calls if call[0] == "POST")[2]
    assert submitted is not None
    assert submitted["symbol"] == "ACME"
    assert submitted["qty"] == "3"
    assert submitted["side"] == "sell"
    assert submitted["client_order_id"].startswith("tsa-stop_loss-")
    assert [(o.side, o.quantity, o.price) for o in broker.orders] == [
        (OrderSide.SELL, 3.0, 10.5)
    ]


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

    broker.order_quantity("ACME", 0, "rebalance_sell")

    assert session.calls == []


def test_terminally_failed_order_raises() -> None:
    """Surface rejected orders as errors instead of assuming a fill."""
    broker, _ = make_broker(
        {
            ("POST", "/v2/orders"): [{"id": "oid-1", "status": "accepted"}],
            ("GET", "/v2/orders/oid-1"): [{"status": "rejected"}],
        }
    )

    with pytest.raises(AlpacaError):
        broker.order_quantity("ACME", 1, "rebalance_buy")


def test_unfilled_order_times_out_without_raising() -> None:
    """Leave a working order in place when the market is closed."""
    broker, _ = make_broker(
        {
            ("POST", "/v2/orders"): [{"id": "oid-1", "status": "accepted"}],
            ("GET", "/v2/orders/oid-1"): [{"status": "new"}],
        },
        fill_timeout=0.0,
    )

    broker.order_quantity("ACME", 2, "rebalance_buy")

    assert [(o.quantity, o.price) for o in broker.orders] == [(2.0, 0.0)]
