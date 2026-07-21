"""Tests for the Financial Modeling Prep data adapters."""

from datetime import date

import pytest

from trading_script_anatomy.data.models import RankedSecurity
from trading_script_anatomy.data.fmp_provider import (
    FMPClient,
    FMPError,
    FMPMarketDataProvider,
    FMPScreenerUniverseProvider,
    ScreenerQuery,
)

from tests.fakes import FakeHTTPSession

AS_OF = date(2026, 7, 21)

QUARTER_STATEMENTS = [
    {
        "date": "2026-06-30",
        "filingDate": "2026-08-10",
        "revenue": 9e6,
        "netIncome": 9e5,
    },
    {
        "date": "2026-03-31",
        "filingDate": "2026-05-12",
        "revenue": 8e6,
        "netIncome": 7e5,
    },
]


def make_provider(
    payloads: dict[str, object],
) -> tuple[FMPMarketDataProvider, FakeHTTPSession]:
    """Build a market-data provider backed by a scripted fake session."""
    session = FakeHTTPSession(
        {("GET", fragment): [payload] for fragment, payload in payloads.items()}
    )
    client = FMPClient(api_key="test-key", session=session)
    return FMPMarketDataProvider(client), session


def test_client_requires_an_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail fast when no API key is configured anywhere."""
    monkeypatch.delenv("FMP_API_KEY", raising=False)

    with pytest.raises(FMPError):
        FMPClient()


def test_client_raises_on_error_payload_despite_http_200() -> None:
    """Treat FMP's 200-with-error-message responses as failures."""
    session = FakeHTTPSession(
        {("GET", "profile"): [{"Error Message": "Invalid API key"}]}
    )
    client = FMPClient(api_key="bad-key", session=session)

    with pytest.raises(FMPError):
        client.get_json("profile", {"symbol": "AAPL"})


def test_daily_bars_orders_ascending_and_limits_rows() -> None:
    """Normalize FMP's newest-first rows to the protocol's oldest-first frame."""
    provider, _ = make_provider(
        {
            "historical-price-eod": [
                {"date": "2026-07-20", "open": 11.0, "close": 12.0},
                {"date": "2026-07-17", "open": 10.0, "close": 11.0},
                {"date": "2026-07-16", "open": 9.0, "close": 10.0},
            ]
        }
    )

    result = provider.daily_bars("IWM", periods=2, as_of=AS_OF)

    assert list(result.columns) == ["open", "close"]
    assert len(result) == 2
    assert result.iloc[-1]["close"] == 12.0
    assert result.index.is_monotonic_increasing


def test_security_info_maps_profile_fields() -> None:
    """Expose the profile fields the US eligibility filter relies on."""
    provider, _ = make_provider(
        {
            "profile": [
                {
                    "companyName": "Acme Micro Corp",
                    "ipoDate": "2019-05-01",
                    "exchangeShortName": "NASDAQ",
                    "isEtf": False,
                    "isAdr": False,
                    "isActivelyTrading": True,
                }
            ]
        }
    )

    info = provider.security_info("ACME")

    assert info is not None
    assert info.name == "Acme Micro Corp"
    assert info.listed_on == date(2019, 5, 1)
    assert info.exchange == "NASDAQ"
    assert info.is_actively_trading is True


def test_financial_snapshot_selects_latest_statement_filed_on_or_before() -> None:
    """Ignore statements filed after the evaluation date to avoid look-ahead."""
    provider, _ = make_provider(
        {
            "profile": [{"marketCap": 2.5e8}],
            "income-statement": QUARTER_STATEMENTS,
        }
    )

    snapshot = provider.financial_snapshot("ACME", AS_OF)

    assert snapshot is not None
    assert snapshot.market_value == 2.5e8
    assert snapshot.operating_revenue == 8e6
    assert snapshot.net_profit == 7e5
    assert snapshot.net_profit_parent == 7e5


def test_financial_snapshot_returns_none_without_a_filed_statement() -> None:
    """Return no snapshot when every statement postdates the evaluation date."""
    provider, _ = make_provider(
        {
            "profile": [{"marketCap": 2.5e8}],
            "income-statement": [QUARTER_STATEMENTS[0]],
        }
    )

    assert provider.financial_snapshot("ACME", AS_OF) is None


def test_profitability_does_not_require_market_cap() -> None:
    """Keep the profitability path independent of profile market-cap data."""
    provider, _ = make_provider(
        {
            "profile": [{"companyName": "Acme, no cap reported"}],
            "income-statement": QUARTER_STATEMENTS,
        }
    )

    profit = provider.profitability("ACME", AS_OF)

    assert profit is not None
    assert profit.operating_revenue == 8e6
    assert profit.net_profit_parent == 7e5
    assert provider.financial_snapshot("ACME", AS_OF) is None


def test_profile_is_cached_across_protocol_calls() -> None:
    """Fetch each symbol's profile once even when both methods need it."""
    provider, session = make_provider(
        {
            "profile": [{"companyName": "Acme", "marketCap": 2.5e8}],
            "income-statement": [QUARTER_STATEMENTS[1]],
        }
    )

    provider.security_info("ACME")
    provider.financial_snapshot("ACME", AS_OF)

    profile_calls = [call for call in session.calls if "profile" in call[1]]
    assert len(profile_calls) == 1


def test_screener_universe_sorts_ascending_and_caches_per_day() -> None:
    """Return smallest-cap-first symbols and reuse the same-day result."""
    session = FakeHTTPSession(
        {
            ("GET", "company-screener"): [
                [
                    {"symbol": "BIGG", "marketCap": 4.0e8},
                    {"symbol": "TINY", "marketCap": 6.0e7},
                    {"symbol": "MIDD", "marketCap": 2.0e8},
                    {"symbol": "BROKEN", "marketCap": None},
                ]
            ]
        }
    )
    client = FMPClient(api_key="test-key", session=session)
    provider = FMPScreenerUniverseProvider(
        client, {"IWM": ScreenerQuery(5e7, 5e8)}
    )

    first = provider.constituents("IWM", AS_OF)
    second = provider.constituents("IWM", AS_OF)
    ranked = provider.ranked_constituents("IWM", AS_OF)

    assert first == ("TINY", "MIDD", "BIGG")
    assert second == first
    assert ranked[0] == RankedSecurity("TINY", 6.0e7)
    assert len(session.calls) == 1


def test_screener_universe_rejects_unconfigured_index() -> None:
    """Raise KeyError for indices without a configured screen."""
    client = FMPClient(api_key="test-key", session=FakeHTTPSession({}))
    provider = FMPScreenerUniverseProvider(client, {})

    with pytest.raises(KeyError):
        provider.constituents("IWM", AS_OF)
