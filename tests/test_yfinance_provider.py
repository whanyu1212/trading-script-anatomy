"""Tests for the Yahoo Finance data adapter."""

from datetime import date
import logging

import pandas as pd
import pytest

from trading_script_anatomy.data.yfinance_provider import YFinanceMarketDataProvider


class FakeTicker:
    """Provide deterministic Yahoo Finance responses for adapter tests."""

    def history(self, **kwargs: object) -> pd.DataFrame:
        """Return representative Yahoo Finance daily history.

        Args:
            **kwargs: Request options accepted for interface compatibility.

        Returns:
            Daily bars with Yahoo Finance column casing.
        """
        del kwargs
        return pd.DataFrame(
            {
                "Open": [10.0, 11.0],
                "Close": [11.0, 12.0],
            },
            index=pd.date_range("2025-06-01", periods=2, tz="Asia/Shanghai"),
        )


class NoFinancialsTicker:
    """Provide an empty financial statement for warning tests."""

    financials = pd.DataFrame()


def test_daily_bars_normalizes_column_names_and_limits_rows() -> None:
    """Return lower-case trailing bars that meet the data-provider protocol."""
    provider = YFinanceMarketDataProvider(ticker_factory=lambda symbol: FakeTicker())

    result = provider.daily_bars("399106.SZ", periods=1, as_of=date(2025, 6, 3))

    assert list(result.columns) == ["open", "close"]
    assert len(result) == 1
    assert result.iloc[0]["close"] == 12.0
    assert result.index.tz is None


def test_historical_fundamentals_warn_about_look_ahead_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Emit one look-ahead warning per provider for historical requests."""
    provider = YFinanceMarketDataProvider(
        ticker_factory=lambda symbol: NoFinancialsTicker()
    )

    with caplog.at_level(logging.WARNING):
        provider.profitability("AAPL", date(2020, 1, 2))
        provider.profitability("AAPL", date(2020, 1, 3))

    warnings = [r for r in caplog.records if "point-in-time" in r.message]
    assert len(warnings) == 1
