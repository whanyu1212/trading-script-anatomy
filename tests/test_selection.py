"""Tests for stock-universe filtering and selection."""

from datetime import date

from trading_script_anatomy.config import StrategyConfig
from trading_script_anatomy.data.models import (
    FinancialSnapshot,
    RankedSecurity,
    SecurityInfo,
)
from trading_script_anatomy.strategy.state import StrategyState
from trading_script_anatomy.strategy.selection import StockSelector

from tests.fakes import FakeMarketData, FakeRankedUniverse, FakeUniverse, bars


AS_OF = date(2025, 6, 3)
OLD_LISTING_DATE = date(2023, 1, 1)


def test_selector_filters_universe_then_orders_by_market_value() -> None:
    """Keep eligible companies in ascending market-value order."""
    symbols = (
        "000001.SZ",
        "000002.SZ",
        "300001.SZ",
        "000003.SZ",
        "000004.SZ",
    )
    data = FakeMarketData(
        infos={
            "000001.SZ": SecurityInfo("000001.SZ", "Alpha", OLD_LISTING_DATE),
            "000002.SZ": SecurityInfo("000002.SZ", "Beta", OLD_LISTING_DATE),
            "300001.SZ": SecurityInfo("300001.SZ", "Growth", OLD_LISTING_DATE),
            "000003.SZ": SecurityInfo("000003.SZ", "ST Gamma", OLD_LISTING_DATE),
            "000004.SZ": SecurityInfo("000004.SZ", "Recent", date(2025, 1, 1)),
        },
        financials={
            "000001.SZ": FinancialSnapshot(2e9, 2e8, 1e7, 1e7),
            "000002.SZ": FinancialSnapshot(1.5e9, 2e8, 1e7, 1e7),
        },
        bars={
            "399106.SZ": bars([100.0] * 10),
            "000001.SZ": bars([10.0]),
            "000002.SZ": bars([11.0]),
        },
    )
    selector = StockSelector(StrategyConfig(), data, FakeUniverse(symbols))
    state = StrategyState(stock_count=2)

    targets = selector.select_targets(AS_OF, set(), state)

    assert targets == ["000002.SZ", "000001.SZ"]
    assert state.candidates == targets


def test_existing_holding_bypasses_new_purchase_price_ceiling() -> None:
    """Preserve the legacy allowance for an expensive existing holding."""
    symbol = "000001.SZ"
    selector = StockSelector(
        StrategyConfig(highest_price=50.0),
        FakeMarketData(bars={symbol: bars([80.0])}),
        FakeUniverse(()),
    )

    assert selector.filter_price([symbol], {symbol}, AS_OF) == [symbol]
    assert selector.filter_price([symbol], set(), AS_OF) == []


def test_ranked_walk_stops_fetching_after_enough_candidates() -> None:
    """Fetch per-symbol data only until the candidate pool is full."""
    config = StrategyConfig(
        min_market_value=100, max_market_value=1000, min_operating_revenue=10
    )
    symbols = ["S1", "S2", "S3", "S4", "S5", "S6"]
    data = FakeMarketData(
        infos={s: SecurityInfo(s, s, OLD_LISTING_DATE) for s in symbols},
        financials={s: FinancialSnapshot(0, 100, 5, 5) for s in symbols},
    )
    universe = FakeRankedUniverse(
        [RankedSecurity(s, 200.0 + i) for i, s in enumerate(symbols)]
    )
    selector = StockSelector(config, data, universe)

    pool = selector.filter_ranked_universe(AS_OF, StrategyState(stock_count=1))

    assert pool == ["S1", "S2", "S3"]
    assert data.info_calls == ["S1", "S2", "S3"]
    assert data.financial_calls == ["S1", "S2", "S3"]


def test_ranked_walk_skips_below_band_and_stops_above_band() -> None:
    """Never fetch data for symbols outside the market-value band."""
    config = StrategyConfig(
        min_market_value=100, max_market_value=1000, min_operating_revenue=10
    )
    data = FakeMarketData(
        infos={"OK": SecurityInfo("OK", "In Band", OLD_LISTING_DATE)},
        financials={"OK": FinancialSnapshot(0, 100, 5, 5)},
    )
    universe = FakeRankedUniverse(
        [
            RankedSecurity("TOOSMALL", 50.0),
            RankedSecurity("OK", 200.0),
            RankedSecurity("TOOBIG", 2000.0),
            RankedSecurity("BIGGER", 3000.0),
        ]
    )
    selector = StockSelector(config, data, universe)

    pool = selector.filter_ranked_universe(AS_OF, StrategyState(stock_count=2))

    assert pool == ["OK"]
    assert data.info_calls == ["OK"]
    assert data.financial_calls == ["OK"]


def test_ranked_and_exhaustive_paths_select_the_same_targets() -> None:
    """Keep the lazy walk behaviorally equivalent to the exhaustive funnel."""
    infos = {
        "000001.SZ": SecurityInfo("000001.SZ", "Alpha", OLD_LISTING_DATE),
        "000002.SZ": SecurityInfo("000002.SZ", "Beta", OLD_LISTING_DATE),
        "000003.SZ": SecurityInfo("000003.SZ", "ST Gamma", OLD_LISTING_DATE),
    }
    financials = {
        "000001.SZ": FinancialSnapshot(2e9, 2e8, 1e7, 1e7),
        "000002.SZ": FinancialSnapshot(1.5e9, 2e8, 1e7, 1e7),
        "000003.SZ": FinancialSnapshot(1.2e9, 2e8, 1e7, 1e7),
    }
    price_bars = {
        "399106.SZ": bars([100.0] * 10),
        "000001.SZ": bars([10.0]),
        "000002.SZ": bars([11.0]),
    }
    ranked = FakeRankedUniverse(
        [
            RankedSecurity("000003.SZ", 1.2e9),
            RankedSecurity("000002.SZ", 1.5e9),
            RankedSecurity("000001.SZ", 2e9),
        ]
    )
    exhaustive = FakeUniverse(("000001.SZ", "000003.SZ", "000002.SZ"))

    targets: list[list[str]] = []
    for universe in (ranked, exhaustive):
        data = FakeMarketData(infos=infos, financials=financials, bars=price_bars)
        selector = StockSelector(StrategyConfig(), data, universe)
        targets.append(
            selector.select_targets(AS_OF, set(), StrategyState(stock_count=2))
        )

    assert targets[0] == targets[1] == ["000002.SZ", "000001.SZ"]


def test_custom_eligibility_filter_replaces_a_share_rules() -> None:
    """Let a market-specific filter override the default A-share checks."""
    data = FakeMarketData(
        infos={
            "FE": SecurityInfo("FE", "FirstEnergy", OLD_LISTING_DATE),
            "XYZ": SecurityInfo("XYZ", "Excluded Corp", OLD_LISTING_DATE),
        }
    )
    selector = StockSelector(
        StrategyConfig(),
        data,
        FakeUniverse(("FE", "XYZ")),
        eligibility=lambda symbol, info: symbol != "XYZ",
    )

    assert selector.filter_basic_stock_pool(AS_OF) == ["FE"]


def test_default_a_share_rules_exclude_st_names() -> None:
    """Preserve the legacy behavior when no filter is supplied."""
    data = FakeMarketData(
        infos={"000003.SZ": SecurityInfo("000003.SZ", "ST Gamma", OLD_LISTING_DATE)}
    )
    selector = StockSelector(StrategyConfig(), data, FakeUniverse(("000003.SZ",)))

    assert selector.filter_basic_stock_pool(AS_OF) == []


def test_price_limit_filter_is_skipped_when_provider_lacks_limit_columns() -> None:
    """Accept a valid price when the provider cannot supply price limits."""
    symbol = "000001.SZ"
    selector = StockSelector(
        StrategyConfig(),
        FakeMarketData(bars={symbol: bars([20.0])}),
        FakeUniverse(()),
    )

    assert selector.filter_price([symbol], set(), AS_OF) == [symbol]
