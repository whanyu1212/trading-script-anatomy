"""Stock-universe filtering and target selection."""

from collections.abc import Callable, Collection, Sequence
from datetime import date
import logging

import pandas as pd

from trading_script_anatomy.config import StrategyConfig
from trading_script_anatomy.data.models import (
    FinancialSnapshot,
    ProfitabilitySnapshot,
    SecurityInfo,
)
from trading_script_anatomy.data.protocols import (
    IndexUniverseProvider,
    MarketDataProvider,
    RankedUniverseProvider,
)
from trading_script_anatomy.strategy.cn_filters import a_share_eligibility
from trading_script_anatomy.strategy.state import StrategyState
from trading_script_anatomy.values import latest_number, numeric_column

EligibilityFilter = Callable[[str, SecurityInfo], bool]

MAX_RANKED_WALK = 400
"""Per-symbol data requests permitted per ranked walk before giving up.

Bounds the worst case where almost nothing in the universe qualifies, which
would otherwise degenerate back into fetching fundamentals for every symbol.
"""


class StockSelector:
    """Apply the strategy's universe, fundamental, and price filters.

    Args:
        config: Immutable strategy parameters.
        market_data: Source for security, price, and fundamental data.
        universe: Source for benchmark constituent symbols.
        logger: Optional logger used for filter diagnostics.
        eligibility: Market-specific security filter. Defaults to the
            mainland-China name and board rules; US strategies must supply
            their own because the A-share checks misfire on US names.
    """

    def __init__(
        self,
        config: StrategyConfig,
        market_data: MarketDataProvider,
        universe: IndexUniverseProvider,
        logger: logging.Logger | None = None,
        eligibility: EligibilityFilter | None = None,
    ) -> None:
        self._config = config
        self._market_data = market_data
        self._universe = universe
        self._logger = logger or logging.getLogger(__name__)
        self._eligibility = eligibility or a_share_eligibility

    def select_targets(
        self,
        as_of: date,
        held_symbols: Collection[str],
        state: StrategyState,
    ) -> list[str]:
        """Select target holdings and update the supplied strategy state.

        Args:
            as_of: Date on which selection is evaluated.
            held_symbols: Symbols already held by the portfolio.
            state: Mutable state to update with candidates and desired count.

        Returns:
            Up to the calculated desired number of selected symbols, ordered by
            ascending market value.
        """
        if isinstance(self._universe, RankedUniverseProvider):
            finance_pool = self.filter_ranked_universe(as_of, state)
        else:
            basic_pool = self.filter_basic_stock_pool(as_of)
            finance_pool = (
                self.filter_finance_and_market_value(basic_pool, as_of, state)
                if basic_pool
                else []
            )
        if not finance_pool:
            state.candidates = []
            return []

        price_pool = self.filter_price(finance_pool, held_symbols, as_of)
        state.candidates = price_pool
        if not price_pool:
            return []

        target_count = self.adjust_position_count(as_of, state)
        return price_pool[:target_count]

    def filter_basic_stock_pool(self, as_of: date) -> list[str]:
        """Remove ST, delisting, excluded-board, and recently listed symbols.

        Args:
            as_of: Date used to calculate listing age and retrieve constituents.

        Returns:
            Eligible benchmark constituent symbols in universe-provider order.
        """
        return [
            symbol
            for symbol in self._universe.constituents(
                self._config.benchmark_symbol, as_of
            )
            if self._passes_basic_filters(symbol, as_of)
        ]

    def filter_ranked_universe(
        self, as_of: date, state: StrategyState
    ) -> list[str]:
        """Walk a ranked universe and stop once enough candidates qualify.

        Produces the same smallest-qualifying symbols as the exhaustive
        ``filter_basic_stock_pool`` and ``filter_finance_and_market_value``
        pair, but examines symbols in ascending market-value order and stops
        after ``stock_count * finance_candidate_multiplier`` qualifiers, so
        per-symbol data requests cover dozens of symbols instead of the whole
        universe. The universe's ranking values are authoritative for the
        market-value band; fundamentals are fetched only for profitability.

        Args:
            as_of: Date on which selection is evaluated.
            state: State whose desired stock count determines pool breadth.

        Returns:
            Eligible symbols in ascending market-value order.
        """
        try:
            ranked = self._universe.ranked_constituents(
                self._config.benchmark_symbol, as_of
            )
        except Exception as error:
            self._logger.warning("Unable to read ranked universe: %s", error)
            return []

        target_count = state.stock_count * self._config.finance_candidate_multiplier
        selected: list[str] = []
        examined = 0
        for entry in ranked:
            if len(selected) >= target_count:
                break
            if entry.market_value < self._config.min_market_value:
                continue
            if entry.market_value > self._config.max_market_value:
                break
            if examined >= MAX_RANKED_WALK:
                self._logger.warning(
                    "Ranked walk stopped after %d symbols with %d of %d "
                    "candidates; universe may be mostly ineligible",
                    examined,
                    len(selected),
                    target_count,
                )
                break
            examined += 1
            if self._passes_basic_filters(entry.symbol, as_of) and self._is_profitable(
                entry.symbol, as_of
            ):
                selected.append(entry.symbol)
        return selected

    def _passes_basic_filters(self, symbol: str, as_of: date) -> bool:
        """Return whether metadata, eligibility, and listing age permit a symbol."""
        try:
            info = self._market_data.security_info(symbol)
        except Exception as error:
            self._logger.warning(
                "Unable to read security info for %s: %s", symbol, error
            )
            return False

        if info is None:
            return False
        if not self._eligibility(symbol, info):
            return False
        return not (
            info.listed_on is not None
            and (as_of - info.listed_on).days < self._config.min_listing_days
        )

    def _is_profitable(self, symbol: str, as_of: date) -> bool:
        """Return whether a symbol's fundamentals pass the profitability rules."""
        try:
            snapshot = self._market_data.profitability(symbol, as_of)
        except Exception as error:
            self._logger.warning(
                "Unable to read profitability for %s: %s", symbol, error
            )
            return False
        return snapshot is not None and self._snapshot_is_profitable(snapshot)

    def _snapshot_is_profitable(
        self, snapshot: FinancialSnapshot | ProfitabilitySnapshot
    ) -> bool:
        """Return whether financial values pass the profitability rules."""
        return not (
            snapshot.net_profit <= 0
            or snapshot.net_profit_parent <= 0
            or snapshot.operating_revenue <= self._config.min_operating_revenue
        )

    def filter_finance_and_market_value(
        self,
        symbols: Sequence[str],
        as_of: date,
        state: StrategyState,
    ) -> list[str]:
        """Filter financial quality and retain the smallest eligible companies.

        Args:
            symbols: Symbols that passed the basic universe filter.
            as_of: Date at which financial values are requested.
            state: State whose desired stock count determines pool breadth.

        Returns:
            Eligible symbols sorted by ascending market value.
        """
        eligible: list[tuple[str, float]] = []
        for symbol in symbols:
            try:
                snapshot = self._market_data.financial_snapshot(symbol, as_of)
            except Exception as error:
                self._logger.warning(
                    "Unable to read financial snapshot for %s: %s", symbol, error
                )
                continue

            if snapshot is None:
                continue
            if not (
                self._config.min_market_value
                <= snapshot.market_value
                <= self._config.max_market_value
            ):
                continue
            if not self._snapshot_is_profitable(snapshot):
                continue
            eligible.append((symbol, snapshot.market_value))

        eligible.sort(key=lambda item: item[1])
        candidate_count = state.stock_count * self._config.finance_candidate_multiplier
        return [symbol for symbol, _ in eligible[:candidate_count]]

    def filter_price(
        self,
        symbols: Sequence[str],
        held_symbols: Collection[str],
        as_of: date,
    ) -> list[str]:
        """Apply price and optional exchange-limit filters to candidates.

        Yahoo Finance does not expose ``high_limit`` and ``low_limit`` columns.
        When those columns are unavailable, only the price-ceiling rule applies.

        Args:
            symbols: Fundamentally eligible symbols.
            held_symbols: Symbols already held, which bypass the price ceiling.
            as_of: Date used to retrieve the latest bar.

        Returns:
            Symbols that pass applicable price and limit filters.
        """
        valid_symbols: list[str] = []
        for symbol in symbols:
            try:
                bars = self._market_data.daily_bars(symbol, periods=1, as_of=as_of)
                close = latest_number(bars, "close")
            except Exception as error:
                self._logger.warning("Unable to read price for %s: %s", symbol, error)
                continue

            if close is None or close <= 0:
                continue
            is_held = symbol in held_symbols
            if not is_held and _is_price_limited(bars, close):
                continue
            if is_held or close <= self._config.highest_price:
                valid_symbols.append(symbol)
        return valid_symbols

    def adjust_position_count(self, as_of: date, state: StrategyState) -> int:
        """Update the desired count from benchmark price versus its 10-day mean.

        Args:
            as_of: Date used to retrieve benchmark history.
            state: Mutable state whose stock count is updated.

        Returns:
            Updated desired equity-position count.
        """
        try:
            bars = self._market_data.daily_bars(
                self._config.benchmark_symbol, periods=10, as_of=as_of
            )
            closes = numeric_column(bars, "close")
        except Exception as error:
            self._logger.warning("Unable to read benchmark history: %s", error)
            return state.stock_count

        if closes.empty:
            return state.stock_count
        close = float(closes.iloc[-1])
        average = float(closes.mean())
        if close == 0 or average == 0:
            return state.stock_count

        difference = close - average
        for upper, lower, stock_count in self._config.position_mapping:
            if difference < upper and difference >= lower:
                state.stock_count = stock_count
                break
        return state.stock_count


def _is_price_limited(frame: pd.DataFrame, close: float) -> bool:
    """Return whether the closing price is at an available price limit."""
    high_limit = latest_number(frame, "high_limit")
    low_limit = latest_number(frame, "low_limit")
    return (
        high_limit is not None
        and high_limit > 0
        and close >= high_limit * 0.998
    ) or (
        low_limit is not None
        and low_limit > 0
        and close <= low_limit * 1.002
    )
