"""Mutable orchestration state owned by the strategy layer."""

from dataclasses import dataclass, field
from datetime import date


@dataclass(slots=True)
class StrategyState:
    """Mutable state that replaces the legacy platform-global ``g`` object.

    Attributes:
        stock_count: Desired number of equity holdings.
        candidates: Latest screened candidates in selection order.
        target_positions: Latest desired holdings.
        stopped_out: Whether a risk control liquidated equities today.
        stop_loss_etf_bought: Whether defensive cash was invested in the ETF.
        stop_loss_etf_order_reference: Identifier for an unresolved defensive order.
        pending_exit_orders: Unresolved risk-exit identifiers by symbol.
        incomplete_exit_reasons: Risk-exit reasons to retry by held symbol.
        pending_weekly_sale_orders: Unresolved weekly-sale identifiers by symbol.
        last_rebalance_date: Date of the latest completed weekly rebalance.
    """

    stock_count: int
    candidates: list[str] = field(default_factory=list)
    target_positions: list[str] = field(default_factory=list)
    stopped_out: bool = False
    stop_loss_etf_bought: bool = False
    stop_loss_etf_order_reference: str | None = None
    pending_exit_orders: dict[str, str] = field(default_factory=dict)
    incomplete_exit_reasons: dict[str, str] = field(default_factory=dict)
    pending_weekly_sale_orders: dict[str, str] = field(default_factory=dict)
    last_rebalance_date: date | None = None
