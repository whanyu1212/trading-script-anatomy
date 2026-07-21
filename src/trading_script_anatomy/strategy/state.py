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
        last_rebalance_date: Date of the latest completed weekly rebalance.
    """

    stock_count: int
    candidates: list[str] = field(default_factory=list)
    target_positions: list[str] = field(default_factory=list)
    stopped_out: bool = False
    stop_loss_etf_bought: bool = False
    last_rebalance_date: date | None = None
