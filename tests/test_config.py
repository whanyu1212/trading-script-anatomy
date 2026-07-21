"""Tests for immutable strategy configuration."""

from trading_script_anatomy.config import StrategyConfig


def test_default_benchmark_uses_yahoo_finance_shenzhen_suffix() -> None:
    """Use the Yahoo Finance suffix for the configured Shenzhen benchmark."""
    assert StrategyConfig().benchmark_symbol == "399106.SZ"
