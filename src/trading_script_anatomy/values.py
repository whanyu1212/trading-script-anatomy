"""Value-coercion helpers shared by data adapters, strategy, and broker code."""

import pandas as pd


def to_finite_float(value: object) -> float | None:
    """Convert a provider value to a finite float when possible.

    Args:
        value: Numeric, string, or missing provider value.

    Returns:
        The finite float, or ``None`` when conversion is impossible.
    """
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if pd.notna(number) else None


def numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return finite numeric values from a data-frame column.

    Args:
        frame: Daily-bars frame from a market-data provider.
        column: Column name to extract.

    Returns:
        Finite values in frame order; empty when the column is absent.
    """
    if column not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").dropna()


def latest_number(frame: pd.DataFrame, column: str) -> float | None:
    """Return the last finite value from a data-frame column.

    Non-finite trailing values are skipped, so on multi-row frames this is
    the latest *finite* value rather than strictly the final row's value.

    Args:
        frame: Daily-bars frame from a market-data provider.
        column: Column name to extract.

    Returns:
        The most recent finite value, or ``None`` when none exists.
    """
    values = numeric_column(frame, column)
    return None if values.empty else float(values.iloc[-1])
