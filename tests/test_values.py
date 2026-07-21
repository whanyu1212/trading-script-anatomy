"""Tests for shared provider-value coercion helpers."""

import pandas as pd
import pytest

from trading_script_anatomy.values import latest_number, numeric_column, to_finite_float


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, 1.0),
        ("2.5", 2.5),
    ],
)
def test_to_finite_float_accepts_finite_numbers(
    value: object, expected: float
) -> None:
    """Convert finite numeric inputs and numeric strings."""
    assert to_finite_float(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "not-a-number",
        True,
        False,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_to_finite_float_rejects_invalid_or_non_finite_values(value: object) -> None:
    """Reject missing, malformed, NaN, and infinite provider values."""
    assert to_finite_float(value) is None


def test_numeric_column_keeps_only_finite_values_in_source_order() -> None:
    """Coerce numeric values while removing malformed and non-finite entries."""
    frame = pd.DataFrame(
        {
            "close": [
                "1.5",
                float("inf"),
                True,
                None,
                "broken",
                float("-inf"),
                2.0,
            ]
        }
    )

    result = numeric_column(frame, "close")

    assert result.tolist() == [1.5, 2.0]
    assert result.index.tolist() == [0, 6]


def test_numeric_column_preserves_numeric_dtype_when_every_value_is_invalid() -> None:
    """Return a numeric empty series after rejecting every observation."""
    frame = pd.DataFrame({"close": [True, "broken", float("inf")]})

    result = numeric_column(frame, "close")

    assert result.empty
    assert pd.api.types.is_numeric_dtype(result.dtype)


def test_latest_number_skips_trailing_non_finite_values() -> None:
    """Return the most recent finite observation rather than infinity."""
    frame = pd.DataFrame({"close": [1.0, 2.0, float("inf"), float("nan")]})

    assert latest_number(frame, "close") == 2.0


def test_latest_number_returns_none_without_finite_values() -> None:
    """Return no value when a column contains only invalid observations."""
    frame = pd.DataFrame({"close": [float("inf"), float("nan")]})

    assert latest_number(frame, "close") is None
