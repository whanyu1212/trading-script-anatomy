"""Index-universe adapters."""

from collections.abc import Mapping, Sequence
from datetime import date


class StaticIndexUniverseProvider:
    """Provide a caller-managed static universe for one or more indices.

    This adapter is suitable for deterministic tests and research snapshots.
    Point-in-time historical research should use a provider with historical
    constituent data.

    Args:
        constituents_by_index: Mapping from index symbol to ordered constituents.
    """

    def __init__(self, constituents_by_index: Mapping[str, Sequence[str]]) -> None:
        self._constituents_by_index = {
            index: tuple(symbols) for index, symbols in constituents_by_index.items()
        }

    def constituents(self, index_symbol: str, as_of: date) -> tuple[str, ...]:
        """Return configured constituents for an index.

        Args:
            index_symbol: Provider-specific index ticker.
            as_of: Requested date, retained for protocol compatibility.

        Returns:
            Configured constituent ticker symbols.

        Raises:
            KeyError: If no universe was configured for ``index_symbol``.
        """
        del as_of
        return self._constituents_by_index[index_symbol]
