# Repository Guide

## Toolchain and Checks

- This is one Python 3.12+ package under `src/trading_script_anatomy`; use the committed `uv.lock` via `uv sync` and `uv run ...`.
- The complete network-free check is `uv run pytest` (66 tests). Run one file with `uv run pytest tests/test_engine.py` or one case with `uv run pytest tests/test_engine.py::test_weekly_rebalance_buys_selected_positions_equally`.
- There is no configured formatter, linter, type checker, code generator, CI workflow, or pre-commit hook. Do not invent one as a required verification step.

## Architecture Contracts

- `StrategyEngine` is the scheduler-independent orchestration entrypoint. It must not read the clock or schedule itself; callers pass every `date`/`datetime` and invoke the lifecycle methods.
- `StrategyConfig()` and `StockSelector` default to the legacy A-share behavior. US flows need both `us_strategy_config()` and an explicit `us_eligibility` passed to `StrategyEngine` or `Backtester`; the US config alone does not replace the eligibility filter.
- Keep strategy code dependent on the protocols in `data/protocols.py` and `broker/protocols.py`. Adapter seams are exercised with deterministic fakes in `tests/fakes.py`; the unit suite must not require credentials or network access.
- Data adapters must return date-indexed bars oldest-to-newest with lower-case `open` and `close`, ending inclusively at `as_of`. Fundamentals must exclude reports filed after `as_of` or clearly warn when they cannot honor that contract.
- A `RankedUniverseProvider` must return ascending market value. Selection skips entries below the band, stops at the first entry above it, and caps the per-symbol walk at 400.
- Preserve the backtest clock split: on day D, `DelayedMarketData` lets strategy decisions see bars only through D-1; the driver fills at D's open and marks equity at D's close.
- Historical FMP results are not fully point-in-time: filed income statements are, but profile market caps are current and the screener is current-listings-only (`as_of` is merely its cache key). Treat such runs as machinery checks, not performance evidence.

## Environment and Live Scripts

- Library code never loads `.env`. Executables must opt in with `load_dotenv`; existing process variables win over file values. The expected keys are in `.env.example`.
- `uv run python examples/check_fmp.py` makes live FMP calls, and its screener path requires an FMP Starter plan.
- `uv run python examples/demo_backtest.py` makes roughly 100 FMP calls and deliberately substitutes a static large-cap universe and SPY for SGOV; do not reuse those demo substitutions as live strategy settings.
- `uv run python examples/check_alpaca.py` targets Alpaca paper trading by default, but when the US market is open it submits a $2 notional SPY buy. Do not run it as a read-only verification during market hours.
