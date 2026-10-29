"""Runtime configuration (environment-overridable, immutable).

Nothing in here is a per-product pricing input: every number the optimiser needs about a product is derived
from the store database (see `ops/state.py`). These settings describe the solver and how the service runs.
"""

from __future__ import annotations

import os
from datetime import date
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


def _env(name: str, default: str) -> str:
    return os.environ.get(f"PRICING_{name}", default)


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    # --- store database
    db_path: Path = Field(default_factory=lambda: Path(_env("DB_PATH", "./data/store.db")))
    random_seed: int = Field(default_factory=lambda: int(_env("SEED", "42")))
    history_days: int = Field(default_factory=lambda: int(_env("HISTORY_DAYS", "730")))
    demand_scale: float = Field(default_factory=lambda: float(_env("DEMAND_SCALE", "0.6")))
    # demo store only: history covers the `history_days` before this date
    start_date: date = Field(default_factory=lambda: date.fromisoformat(_env("START_DATE", date.today().isoformat())))

    # --- service
    demo_data: bool = Field(default_factory=lambda: _env("DEMO_DATA", "1") not in ("0", "false", "no"))
    api_key: str = Field(default_factory=lambda: _env("API_KEY", ""))          # empty = write endpoints open
    daily_run_at: str = Field(default_factory=lambda: _env("DAILY_RUN_AT", ""))  # "06:00" = compute every morning
    workers: int = Field(default_factory=lambda: int(_env("WORKERS", "4")))
    refit_every_days: int = 7

    # --- demand estimation
    fit_window_days: int = 730
    holdout_days: int = 60
    ridge: float = 1e-3
    elasticity_bounds: tuple[float, float] = (-8.0, -0.3)
    cross_elasticity_bounds: tuple[float, float] = (0.0, 5.0)

    # --- competition (MNL calibration)
    market_coverage: float = Field(default_factory=lambda: float(_env("MARKET_COVERAGE", "0.6")))
    price_ratio_clip: tuple[float, float] = (0.2, 5.0)
    nash_max_iter: int = 500
    nash_tol: float = 1e-9
    reaction_threshold: float = 0.12          # estimated competitor reaction slope above which it is a follower

    # --- stochastic DP
    price_grid_size: int = Field(default_factory=lambda: int(_env("PRICE_GRID", "72")))
    max_inventory_states: int = Field(default_factory=lambda: int(_env("MAX_STATES", "600")))
    max_horizon_days: int = 30

    # --- simulation / guardrails
    mc_paths: int = Field(default_factory=lambda: int(_env("MC_PATHS", "1500")))
    kkt_tolerance: float = 1e-4


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
