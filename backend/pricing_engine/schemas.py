"""Domain contracts. `ProductState` is assembled from the store database by `ops/state.py`; nobody types it in."""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

Money = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
PositiveMoney = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class CompetitorResponse(str, Enum):
    STACKELBERG = "stackelberg"  # rival follows our price; we move first
    STATIC = "static"            # rival price treated as fixed


class ProductState(_Strict):
    sku: str = Field(min_length=1, max_length=64)

    # unit economics (purchase lots + product master + category policy)
    unit_cost: PositiveMoney
    handling_cost_per_unit: Money = 0.0
    storage_cost_per_unit_day: Money = 0.0
    days_in_stock: Annotated[int, Field(ge=0, le=3650)] = 0
    min_margin_pct: Annotated[float, Field(ge=0.0, le=5.0, allow_inf_nan=False)] = 0.05
    salvage_value_per_unit: Annotated[float, Field(ge=-1e6, le=1e6, allow_inf_nan=False)] = 0.0

    # market (price book, MRP, competitor feed)
    current_price: PositiveMoney
    mrp: PositiveMoney
    competitor_price: PositiveMoney
    max_price_change_pct: Annotated[float, Field(gt=0.0, le=1.0, allow_inf_nan=False)] = 0.1

    # stock (lots and write-off history)
    inventory: Annotated[int, Field(ge=0, le=5_000_000)]
    shelf_life_days: Annotated[int, Field(ge=1, le=3650)]
    perishable: bool = False
    daily_spoilage_rate: Annotated[float, Field(ge=0.0, lt=1.0, allow_inf_nan=False)] = 0.0
    replenishment_days: Annotated[int, Field(ge=1, le=365)] = 7   # supplier lead time + 1 review day

    competitor_response: CompetitorResponse = CompetitorResponse.STACKELBERG

    @property
    def full_unit_cost(self) -> float:
        return self.unit_cost + self.handling_cost_per_unit + self.storage_cost_per_unit_day * self.days_in_stock


class KKTReport(_Strict):
    price: float
    lower_bound: float
    upper_bound: float
    gradient: float
    mu_floor: float
    mu_ceiling: float
    active_constraints: list[str]
    stationarity_residual: float
    complementary_slackness: float
    dual_feasible: bool
    primal_feasible: bool
    floor_shadow_price: float
