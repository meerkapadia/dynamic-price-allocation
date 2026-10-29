"""Loss-prevention layer: hard profit floor + KKT certification of the constrained price decision.

Problem at t = 0:   max_p f(p)   s.t.   g₁(p) = p_floor − p ≤ 0,   g₂(p) = p − p_upper ≤ 0
KKT (maximisation): f'(p*) = −μ₁ + μ₂,  μ ≥ 0,  μᵢ gᵢ(p*) = 0,  gᵢ(p*) ≤ 0.

p_floor_t = (unit_cost + handling + storage_rate · (days_in_stock + t)) · (1 + min_margin)
p_ceiling  = MRP (selling above the printed MRP is illegal in India)
The floor is never relaxed: rate limits, ceilings and rounding are applied *subject to* it.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..schemas import KKTReport, ProductState

TICK = 1.0  # prices are set in whole rupees


def full_cost_path(state: ProductState, horizon: int) -> np.ndarray:
    t = np.arange(horizon, dtype=np.float64)
    return state.unit_cost + state.handling_cost_per_unit + state.storage_cost_per_unit_day * (state.days_in_stock + t)


def floor_path(state: ProductState, horizon: int) -> np.ndarray:
    return full_cost_path(state, horizon) * (1.0 + state.min_margin_pct)


def ceil_tick(x: float) -> float:
    return math.ceil(round(x / TICK, 6)) * TICK


@dataclass(frozen=True, slots=True)
class ActionBounds:
    lower: float
    upper: float
    floor: float
    rate_limit_overridden: bool


def action_bounds(state: ProductState, floor0: float, ceiling: float) -> ActionBounds:
    """Intersect [floor, ceiling] with the anti-spike rate limit; the floor always wins."""
    floor0 = ceil_tick(floor0)
    rl_lo = state.current_price * (1.0 - state.max_price_change_pct)
    rl_hi = state.current_price * (1.0 + state.max_price_change_pct)
    lo = max(floor0, rl_lo)
    hi = min(ceiling, rl_hi)
    overridden = False
    if hi < lo:
        # rate limit incompatible with the floor (cost shock) → jump straight to the floor
        lo = hi = floor0
        overridden = True
    hi = max(hi, lo)
    return ActionBounds(lower=lo, upper=hi, floor=floor0, rate_limit_overridden=overridden)


def certify(objective: Callable[[float], float], price: float, bounds: ActionBounds, tol: float) -> KKTReport:
    span = max(bounds.upper - bounds.lower, 1e-9)
    h = max(1e-6 * max(1.0, price), 1e-4 * span)
    lo, hi = bounds.lower, bounds.upper
    # second-order one-sided differences at active bounds so we never evaluate outside the feasible set
    if price - h < lo:
        if price + 2 * h <= hi + 1e-12:
            grad = (-3 * objective(price) + 4 * objective(price + h) - objective(price + 2 * h)) / (2 * h)
        else:  # degenerate interval (lo == hi): probe outward only to sign the multiplier
            grad = (objective(price + h) - objective(price)) / h
    elif price + h > hi:
        grad = (3 * objective(price) - 4 * objective(price - h) + objective(price - 2 * h)) / (2 * h)
    else:
        grad = (objective(price + h) - objective(price - h)) / (2 * h)
    if not math.isfinite(grad):
        grad = 0.0

    scale = max(1.0, abs(objective(price)))
    at_lo = abs(price - lo) <= 1e-6 * max(1.0, price)
    at_hi = abs(price - hi) <= 1e-6 * max(1.0, price)
    mu1 = max(0.0, -grad) if at_lo else 0.0
    mu2 = max(0.0, grad) if at_hi else 0.0
    active: list[str] = []
    if at_lo:
        active.append("profit_floor" if abs(lo - bounds.floor) < 1e-9 else "rate_limit_lower")
    if at_hi:
        active.append("ceiling_or_rate_limit_upper")
    stationarity = abs(grad + mu1 - mu2) / scale
    comp_slack = abs(mu1 * (bounds.lower - price)) + abs(mu2 * (price - bounds.upper))
    primal = (price >= bounds.floor - 1e-9) and (lo - 1e-9 <= price <= hi + 1e-9)
    # dual feasibility: an interior point must be stationary; a bound point must have the sign that
    # pushes against the bound. Tolerance is relative to objective scale per unit price.
    rel_tol = tol * scale / max(1.0, price)
    dual_ok = stationarity <= tol * 10 or abs(grad + mu1 - mu2) <= rel_tol * 10
    return KKTReport(
        price=price, lower_bound=lo, upper_bound=hi, gradient=grad, mu_floor=mu1, mu_ceiling=mu2,
        active_constraints=active, stationarity_residual=stationarity, complementary_slackness=comp_slack,
        dual_feasible=bool(dual_ok), primal_feasible=bool(primal),
        floor_shadow_price=mu1 if "profit_floor" in active else 0.0,
    )


def finalize_price(raw: float, bounds: ActionBounds) -> float:
    """Round to whole rupees without ever undercutting the floor; NaN → safest feasible price."""
    if not math.isfinite(raw):
        return ceil_tick(max(bounds.lower, bounds.floor))
    p = min(max(raw, bounds.lower), bounds.upper)
    p = round(p / TICK) * TICK
    if p > bounds.upper + 1e-9:
        p = math.floor(bounds.upper / TICK + 1e-6) * TICK
    if p < bounds.floor:
        p = ceil_tick(bounds.floor)
    return p
