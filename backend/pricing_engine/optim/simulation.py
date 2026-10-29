"""Vectorised Monte-Carlo policy evaluation with common random numbers (CRN).

All policies face the *same* demand uniforms and spoilage stream, so differences in outcomes are due to
pricing alone (variance-reduced paired comparison). Loops only over time; paths are vectorised.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.stats import poisson

PricePolicy = Callable[[int, np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class SimEnv:
    demand_fn: Callable[[np.ndarray], np.ndarray]
    demand_mult: np.ndarray       # (T,)
    full_cost: np.ndarray          # (T,) per-unit full cost on day t (for loss accounting)
    floors: np.ndarray             # (T,)
    sunk_cost_per_unit: float      # unit cost + storage already accrued
    handling_cost: float
    holding_cost: float
    salvage_value: float
    spoilage_rate: float
    inventory: int


@dataclass(slots=True)
class SimResult:
    name: str
    profit: np.ndarray             # (N,)
    revenue: np.ndarray
    sold: np.ndarray
    wasted: np.ndarray
    loss_units: np.ndarray
    price_sum: np.ndarray
    price_days: np.ndarray
    min_price: float
    inv_paths: np.ndarray          # (T+1, N) start-of-day inventory
    price_paths: np.ndarray        # (T, N) (nan when out of stock)


def simulate(env: SimEnv, policies: dict[str, PricePolicy], n_paths: int, seed: int) -> dict[str, SimResult]:
    T = env.demand_mult.size
    rng = np.random.default_rng(seed)
    U = rng.random((T, n_paths))
    spoil_seed = int(rng.integers(0, 2**31 - 1))
    out: dict[str, SimResult] = {}
    for name, policy in policies.items():
        sp_rng = np.random.default_rng(spoil_seed)
        inv = np.full(n_paths, env.inventory, dtype=np.int64)
        inv_paths = np.empty((T + 1, n_paths), dtype=np.int64)
        price_paths = np.full((T, n_paths), np.nan)
        revenue = np.zeros(n_paths)
        cash = np.zeros(n_paths)
        sold = np.zeros(n_paths)
        spoiled = np.zeros(n_paths)
        loss_units = np.zeros(n_paths)
        min_price = np.inf
        for t in range(T):
            inv_paths[t] = inv
            p = np.asarray(policy(t, inv), dtype=np.float64)
            if p.shape != (n_paths,) or not np.all(np.isfinite(p)):
                raise ValueError(f"policy {name!r} produced invalid prices at t={t}")
            lam = np.maximum(env.demand_fn(p) * env.demand_mult[t], 1e-12)
            d = poisson.ppf(U[t], lam).astype(np.int64)
            s = np.minimum(d, inv)
            has = inv > 0
            price_paths[t, has] = p[has]
            if has.any():
                min_price = min(min_price, float(p[has].min()))
            revenue += p * s
            cash += (p - env.handling_cost) * s
            sold += s
            loss_units += np.where(p < env.full_cost[t] - 1e-9, s, 0)
            left = inv - s
            cash -= env.holding_cost * left
            if env.spoilage_rate > 0:
                surv = sp_rng.binomial(left, 1.0 - env.spoilage_rate)
                spoiled += left - surv
                left = surv
            inv = left
        inv_paths[T] = inv
        cash += env.salvage_value * inv
        profit = cash - env.sunk_cost_per_unit * env.inventory
        days = np.sum(~np.isnan(price_paths), axis=0)
        out[name] = SimResult(
            name=name, profit=profit, revenue=revenue, sold=sold, wasted=spoiled + inv, loss_units=loss_units,
            price_sum=np.nansum(price_paths, axis=0), price_days=days,
            min_price=float(min_price) if np.isfinite(min_price) else 0.0,
            inv_paths=inv_paths, price_paths=price_paths,
        )
    return out


def constant_policy(price_fn: Callable[[int], float]) -> PricePolicy:
    return lambda t, inv: np.full(inv.shape, price_fn(t), dtype=np.float64)
