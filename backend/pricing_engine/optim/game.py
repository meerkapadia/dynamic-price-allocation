"""Duopoly price competition under multinomial-logit demand with an outside option.

Shares:  s_i = exp(a_i - b p_i) / (1 + exp(a_0 - b p_0) + exp(a_1 - b p_1))

Calibration (from the ML layer): own elasticity e0 = -b p0 (1 - s0); cross elasticity g = b p1 s1;
inside-share coverage theta = s0 + s1  ⇒  closed-form s0, then b, a_0, a_1, market size M.

Best response (exact, closed form, overflow-safe):
    FOC  b (p_i - c_i)(1 - s_i) = 1   ⇒   p_i* = c_i + (1 + W(e^z)) / b,
    z = a_i - b c_i - 1 - log(1 + e^{a_j - b p_j}),   W(e^z) = wrightomega(z).
Logit profit is strictly quasi-concave in own price, so the FOC root is the global best response.

Bertrand-Nash: damped best-response iteration (contraction for logit; convergence is checked, not assumed).
Stackelberg (we lead): maximise π_0(p, BR_1(p)) — vectorised global grid + bounded Brent refinement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import wrightomega

from ..config import Settings


@dataclass(frozen=True, slots=True)
class LogitMarket:
    b: float
    a0: float
    a1: float
    market_size: float
    c0: float  # our marginal cost relevant for competition (full unit cost)
    c1: float  # competitor marginal cost
    p1_ref: float  # competitor price used for calibration (after spike clipping)

    def shares(self, p0: np.ndarray | float, p1: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        u0 = self.a0 - self.b * np.asarray(p0, dtype=np.float64)
        u1 = self.a1 - self.b * np.asarray(p1, dtype=np.float64)
        m = np.maximum(np.maximum(u0, u1), 0.0)  # log-sum-exp stabilisation
        e0, e1, eo = np.exp(u0 - m), np.exp(u1 - m), np.exp(-m)
        den = eo + e0 + e1
        return e0 / den, e1 / den

    def demand0(self, p0: np.ndarray | float, p1: np.ndarray | float) -> np.ndarray:
        return self.market_size * self.shares(p0, p1)[0]

    def profit0(self, p0: np.ndarray | float, p1: np.ndarray | float) -> np.ndarray:
        return (np.asarray(p0) - self.c0) * self.demand0(p0, p1)

    def profit1(self, p0: np.ndarray | float, p1: np.ndarray | float) -> np.ndarray:
        return (np.asarray(p1) - self.c1) * self.market_size * self.shares(p0, p1)[1]

    def best_response1(self, p0: np.ndarray | float) -> np.ndarray:
        u0 = self.a0 - self.b * np.asarray(p0, dtype=np.float64)
        z = self.a1 - self.b * self.c1 - 1.0 - np.logaddexp(0.0, u0)
        return self.c1 + (1.0 + np.real(wrightomega(z))) / self.b

    def best_response0(self, p1: np.ndarray | float) -> np.ndarray:
        u1 = self.a1 - self.b * np.asarray(p1, dtype=np.float64)
        z = self.a0 - self.b * self.c0 - 1.0 - np.logaddexp(0.0, u1)
        return self.c0 + (1.0 + np.real(wrightomega(z))) / self.b


@dataclass(slots=True)
class CompetitionResult:
    market: LogitMarket
    our_share: float
    competitor_share: float
    implied_cross_elasticity: float
    nash: tuple[float, float]
    nash_iterations: int
    nash_profit0: float
    stackelberg: tuple[float, float]
    stackelberg_profit0: float
    warnings: list[str] = field(default_factory=list)


def calibrate(
    *,
    our_price: float,
    competitor_price: float,
    base_demand: float,
    own_elasticity: float,
    cross_elasticity: float,
    our_cost: float,
    competitor_cost: float | None,
    settings: Settings,
) -> tuple[LogitMarket, float, float, list[str]]:
    warnings: list[str] = []
    if not (np.isfinite([our_price, competitor_price, base_demand, own_elasticity, cross_elasticity]).all()):
        raise ValueError("non-finite calibration input")
    if our_price <= 0 or competitor_price <= 0 or base_demand <= 0:
        raise ValueError("calibration requires positive prices and demand")
    lo, hi = settings.price_ratio_clip
    ratio = competitor_price / our_price
    if not (lo <= ratio <= hi):
        warnings.append(f"Competitor price is {ratio:.1f}× ours (a spike); clipped to [{lo}, {hi}] for the market model.")
        competitor_price = our_price * float(np.clip(ratio, lo, hi))

    theta = float(np.clip(settings.market_coverage, 0.05, 0.95))
    e0 = -min(own_elasticity, -1e-3)
    g = max(cross_elasticity, 1e-3)
    # (1 - s0)/s1 = R with s1 = theta - s0
    R = (e0 / our_price) / (g / competitor_price)
    s0 = (R * theta - 1.0) / (R - 1.0) if abs(R - 1.0) > 1e-9 else np.nan
    margin = 0.02 * theta
    if not np.isfinite(s0) or not (margin < s0 < theta - margin):
        warnings.append("Own and competitor elasticities do not fit a two-seller logit exactly; assumed an even split of shoppers.")
        s0 = theta / 2.0
    s1 = theta - s0
    s_out = 1.0 - theta
    b = e0 / (our_price * (1.0 - s0))
    a0 = np.log(s0 / s_out) + b * our_price
    a1 = np.log(s1 / s_out) + b * competitor_price
    market_size = base_demand / s0

    if competitor_cost is None:
        # invert competitor's FOC assuming it currently best-responds: c1 = p1 - 1 / (b (1 - s1))
        c1 = competitor_price - 1.0 / (b * (1.0 - s1))
        floor = 0.3 * competitor_price
        if c1 < floor:
            warnings.append("Competitor cost implied by its price looked too low; assumed at least 30% of its price.")
            c1 = floor
    else:
        c1 = competitor_cost
    market = LogitMarket(b=float(b), a0=float(a0), a1=float(a1), market_size=float(market_size), c0=our_cost, c1=float(c1),
                         p1_ref=float(competitor_price))
    return market, float(s0), float(s1), warnings


def bertrand_nash(market: LogitMarket, p0: float, p1: float, settings: Settings) -> tuple[float, float, int, bool]:
    damp = 0.7
    for it in range(1, settings.nash_max_iter + 1):
        n0 = float(market.best_response0(p1))
        n1 = float(market.best_response1(p0))
        n0 = damp * n0 + (1 - damp) * p0
        n1 = damp * n1 + (1 - damp) * p1
        if abs(n0 - p0) + abs(n1 - p1) < settings.nash_tol * max(1.0, p0 + p1):
            return n0, n1, it, True
        p0, p1 = n0, n1
    return p0, p1, settings.nash_max_iter, False


def stackelberg_leader(market: LogitMarket, lo: float, hi: float, n_grid: int = 400) -> tuple[float, float, float]:
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        raise ValueError("invalid Stackelberg search interval")
    grid = np.linspace(lo, hi, n_grid)
    vals = market.profit0(grid, market.best_response1(grid))
    k = int(np.nanargmax(vals))
    a, b = grid[max(k - 1, 0)], grid[min(k + 1, n_grid - 1)]
    if b > a:
        res = minimize_scalar(lambda p: -float(market.profit0(p, market.best_response1(p))), bounds=(a, b),
                              method="bounded", options={"xatol": 1e-8 * max(1.0, hi)})
        p = float(res.x) if res.success and -res.fun >= vals[k] else float(grid[k])
    else:
        p = float(grid[k])
    p1 = float(market.best_response1(p))
    return p, p1, float(market.profit0(p, p1))


def analyse(
    *,
    our_price: float,
    competitor_price: float,
    base_demand: float,
    own_elasticity: float,
    cross_elasticity: float,
    our_cost: float,
    competitor_cost: float | None,
    settings: Settings,
) -> CompetitionResult:
    market, s0, s1, warnings = calibrate(
        our_price=our_price, competitor_price=competitor_price, base_demand=base_demand,
        own_elasticity=own_elasticity, cross_elasticity=cross_elasticity, our_cost=our_cost,
        competitor_cost=competitor_cost, settings=settings,
    )
    n0, n1, iters, ok = bertrand_nash(market, our_price, market.p1_ref, settings)
    if not ok:
        warnings.append("Nash price search did not converge; showing the last iterate.")
    hi = max(our_price, n0, market.c0) * 4.0 + 10.0 / market.b
    st0, st1, st_pi = stackelberg_leader(market, market.c0, hi)
    implied_cross = float(market.b * market.p1_ref * s1)
    return CompetitionResult(
        market=market, our_share=s0, competitor_share=s1, implied_cross_elasticity=implied_cross,
        nash=(n0, n1), nash_iterations=iters, nash_profit0=float(market.profit0(n0, n1)),
        stackelberg=(st0, st1), stackelberg_profit0=st_pi, warnings=warnings,
    )


def reaction_demand(market: LogitMarket, prices: np.ndarray, react: bool) -> tuple[np.ndarray, np.ndarray]:
    """Demand for our prices when competitor reacts (Stackelberg follower) or holds its (calibrated) price."""
    prices = np.asarray(prices, dtype=np.float64)
    p1 = market.best_response1(prices) if react else np.full_like(prices, market.p1_ref)
    return market.demand0(prices, p1), p1


__all__ = ["LogitMarket", "CompetitionResult", "analyse", "calibrate", "bertrand_nash", "stackelberg_leader",
           "reaction_demand"]
