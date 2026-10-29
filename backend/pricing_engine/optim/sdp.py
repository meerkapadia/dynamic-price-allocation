"""Finite-horizon stochastic dynamic program for perishable inventory pricing.

State x_t ∈ {0..X} (units or lots), action p ∈ P_t = {p ∈ grid : p ≥ floor_t}, demand D_t ~ Poisson(λ(p)·m_t),
where m_t is the forecast seasonal multiplier for day t (yearly season × weekday × festival × trend, from the
fitted demand model) so the policy anticipates a Diwali ramp or the end of mango season.

    V_t(x) = max_{p ∈ P_t}  (p − h_c)·E[min(D,x)] − h_s·E[x − min(D,x)] + E[ Ṽ_{t+1}(x − min(D,x)) ]
    Ṽ_{t+1}(y) = E[ V_{t+1}(Binomial(y, 1−δ)) ]           (daily spoilage)
    V_T(x) = salvage · x                                   (expiry: salvage > 0, disposal < 0)

Exact backward induction (finite horizon ⇒ no fixed-point convergence risk). Each stage is fully
vectorised over (price × inventory): the continuation Σ_{d<x} pmf(d) Ṽ(x−d) is a causal convolution
evaluated as one GEMM against a zero-copy Toeplitz view of Ṽ, with the Poisson pmf truncated at its
1e-13 quantile (O(|P|·X·d_max), exact to float64); E[min(D,x)] = Σ_{k<x} P(D>k) is a cumulative sum.
The loss-prevention floor is imposed *inside* the Bellman operator (infeasible actions = −∞), so the
policy is optimal for the constrained problem, not a clipped unconstrained one.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import minimize_scalar
from scipy.stats import binom, poisson

DemandFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class SDPProblem:
    price_grid: np.ndarray          # (P,) strictly increasing
    demand_fn: DemandFn             # price -> expected units/day at weekday multiplier 1
    demand_mult: np.ndarray         # (T,) forecast demand multiplier per period (m_0 = 1)
    floors: np.ndarray              # (T,) per-period hard price floor
    inventory: int                  # initial units
    handling_cost: float            # per unit sold
    holding_cost: float             # per unit per day carried overnight
    salvage_value: float            # per unit left at expiry (negative = disposal fee)
    spoilage_rate: float            # δ ∈ [0, 1)
    max_states: int = 2000


@dataclass(slots=True)
class SDPSolution:
    values: np.ndarray        # (T+1, X+1) value per lot-state
    policy_idx: np.ndarray    # (T, X+1) index into price_grid (-1 = no stock)
    price_grid: np.ndarray
    lot_size: int
    lam_grid: np.ndarray      # (P,) λ(p) units/day at multiplier 1
    problem: SDPProblem
    solve_ms: float
    v_tilde_next0: np.ndarray  # Ṽ_1 = E[V_1(after spoilage)], cached for O(x) t=0 action evaluation

    @property
    def horizon(self) -> int:
        return self.policy_idx.shape[0]

    def policy_price(self, t: int, units: np.ndarray) -> np.ndarray:
        lots = np.clip(np.rint(np.asarray(units) / self.lot_size).astype(np.int64), 0, self.values.shape[1] - 1)
        idx = self.policy_idx[t, lots]
        prices = self.price_grid[np.maximum(idx, 0)]
        return np.maximum(prices, self.problem.floors[t])

    def value_at(self, units: int) -> float:
        x = min(int(round(units / self.lot_size)), self.values.shape[1] - 1)
        return float(self.values[0, x])


class _StageKernel:
    """Caches truncated Poisson pmf / E[sales] / tail matrices per distinct demand multiplier (rounded to 0.1%).

    The pmf is truncated at d_max = F⁻¹(1 − 1e-13); mass beyond it is < 1e-13 and the exact tail
    P(D ≥ x) is still used for the stock-out term, so truncation error is below float64 noise.
    """

    _TAIL_EPS = 1e-13

    def __init__(self, lam_lots: np.ndarray, n_states: int) -> None:
        self.lam = lam_lots
        self.n = n_states
        self.k = np.arange(n_states)
        self._cache: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def get(self, mult: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = round(float(mult), 3)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        lam = (self.lam * key)[:, None]
        d_max = int(min(self.n, poisson.isf(self._TAIL_EPS, float(lam.max()) + 1e-12) + 2))
        pmf = poisson.pmf(self.k[None, :d_max], lam)                # P(D = d), d < d_max
        sf = poisson.sf(self.k[None, :], lam)                       # P(D > k)
        e_sales = np.zeros_like(sf)
        np.cumsum(sf[:, :-1], axis=1, out=e_sales[:, 1:])           # E[min(D,x)] = Σ_{k<x} P(D>k)
        tail = np.ones_like(sf)
        tail[:, 1:] = sf[:, :-1]                                    # P(D ≥ x)
        out = (pmf, e_sales, tail)
        self._cache[key] = out
        return out


def _continuation(pmf: np.ndarray, tail: np.ndarray, v: np.ndarray) -> np.ndarray:
    """cont[p, x] = Σ_{d<x} pmf[p,d]·v[x−d] + P(D≥x)·v[0]   via one BLAS GEMM on a Toeplitz view."""
    n, d_max = v.size, pmf.shape[1]
    padded = np.concatenate([np.zeros(d_max - 1), v])
    toeplitz = sliding_window_view(padded, d_max)[:, ::-1]           # [x, d] = v[x−d] (0 when d > x)
    cont = pmf @ toeplitz.T                                          # Σ_{d≤x} pmf(d)·v(x−d)
    diag = np.arange(min(n, d_max))
    cont[:, diag] -= pmf[:, diag] * v[0]                             # drop d = x (stock-out handled by tail)
    cont += tail * v[0]
    return cont


def _spoilage_matrix(n_states: int, delta: float) -> np.ndarray | None:
    if delta <= 0.0:
        return None
    y = np.arange(n_states)
    return binom.pmf(y[None, :], y[:, None], 1.0 - delta)  # B[y, s] = P(survivors = s | y)


def solve(problem: SDPProblem) -> SDPSolution:
    t0 = time.perf_counter()
    grid = np.asarray(problem.price_grid, dtype=np.float64)
    T = int(problem.demand_mult.size)
    if grid.ndim != 1 or grid.size < 2 or not np.all(np.diff(grid) > 0):
        raise ValueError("price grid must be 1-D, strictly increasing, size ≥ 2")
    if T < 1 or problem.floors.shape != (T,):
        raise ValueError("floors must align with horizon")
    if not np.all(np.isfinite(grid)) or not np.all(np.isfinite(problem.floors)):
        raise ValueError("non-finite grid or floors")
    if grid[-1] < problem.floors.max():
        raise ValueError("price grid does not reach the maximum floor; no feasible action in some period")
    if problem.inventory < 0:
        raise ValueError("inventory must be non-negative")

    lot = max(1, int(np.ceil(problem.inventory / problem.max_states)))
    X = int(np.ceil(problem.inventory / lot))
    n = X + 1

    lam = np.asarray(problem.demand_fn(grid), dtype=np.float64)
    if lam.shape != grid.shape or not np.all(np.isfinite(lam)):
        raise ValueError("demand function returned invalid values")
    lam = np.maximum(lam, 0.0)

    kernel = _StageKernel(lam / lot, n)
    B = _spoilage_matrix(n, problem.spoilage_rate)
    x = np.arange(n, dtype=np.float64)
    unit_margin = (grid - problem.handling_cost)[:, None] * lot
    hold = problem.holding_cost * lot

    V = np.empty((T + 1, n))
    V[T] = problem.salvage_value * lot * x
    policy = np.empty((T, n), dtype=np.int64)
    v_next = V[T]

    for t in range(T - 1, -1, -1):
        pmf, e_sales, tail = kernel.get(problem.demand_mult[t])
        v_next = V[t + 1] if B is None else B @ V[t + 1]
        cont = _continuation(pmf, tail, v_next)
        Q = unit_margin * e_sales - hold * (x[None, :] - e_sales) + cont
        Q[grid < problem.floors[t] - 1e-12, :] = -np.inf
        best = np.argmax(Q, axis=0)
        V[t] = Q[best, np.arange(n)]
        best[0] = -1
        policy[t] = best

    if not np.all(np.isfinite(V)):
        raise FloatingPointError("Bellman recursion produced non-finite values")
    return SDPSolution(V, policy, grid, lot, lam, problem, (time.perf_counter() - t0) * 1e3, v_next.copy())


def q_value(sol: SDPSolution, price: float, units: int) -> float:
    """Exact first-period action value Q_0(p, x) for any continuous price, O(x) (refinement + KKT)."""
    t = 0
    pb = sol.problem
    lot = sol.lot_size
    xl = min(int(round(units / lot)), sol.values.shape[1] - 1)
    if xl == 0:
        return float(sol.values[t, 0])
    lam = float(np.maximum(pb.demand_fn(np.array([price]))[0], 0.0)) * pb.demand_mult[t] / lot
    k = np.arange(xl + 1)
    pmf = poisson.pmf(k, lam)
    sf = poisson.sf(k, lam)
    e_sales = float(sf[:xl].sum())
    v_next = sol.v_tilde_next0
    cont = float(np.dot(pmf[:xl], v_next[xl - np.arange(xl)]) + (sf[xl - 1]) * v_next[0])
    return (price - pb.handling_cost) * lot * e_sales - pb.holding_cost * lot * (xl - e_sales) + cont


def refine_first_action(sol: SDPSolution, units: int, lo: float, hi: float) -> tuple[float, float]:
    """Global-on-grid then bounded-Brent continuous refinement of the t=0 action within [lo, hi]."""
    if hi < lo:
        raise ValueError("empty action interval")
    if hi - lo < 1e-9:
        return lo, q_value(sol, lo, units)
    cand = np.unique(np.concatenate([np.linspace(lo, hi, 64), sol.price_grid[(sol.price_grid >= lo) & (sol.price_grid <= hi)]]))
    vals = np.array([q_value(sol, float(p), units) for p in cand])
    k = int(np.argmax(vals))
    a, b = cand[max(k - 1, 0)], cand[min(k + 1, cand.size - 1)]
    best_p, best_v = float(cand[k]), float(vals[k])
    if b > a:
        res = minimize_scalar(lambda p: -q_value(sol, float(p), units), bounds=(a, b), method="bounded",
                              options={"xatol": 1e-6 * max(1.0, hi)})
        if res.success and -res.fun > best_v:
            best_p, best_v = float(res.x), float(-res.fun)
    return best_p, best_v
