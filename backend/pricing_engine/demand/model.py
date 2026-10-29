"""Demand estimation from sales history: Poisson maximum likelihood solved with damped Newton's method.

For one SKU and day t with our price p, competitor price c and printed MRP m:

    log E[q_t] = θ₀ + β·log(p/m) + γ·log(c/m)                 price response (β = own, γ = cross elasticity)
               + Σ_k a_k sin(2πk·doy/365.25) + b_k cos(…)      yearly seasonality, k = 1..3
               + Σ_d δ_d·[weekday = d]                         weekly pattern (Monday baseline)
               + Σ_e λ_e·w_e(t)                                festival / sale windows from the calendar
               + τ·years + π·promo                             trend and the offer-badge effect

The negative log-likelihood ℓ(θ) = Σ μ_t − y_t·log μ_t (+ ridge) is convex, so Newton's method with step halving
converges to the global optimum, typically in 6–10 iterations; the inverse Hessian (Fisher information) gives
standard errors, scaled by the Pearson dispersion because daily sales are over-dispersed.

Days that ended with zero stock are dropped: the store sold everything it had, so sales understate demand
(a censored observation). Fitting on them would bias seasonality peaks and price elasticity downward.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import time
from dataclasses import dataclass

import numpy as np

from ..config import Settings
from ..data.calendar import EVENT_KEYS, day_of_year, event_weights, weekday
from ..data.market import EPOCH

log = logging.getLogger(__name__)

HARMONICS = 3
WEEKDAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
FEATURES: tuple[str, ...] = (
    "intercept", "log_price", "log_competitor",
    *[f"{f}{k}" for k in range(1, HARMONICS + 1) for f in ("sin", "cos")],
    *[f"wd_{d}" for d in WEEKDAY_NAMES[1:]],
    *[f"ev_{e}" for e in EVENT_KEYS],
    "trend", "promo",
)
IDX = {f: i for i, f in enumerate(FEATURES)}
_SEASON = [IDX[f"{f}{k}"] for k in range(1, HARMONICS + 1) for f in ("sin", "cos")]
_WEEK = [IDX[f"wd_{d}"] for d in WEEKDAY_NAMES[1:]]
_EVENTS = [IDX[f"ev_{e}"] for e in EVENT_KEYS]


def _days(ords: np.ndarray) -> np.ndarray:
    return (np.asarray(ords, dtype=np.int64) - 719163).astype("datetime64[D]")


def time_features(ords: np.ndarray) -> np.ndarray:
    """Columns of the design matrix that depend only on the date (n × len(FEATURES), price columns zero)."""
    ords = np.asarray(ords, dtype=np.int64)
    d = _days(ords)
    n = ords.size
    X = np.zeros((n, len(FEATURES)))
    X[:, 0] = 1.0
    doy = day_of_year(d).astype(float)
    for k in range(1, HARMONICS + 1):
        ang = 2 * np.pi * k * doy / 365.25
        X[:, IDX[f"sin{k}"]] = np.sin(ang)
        X[:, IDX[f"cos{k}"]] = np.cos(ang)
    wd = weekday(d)
    for j, name in enumerate(WEEKDAY_NAMES[1:], start=1):
        X[:, IDX[f"wd_{name}"]] = (wd == j).astype(float)
    X[:, _EVENTS] = event_weights(d)
    X[:, IDX["trend"]] = (ords - EPOCH) / 365.25
    return X


def design(ords, price, comp, mrp: float, promo) -> np.ndarray:
    X = time_features(ords)
    X[:, IDX["log_price"]] = np.log(np.asarray(price, float) / mrp)
    X[:, IDX["log_competitor"]] = np.log(np.asarray(comp, float) / mrp)
    X[:, IDX["promo"]] = np.asarray(promo, float)
    return X


@dataclass(slots=True)
class NewtonResult:
    coef: np.ndarray
    cov: np.ndarray
    iterations: int
    dispersion: float
    converged: bool


def poisson_newton(X: np.ndarray, y: np.ndarray, ridge: float, max_iter: int = 60, tol: float = 1e-9) -> NewtonResult:
    """Damped Newton–Raphson for ridge-penalised Poisson regression (log link)."""
    n, k = X.shape
    R = np.full(k, ridge * n)
    R[0] = 0.0                                       # intercept unpenalised
    theta = np.zeros(k)
    theta[0] = math.log(max(y.mean(), 1e-3))

    def objective(t: np.ndarray) -> float:
        eta = np.clip(X @ t, -30, 30)
        return float(np.sum(np.exp(eta) - y * eta) + 0.5 * np.sum(R * t * t))

    f = objective(theta)
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        eta = np.clip(X @ theta, -30, 30)
        mu = np.exp(eta)
        grad = X.T @ (mu - y) + R * theta
        H = (X * mu[:, None]).T @ X + np.diag(R)
        step = np.linalg.solve(H + 1e-10 * np.eye(k), grad)
        t = 1.0
        while t > 1e-6:                                # Armijo backtracking keeps every iterate a descent step
            cand = theta - t * step
            fc = objective(cand)
            if fc <= f - 1e-4 * t * float(grad @ step):
                break
            t *= 0.5
        theta, f_prev, f = cand, f, fc
        if np.max(np.abs(t * step)) < 1e-7 or abs(f_prev - f) < tol * max(1.0, abs(f)):
            converged = True
            break
    mu = np.exp(np.clip(X @ theta, -30, 30))
    H = (X * mu[:, None]).T @ X + np.diag(R)
    dof = max(n - k, 1)
    dispersion = float(max(np.sum((y - mu) ** 2 / np.maximum(mu, 1e-9)) / dof, 1.0))
    cov = np.linalg.inv(H) * dispersion
    return NewtonResult(theta, cov, it, dispersion, converged)


# ====================================================================== fitted model


@dataclass(slots=True)
class SkuDemand:
    sku: str
    mrp: float
    coef: np.ndarray
    se: np.ndarray
    fitted_on: str
    n_obs: int
    n_censored: int
    holdout_wape: float
    dispersion: float
    iterations: int
    competitor_reaction: float
    mean_x: tuple[float, float] = (0.0, 0.0)   # average log(p/m), log(c/m) in the fit window (for re-centring)

    # ---- prediction
    def mean(self, ords, price, comp, promo=0) -> np.ndarray:
        ords = np.atleast_1d(np.asarray(ords, dtype=np.int64))
        n = ords.size
        X = design(ords, np.broadcast_to(price, (n,)), np.broadcast_to(comp, (n,)), self.mrp,
                   np.broadcast_to(promo, (n,)))
        return np.exp(np.clip(X @ self.coef, -30, 30))

    def time_effect(self, ords) -> np.ndarray:
        """exp(seasonal + weekday + event + trend) part of log demand for each date (price terms excluded)."""
        X = time_features(np.atleast_1d(np.asarray(ords, dtype=np.int64)))
        keep = _SEASON + _WEEK + _EVENTS + [IDX["trend"]]
        return np.exp(X[:, keep] @ self.coef[keep])

    @property
    def own_elasticity(self) -> float:
        return float(self.coef[IDX["log_price"]])

    @property
    def cross_elasticity(self) -> float:
        return float(self.coef[IDX["log_competitor"]])

    # ---- decomposition for reporting
    def yearly_index(self) -> list[float]:
        doy = np.arange(1, 366, dtype=float)
        s = np.zeros_like(doy)
        for k in range(1, HARMONICS + 1):
            ang = 2 * np.pi * k * doy / 365.25
            s += self.coef[IDX[f"sin{k}"]] * np.sin(ang) + self.coef[IDX[f"cos{k}"]] * np.cos(ang)
        v = np.exp(s)
        return (v / v.mean()).tolist()

    def weekday_index(self) -> list[float]:
        v = np.exp(np.concatenate([[0.0], self.coef[_WEEK]]))
        return (v / v.mean()).tolist()

    def event_lifts(self) -> dict[str, tuple[float, float, float]]:
        """Peak-day multiplier with a 90% interval for each calendar event."""
        out = {}
        for e, j in zip(EVENT_KEYS, _EVENTS):
            c, s = self.coef[j], self.se[j]
            out[e] = (math.exp(c), math.exp(c - 1.645 * s), math.exp(c + 1.645 * s))
        return out

    def annual_growth(self) -> float:
        return math.exp(self.coef[IDX["trend"]]) - 1.0

    def to_row(self) -> tuple:
        return (self.sku, self.fitted_on, self.n_obs, self.n_censored, json.dumps(self.coef.tolist()),
                json.dumps(self.se.tolist()), self.holdout_wape, self.dispersion, self.iterations,
                self.competitor_reaction)

    @staticmethod
    def from_row(row: sqlite3.Row, mrp: float) -> "SkuDemand":
        return SkuDemand(row["sku"], mrp, np.array(json.loads(row["coef"])), np.array(json.loads(row["se"])),
                         row["fitted_on"], row["n_obs"], row["n_censored"], row["holdout_wape"], row["dispersion"],
                         row["newton_iters"], row["competitor_reaction"])


def _history(conn: sqlite3.Connection, sku: str, first: str, last: str, mrp: float) -> dict[str, np.ndarray]:
    rows = conn.execute(
        """SELECT s.date, s.sold, s.closing, s.stocked_out, p.price, p.promo, c.price AS comp
           FROM stock_daily s
           JOIN price_book p ON p.sku = s.sku AND p.date = s.date
           LEFT JOIN competitor_prices c ON c.sku = s.sku AND c.date = s.date
           WHERE s.sku = ? AND s.date BETWEEN ? AND ? ORDER BY s.date""",
        (sku, first, last),
    ).fetchall()
    if not rows:
        return {}
    cols = list(zip(*rows))
    from datetime import date as _d
    comp = np.array([np.nan if x is None else x for x in cols[6]], float)
    has_comp = bool(np.isfinite(comp).any())
    if has_comp:  # carry the last known competitor price forward over gaps in the feed
        idx = np.where(np.isfinite(comp), np.arange(comp.size), 0)
        np.maximum.accumulate(idx, out=idx)
        comp = comp[idx]
        first_known = comp[np.isfinite(comp)][0]
        comp = np.where(np.isfinite(comp), comp, first_known)
    else:          # no competitor feed for this product: the competitor term drops out (constant)
        comp = np.full(comp.size, mrp)
    return {
        "ord": np.array([_d.fromisoformat(x).toordinal() for x in cols[0]], dtype=np.int64),
        "sold": np.array(cols[1], float), "closing": np.array(cols[2], float),
        "stocked_out": np.array(cols[3], float), "price": np.array(cols[4], float),
        "promo": np.array(cols[5], float), "comp": comp, "has_comp": np.array([has_comp]),
    }


def competitor_reaction(h: dict[str, np.ndarray]) -> float:
    """Long-run slope of the rival's log price on our lagged log price: log c_t = a + ρ·log c_{t−1} + θ·log p_{t−1}."""
    if h["comp"].size < 60 or not bool(h["has_comp"][0]):
        return 0.0
    lc, lp = np.log(h["comp"]), np.log(h["price"])
    A = np.column_stack([np.ones(lc.size - 1), lc[:-1], lp[:-1]])
    coef, *_ = np.linalg.lstsq(A, lc[1:], rcond=None)
    rho, theta = float(coef[1]), float(coef[2])
    if rho >= 0.995:
        return 0.0
    return float(np.clip(theta / (1 - rho), 0.0, 1.5))


def fit_sku(conn: sqlite3.Connection, sku: str, mrp: float, last_ord: int, settings: Settings) -> SkuDemand:
    from datetime import date as _d
    first = _d.fromordinal(last_ord - settings.fit_window_days + 1).isoformat()
    last = _d.fromordinal(last_ord).isoformat()
    h = _history(conn, sku, first, last, mrp)
    if not h or h["ord"].size < 120:
        raise ValueError(f"{sku}: not enough sales history to estimate demand")
    # a day that ended with no stock (or was flagged as a stock-out) only shows a lower bound on demand
    censored = (h["stocked_out"] > 0) | (h["closing"] == 0)
    keep = ~censored
    X = design(h["ord"][keep], h["price"][keep], h["comp"][keep], mrp, h["promo"][keep])
    y = h["sold"][keep]

    # out-of-sample check: fit without the last `holdout_days`, score them, then refit on everything
    cut = last_ord - settings.holdout_days
    tr = h["ord"][keep] <= cut
    wape = float("nan")
    if tr.sum() > 120 and (~tr).sum() > 10:
        r = poisson_newton(X[tr], y[tr], settings.ridge)
        pred = np.exp(np.clip(X[~tr] @ r.coef, -30, 30))
        wape = float(np.sum(np.abs(y[~tr] - pred)) / max(np.sum(y[~tr]), 1e-9))
    res = poisson_newton(X, y, settings.ridge)
    return SkuDemand(
        sku=sku, mrp=mrp, coef=res.coef, se=np.sqrt(np.maximum(np.diag(res.cov), 0.0)), fitted_on=last,
        n_obs=int(keep.sum()), n_censored=int(censored.sum()), holdout_wape=wape if math.isfinite(wape) else 0.0,
        dispersion=res.dispersion, iterations=res.iterations, competitor_reaction=competitor_reaction(h),
        mean_x=(float(X[:, IDX["log_price"]].mean()), float(X[:, IDX["log_competitor"]].mean())),
    )


PRIOR_SD = 0.8  # how far a SKU's elasticity is expected to sit from its category average


def shrink_price_response(models: dict[str, SkuDemand], cats: dict[str, list[str]], settings: Settings) -> None:
    """Empirical-Bayes partial pooling of own/cross elasticity toward the category's precision-weighted mean.

    Slow sellers (a diya set selling 0.7 units a day) carry little price information; the posterior mean
    (β̂/se² + β̄/τ²)/(1/se² + 1/τ²) borrows strength from sibling SKUs, while fast sellers keep their own estimate.
    The intercept is re-solved so the fitted mean at the average observed price is unchanged.
    """
    lo, hi = settings.elasticity_bounds
    clo, chi = settings.cross_elasticity_bounds
    for skus in cats.values():
        for j, (a, b) in ((IDX["log_price"], (lo, hi)), (IDX["log_competitor"], (clo, chi))):
            est = np.array([models[s].coef[j] for s in skus])
            w = np.array([1.0 / max(models[s].se[j], 1e-3) ** 2 for s in skus])
            prior = float(np.sum(w * est) / np.sum(w))
            tau_w = 1.0 / PRIOR_SD**2
            for s, e, wi in zip(skus, est, w):
                post = float(np.clip((wi * e + tau_w * prior) / (wi + tau_w), a, b))
                m = models[s]
                xbar = m.mean_x[0] if j == IDX["log_price"] else m.mean_x[1]
                m.coef[0] += (m.coef[j] - post) * xbar
                m.coef[j] = post
                models[s].se[j] = math.sqrt(1.0 / (wi + tau_w))


def fit_all(conn: sqlite3.Connection, last_ord: int, settings: Settings) -> dict[str, SkuDemand]:
    t0 = time.perf_counter()
    models = {}
    cats: dict[str, list[str]] = {}
    skipped = []
    for row in conn.execute("SELECT sku, mrp, category FROM products ORDER BY sku").fetchall():
        try:
            models[row["sku"]] = fit_sku(conn, row["sku"], float(row["mrp"]), last_ord, settings)
        except (ValueError, np.linalg.LinAlgError) as exc:  # too little or unusable history: price it later
            skipped.append(row["sku"])
            log.warning("no demand model for %s: %s", row["sku"], exc)
            continue
        cats.setdefault(row["category"], []).append(row["sku"])
    shrink_price_response(models, cats, settings)
    conn.executemany("INSERT OR REPLACE INTO demand_models VALUES (?,?,?,?,?,?,?,?,?,?)",
                     [m.to_row() for m in models.values()])
    log.info("fitted %d demand models in %.2fs", len(models), time.perf_counter() - t0)
    return models


def load_all(conn: sqlite3.Connection) -> dict[str, SkuDemand]:
    mrp = {r["sku"]: float(r["mrp"]) for r in conn.execute("SELECT sku, mrp FROM products")}
    return {r["sku"]: SkuDemand.from_row(r, mrp[r["sku"]]) for r in conn.execute("SELECT * FROM demand_models")}
