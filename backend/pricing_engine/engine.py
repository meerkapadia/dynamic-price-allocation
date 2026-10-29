"""Pricing optimiser for one SKU on one business day.

    database ──► derived ProductState (ops/state.py) + fitted demand model (demand/model.py)
                    │   λ₀ = model forecast today at current prices;  β, γ = estimated elasticities
                    ▼
    MNL duopoly calibrated to (λ₀, β, γ) ──► Bertrand–Nash · Stackelberg · rival best response
                    │   λ(p) = M·s₀(p, BR₁(p))   (or rival held fixed if it does not react)
                    ▼
    Bellman backward induction over (day × stock) with seasonal multipliers m_t from the model,
    floor_t enforced inside the max operator, horizon = next delivery (leftovers worth cost) or, for perishable excess, expiry (worth salvage)
                    ▼
    first-day continuous refinement on [max(floor, −Δ), min(MRP, +Δ)] ─► KKT certificate ─► whole-rupee price

`decide` is the fast path a pricing run calls for every product. `explain` adds the curves, game report,
policy surface and a Monte-Carlo comparison against simple rules for the product page.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize_scalar

from .config import Settings
from .demand.model import SkuDemand
from .optim.game import CompetitionResult, analyse, reaction_demand
from .optim.kkt import ActionBounds, action_bounds, ceil_tick, certify, finalize_price, floor_path, full_cost_path
from .optim.sdp import SDPProblem, SDPSolution, q_value, refine_first_action, solve
from .optim.simulation import SimEnv, constant_policy, simulate
from .ops.state import SkuContext
from .schemas import CompetitorResponse, KKTReport

log = logging.getLogger(__name__)


class PricingError(ValueError):
    """Business-rule violation that makes a decision infeasible."""


def _f(x: float) -> float:
    x = float(x)
    return x if math.isfinite(x) else 0.0


BINDING_LABEL = {
    "profit_floor": "Loss floor",
    "rate_limit_lower": "Daily move limit (down)",
    "rate_limit_upper": "Daily move limit (up)",
    "mrp": "MRP ceiling",
    "none": "None",
}


@dataclass(slots=True)
class Decision:
    sku: str
    price: float
    prev_price: float
    floor: float
    ceiling: float
    competitor_price: float
    expected_units: float
    expected_units_at_prev: float
    unit_margin: float
    binding: str
    horizon: int
    horizon_mode: str          # "replenish" (plan to next delivery) | "expiry" (plan to expiry, clear excess)
    kkt: KKTReport
    response: str
    lam0: float
    warnings: list[str]
    # internals reused by explain()
    comp: CompetitionResult
    bounds: ActionBounds
    mult: np.ndarray
    floors: np.ndarray
    full_costs: np.ndarray
    salvage: float
    grid: np.ndarray
    sol: SDPSolution | None = None
    demand_fn: object = field(default=None, repr=False)
    advice: dict = field(default_factory=dict)


class PricingEngine:
    def __init__(self, settings: Settings) -> None:
        self.s = settings

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _demand_fn(comp: CompetitionResult, react: bool):
        def fn(prices: np.ndarray) -> np.ndarray:
            q, _ = reaction_demand(comp.market, np.asarray(prices, dtype=np.float64), react)
            return q
        return fn

    @staticmethod
    def _myopic(demand_fn, cost: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return lo
        grid = np.geomspace(lo, hi, 256)
        prof = (grid - cost) * demand_fn(grid)
        k = int(np.argmax(prof))
        a, b = grid[max(k - 1, 0)], grid[min(k + 1, grid.size - 1)]
        res = minimize_scalar(lambda p: -float((p - cost) * demand_fn(np.array([p]))[0]), bounds=(a, b), method="bounded")
        return float(res.x) if res.success and -res.fun >= prof[k] else float(grid[k])

    # ------------------------------------------------------------------ decision
    def decide(self, ctx: SkuContext, dm: SkuDemand, o: int) -> Decision:
        s, st = self.s, ctx.state
        warnings: list[str] = []
        mrp = st.mrp

        # ---- demand today: model forecast at current prices, calibrated into the competition model
        lam0 = float(dm.mean(o, st.current_price, st.competitor_price)[0])
        own = float(np.clip(dm.own_elasticity, *s.elasticity_bounds))
        cross = float(np.clip(dm.cross_elasticity, *s.cross_elasticity_bounds))
        if lam0 < 1e-3:
            warnings.append("Forecast demand is close to zero; the price is set by the floor and the season ahead.")
        marginal_cost = st.unit_cost + st.handling_cost_per_unit
        comp = analyse(our_price=st.current_price, competitor_price=st.competitor_price, base_demand=max(lam0, 1e-4),
                       own_elasticity=own, cross_elasticity=cross, our_cost=marginal_cost, competitor_cost=None,
                       settings=s)
        warnings += comp.warnings
        react = st.competitor_response == CompetitorResponse.STACKELBERG
        demand_fn = self._demand_fn(comp, react)

        # ---- planning horizon. Stock is only scarce until the next delivery can land, so by default the plan runs
        # to the replenishment cycle and leftovers are worth their replacement cost. Perishable stock that cannot be
        # sold before it expires is different: the plan then runs to expiry and leftovers are worth only salvage.
        life = max(1, min(st.shelf_life_days, s.max_horizon_days))
        repl = max(1, min(st.replenishment_days, s.max_horizon_days))
        te_all = dm.time_effect(o + np.arange(max(life, repl)))
        mult_all = te_all / te_all[0]
        if not st.perishable or st.shelf_life_days > max(s.max_horizon_days, repl):
            horizon, salvage, mode = repl, st.unit_cost, "replenish"
        elif life <= repl:
            horizon, salvage, mode = life, st.salvage_value_per_unit, "expiry"
        else:
            c0 = st.full_unit_cost
            p_m = self._myopic(demand_fn, c0, max(c0, 0.01), mrp)
            sell_through = float(demand_fn(np.array([p_m]))[0] * mult_all[:life].sum())
            if st.inventory <= sell_through:
                horizon, salvage, mode = repl, st.unit_cost, "replenish"
            else:
                horizon, salvage, mode = life, st.salvage_value_per_unit, "expiry"
        mult = mult_all[:horizon]

        floors = floor_path(st, horizon)
        full_costs = full_cost_path(st, horizon)
        if floors.max() > mrp:
            if floors[0] > mrp:
                warnings.append(f"Cost plus minimum margin (₹{floors[0]:.2f}) is above MRP; price held at MRP. "
                                "Review the supplier cost.")
            else:
                k = int(np.argmax(floors > mrp))
                warnings.append(f"From day {k + 1} the loss floor exceeds MRP; the plan clears stock before then.")
            floors = np.minimum(floors, mrp)
        floor0 = min(ceil_tick(float(floors[0])), mrp)

        ceiling = mrp
        lo_grid = max(min(float(floors.min()), st.current_price) * 0.98, 0.01)
        base_grid = np.geomspace(lo_grid, ceiling, s.price_grid_size)
        extra = np.concatenate([floors, [st.current_price, st.competitor_price, *comp.stackelberg, *comp.nash]])
        extra = extra[(extra >= lo_grid) & (extra <= ceiling)]
        grid = np.unique(np.concatenate([base_grid, extra]))
        grid = grid[np.concatenate([[True], np.diff(grid) > 1e-9 * grid[1:]])]

        bounds = action_bounds(st, floor0, ceiling)
        if bounds.rate_limit_overridden:
            warnings.append("Loss floor is outside today's move limit; the floor takes precedence.")
        if st.competitor_price < floor0:
            warnings.append(f"Competitor at ₹{st.competitor_price:.0f} is below our loss floor ₹{floor0:.0f}; not matching.")

        sol: SDPSolution | None = None
        if st.inventory == 0:
            warnings.append("Out of stock: listing price set to the single-day optimum until stock arrives.")
            raw = self._myopic(demand_fn, float(full_costs[0]), bounds.lower, bounds.upper)

            def objective(p: float) -> float:
                return float((p - full_costs[0]) * demand_fn(np.array([p]))[0])
        else:
            problem = SDPProblem(
                price_grid=grid, demand_fn=demand_fn, demand_mult=mult, floors=floors, inventory=st.inventory,
                handling_cost=st.handling_cost_per_unit, holding_cost=st.storage_cost_per_unit_day,
                salvage_value=salvage, spoilage_rate=st.daily_spoilage_rate, max_states=s.max_inventory_states,
            )
            sol = solve(problem)
            raw, _ = refine_first_action(sol, st.inventory, bounds.lower, bounds.upper)

            def objective(p: float) -> float:
                return q_value(sol, p, st.inventory)

        kkt = certify(objective, raw, bounds, s.kkt_tolerance)
        price = finalize_price(raw, bounds)
        if price < bounds.floor - 1e-9:
            raise PricingError("guardrail invariant violated")

        tol = max(0.51, 1e-6 * price)
        if abs(price - bounds.floor) <= tol and bounds.lower <= bounds.floor + 1e-9:
            binding = "profit_floor"
        elif abs(price - bounds.lower) <= tol and bounds.lower > bounds.floor:
            binding = "rate_limit_lower"
        elif abs(price - bounds.upper) <= tol:
            binding = "mrp" if abs(bounds.upper - mrp) < 1e-9 else "rate_limit_upper"
        else:
            binding = "none"

        q_new = float(demand_fn(np.array([price]))[0])
        q_old = float(demand_fn(np.array([st.current_price]))[0])
        dec = Decision(
            sku=st.sku, price=price, prev_price=st.current_price, floor=bounds.floor, ceiling=ceiling,
            competitor_price=st.competitor_price, expected_units=q_new, expected_units_at_prev=q_old,
            unit_margin=price - float(full_costs[0]), binding=BINDING_LABEL[binding], horizon=horizon,
            horizon_mode=mode, kkt=kkt,
            response=st.competitor_response.value, lam0=lam0, warnings=list(dict.fromkeys(warnings)), comp=comp,
            bounds=bounds, mult=mult, floors=floors, full_costs=full_costs, salvage=salvage, grid=grid, sol=sol,
            demand_fn=demand_fn,
        )
        dec.advice = self._advice(dec, st, own, react)
        return dec

    # ------------------------------------------------------------------ recommendation & factor attribution
    def _advice(self, dec: Decision, st, own: float, react: bool) -> dict:
        """Why the price moves: re-solve the problem adding one factor at a time and attribute each step.

        1. cost + price sensitivity     single-day profit optimum, competitor price held, no range limits
        2. competitor reaction          same, but the competitor answers with its best reply (if it follows us)
        3. stock, expiry, season        first-day action of the dynamic programme (stock and seasonal forecast)
        4. allowed range                loss floor, MRP and today's move limit, then whole-rupee rounding
        The four steps add up exactly to (recommended − current).
        """
        cur, price, comp, mrp = st.current_price, dec.price, dec.comp, st.mrp
        c0 = float(dec.full_costs[0])
        fixed_fn = self._demand_fn(comp, False)
        lo = max(min(c0 * 1.0005, cur * 0.5), 0.01)
        hi = max(mrp, cur, c0) * 1.6
        p1 = self._myopic(fixed_fn, c0, lo, hi)
        p2 = self._myopic(dec.demand_fn, c0, lo, hi) if react else p1
        if dec.sol is not None and st.inventory > 0:
            p3, _ = refine_first_action(dec.sol, st.inventory, max(dec.grid[0] * 0.6, 0.01), hi)
        else:
            p3 = p2
        target = float(np.clip(round(p3), dec.bounds.floor, mrp))

        # context numbers for each factor
        e = abs(own)
        h = dec.mult.size
        exp_sales = float(dec.demand_fn(np.array([p2]))[0] * dec.mult.sum())
        week = float(np.mean(dec.mult[: min(7, h)]))
        stock_ratio = st.inventory / exp_sales if exp_sales > 0 else float("inf")
        if st.inventory == 0:
            stock_text = "No stock on hand, so only today's demand matters."
        elif stock_ratio > 1.15:
            stock_text = (f"{st.inventory:,} units on hand but only about {exp_sales:,.0f} expected to sell "
                          f"{'before they expire' if dec.horizon_mode == 'expiry' else 'before the next delivery'}; "
                          "a lower price clears the surplus.")
        elif stock_ratio < 0.85:
            stock_text = (f"Only {st.inventory:,} units on hand against about {exp_sales:,.0f} expected sales before "
                          "the next delivery; a higher price stretches the stock.")
        else:
            stock_text = f"{st.inventory:,} units on hand, close to the {exp_sales:,.0f} expected to sell."
        season_text = (f"Demand over the next {min(7, h)} days is forecast at {week:.2f}× today's"
                       + (" (rising)" if week > 1.05 else " (falling)" if week < 0.95 else "") + ".")
        b = dec.bounds
        factors = [
            {
                "key": "economics", "label": "Cost and price sensitivity", "delta": p1 - cur, "price_after": p1,
                "detail": (f"Full unit cost ₹{c0:,.2f}. A 1% price rise loses about {e:.1f}% of buyers, so the "
                           f"profit-maximising price for today alone is ₹{p1:,.2f}."),
            },
            {
                "key": "competitor", "label": "Competitor's likely reaction", "delta": p2 - p1, "price_after": p2,
                "detail": (f"The competitor (₹{st.competitor_price:,.0f}) follows our moves, so a higher price loses "
                           "fewer shoppers than it would otherwise." if react else
                           f"The competitor (₹{st.competitor_price:,.0f}) does not follow our price; no adjustment."),
            },
            {
                "key": "stock", "label": "Stock, expiry and the season ahead", "delta": p3 - p2, "price_after": p3,
                "detail": f"{stock_text} {season_text}",
            },
            {
                "key": "range", "label": "Allowed price range today", "delta": price - p3, "price_after": price,
                "detail": (f"Must stay between ₹{b.lower:,.0f} and ₹{b.upper:,.0f}: loss floor ₹{b.floor:,.0f}, "
                           f"MRP ₹{mrp:,.0f}, at most {st.max_price_change_pct * 100:.0f}% move from yesterday. "
                           "Rounded to whole rupees."),
            },
        ]

        # profit impact over the plan horizon: hold today's price vs follow the recommendation (same shoppers)
        hold_profit = new_profit = None
        if dec.sol is not None and st.inventory > 0:
            sol = dec.sol
            env = SimEnv(demand_fn=dec.demand_fn, demand_mult=dec.mult, full_cost=dec.full_costs, floors=dec.floors,
                         sunk_cost_per_unit=st.unit_cost, handling_cost=st.handling_cost_per_unit,
                         holding_cost=st.storage_cost_per_unit_day,
                         salvage_value=dec.salvage, spoilage_rate=st.daily_spoilage_rate, inventory=st.inventory)

            def plan(t: int, inv: np.ndarray) -> np.ndarray:
                return np.full(inv.shape, price) if t == 0 else sol.policy_price(t, inv)

            res = simulate(env, {"plan": plan, "hold": constant_policy(lambda t: cur)}, 300, 11)
            new_profit, hold_profit = float(res["plan"].profit.mean()), float(res["hold"].profit.mean())
        day_new = (price - c0) * dec.expected_units
        day_old = (cur - c0) * dec.expected_units_at_prev
        change = (price / cur - 1.0) * 100.0
        action = "hold" if abs(price - cur) < 0.5 else ("increase" if price > cur else "decrease")
        return {
            "action": action, "change_pct": change, "change_abs": price - cur, "current_price": cur,
            "recommended_price": price, "target_price": target,
            "target_change_pct": (target / cur - 1.0) * 100.0,
            "factors": [{k: (_f(v) if isinstance(v, float) else v) for k, v in f.items()} for f in factors],
            "units_today_current": _f(dec.expected_units_at_prev), "units_today_new": _f(dec.expected_units),
            "profit_today_current": _f(day_old), "profit_today_new": _f(day_new),
            "horizon_days": h, "horizon_mode": dec.horizon_mode,
            "profit_horizon_hold": None if hold_profit is None else _f(hold_profit),
            "profit_horizon_new": None if new_profit is None else _f(new_profit),
            "profit_gain": _f((new_profit - hold_profit) if new_profit is not None else (day_new - day_old)),
            "unit_margin_new": _f(price - c0), "unit_margin_current": _f(cur - c0), "full_unit_cost": _f(c0),
        }

    # ------------------------------------------------------------------ explanation for the product page
    def explain(self, dec: Decision, ctx: SkuContext, dm: SkuDemand, o: int, n_paths: int | None = None,
                seed: int = 7) -> dict:
        st, comp = ctx.state, dec.comp
        demand_fn = dec.demand_fn
        lo = max(0.85 * min(st.current_price, dec.floor, st.competitor_price), 0.01)
        cp = np.geomspace(lo, dec.ceiling, 60)
        q_static, _ = reaction_demand(comp.market, cp, react=False)
        q_react, p1 = reaction_demand(comp.market, cp, react=True)
        q_glm = dm.mean(np.full(cp.size, o), cp, st.competitor_price)
        used = demand_fn(cp)
        c0 = float(dec.full_costs[0])
        curve = [
            {"price": _f(p), "demand_static": _f(a), "demand_with_response": _f(b), "demand_model": _f(g),
             "competitor_response_price": _f(r), "daily_profit": _f((p - c0) * u)}
            for p, a, b, g, r, u in zip(cp, q_static, q_react, q_glm, p1, used)
        ]
        br_grid = np.linspace(max(comp.market.c0, 0.01), dec.ceiling, 40)
        game = {
            "logit_price_sensitivity": _f(comp.market.b), "market_size": _f(comp.market.market_size),
            "our_share": _f(comp.our_share), "competitor_share": _f(comp.competitor_share),
            "competitor_cost_estimate": _f(comp.market.c1),
            "nash": [_f(comp.nash[0]), _f(comp.nash[1])], "nash_iterations": comp.nash_iterations,
            "nash_profit": _f(comp.nash_profit0), "stackelberg": [_f(comp.stackelberg[0]), _f(comp.stackelberg[1])],
            "stackelberg_profit": _f(comp.stackelberg_profit0),
            "predicted_competitor_reply": _f(comp.market.best_response1(dec.price)),
            "best_response": [[_f(a), _f(b)] for a, b in zip(br_grid, comp.market.best_response1(br_grid))],
        }
        out: dict = {"curve": curve, "game": game, "policy": None, "strategies": [], "plan": [], "objective": []}
        # what-if: expected profit over the plan if today's price were p (the plan continues optimally after today)
        wp = np.round(np.linspace(max(0.85 * min(dec.floor, st.current_price), 1.0), dec.ceiling, 90))
        wp = np.unique(np.concatenate([wp, [dec.price, st.current_price]]))
        for p in wp:
            u = float(demand_fn(np.array([p]))[0])
            hp = q_value(dec.sol, float(p), st.inventory) if dec.sol is not None and st.inventory > 0 else None
            if hp is not None:
                hp -= st.unit_cost * st.inventory
            out["objective"].append({"price": _f(p), "units_today": _f(u), "profit_today": _f((p - c0) * u),
                                     "profit_horizon": None if hp is None else _f(hp),
                                     "below_floor": bool(p < dec.floor - 1e-9)})
        sol = dec.sol
        if sol is None:
            return out

        horizon = dec.mult.size
        days_idx = np.unique(np.linspace(0, horizon - 1, min(horizon, 30)).round().astype(int))
        n_states = sol.values.shape[1]
        inv_idx = np.unique(np.linspace(1, n_states - 1, min(n_states - 1, 24)).round().astype(int)) if n_states > 1 else np.array([0])
        out["policy"] = {
            "days": [int(d) for d in days_idx], "inventory": [int(i * sol.lot_size) for i in inv_idx],
            "prices": [[_f(sol.price_grid[max(sol.policy_idx[t, x], 0)]) for x in inv_idx] for t in days_idx],
            "lot_size": sol.lot_size, "solve_ms": _f(sol.solve_ms),
            "expected_value": _f(sol.value_at(st.inventory)),
            "marginal_value_of_stock": _f((sol.values[0, -1] - sol.values[0, -2]) / sol.lot_size) if n_states > 1 else 0.0,
        }

        floors = dec.floors
        myopic = self._myopic(demand_fn, c0, float(floors[0]), dec.ceiling)
        price = dec.price

        def sdp_policy(t: int, inv: np.ndarray) -> np.ndarray:
            return np.full(inv.shape, price) if t == 0 else sol.policy_price(t, inv)

        policies = {
            "Recommended plan (dynamic programme)": sdp_policy,
            "Hold yesterday's price": constant_policy(lambda t: st.current_price),
            "Best single-day price, held": constant_policy(lambda t: max(myopic, float(floors[t]))),
            "Cost plus 25%": constant_policy(lambda t: min(float(dec.full_costs[t]) * 1.25, dec.ceiling)),
            "Match the competitor": constant_policy(lambda t: st.competitor_price),
        }
        sunk = st.unit_cost  # storage already paid is sunk whatever we do; leave it out of the comparison
        env = SimEnv(demand_fn=demand_fn, demand_mult=dec.mult, full_cost=dec.full_costs, floors=floors,
                     sunk_cost_per_unit=sunk, handling_cost=st.handling_cost_per_unit,
                     holding_cost=st.storage_cost_per_unit_day, salvage_value=dec.salvage,
                     spoilage_rate=st.daily_spoilage_rate, inventory=st.inventory)
        res = simulate(env, policies, n_paths or self.s.mc_paths, seed)
        x0 = st.inventory
        for r in res.values():
            days_sold = r.price_days.sum()
            out["strategies"].append({
                "name": r.name, "revenue": _f(r.revenue.mean()), "profit": _f(r.profit.mean()),
                "profit_p05": _f(np.percentile(r.profit, 5)), "profit_p95": _f(np.percentile(r.profit, 95)),
                "units_sold": _f(r.sold.mean()), "units_wasted": _f(r.wasted.mean()),
                "sell_through_pct": _f(100.0 * r.sold.mean() / x0) if x0 else 0.0,
                "avg_price": _f(r.price_sum.sum() / days_sold) if days_sold else 0.0,
                "loss_making_units": _f(r.loss_units.mean()),
            })
        opt = res["Recommended plan (dynamic programme)"]
        hold = res["Hold yesterday's price"]
        for t in range(horizon + 1):
            pr = opt.price_paths[t] if t < horizon else np.array([np.nan])
            pv = pr[~np.isnan(pr)]
            out["plan"].append({
                "day": t, "date_ordinal": o + t, "stock_mean": _f(opt.inv_paths[t].mean()),
                "stock_p10": _f(np.percentile(opt.inv_paths[t], 10)), "stock_p90": _f(np.percentile(opt.inv_paths[t], 90)),
                "price_mean": _f(pv.mean()) if pv.size else None,
                "floor": _f(floors[min(t, horizon - 1)]), "multiplier": _f(dec.mult[min(t, horizon - 1)]),
                "stock_hold": _f(hold.inv_paths[t].mean()),
            })
        return out
