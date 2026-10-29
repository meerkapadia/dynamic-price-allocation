"""Correctness tests (stdlib unittest; also collected by pytest).  Run: python -m pytest  or  python -m unittest

A small store (200 days of history) is generated once into a temp folder and shared by the tests.
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np

from pricing_engine.config import Settings
from pricing_engine.data.calendar import EVENT_KEYS, event_weights, weekday
from pricing_engine.data.generate import generate_store
from pricing_engine.db import connect, meta_get
from pricing_engine.demand.model import IDX, design, fit_all, poisson_newton
from pricing_engine.engine import PricingEngine
from pricing_engine.ops.imports import ImportError_, run_import, template
from pricing_engine.ops.service import PricingService, ServiceError
from pricing_engine.ops.state import derive
from pricing_engine.optim.game import analyse, bertrand_nash, calibrate
from pricing_engine.optim.kkt import ActionBounds, action_bounds, certify, finalize_price
from pricing_engine.optim.sdp import SDPProblem, _spoilage_matrix, q_value, refine_first_action, solve
from pricing_engine.optim.simulation import SimEnv, simulate
from pricing_engine.schemas import ProductState

_TMP = tempfile.TemporaryDirectory()
START = date(2026, 10, 28)
SETTINGS = Settings(db_path=Path(_TMP.name) / "store.db", history_days=240, start_date=START, mc_paths=400,
                    fit_window_days=240, holdout_days=30, workers=4)
generate_store(SETTINGS)
_CONN = connect(SETTINGS.db_path)
MODELS = fit_all(_CONN, START.toordinal() - 1, SETTINGS)
_CONN.commit()
TRUTH = {r["sku"]: json.loads(r["params"]) for r in _CONN.execute("SELECT sku, params FROM market_truth")}


def _state(**kw) -> ProductState:
    base = dict(sku="T-1", unit_cost=20.0, handling_cost_per_unit=1.0, storage_cost_per_unit_day=0.05, days_in_stock=2,
                min_margin_pct=0.05, salvage_value_per_unit=-0.5, current_price=34.0, mrp=40.0, competitor_price=33.0,
                inventory=300, shelf_life_days=14, perishable=True, max_price_change_pct=0.1)
    base.update(kw)
    return ProductState(**base)


class TestCalendar(unittest.TestCase):
    def test_weights_bounded_and_peak_is_one(self) -> None:
        w = event_weights(np.array(["2026-11-08", "2026-10-25", "2026-06-01"], dtype="datetime64[D]"))
        self.assertTrue(np.all((w >= 0) & (w <= 1)))
        self.assertAlmostEqual(w[0, EVENT_KEYS.index("diwali")], 1.0)
        self.assertGreater(w[1, EVENT_KEYS.index("diwali")], 0.0)
        self.assertEqual(w[2].sum(), 0.0)

    def test_weekday(self) -> None:
        self.assertEqual(int(weekday(np.array(["2026-10-28"], dtype="datetime64[D]"))[0]), 2)  # Wednesday


class TestEstimation(unittest.TestCase):
    def test_newton_recovers_known_coefficients(self) -> None:
        rng = np.random.default_rng(0)
        X = np.column_stack([np.ones(4000), rng.normal(size=(4000, 3))])
        theta = np.array([1.5, -0.8, 0.4, 0.2])
        y = rng.poisson(np.exp(X @ theta))
        r = poisson_newton(X, y.astype(float), ridge=0.0)
        self.assertTrue(r.converged)
        self.assertLess(r.iterations, 15)
        np.testing.assert_allclose(r.coef, theta, atol=0.05)
        grad = X.T @ (np.exp(X @ r.coef) - y)
        self.assertLess(np.abs(grad).max() / len(y), 1e-6)

    def test_store_elasticities_recovered_for_fast_sellers(self) -> None:
        avg = dict(_CONN.execute("SELECT sku, AVG(sold) FROM stock_daily GROUP BY sku").fetchall())
        fast = [s for s in MODELS if avg[s] > 30]
        self.assertGreater(len(fast), 10)
        err = [abs(MODELS[s].own_elasticity - TRUTH[s]["beta"]) for s in fast]
        self.assertLess(float(np.median(err)), 1.0)  # 240 days of history; the full 730 gives ~0.5

    def test_sold_out_days_are_excluded(self) -> None:
        total = _CONN.execute("SELECT COUNT(*) FROM stock_daily WHERE sku = 'DB-MILK-1L'").fetchone()[0]
        m = MODELS["DB-MILK-1L"]
        self.assertEqual(m.n_obs + m.n_censored, min(total, SETTINGS.fit_window_days))

    def test_forecast_is_positive_and_seasonal(self) -> None:
        m = MODELS["SW-KAJU-500"]
        days = np.arange(START.toordinal(), START.toordinal() + 20)
        te = m.time_effect(days)
        self.assertTrue(np.all(te > 0))
        diwali = date(2026, 11, 8).toordinal() - START.toordinal()
        self.assertGreater(te[diwali], te[0])  # pre-Diwali build-up is forecast

    def test_design_matches_feature_list(self) -> None:
        X = design(np.array([START.toordinal()]), [50.0], [55.0], 60.0, [1])
        self.assertEqual(X.shape[1], len(IDX))
        self.assertAlmostEqual(X[0, IDX["log_price"]], math.log(50 / 60))


class TestDerivedState(unittest.TestCase):
    def test_every_product_state_comes_from_the_database(self) -> None:
        o = START.toordinal()
        for sku, m in MODELS.items():
            with self.subTest(sku=sku):
                ctx = derive(_CONN, sku, o, m, SETTINGS)
                lots = _CONN.execute("SELECT COALESCE(SUM(qty_on_hand),0) FROM lots WHERE sku = ?", (sku,)).fetchone()[0]
                self.assertEqual(ctx.state.inventory, lots)
                mrp = _CONN.execute("SELECT mrp FROM products WHERE sku = ?", (sku,)).fetchone()[0]
                self.assertEqual(ctx.state.mrp, mrp)
                self.assertGreater(ctx.state.unit_cost, 0)
                self.assertTrue(all(i.source for i in ctx.inputs))


class TestGameTheory(unittest.TestCase):
    def setUp(self) -> None:
        self.market, *_ = calibrate(our_price=30.0, competitor_price=28.0, base_demand=40.0, own_elasticity=-2.2,
                                    cross_elasticity=0.6, our_cost=18.0, competitor_cost=17.0, settings=SETTINGS)

    def test_calibration_reproduces_inputs(self) -> None:
        m = self.market
        self.assertAlmostEqual(float(m.demand0(30.0, 28.0)), 40.0, places=8)
        s0, s1 = m.shares(30.0, 28.0)
        self.assertAlmostEqual(float(-m.b * 30.0 * (1 - s0)), -2.2, places=8)
        self.assertAlmostEqual(float(m.b * 28.0 * s1), 0.6, places=6)

    def test_closed_form_best_response_is_global_argmax(self) -> None:
        m = self.market
        for p0 in (20.0, 30.0, 55.0):
            grid = np.linspace(m.c1, m.c1 + 80, 200_001)
            self.assertAlmostEqual(float(m.best_response1(p0)), float(grid[np.argmax(m.profit1(p0, grid))]), delta=1e-3)

    def test_nash_foc_and_stackelberg_dominates(self) -> None:
        m = self.market
        n0, n1, _, ok = bertrand_nash(m, 30.0, 28.0, SETTINGS)
        self.assertTrue(ok)
        s0, s1 = m.shares(n0, n1)
        self.assertAlmostEqual(float(m.b * (n0 - m.c0) * (1 - s0)), 1.0, places=6)
        self.assertAlmostEqual(float(m.b * (n1 - m.c1) * (1 - s1)), 1.0, places=6)
        res = analyse(our_price=30.0, competitor_price=28.0, base_demand=40.0, own_elasticity=-2.2,
                      cross_elasticity=0.6, our_cost=18.0, competitor_cost=17.0, settings=SETTINGS)
        self.assertGreaterEqual(res.stackelberg_profit0, res.nash_profit0 - 1e-9)


def _toy(inventory: int, T: int, floor: float = 5.0, spoil: float = 0.0, salvage: float = 0.0, mult=None) -> SDPProblem:
    return SDPProblem(price_grid=np.linspace(4.0, 30.0, 80), demand_fn=lambda p: 40.0 * np.exp(-0.12 * np.asarray(p)),
                      demand_mult=np.ones(T) if mult is None else mult, floors=np.full(T, floor), inventory=inventory,
                      handling_cost=0.5, holding_cost=0.02, salvage_value=salvage, spoilage_rate=spoil, max_states=2000)


class TestSDP(unittest.TestCase):
    def test_single_period_matches_brute_force(self) -> None:
        from scipy.stats import poisson

        prob = _toy(25, 1)
        sol = solve(prob)
        best = -np.inf
        for p in prob.price_grid[prob.price_grid >= 5.0]:
            d = np.arange(0, 400)
            pmf = poisson.pmf(d, 40.0 * math.exp(-0.12 * p))
            sales = np.minimum(d, 25)
            best = max(best, float(np.sum(pmf * ((p - 0.5) * sales - 0.02 * (25 - sales)))))
        self.assertAlmostEqual(sol.values[0, 25], best, places=6)

    def test_bellman_value_matches_monte_carlo_with_seasonal_demand(self) -> None:
        mult = np.array([1.0, 1.1, 1.3, 1.6, 2.0, 1.2, 0.8, 0.7, 0.9, 1.0])
        prob = _toy(120, 10, spoil=0.03, salvage=-0.4, mult=mult)
        sol = solve(prob)
        env = SimEnv(demand_fn=prob.demand_fn, demand_mult=mult, full_cost=np.zeros(10), floors=prob.floors,
                     sunk_cost_per_unit=0.0, handling_cost=0.5, holding_cost=0.02, salvage_value=-0.4,
                     spoilage_rate=0.03, inventory=120)
        r = simulate(env, {"opt": lambda t, inv: sol.policy_price(t, inv)}, n_paths=20_000, seed=3)["opt"]
        se = r.profit.std() / math.sqrt(r.profit.size)
        self.assertLess(abs(r.profit.mean() - sol.values[0, 120]), 4 * se + 1e-6)

    def test_floor_is_hard_in_every_state(self) -> None:
        sol = solve(_toy(200, 15, floor=12.0))
        self.assertTrue(np.all(sol.price_grid[sol.policy_idx[:, 1:]] >= 12.0))

    def test_spoilage_matrix_is_stochastic(self) -> None:
        B = _spoilage_matrix(60, 0.1)
        self.assertTrue(np.allclose(B.sum(axis=1), 1.0))

    def test_q_value_consistent(self) -> None:
        sol = solve(_toy(80, 6, spoil=0.02))
        p_star = sol.price_grid[sol.policy_idx[0, 80]]
        self.assertAlmostEqual(q_value(sol, float(p_star), 80), sol.values[0, 80], places=6)
        _, v = refine_first_action(sol, 80, 5.0, 30.0)
        self.assertGreaterEqual(v, sol.values[0, 80] - 1e-9)

    def test_large_inventory_lot_aggregation(self) -> None:
        sol = solve(_toy(50_000, 5))
        self.assertGreater(sol.lot_size, 1)
        self.assertTrue(np.isfinite(sol.values).all())


class TestGuardrails(unittest.TestCase):
    def test_floor_dominates_rate_limit(self) -> None:
        b = action_bounds(_state(unit_cost=60.0, mrp=100.0), floor0=70.0, ceiling=100.0)
        self.assertTrue(b.rate_limit_overridden)
        self.assertEqual(b.lower, 70.0)

    def test_kkt_floor_multiplier(self) -> None:
        b = ActionBounds(lower=10.0, upper=20.0, floor=10.0, rate_limit_overridden=False)
        rep = certify(lambda p: -(p - 5.0) ** 2, 10.0, b, 1e-4)
        self.assertIn("profit_floor", rep.active_constraints)
        self.assertAlmostEqual(rep.mu_floor, 10.0, places=3)
        self.assertTrue(rep.dual_feasible and rep.primal_feasible)

    def test_rounding_never_below_floor(self) -> None:
        b = ActionBounds(lower=10.2, upper=20.0, floor=10.2, rate_limit_overridden=False)
        self.assertGreaterEqual(finalize_price(10.2, b), 10.2)
        self.assertEqual(finalize_price(10.2, b) % 1, 0)
        self.assertGreaterEqual(finalize_price(float("nan"), b), 10.2)


class TestService(unittest.TestCase):
    """End-to-end: run pricing, approve, override, reject, export; nothing changes without approval."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.svc = PricingService(SETTINGS)
        cls.run_id = cls.svc.start_run(requested_by="test", wait=True)

    def test_run_prices_every_product_within_limits(self) -> None:
        conn = connect(SETTINGS.db_path, readonly=True)
        try:
            run = conn.execute("SELECT * FROM pricing_runs WHERE run_id = ?", (self.run_id,)).fetchone()
            self.assertEqual(run["status"], "done")
            self.assertEqual(run["priced"], len(MODELS))
            self.assertEqual(run["price_date"], START.isoformat())
            untouched = ("DB-MILK-1L", "DB-CURD-400")  # the approval test may have run first
            for r in conn.execute("SELECT sku, recommended_price, floor, ceiling, status FROM recommendations "
                                  "WHERE run_id = ?", (self.run_id,)):
                self.assertGreaterEqual(r["recommended_price"], min(r["floor"], r["ceiling"]) - 1e-9)
                self.assertLessEqual(r["recommended_price"], r["ceiling"] + 1e-9)
                if r["sku"] not in untouched:
                    self.assertEqual(r["status"], "pending")
            # nothing reaches the price book until someone approves it
            self.assertIsNone(conn.execute("SELECT 1 FROM price_book WHERE date = ? AND sku NOT IN (?, ?)",
                                           (START.isoformat(), *untouched)).fetchone())
        finally:
            conn.close()

    def test_approve_override_reject_and_audit(self) -> None:
        s = self.svc
        with self.assertRaises(ServiceError):
            s.decide("DB-MILK-1L", "approve", user="")
        res = s.decide("DB-MILK-1L", "approve", user="Asha")
        self.assertEqual(res["status"], "approved")
        conn = connect(SETTINGS.db_path, readonly=True)
        rec = conn.execute("SELECT * FROM recommendations WHERE run_id = ? AND sku = 'DB-MILK-1L'", (self.run_id,)).fetchone()
        pb = conn.execute("SELECT price, set_by FROM price_book WHERE sku = 'DB-MILK-1L' AND date = ?",
                          (START.isoformat(),)).fetchone()
        conn.close()
        self.assertEqual(pb["price"], rec["recommended_price"])
        self.assertEqual(pb["set_by"], "Asha")
        with self.assertRaises(ServiceError):  # below the loss floor
            s.decide("DB-MILK-1L", "override", user="Asha", price=rec["floor"] - 5)
        with self.assertRaises(ServiceError):  # above MRP
            s.decide("DB-MILK-1L", "override", user="Asha", price=rec["ceiling"] + 5)
        ok = s.decide("DB-MILK-1L", "override", user="Asha", price=rec["ceiling"])
        self.assertIn(ok["status"], ("overridden", "approved"))
        s.decide("DB-MILK-1L", "reject", user="Asha")
        conn = connect(SETTINGS.db_path, readonly=True)
        self.assertIsNone(conn.execute("SELECT 1 FROM price_book WHERE sku = 'DB-MILK-1L' AND date = ?",
                                       (START.isoformat(),)).fetchone())
        conn.close()
        bulk = s.decide_many(["DB-CURD-400", "NOT-A-SKU"], "approve", "Asha")
        self.assertEqual(bulk["updated"], 1)
        self.assertEqual(len(bulk["failed"]), 1)

    def test_explain(self) -> None:
        conn = connect(SETTINGS.db_path, readonly=True)
        try:
            ex = self.svc.explain(conn, "SW-KAJU-500", self.run_id)
        finally:
            conn.close()
        self.assertTrue(ex["curve"] and ex["objective"])


class TestImports(unittest.TestCase):
    def setUp(self) -> None:
        self.path = Path(_TMP.name) / "real.db"
        self.path.unlink(missing_ok=True)
        self.conn = connect(self.path)
        from pricing_engine.db import init_schema

        init_schema(self.conn)

    def tearDown(self) -> None:
        self.conn.close()

    def test_templates_import_cleanly(self) -> None:
        for kind in ("products", "sales", "stock", "competitor", "write_offs"):
            with self.subTest(kind=kind):
                res = run_import(self.conn, kind, template(kind))
                self.assertGreater(res["written"], 0)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM products").fetchone()[0], 2)
        self.assertEqual(self.conn.execute("SELECT stocked_out FROM stock_daily WHERE sku = 'MILK-1L'").fetchone()[0], 0)

    def test_bad_file_is_rejected_whole(self) -> None:
        run_import(self.conn, "products", template("products"))
        bad = "sku,date,units_sold,price\nMILK-1L,2026-10-01,10,66\nUNKNOWN,2026-10-01,5,10\nMILK-1L,yesterday,x,66\n"
        with self.assertRaises(ImportError_) as ctx:
            run_import(self.conn, "sales", bad)
        self.assertGreaterEqual(len(ctx.exception.errors), 3)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM stock_daily").fetchone()[0], 0)

    def test_real_store_flow_from_csv(self) -> None:
        """A store with no demo data: import files, calculate, approve, export."""
        run_import(self.conn, "products", template("products"))
        rng = np.random.default_rng(5)
        lines = ["sku,date,units_sold,price,closing_stock"]
        start = date(2026, 3, 1).toordinal()
        for sku, base, p0 in (("MILK-1L", 300, 64.0), ("RICE-5KG", 35, 640.0)):
            for t in range(240):
                p = round(p0 * float(rng.choice([0.9, 0.95, 1.0, 1.0, 1.05])))
                q = int(rng.poisson(base * (p / p0) ** -3.0))
                lines.append(f"{sku},{date.fromordinal(start + t).isoformat()},{q},{p},{50 + q % 7}")
        run_import(self.conn, "sales", "\n".join(lines))
        run_import(self.conn, "stock", "sku,qty_on_hand,unit_cost,received_on\nMILK-1L,600,55,2026-10-26\nRICE-5KG,300,520,2026-10-01\n")
        settings = SETTINGS.model_copy(update={"db_path": self.path, "workers": 2})
        svc = PricingService(settings)
        run_id = svc.start_run(requested_by="test", wait=True)
        rows = self.conn.execute("SELECT sku, recommended_price, floor, ceiling FROM recommendations WHERE run_id = ?",
                                 (run_id,)).fetchall()
        self.assertEqual(len(rows), 2)
        for r in rows:
            self.assertTrue(r["floor"] - 1e-9 <= r["recommended_price"] <= r["ceiling"] + 1e-9)
        self.assertEqual(svc.decide("MILK-1L", "approve", user="Owner")["status"], "approved")

    def test_missing_columns(self) -> None:
        with self.assertRaises(ImportError_):
            run_import(self.conn, "products", "sku,name\nA,B\n")


class TestEngineGuards(unittest.TestCase):
    def test_engine_never_prices_below_floor(self) -> None:
        eng = PricingEngine(SETTINGS)
        o = START.toordinal()
        for sku in ("DB-MILK-1L", "EL-EARBUDS-X2", "SW-KAJU-500", "FV-TOMATO-1K"):
            ctx = derive(_CONN, sku, o, MODELS[sku], SETTINGS)
            dec = eng.decide(ctx, MODELS[sku], o)
            self.assertGreaterEqual(dec.price, dec.bounds.floor - 1e-9)
            self.assertLessEqual(dec.price, ctx.state.mrp + 1e-9)
            self.assertTrue(dec.kkt.primal_feasible)
            a = dec.advice
            self.assertIn(a["action"], ("increase", "decrease", "hold"))
            # the four factor steps add up exactly to the recommended change
            self.assertAlmostEqual(sum(f["delta"] for f in a["factors"]), dec.price - ctx.state.current_price, places=6)
            if a["profit_horizon_new"] is not None and ctx.state.current_price >= dec.bounds.floor:
                # holding a price below the loss floor is not allowed, so only compare when holding is legal
                self.assertGreaterEqual(a["profit_horizon_new"], a["profit_horizon_hold"] - 0.02 * abs(a["profit_horizon_hold"]) - 5)


if __name__ == "__main__":
    unittest.main()
