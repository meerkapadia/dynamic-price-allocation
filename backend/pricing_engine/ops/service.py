"""Pricing service: compute recommendations on request, let people approve them, keep an audit trail.

    run       derive every product's state from the database, refit stale demand models, solve each product's
              pricing problem in a worker pool, store one recommendation per product (status: pending)
    decide    approve, override (with a price inside the allowed range) or reject; approved prices are written to
              the price book for the price date, which is what the export and the next run read
    explain   rebuild one recommendation's full working (curves, game, plan, strategy comparison) for its page

Nothing changes a live price without a person approving it. Products are priced independently, so a run scales
with workers, and one product failing (bad data) is recorded without stopping the others.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from ..config import Settings
from ..db import connect, meta_get, meta_set, transaction
from ..demand.model import SkuDemand, fit_all, load_all
from ..engine import PricingEngine
from ..schemas import ProductState
from .state import derive

log = logging.getLogger(__name__)


class ServiceError(ValueError):
    """A request that cannot be carried out (mapped to HTTP 4xx)."""


@dataclass(slots=True)
class _Snapshot:
    state: ProductState


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class PricingService:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.engine = PricingEngine(settings)
        self._run_lock = threading.Lock()
        self._models: dict[str, SkuDemand] | None = None
        self._models_version: str | None = None
        self._explain_cache: dict[tuple[str, int], dict] = {}
        self.progress: dict = {"running": False}
        self._stop = threading.Event()
        self._scheduler: threading.Thread | None = None

    # ------------------------------------------------------------------ helpers
    def conn(self, readonly: bool = True):
        return connect(self.s.db_path, readonly=readonly)

    def default_price_date(self, conn) -> date:
        last = conn.execute("SELECT MAX(date) FROM stock_daily").fetchone()[0]
        return date.fromisoformat(last) + timedelta(days=1) if last else date.today()

    def models(self, conn) -> dict[str, SkuDemand]:
        version = meta_get(conn, "model_fitted_on")
        if self._models is None or self._models_version != version:
            self._models = load_all(conn)
            self._models_version = version
        return self._models

    def models_stale(self, conn, price_ord: int) -> bool:
        fitted = meta_get(conn, "model_fitted_on")
        if not fitted:
            return True
        if meta_get(conn, "data_version", "0") != meta_get(conn, "model_data_version", "0"):
            return True
        return price_ord - date.fromisoformat(fitted).toordinal() > self.s.refit_every_days

    # ------------------------------------------------------------------ runs
    def start_run(self, price_date: date | None = None, refit: bool = False, requested_by: str = "system",
                  wait: bool = False) -> int:
        if not self._run_lock.acquire(blocking=False):
            raise ServiceError("A pricing run is already in progress.")
        try:
            conn = self.conn(readonly=False)
            try:
                pd = price_date or self.default_price_date(conn)
                n = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
                if n == 0:
                    raise ServiceError("No products on file. Import a product list first.")
                with transaction(conn):
                    cur = conn.execute("INSERT INTO pricing_runs(price_date, started_at, status, products, requested_by) "
                                       "VALUES (?,?,?,?,?)", (pd.isoformat(), _now(), "running", n, requested_by))
                    run_id = int(cur.lastrowid)
            finally:
                conn.close()
        except BaseException:
            self._run_lock.release()
            raise
        self.progress = {"running": True, "run_id": run_id, "done": 0, "total": n, "phase": "Starting"}
        if wait:
            self._run(run_id, pd, refit)
        else:
            threading.Thread(target=self._run, args=(run_id, pd, refit), name=f"run-{run_id}", daemon=True).start()
        return run_id

    def _run(self, run_id: int, pd: date, refit: bool) -> None:
        t0 = time.perf_counter()
        o = pd.toordinal()
        errors: list[dict] = []
        try:
            conn = self.conn(readonly=False)
            try:
                if refit or self.models_stale(conn, o):
                    self.progress["phase"] = "Learning demand from sales history"
                    with transaction(conn):
                        fit_all(conn, o - 1, self.s)
                        meta_set(conn, "model_fitted_on", pd.isoformat())
                        meta_set(conn, "model_data_version", meta_get(conn, "data_version", "0"))
                models = self.models(conn)
                skus = [r[0] for r in conn.execute("SELECT sku FROM products ORDER BY sku")]
            finally:
                conn.close()

            self.progress["phase"] = "Pricing products"
            local = threading.local()

            def one(sku: str):
                c = getattr(local, "conn", None)
                if c is None:
                    c = local.conn = self.conn()
                try:
                    dm = models.get(sku)
                    if dm is None:
                        raise ServiceError("no demand model (not enough sales history)")
                    ctx = derive(c, sku, o, dm, self.s)
                    dec = self.engine.decide(ctx, dm, o)
                    return sku, ctx, dec, None
                except Exception as exc:  # isolate bad products
                    log.warning("pricing %s failed: %s", sku, exc)
                    return sku, None, None, f"{type(exc).__name__}: {exc}"
                finally:
                    self.progress["done"] = self.progress.get("done", 0) + 1

            with ThreadPoolExecutor(max_workers=max(1, self.s.workers)) as pool:
                results = list(pool.map(one, skus))

            rows = []
            for sku, ctx, dec, error in results:
                if error:
                    errors.append({"sku": sku, "error": error})
                    continue
                a = dec.advice
                payload = {
                    "state": ctx.state.model_dump(mode="json"), "inputs": [i.as_dict() for i in ctx.inputs],
                    "advice": a, "kkt": dec.kkt.model_dump(), "warnings": dec.warnings,
                    "game": {"nash": dec.comp.nash, "stackelberg": dec.comp.stackelberg},
                    "expected_units": dec.expected_units, "expected_units_at_prev": dec.expected_units_at_prev,
                    "unit_margin": dec.unit_margin, "response": dec.response, "horizon_mode": dec.horizon_mode,
                }
                rows.append((run_id, sku, pd.isoformat(), dec.prev_price, dec.price, dec.floor, dec.ceiling,
                             dec.competitor_price, a["action"], a["change_pct"], a["profit_gain"], dec.binding,
                             dec.horizon, json.dumps(payload)))
            conn = self.conn(readonly=False)
            try:
                with transaction(conn):
                    conn.executemany(
                        "INSERT INTO recommendations(run_id, sku, price_date, current_price, recommended_price, floor, "
                        "ceiling, competitor_price, action, change_pct, profit_gain, binding, horizon_days, payload) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
                    conn.execute("UPDATE pricing_runs SET finished_at = ?, status = 'done', priced = ?, errors = ?, "
                                 "model_fitted_on = ? WHERE run_id = ?",
                                 (_now(), len(rows), json.dumps(errors), meta_get(conn, "model_fitted_on"), run_id))
            finally:
                conn.close()
            log.info("pricing run %d for %s: %d products in %.1fs, %d errors", run_id, pd, len(rows),
                     time.perf_counter() - t0, len(errors))
        except Exception as exc:
            log.exception("pricing run %d failed", run_id)
            conn = self.conn(readonly=False)
            try:
                with transaction(conn):
                    conn.execute("UPDATE pricing_runs SET finished_at = ?, status = 'failed', errors = ? WHERE run_id = ?",
                                 (_now(), json.dumps([{"sku": None, "error": f"{type(exc).__name__}: {exc}"}]), run_id))
            finally:
                conn.close()
        finally:
            self.progress = {"running": False, "run_id": run_id}
            self._explain_cache.clear()
            self._run_lock.release()

    # ------------------------------------------------------------------ approvals
    def decide(self, sku: str, decision: str, user: str, price: float | None = None, note: str = "",
               run_id: int | None = None) -> dict:
        if decision not in ("approve", "override", "reject", "reset"):
            raise ServiceError("decision must be approve, override, reject or reset")
        user = (user or "").strip()[:80]
        if not user:
            raise ServiceError("Enter your name before approving prices.")
        conn = self.conn(readonly=False)
        try:
            run_id = run_id or self.latest_run_id(conn)
            rec = conn.execute("SELECT * FROM recommendations WHERE run_id = ? AND sku = ?", (run_id, sku)).fetchone()
            if rec is None:
                raise ServiceError(f"No recommendation for {sku} in run {run_id}.")
            final = None
            if decision == "approve":
                final, status = float(rec["recommended_price"]), "approved"
            elif decision == "override":
                if price is None or not math.isfinite(price):
                    raise ServiceError("Give the price to use.")
                final = float(round(price))
                if final < rec["floor"] - 1e-9:
                    raise ServiceError(f"₹{final:,.0f} is below the loss floor of ₹{rec['floor']:,.2f} for {sku}.")
                if final > rec["ceiling"] + 1e-9:
                    raise ServiceError(f"₹{final:,.0f} is above the MRP of ₹{rec['ceiling']:,.0f} for {sku}.")
                status = "approved" if abs(final - rec["recommended_price"]) < 0.5 else "overridden"
            elif decision == "reject":
                status = "rejected"
            else:
                status = "pending"
            with transaction(conn):
                conn.execute("UPDATE recommendations SET status = ?, final_price = ?, decided_by = ?, decided_at = ?, "
                             "note = ? WHERE run_id = ? AND sku = ?",
                             (status, final, user if status != "pending" else None,
                              _now() if status != "pending" else None, note[:500] or None, run_id, sku))
                d = rec["price_date"]
                conn.execute("DELETE FROM price_book WHERE sku = ? AND date = ? AND set_by NOT IN ('pricing team', 'import')",
                             (sku, d))
                if final is not None:
                    conn.execute("INSERT OR REPLACE INTO price_book VALUES (?,?,?,?,?)", (sku, d, final, 0, user))
            return {"sku": sku, "run_id": run_id, "status": status, "final_price": final}
        finally:
            conn.close()

    def decide_many(self, skus: list[str], decision: str, user: str, run_id: int | None = None) -> dict:
        if decision not in ("approve", "reject", "reset"):
            raise ServiceError("bulk decision must be approve, reject or reset")
        done, failed = [], []
        for sku in skus:
            try:
                done.append(self.decide(sku, decision, user, run_id=run_id))
            except ServiceError as exc:
                failed.append({"sku": sku, "error": str(exc)})
        return {"updated": len(done), "failed": failed}

    @staticmethod
    def latest_run_id(conn) -> int | None:
        r = conn.execute("SELECT MAX(run_id) FROM pricing_runs WHERE status = 'done'").fetchone()
        return r[0] if r and r[0] is not None else None

    # ------------------------------------------------------------------ explanation
    def explain(self, conn, sku: str, run_id: int) -> dict | None:
        key = (sku, run_id)
        if key in self._explain_cache:
            return self._explain_cache[key]
        row = conn.execute("SELECT price_date, payload FROM recommendations WHERE run_id = ? AND sku = ?",
                           (run_id, sku)).fetchone()
        dm = self.models(conn).get(sku)
        if row is None or dm is None:
            return None
        state = ProductState(**json.loads(row["payload"])["state"])
        o = date.fromisoformat(row["price_date"]).toordinal()
        snap = _Snapshot(state)
        dec = self.engine.decide(snap, dm, o)  # type: ignore[arg-type]
        out = self.engine.explain(dec, snap, dm, o)  # type: ignore[arg-type]
        for p in out["plan"]:
            p["date"] = date.fromordinal(p.pop("date_ordinal")).isoformat()
        if len(self._explain_cache) > 512:
            self._explain_cache.clear()
        self._explain_cache[key] = out
        return out

    # ------------------------------------------------------------------ optional daily schedule
    def start_schedule(self) -> None:
        if not self.s.daily_run_at or self._scheduler is not None:
            return
        try:
            hh, mm = (int(x) for x in self.s.daily_run_at.split(":"))
        except ValueError:
            log.error("PRICING_DAILY_RUN_AT must look like 06:00; schedule disabled")
            return

        def loop() -> None:
            while not self._stop.wait(30):
                now = datetime.now()
                if (now.hour, now.minute) < (hh, mm):
                    continue
                conn = self.conn()
                try:
                    pd = self.default_price_date(conn)
                    done = conn.execute("SELECT 1 FROM pricing_runs WHERE price_date = ? AND status != 'failed'",
                                        (pd.isoformat(),)).fetchone()
                finally:
                    conn.close()
                if not done:
                    try:
                        self.start_run(pd, requested_by="daily schedule")
                    except ServiceError:
                        pass

        self._scheduler = threading.Thread(target=loop, name="schedule", daemon=True)
        self._scheduler.start()

    def stop(self) -> None:
        self._stop.set()
