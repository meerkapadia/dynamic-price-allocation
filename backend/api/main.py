"""HTTP API for the pricing service.

Run:  uvicorn api.main:app --port 8000
Write endpoints (runs, approvals, imports) require the X-API-Key header when PRICING_API_KEY is set.
Built on Starlette (FastAPI's foundation) to keep the dependency surface small.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import os
import threading
from contextlib import asynccontextmanager
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from pricing_engine.config import get_settings
from pricing_engine.data.calendar import EVENTS
from pricing_engine.data.catalog import STORE_NAME
from pricing_engine.data.generate import generate_store
from pricing_engine.db import connect, init_schema, meta_get
from pricing_engine.ops.imports import SPECS, ImportError_, run_import, template
from pricing_engine.ops.service import PricingService, ServiceError

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("pricing.api")
SETTINGS = get_settings()
EVENT_LABEL = {e.key: e.label for e in EVENTS}
VERSION = "3.0.0"
MAX_UPLOAD = 50 * 1024 * 1024


class _State:
    service: PricingService | None = None
    ready = False
    phase = "Starting"
    error: str | None = None


STATE = _State()


def _boot() -> None:
    try:
        if SETTINGS.demo_data and _is_demo_or_missing():
            STATE.phase = "Building the demo store (two years of sample history)"
            generate_store(SETTINGS)  # no-op when the demo store is already current
        conn = connect(SETTINGS.db_path)
        init_schema(conn)
        n_products = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        n_runs = conn.execute("SELECT COUNT(*) FROM pricing_runs").fetchone()[0]
        conn.close()
        svc = PricingService(SETTINGS)
        STATE.service = svc
        if n_products and not n_runs and conn_has_sales():
            STATE.phase = "Calculating the first price recommendations"
            svc.start_run(requested_by="first start", wait=True)
        STATE.ready = True
        STATE.phase = "Ready"
        svc.start_schedule()
    except Exception as exc:  # pragma: no cover - surfaced through /api/status
        log.exception("startup failed")
        STATE.error = f"{type(exc).__name__}: {exc}"


def _is_demo_or_missing() -> bool:
    """Only ever (re)build a database that is missing or is itself the demo store; never touch real data."""
    if not SETTINGS.db_path.exists():
        return True
    c = connect(SETTINGS.db_path, readonly=True)
    try:
        has_meta = c.execute("SELECT 1 FROM sqlite_master WHERE name = 'store_meta'").fetchone()
        return bool(has_meta) and meta_get(c, "generated") == "1"
    except Exception:
        return False
    finally:
        c.close()


def conn_has_sales() -> bool:
    c = connect(SETTINGS.db_path, readonly=True)
    try:
        return c.execute("SELECT 1 FROM stock_daily LIMIT 1").fetchone() is not None
    finally:
        c.close()


@asynccontextmanager
async def lifespan(_app):
    threading.Thread(target=_boot, name="boot", daemon=True).start()
    yield
    if STATE.service:
        STATE.service.stop()


# ====================================================================== helpers


def _clean(x):
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def ok(data, status: int = 200) -> Response:
    return Response(json.dumps(_clean(data), separators=(",", ":")), status_code=status, media_type="application/json")


def err(msg: str, status: int, **extra) -> Response:
    return JSONResponse({"detail": msg, **extra}, status_code=status)


def _conn():
    return connect(SETTINGS.db_path, readonly=True)


def _rows(conn, sql: str, args=()) -> list[dict]:
    return [dict(r) for r in conn.execute(sql, args)]


def _ready() -> Response | None:
    if STATE.error:
        return err(STATE.error, 500)
    if not STATE.ready:
        return err(STATE.phase, 503)
    return None


def _authorised(request: Request) -> Response | None:
    if SETTINGS.api_key and request.headers.get("x-api-key") != SETTINGS.api_key:
        return err("This action needs the store's API key.", 401)
    return None


async def _json(request: Request) -> dict:
    try:
        body = await request.json()
    except Exception:
        return {}
    return body if isinstance(body, dict) else {}


def _int(q, name: str, default: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(q.get(name, default))))
    except (TypeError, ValueError):
        return default


def _run_id(conn, request: Request) -> int | None:
    raw = request.query_params.get("run_id")
    if raw:
        try:
            return int(raw)
        except ValueError:
            return None
    return PricingService.latest_run_id(conn)


# ====================================================================== status & runs


async def health(_: Request) -> Response:
    return ok({"status": "ok" if STATE.ready else ("error" if STATE.error else "starting"), "phase": STATE.phase})


def status(_: Request) -> Response:
    base = {"store": STORE_NAME, "version": VERSION, "ready": STATE.ready, "phase": STATE.phase, "error": STATE.error,
            "api_key_required": bool(SETTINGS.api_key), "daily_run_at": SETTINGS.daily_run_at or None}
    if not STATE.ready:
        return ok(base)
    svc = STATE.service
    conn = _conn()
    try:
        cov = conn.execute("SELECT MIN(date), MAX(date), COUNT(*) FROM stock_daily").fetchone()
        last_run = conn.execute("SELECT * FROM pricing_runs ORDER BY run_id DESC LIMIT 1").fetchone()
        base.update({
            "demo": conn.execute("SELECT COUNT(*) FROM market_truth").fetchone()[0] > 0,
            "products": conn.execute("SELECT COUNT(*) FROM products").fetchone()[0],
            "sales_from": cov[0], "sales_to": cov[1], "sales_rows": cov[2],
            "next_price_date": svc.default_price_date(conn).isoformat() if svc else None,
            "model_fitted_on": meta_get(conn, "model_fitted_on"),
            "models_stale": svc.models_stale(conn, svc.default_price_date(conn).toordinal()) if svc else None,
            "last_run": dict(last_run) if last_run else None,
            "progress": svc.progress if svc else None,
        })
        return ok(base)
    finally:
        conn.close()


def runs(_: Request) -> Response:
    if (r := _ready()) is not None:
        return r
    conn = _conn()
    try:
        rows = _rows(conn, """SELECT r.*, (SELECT COUNT(*) FROM recommendations x WHERE x.run_id = r.run_id
                   AND x.status != 'pending') AS decided FROM pricing_runs r ORDER BY run_id DESC LIMIT 50""")
        for x in rows:
            x["errors"] = json.loads(x["errors"] or "[]")
        return ok(rows)
    finally:
        conn.close()


async def start_run(request: Request) -> Response:
    if (r := _ready() or _authorised(request)) is not None:
        return r
    body = await _json(request)
    pd = None
    if body.get("price_date"):
        try:
            pd = date.fromisoformat(str(body["price_date"]))
        except ValueError:
            return err("price_date must be YYYY-MM-DD", 422)
    try:
        run_id = STATE.service.start_run(pd, refit=bool(body.get("refit")), requested_by=str(body.get("user") or "web"))
    except ServiceError as exc:
        return err(str(exc), 409)
    return ok({"run_id": run_id}, 202)


# ====================================================================== recommendations

REC_LIST_SQL = """
SELECT p.sku, p.name, p.category, p.brand, p.mrp, p.shelf_life_days,
       r.run_id, r.price_date, r.current_price, r.recommended_price, r.floor, r.ceiling, r.competitor_price, r.action,
       r.change_pct, r.profit_gain, r.binding, r.horizon_days, r.status, r.final_price, r.decided_by, r.decided_at,
       json_extract(r.payload, '$.advice.target_price') AS target_price,
       json_extract(r.payload, '$.advice.factors') AS factors,
       json_extract(r.payload, '$.expected_units') AS expected_units,
       COALESCE(l.on_hand, 0) AS on_hand, COALESCE(l.expiring, 0) AS expiring_3d,
       COALESCE(s.sold_28, 0) AS sold_28d
FROM recommendations r
JOIN products p ON p.sku = r.sku
LEFT JOIN (SELECT sku, SUM(qty_on_hand) AS on_hand,
                  SUM(CASE WHEN expires_on <= :exp THEN qty_on_hand ELSE 0 END) AS expiring
           FROM lots WHERE qty_on_hand > 0 GROUP BY sku) l ON l.sku = p.sku
LEFT JOIN (SELECT sku, SUM(sold) AS sold_28 FROM stock_daily WHERE date >= :since AND date < :pd GROUP BY sku) s
       ON s.sku = p.sku
WHERE r.run_id = :run
ORDER BY p.category, p.name
"""


def _main_driver(action: str, factors: list[dict]) -> tuple[str | None, bool]:
    sign = {"increase": 1, "decrease": -1}.get(action, 0)
    pool = [f for f in factors if sign == 0 or f["delta"] * sign > 0] or factors
    main = max(pool, key=lambda f: abs(f["delta"]), default=None)
    if not main or abs(main["delta"]) < 0.5:
        return None, False
    return main["key"], main["delta"] > 0


def recommendations(request: Request) -> Response:
    if (r := _ready()) is not None:
        return r
    conn = _conn()
    try:
        run_id = _run_id(conn, request)
        if run_id is None:
            return ok({"run": None, "rows": []})
        run = conn.execute("SELECT * FROM pricing_runs WHERE run_id = ?", (run_id,)).fetchone()
        if run is None:
            return err(f"No pricing run {run_id}", 404)
        pd = date.fromisoformat(run["price_date"])
        rows = _rows(conn, REC_LIST_SQL, {"run": run_id, "pd": pd.isoformat(),
                                          "exp": (pd + timedelta(days=2)).isoformat(),
                                          "since": (pd - timedelta(days=28)).isoformat()})
        for x in rows:
            factors = json.loads(x.pop("factors") or "[]")
            x["main_driver"], x["main_driver_up"] = _main_driver(x["action"], factors)
            daily = x["sold_28d"] / 28.0
            x["days_of_cover"] = x["on_hand"] / daily if daily > 0 else None
        run_d = dict(run)
        run_d["errors"] = json.loads(run_d["errors"] or "[]")
        return ok({"run": run_d, "rows": rows})
    finally:
        conn.close()


async def bulk_decision(request: Request) -> Response:
    if (r := _ready() or _authorised(request)) is not None:
        return r
    body = await _json(request)
    skus = body.get("skus")
    if not isinstance(skus, list) or not skus or not all(isinstance(s, str) for s in skus):
        return err("skus must be a non-empty list of SKU codes", 422)
    try:
        res = STATE.service.decide_many(skus[:5000], str(body.get("decision", "")), str(body.get("user", "")),
                                        run_id=body.get("run_id"))
    except ServiceError as exc:
        return err(str(exc), 422)
    return ok(res)


async def decision(request: Request) -> Response:
    if (r := _ready() or _authorised(request)) is not None:
        return r
    body = await _json(request)
    price = body.get("price")
    try:
        price = None if price is None or price == "" else float(price)
    except (TypeError, ValueError):
        return err("price must be a number", 422)
    try:
        res = STATE.service.decide(request.path_params["sku"], str(body.get("decision", "")), str(body.get("user", "")),
                                   price=price, note=str(body.get("note") or ""), run_id=body.get("run_id"))
    except ServiceError as exc:
        return err(str(exc), 422)
    return ok(res)


def product(request: Request) -> Response:
    if (r := _ready()) is not None:
        return r
    sku = request.path_params["sku"]
    conn = _conn()
    svc = STATE.service
    try:
        p = conn.execute("""SELECT p.*, s.name AS supplier, s.lead_time_days, c.min_margin_pct, c.max_daily_change_pct
                   FROM products p JOIN suppliers s ON s.supplier_id = p.supplier_id JOIN categories c ON c.name = p.category
                   WHERE p.sku = ?""", (sku,)).fetchone()
        if p is None:
            return err(f"No product with SKU {sku}", 404)
        run_id = _run_id(conn, request)
        rec = conn.execute("SELECT * FROM recommendations WHERE run_id = ? AND sku = ?", (run_id, sku)).fetchone() \
            if run_id else None
        recommendation = None
        explanation = None
        if rec:
            recommendation = {k: rec[k] for k in rec.keys() if k != "payload"} | json.loads(rec["payload"])
            explanation = svc.explain(conn, sku, run_id)
        dm = svc.models(conn).get(sku)
        last = conn.execute("SELECT MAX(date) FROM stock_daily WHERE sku = ?", (sku,)).fetchone()[0]
        last_d = date.fromisoformat(last) if last else date.today()

        weekly = _rows(conn, """SELECT date(s.date, 'weekday 0', '-6 days') AS week, SUM(s.sold) AS units,
                SUM(s.sold * pb.price) / NULLIF(SUM(s.sold), 0) AS avg_price, AVG(c.price) AS competitor,
                SUM(s.stocked_out) AS stockout_days
                FROM stock_daily s JOIN price_book pb ON pb.sku = s.sku AND pb.date = s.date
                LEFT JOIN competitor_prices c ON c.sku = s.sku AND c.date = s.date
                WHERE s.sku = ? GROUP BY week ORDER BY week""", (sku,))
        daily = _rows(conn, """SELECT s.date, s.sold, s.stocked_out, pb.price, c.price AS competitor FROM stock_daily s
                JOIN price_book pb ON pb.sku = s.sku AND pb.date = s.date
                LEFT JOIN competitor_prices c ON c.sku = s.sku AND c.date = s.date
                WHERE s.sku = ? AND s.date > ? ORDER BY s.date""", (sku, (last_d - timedelta(days=120)).isoformat()))
        model = None
        if dm is not None:
            if daily:
                comp = np.array([d["competitor"] if d["competitor"] is not None else d["price"] for d in daily])
                fitted = dm.mean(np.array([date.fromisoformat(d["date"]).toordinal() for d in daily], dtype=np.int64),
                                 np.array([d["price"] for d in daily]), comp)
                for d, f in zip(daily, fitted):
                    d["fitted"] = float(f)
            start = (date.fromisoformat(rec["price_date"]) if rec else last_d + timedelta(days=1)).toordinal()
            fdays = start + np.arange(56)
            price = rec["recommended_price"] if rec else p["mrp"]
            comp_p = rec["competitor_price"] if rec else price
            model = {
                "fitted_on": dm.fitted_on, "n_obs": dm.n_obs, "n_censored": dm.n_censored, "holdout_wape": dm.holdout_wape,
                "dispersion": dm.dispersion, "iterations": dm.iterations, "own_elasticity": dm.own_elasticity,
                "own_se": float(dm.se[1]), "cross_elasticity": dm.cross_elasticity, "cross_se": float(dm.se[2]),
                "competitor_reaction": dm.competitor_reaction, "annual_growth": dm.annual_growth(),
                "yearly_index": dm.yearly_index(), "weekday_index": dm.weekday_index(),
                "events": [{"event": k, "label": EVENT_LABEL[k], "lift": v[0], "lo": v[1], "hi": v[2]}
                           for k, v in dm.event_lifts().items()],
                "forecast": [{"date": date.fromordinal(int(o)).isoformat(), "units": float(m)}
                             for o, m in zip(fdays, dm.mean(fdays, price, comp_p))],
                "forecast_price": price,
            }
        lots = _rows(conn, "SELECT lot_id, received_on, expires_on, qty_on_hand, unit_cost FROM lots "
                           "WHERE sku = ? AND qty_on_hand > 0 ORDER BY expires_on", (sku,))
        history = _rows(conn, """SELECT run_id, price_date, current_price, recommended_price, action, status, final_price,
                decided_by, decided_at, note FROM recommendations WHERE sku = ? ORDER BY run_id DESC LIMIT 30""", (sku,))
        return ok({"product": dict(p), "recommendation": recommendation, "explanation": explanation, "model": model,
                   "weekly": weekly, "daily": daily, "lots": lots, "history": history, "last_sales_date": last})
    finally:
        conn.close()


def price_changes(request: Request) -> Response:
    if (r := _ready()) is not None:
        return r
    q = request.query_params
    page, size = _int(q, "page", 1, 1, 100_000), _int(q, "page_size", 50, 10, 200)
    where, args = ["r.status != 'pending'"], []
    if q.get("status") in ("approved", "overridden", "rejected"):
        where.append("r.status = ?"); args.append(q["status"])
    if q.get("sku"):
        where.append("r.sku = ?"); args.append(q["sku"])
    w = " AND ".join(where)
    conn = _conn()
    try:
        total = conn.execute(f"SELECT COUNT(*) FROM recommendations r WHERE {w}", args).fetchone()[0]
        rows = _rows(conn, f"""SELECT r.run_id, r.sku, p.name, r.price_date, r.current_price, r.recommended_price,
                r.final_price, r.status, r.decided_by, r.decided_at, r.note, r.action, r.profit_gain
                FROM recommendations r JOIN products p USING(sku) WHERE {w}
                ORDER BY r.decided_at DESC LIMIT ? OFFSET ?""", [*args, size, (page - 1) * size])
        return ok({"total": total, "page": page, "page_size": size, "rows": rows})
    finally:
        conn.close()


def export_prices(request: Request) -> Response:
    """Price list for the price date of a run: approved prices, everything else unchanged."""
    if (r := _ready()) is not None:
        return r
    conn = _conn()
    try:
        run_id = _run_id(conn, request)
        if run_id is None:
            return err("No pricing run yet.", 404)
        only_changed = request.query_params.get("changed") == "1"
        run = conn.execute("SELECT price_date FROM pricing_runs WHERE run_id = ?", (run_id,)).fetchone()
        rows = conn.execute("""SELECT p.sku, p.name, p.mrp, r.current_price, r.final_price, r.status, r.decided_by
                FROM products p LEFT JOIN recommendations r ON r.sku = p.sku AND r.run_id = ?
                ORDER BY p.category, p.name""", (run_id,)).fetchall()
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["sku", "name", "effective_date", "price", "previous_price", "change_pct", "mrp", "status", "approved_by"])
        for x in rows:
            new = x["final_price"] if x["status"] in ("approved", "overridden") else x["current_price"]
            changed = x["status"] in ("approved", "overridden") and abs((x["final_price"] or 0) - (x["current_price"] or 0)) >= 0.5
            if only_changed and not changed:
                continue
            pct = (new / x["current_price"] - 1) * 100 if x["current_price"] and new else 0.0
            w.writerow([x["sku"], x["name"], run["price_date"], f"{new:.2f}" if new else "", f"{x['current_price'] or 0:.2f}",
                        f"{pct:.2f}", f"{x['mrp']:.2f}", x["status"] or "not priced", x["decided_by"] or ""])
        return Response(buf.getvalue(), media_type="text/csv", headers={
            "Content-Disposition": f'attachment; filename="prices-{run["price_date"]}.csv"'})
    finally:
        conn.close()


# ====================================================================== data


def data_summary(_: Request) -> Response:
    if (r := _ready()) is not None:
        return r
    conn = _conn()
    try:
        out = []
        tables = {"products": ("products", None), "sales": ("stock_daily", "date"), "stock": ("lots", "received_on"),
                  "competitor": ("competitor_prices", "date"), "write_offs": ("write_offs", "date")}
        for kind, (table, col) in tables.items():
            where = " WHERE qty_on_hand > 0" if table == "lots" else ""
            n = conn.execute(f"SELECT COUNT(*) FROM {table}{where}").fetchone()[0]
            span = conn.execute(f"SELECT MIN({col}), MAX({col}) FROM {table}{where}").fetchone() if col else (None, None)
            spec = SPECS[kind]
            out.append({"kind": kind, "label": spec["label"], "help": spec["help"], "required": spec["required"],
                        "optional": spec["optional"], "rows": n, "from": span[0], "to": span[1],
                        "last_import": meta_get(conn, f"last_import_{kind}")})
        return ok(out)
    finally:
        conn.close()


def import_template(request: Request) -> Response:
    kind = request.path_params["kind"]
    if kind not in SPECS:
        return err(f"Unknown data type {kind}", 404)
    return Response(template(kind), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{kind}-template.csv"'})


async def import_data(request: Request) -> Response:
    if (r := _ready() or _authorised(request)) is not None:
        return r
    kind = request.path_params["kind"]
    if kind not in SPECS:
        return err(f"Unknown data type {kind}", 404)
    raw = await request.body()
    if len(raw) > MAX_UPLOAD:
        return err("The file is larger than 50 MB; split it by date.", 413)
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return err("Save the file as UTF-8 CSV and try again.", 422)
    conn = connect(SETTINGS.db_path)
    try:
        res = run_import(conn, kind, text, append=request.query_params.get("mode") == "append")
    except ImportError_ as exc:
        return err(f"Nothing was imported: {len(exc.errors)} problem(s) found.", 422, errors=exc.errors)
    finally:
        conn.close()
    return ok(res)


RECORD_SQL = {
    "products": ("""SELECT p.sku, p.name, p.category, p.brand, p.mrp, p.shelf_life_days, p.storage_class, p.handling_cost,
                    p.storage_cost_per_day, p.salvage_pct, p.case_pack, s.name AS supplier_name, s.lead_time_days,
                    c.min_margin_pct, c.max_daily_change_pct FROM products p
                    JOIN suppliers s ON s.supplier_id = p.supplier_id JOIN categories c ON c.name = p.category""",
                 "p.category, p.name"),
    "sales": ("""SELECT s.sku, p.name, s.date, s.sold AS units_sold, b.price, NULLIF(s.closing, -1) AS closing_stock,
                 s.stocked_out, b.promo FROM stock_daily s JOIN products p USING(sku)
                 LEFT JOIN price_book b ON b.sku = s.sku AND b.date = s.date""", "s.date DESC, s.sku"),
    "stock": ("""SELECT l.sku, p.name, l.received_on, l.expires_on, l.qty_on_hand, l.unit_cost FROM lots l
                 JOIN products p USING(sku) WHERE l.qty_on_hand > 0""", "l.expires_on, l.sku"),
    "competitor": ("""SELECT c.sku, p.name, c.date, c.price FROM competitor_prices c JOIN products p USING(sku)""",
                   "c.date DESC, c.sku"),
    "write_offs": ("""SELECT w.sku, p.name, w.date, w.qty, w.reason, w.unit_cost FROM write_offs w JOIN products p USING(sku)""",
                   "w.date DESC, w.sku"),
}


def records(request: Request) -> Response:
    """Browse any table of the store database (paged, searchable by SKU or name)."""
    if (r := _ready()) is not None:
        return r
    kind = request.path_params["kind"]
    if kind not in RECORD_SQL:
        return err(f"Unknown table {kind}", 404)
    q = request.query_params
    page, size = _int(q, "page", 1, 1, 1_000_000), _int(q, "page_size", 50, 10, 200)
    base, order = RECORD_SQL[kind]
    args: list = []
    if q.get("q"):
        joiner = " AND " if " WHERE " in base else " WHERE "
        base += joiner + "(p.sku LIKE ? OR p.name LIKE ?)"
        args += [f"%{q['q']}%"] * 2
    conn = _conn()
    try:
        total = conn.execute(f"SELECT COUNT(*) FROM ({base})", args).fetchone()[0]
        rows = _rows(conn, f"{base} ORDER BY {order} LIMIT ? OFFSET ?", [*args, size, (page - 1) * size])
        return ok({"total": total, "page": page, "page_size": size, "rows": rows})
    finally:
        conn.close()


def model(_: Request) -> Response:
    if (r := _ready()) is not None:
        return r
    conn = _conn()
    try:
        rows = _rows(conn, """SELECT m.sku, p.name, p.category, m.fitted_on, m.n_obs, m.n_censored, m.holdout_wape,
                m.newton_iters, m.competitor_reaction, json_extract(m.coef, '$[1]') AS own_elasticity,
                json_extract(m.se, '$[1]') AS own_se, json_extract(m.coef, '$[2]') AS cross_elasticity,
                json_extract(m.se, '$[2]') AS cross_se FROM demand_models m JOIN products p USING(sku)
                ORDER BY p.category, p.name""")
        s = SETTINGS
        return ok({"models": rows, "settings": {
            "max_horizon_days": s.max_horizon_days, "price_grid_size": s.price_grid_size,
            "max_inventory_states": s.max_inventory_states, "refit_every_days": s.refit_every_days,
            "fit_window_days": s.fit_window_days, "market_coverage": s.market_coverage,
            "reaction_threshold": s.reaction_threshold, "workers": s.workers}})
    finally:
        conn.close()


routes = [
    Route("/health", health),
    Route("/api/status", status),
    Route("/api/runs", runs, methods=["GET"]),
    Route("/api/runs", start_run, methods=["POST"]),
    Route("/api/recommendations", recommendations, methods=["GET"]),
    Route("/api/recommendations/decisions", bulk_decision, methods=["POST"]),
    Route("/api/products/{sku}", product, methods=["GET"]),
    Route("/api/products/{sku}/decision", decision, methods=["POST"]),
    Route("/api/price-changes", price_changes),
    Route("/api/export/prices.csv", export_prices),
    Route("/api/data", data_summary),
    Route("/api/import/{kind}/template.csv", import_template),
    Route("/api/import/{kind}", import_data, methods=["POST"]),
    Route("/api/model", model),
    Route("/api/records/{kind}", records),
]

_static = Path(os.environ.get("PRICING_STATIC_DIR", Path(__file__).resolve().parents[2] / "frontend" / "dist"))
if _static.is_dir():
    routes.append(Mount("/", StaticFiles(directory=_static, html=True), name="web"))

app = Starlette(
    routes=routes,
    lifespan=lifespan,
    middleware=[
        Middleware(GZipMiddleware, minimum_size=1024),
        Middleware(CORSMiddleware, allow_origins=[o.strip() for o in os.environ.get(
            "CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()],
                   allow_methods=["GET", "POST"], allow_headers=["*"]),
    ],
)
