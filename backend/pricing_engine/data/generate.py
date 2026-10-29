"""Builds the store database: master data plus two years of trading run by a human pricing team and planner.

The human policies are deliberately ordinary (cost-anchored list prices, ad-hoc promotions, clearance markdowns,
order-up-to replenishment from a 14-day moving average). They leave realistic fingerprints the estimator must
cope with: stock-outs that censor demand at season ramps, expiry write-offs, and price variation that is
partly driven by festivals rather than random.
"""

from __future__ import annotations

import logging
import math
import os
import sqlite3
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

from ..config import Settings
from ..db import connect, init_schema, meta_get, meta_set, transaction
from .calendar import EVENT_KEYS, calendar_rows
from .catalog import (
    CATEGORIES,
    CATEGORY_SUPPLIER,
    PRODUCTS,
    SUPPLIERS,
    Spec,
    handling_cost,
    is_perishable,
    salvage_pct,
    storage_class,
    storage_cost_per_day,
)
from .market import Market, Sink, SkuState, build_truth, cal_row, retail_round
from .store import flush

log = logging.getLogger(__name__)
SCHEMA_VERSION = "4"


def _event_pressure(st: SkuState, o: int) -> float:
    ev = cal_row(o)[0]
    return sum(st.truth.events.get(k, 0.0) * ev[j] for j, k in enumerate(EVENT_KEYS))


@dataclass(slots=True)
class _TeamMemory:
    anchor: float = 1.0
    next_review: int = 0
    promo_left: int = 0
    promo_depth: float = 0.0
    sold: deque = field(default_factory=lambda: deque(maxlen=14))
    stockouts: deque = field(default_factory=lambda: deque(maxlen=14))


class PricingTeam:
    """Category managers: a slowly drifting list price, festival promotions and clearance markdowns."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.mem: dict[str, _TeamMemory] = {}

    def prices(self, states: dict[str, SkuState], o: int) -> tuple[dict[str, float], dict[str, int]]:
        prices, promos = {}, {}
        for sku, st in states.items():
            m = self.mem.setdefault(sku, _TeamMemory(next_review=o + int(self.rng.integers(10, 40))))
            r = self.rng
            if o >= m.next_review:
                m.anchor = float(np.clip(m.anchor * math.exp(r.normal(0, 0.035)), 0.92, 1.07))
                m.next_review = o + int(r.integers(21, 50))
            pressure = _event_pressure(st, o)
            if m.promo_left == 0:
                p_start = 0.13 if pressure > 0.6 else 0.016
                if r.random() < p_start:
                    m.promo_left = int(r.integers(4, 10))
                    m.promo_depth = float(r.uniform(0.07, 0.2))
            depth = m.promo_depth if m.promo_left > 0 else 0.0
            if m.promo_left > 0:
                m.promo_left -= 1
            markdown = 0.0
            if is_perishable(st.spec) and st.lots:
                oldest = min(st.lots, key=lambda l: l.expires)
                avg = (sum(m.sold) / len(m.sold)) if m.sold else 1.0
                if oldest.expires <= o and oldest.qty > 1.1 * avg:
                    markdown = float(r.uniform(0.1, 0.25))
            p = st.truth.usual_price * m.anchor * (1 - depth) * (1 - markdown)
            prices[sku] = retail_round(p, st.spec.mrp)
            promos[sku] = int(depth > 0)
        return prices, promos

    def observe(self, sku: str, sold: int, stocked_out: bool) -> None:
        m = self.mem[sku]
        m.sold.append(sold)
        m.stockouts.append(stocked_out)


class Planner:
    """Daily order-up-to review on a moving-average forecast with a rough festival adjustment."""

    def review(self, market: Market, sink: Sink, states: dict[str, SkuState], team: PricingTeam, o: int,
               rng: np.random.Generator) -> None:
        for sku, st in states.items():
            mem = team.mem[sku]
            window = list(mem.sold)[-7:] if is_perishable(st.spec) else list(mem.sold)
            if not window:
                continue
            f = sum(window) / len(window)
            if any(list(mem.stockouts)[-len(window):]):
                f *= 1.15
            L = st.lead_time
            f *= math.exp(0.6 * (_event_pressure(st, o + L + 1) - _event_pressure(st, max(o - 3, 0))))
            f = max(f, 0.05)
            if is_perishable(st.spec):
                cover = 1.0 if st.spec.shelf <= 5 else min(3.0, st.spec.shelf / 4)
                trigger, target = f * (L + 0.6 * cover), f * (L + cover + 0.3)
            else:
                cover = 14.0
                trigger, target = f * (L + 0.5 * cover + 1), f * (L + cover + 1)
            usable = sum(l.qty for l in st.lots if l.expires > o + L)
            ip = usable + st.on_order
            if ip < trigger:
                q = target - ip
                case = st.spec.case
                q = max(case, int(math.ceil(q / case) * case))
                market.raise_po(sink, st, o, q, market.unit_cost(st, o, rng), "planner")


def _seed_master_data(conn: sqlite3.Connection, rng: np.random.Generator, settings: Settings,
                      first: date, last: date) -> dict:
    conn.executemany("INSERT INTO categories VALUES (?,?,?,?,?)", CATEGORIES)
    conn.executemany("INSERT INTO suppliers VALUES (?,?,?,?,?)", SUPPLIERS)
    sup = CATEGORY_SUPPLIER
    truths = {}
    for spec in PRODUCTS:
        conn.execute(
            "INSERT INTO products VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (spec.sku, spec.name, spec.category, spec.brand, spec.mrp, spec.shelf, storage_class(spec),
             handling_cost(spec), storage_cost_per_day(spec), salvage_pct(spec), sup[spec.category], spec.case),
        )
        t = build_truth(spec, rng, settings.demand_scale)
        truths[spec.sku] = t
        conn.execute("INSERT INTO market_truth VALUES (?,?)", (spec.sku, t.to_json()))
    conn.executemany("INSERT INTO calendar_events VALUES (?,?,?,?,?)",
                     calendar_rows(first, date.fromordinal(last.toordinal() + 400)))
    return truths


def _initial_stock(market: Market, sink: Sink, st: SkuState, o: int, rng: np.random.Generator) -> None:
    spec: Spec = st.spec
    lam = math.exp(st.truth.log_demand_at(o, st.truth.usual_price, st.truth.comp_usual_price, 0))
    cover = 2 if is_perishable(spec) else 20
    q = max(spec.case, int(math.ceil(lam * (st.lead_time + cover) / spec.case) * spec.case))
    age = int(rng.integers(0, max(1, spec.shelf // 4)))
    market.add_lot(sink, st, o, q, market.unit_cost(st, o - age, rng), None, age=age)
    st.last_price = retail_round(st.truth.usual_price, spec.mrp)
    st.last_comp = retail_round(st.truth.comp_usual_price, spec.mrp)


def generate_store(settings: Settings) -> Path:
    """Create the database if missing (atomic: written to a temp file, then renamed)."""
    path = settings.db_path
    if path.exists():
        try:
            with connect(path, readonly=True) as c:
                if meta_get(c, "schema_version") == SCHEMA_VERSION and meta_get(c, "generated") == "1":
                    return path
        except sqlite3.DatabaseError:
            pass
        log.warning("store database %s is stale or incomplete; regenerating", path)
        for suffix in ("", "-wal", "-shm"):
            Path(f"{path}{suffix}").unlink(missing_ok=True)

    t0 = time.perf_counter()
    tmp = path.with_suffix(".building")
    for suffix in ("", "-wal", "-shm"):
        Path(f"{tmp}{suffix}").unlink(missing_ok=True)
    conn = connect(tmp)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    init_schema(conn)
    rng = np.random.default_rng(settings.random_seed)
    start = settings.start_date.toordinal()
    first = start - settings.history_days
    with transaction(conn):
        truths = _seed_master_data(conn, rng, settings, date.fromordinal(first), settings.start_date)

    market = Market(truths, settings.random_seed)
    states = {spec.sku: market.new_state(spec) for spec in PRODUCTS}
    sink = Sink()
    team, planner = PricingTeam(rng), Planner()
    for st in states.values():
        _initial_stock(market, sink, st, first, rng)

    for o in range(first, start):
        received = market.morning(sink, states, o)
        prices, promos = team.prices(states, o)
        res = market.close(sink, states, o, prices, promos, received, "pricing team")
        for sku, (sold, _demand, out) in res.by_sku.items():
            team.observe(sku, sold, out)
        planner.review(market, sink, states, team, o, rng)
        if len(sink.sales) > 150_000:
            with transaction(conn):
                flush(conn, sink)
    with transaction(conn):
        flush(conn, sink)
        meta_set(conn, "schema_version", SCHEMA_VERSION)
        meta_set(conn, "business_date", date.fromordinal(start).isoformat())
        meta_set(conn, "history_start", date.fromordinal(first).isoformat())
        meta_set(conn, "seed", str(settings.random_seed))
        meta_set(conn, "generated", "1")
    conn.execute("ANALYZE")
    conn.close()
    os.replace(tmp, path)
    log.info("store database generated in %.1fs at %s", time.perf_counter() - t0, path)
    return path
