"""Moves simulator state in and out of SQLite (flush rows, reload live state for the next business day)."""

from __future__ import annotations

import math
import sqlite3
from datetime import date

from .catalog import PRODUCTS
from .market import PO, Lot, Market, Sink, SkuState, Truth, iso


def flush(conn: sqlite3.Connection, sink: Sink) -> None:
    conn.executemany("INSERT OR REPLACE INTO sales VALUES (?,?,?,?,?,?,?,?,?)", sink.sales)
    conn.executemany("INSERT INTO write_offs(sku, lot_id, date, qty, reason, unit_cost) VALUES (?,?,?,?,?,?)",
                     sink.write_offs)
    conn.executemany("INSERT OR REPLACE INTO price_book VALUES (?,?,?,?,?)", sink.price_book)
    conn.executemany("INSERT OR REPLACE INTO competitor_prices VALUES (?,?,?)", sink.competitor)
    conn.executemany("INSERT OR REPLACE INTO stock_daily VALUES (?,?,?,?,?,?,?,?,?,?)", sink.stock)
    conn.executemany(
        "INSERT OR REPLACE INTO lots VALUES (?,?,?,?,?,?,?,?)",
        [(l.lot_id, l.sku, l.po_id, iso(l.received), iso(l.expires), l.qty_received, l.qty, l.cost)
         for l in sink.lots.values()],
    )
    conn.executemany(
        "INSERT OR REPLACE INTO purchase_orders VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        [(p.po_id, p.sku, p.supplier_id, iso(p.ordered), iso(p.expected), iso(p.received) if p.received else None,
          p.qty, p.qty_received, p.cost, "received" if p.received else "open", p.raised_by) for p in sink.pos.values()],
    )
    sink.sales.clear(); sink.write_offs.clear(); sink.price_book.clear(); sink.competitor.clear()
    sink.stock.clear(); sink.lots.clear(); sink.pos.clear()


def _ord(s: str) -> int:
    return date.fromisoformat(s).toordinal()


def load_market(conn: sqlite3.Connection, seed: int, business_ordinal: int) -> tuple[Market, dict[str, SkuState]]:
    truths = {r["sku"]: Truth.from_json(r["params"]) for r in conn.execute("SELECT sku, params FROM market_truth")}
    market = Market(truths, seed)
    market.next_lot = int(conn.execute("SELECT COALESCE(MAX(lot_id), 0) FROM lots").fetchone()[0]) + 1
    market.po_seq[business_ordinal] = int(
        conn.execute("SELECT COUNT(*) FROM purchase_orders WHERE ordered_on = ?", (iso(business_ordinal),)).fetchone()[0]
    )
    states: dict[str, SkuState] = {}
    for spec in PRODUCTS:
        st = market.new_state(spec)
        st.lots = [
            Lot(r["lot_id"], r["sku"], r["po_id"], _ord(r["received_on"]), _ord(r["expires_on"]), r["qty_received"],
                r["qty_on_hand"], r["unit_cost"])
            for r in conn.execute("SELECT * FROM lots WHERE sku = ? AND qty_on_hand > 0", (spec.sku,))
        ]
        st.pos = [
            PO(r["po_id"], r["sku"], r["supplier_id"], _ord(r["ordered_on"]), _ord(r["expected_on"]), r["qty_ordered"],
               r["unit_cost"], r["raised_by"])
            for r in conn.execute("SELECT * FROM purchase_orders WHERE sku = ? AND status = 'open'", (spec.sku,))
        ]
        lp = conn.execute("SELECT price FROM price_book WHERE sku = ? ORDER BY date DESC LIMIT 1", (spec.sku,)).fetchone()
        lc = conn.execute("SELECT date, price FROM competitor_prices WHERE sku = ? ORDER BY date DESC LIMIT 1",
                          (spec.sku,)).fetchone()
        st.last_price = float(lp[0]) if lp else st.truth.usual_price
        if lc:
            st.last_comp = float(lc["price"])
            st.comp_dev = math.log(st.last_comp / market.competitor_target(st, _ord(lc["date"])))
        states[spec.sku] = st
    return market, states
