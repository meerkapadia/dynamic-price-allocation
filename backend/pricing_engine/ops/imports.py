"""CSV import and export: how a real store's data gets in and approved prices get out.

Each import file is validated as a whole; if any row is wrong nothing is written and the first errors are
returned with their line numbers. Imports upsert, so sending the same day twice simply replaces it.

    products     sku, name, category, mrp, shelf_life_days [, brand, storage_class, handling_cost,
                 storage_cost_per_day, salvage_pct, case_pack, supplier_name, lead_time_days, min_margin_pct,
                 max_daily_change_pct]
    sales        sku, date, units_sold, price [, closing_stock, stocked_out, promo]          one row per product-day
    stock        sku, qty_on_hand, unit_cost [, received_on, expires_on]                     replaces open stock
    competitor   sku, date, price
    write_offs   sku, date, qty [, reason, unit_cost]
"""

from __future__ import annotations

import csv
import io
import math
import sqlite3
from collections.abc import Callable
from datetime import date, timedelta

from ..db import meta_get, meta_set, transaction

MAX_ERRORS = 50


class ImportError_(ValueError):
    def __init__(self, errors: list[dict], rows: int) -> None:
        super().__init__(f"{len(errors)} problem(s) in the file")
        self.errors = errors
        self.rows = rows


SPECS: dict[str, dict] = {
    "products": {
        "required": ["sku", "name", "category", "mrp", "shelf_life_days"],
        "optional": ["brand", "storage_class", "handling_cost", "storage_cost_per_day", "salvage_pct", "case_pack",
                     "supplier_name", "lead_time_days", "min_margin_pct", "max_daily_change_pct"],
        "example": [["MILK-1L", "Toned milk, 1 L", "Dairy", "68", "3", "Dairy Vale", "chilled", "3", "0.17", "-0.05", "12",
                     "Krishna Dairy", "1", "0.04", "0.10"],
                    ["RICE-5KG", "Basmati rice, 5 kg", "Staples", "699", "365", "Harvest Mill", "ambient", "9", "0.28",
                     "0.10", "4", "Shree Balaji Traders", "3", "0.05", "0.05"]],
        "label": "Product list",
        "help": "One row per product. MRP is the legal price ceiling; category policy (minimum margin, largest daily move) "
                "is taken from the first row of each category.",
    },
    "sales": {
        "required": ["sku", "date", "units_sold", "price"],
        "optional": ["closing_stock", "stocked_out", "promo"],
        "example": [["MILK-1L", "2026-10-27", "312", "66", "140", "0", "0"], ["RICE-5KG", "2026-10-27", "37", "649", "", "", "0"]],
        "label": "Daily sales",
        "help": "Units sold and the selling price per product per day. Give closing stock (or stocked_out = 1) on days "
                "you ran out, so the model knows sales were capped. Two years of history gives the best seasonality.",
    },
    "stock": {
        "required": ["sku", "qty_on_hand", "unit_cost"],
        "optional": ["received_on", "expires_on"],
        "example": [["MILK-1L", "180", "56.10", "2026-10-27", "2026-10-29"], ["RICE-5KG", "420", "520.00", "2026-10-01", ""]],
        "label": "Stock on hand",
        "help": "Current stock by batch with its purchase cost. Replaces all open stock for the products in the file. "
                "Expiry defaults to received date plus shelf life.",
    },
    "competitor": {
        "required": ["sku", "date", "price"],
        "optional": [],
        "example": [["MILK-1L", "2026-10-28", "66"], ["RICE-5KG", "2026-10-28", "659"]],
        "label": "Competitor prices",
        "help": "The main competitor's price per product per day, from a price feed or manual checks.",
    },
    "write_offs": {
        "required": ["sku", "date", "qty"],
        "optional": ["reason", "unit_cost"],
        "example": [["MILK-1L", "2026-10-27", "14", "expired", "55.80"]],
        "label": "Write-offs",
        "help": "Units thrown away (expired, spoiled or damaged). Used to estimate daily spoilage.",
    },
}


def template(kind: str) -> str:
    spec = SPECS[kind]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(spec["required"] + spec["optional"])
    for row in spec["example"]:
        w.writerow(row)
    return buf.getvalue()


# ---------------------------------------------------------------------- parsing helpers


class _Row:
    def __init__(self, data: dict[str, str], line: int, errors: list[dict]) -> None:
        self.d, self.line, self.errors = data, line, errors

    def _err(self, col: str, msg: str) -> None:
        if len(self.errors) < MAX_ERRORS:
            self.errors.append({"line": self.line, "column": col, "message": msg})

    def text(self, col: str, required: bool = True, max_len: int = 200) -> str | None:
        v = (self.d.get(col) or "").strip()
        if not v:
            if required:
                self._err(col, "is required")
            return None
        if len(v) > max_len:
            self._err(col, f"is longer than {max_len} characters")
            return None
        return v

    def num(self, col: str, required: bool = True, lo: float | None = None, hi: float | None = None,
            integer: bool = False, default: float | None = None) -> float | None:
        raw = (self.d.get(col) or "").strip().replace(",", "").replace("₹", "")
        if not raw:
            if required:
                self._err(col, "is required")
            return default
        try:
            v = float(raw)
        except ValueError:
            self._err(col, f"'{raw}' is not a number")
            return default
        if not math.isfinite(v):
            self._err(col, "must be a finite number")
            return default
        if integer and abs(v - round(v)) > 1e-9:
            self._err(col, "must be a whole number")
            return default
        if lo is not None and v < lo:
            self._err(col, f"must be at least {lo:g}")
            return default
        if hi is not None and v > hi:
            self._err(col, f"must be at most {hi:g}")
            return default
        return int(round(v)) if integer else v

    def day(self, col: str, required: bool = True) -> str | None:
        raw = (self.d.get(col) or "").strip()
        if not raw:
            if required:
                self._err(col, "is required")
            return None
        try:
            return date.fromisoformat(raw[:10]).isoformat()
        except ValueError:
            self._err(col, f"'{raw}' is not a date (use YYYY-MM-DD)")
            return None


def _read(kind: str, text: str) -> tuple[list[_Row], list[dict]]:
    spec = SPECS[kind]
    text = text.lstrip("﻿")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ImportError_([{"line": 1, "column": None, "message": "The file is empty."}], 0)
    headers = [h.strip().lower() for h in reader.fieldnames]
    missing = [c for c in spec["required"] if c not in headers]
    if missing:
        raise ImportError_([{"line": 1, "column": None,
                             "message": f"Missing column(s): {', '.join(missing)}. Download the template for the layout."}], 0)
    errors: list[dict] = []
    rows = []
    for i, raw in enumerate(reader, start=2):
        data = {k.strip().lower(): (v or "") for k, v in raw.items() if k is not None}
        if not any(v.strip() for v in data.values()):
            continue
        rows.append(_Row(data, i, errors))
    if not rows:
        raise ImportError_([{"line": 2, "column": None, "message": "The file has a header but no rows."}], 0)
    return rows, errors


def _known_skus(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT sku FROM products")}


def _check_sku(r: _Row, skus: set[str]) -> str | None:
    sku = r.text("sku", max_len=64)
    if sku is not None and sku not in skus:
        r._err("sku", f"'{sku}' is not in the product list; import products first")
        return None
    return sku


# ---------------------------------------------------------------------- importers


def _products(conn, rows: list[_Row], errors) -> int:
    cats: dict[str, tuple] = {}
    sups: dict[str, tuple] = {}
    prods = []
    existing_sup = {r["name"]: r["supplier_id"] for r in conn.execute("SELECT supplier_id, name FROM suppliers")}
    n_sup = conn.execute("SELECT COUNT(*) FROM suppliers").fetchone()[0]
    for r in rows:
        sku, name, cat = r.text("sku", max_len=64), r.text("name"), r.text("category", max_len=80)
        mrp = r.num("mrp", lo=0.01)
        shelf = r.num("shelf_life_days", lo=1, hi=3650, integer=True)
        storage = (r.text("storage_class", required=False) or "ambient").lower()
        if storage not in ("ambient", "chilled", "frozen"):
            r._err("storage_class", "must be ambient, chilled or frozen")
        handling = r.num("handling_cost", required=False, lo=0, default=0.0)
        storage_cost = r.num("storage_cost_per_day", required=False, lo=0, default=0.0)
        perishable = shelf is not None and shelf <= 30
        salvage = r.num("salvage_pct", required=False, lo=-5, hi=1, default=-0.05 if perishable else 0.1)
        case = r.num("case_pack", required=False, lo=1, integer=True, default=1)
        sup_name = r.text("supplier_name", required=False) or "Default supplier"
        lead = r.num("lead_time_days", required=False, lo=0, hi=365, integer=True, default=3)
        margin = r.num("min_margin_pct", required=False, lo=0, hi=5, default=0.05)
        move = r.num("max_daily_change_pct", required=False, lo=0.01, hi=1, default=0.10)
        if None in (sku, name, cat, mrp, shelf):
            continue
        if cat not in cats:
            cats[cat] = (cat, storage, int(perishable), margin, move)
        if sup_name not in sups:
            sid = existing_sup.get(sup_name)
            if sid is None:
                n_sup += 1
                sid = f"SUP-{n_sup:02d}"
            sups[sup_name] = (sid, sup_name, "", max(int(lead), 1), 500.0)
        prods.append((sku, name, cat, r.text("brand", required=False) or "", mrp, shelf, storage, handling,
                      storage_cost, salvage, sups[sup_name][0], case))
    if errors:
        return 0
    conn.executemany("INSERT INTO categories VALUES (?,?,?,?,?) ON CONFLICT(name) DO UPDATE SET "
                     "storage_class = excluded.storage_class, perishable = excluded.perishable, "
                     "min_margin_pct = excluded.min_margin_pct, max_daily_change_pct = excluded.max_daily_change_pct",
                     list(cats.values()))
    conn.executemany("INSERT INTO suppliers VALUES (?,?,?,?,?) ON CONFLICT(supplier_id) DO UPDATE SET "
                     "lead_time_days = excluded.lead_time_days", list(sups.values()))
    conn.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(sku) DO UPDATE SET "
                     "name = excluded.name, category = excluded.category, brand = excluded.brand, mrp = excluded.mrp, "
                     "shelf_life_days = excluded.shelf_life_days, storage_class = excluded.storage_class, "
                     "handling_cost = excluded.handling_cost, storage_cost_per_day = excluded.storage_cost_per_day, "
                     "salvage_pct = excluded.salvage_pct, supplier_id = excluded.supplier_id, case_pack = excluded.case_pack",
                     prods)
    return len(prods)


def _sales(conn, rows, errors) -> int:
    skus = _known_skus(conn)
    out_stock, out_price = [], []
    for r in rows:
        sku = _check_sku(r, skus)
        d = r.day("date")
        units = r.num("units_sold", lo=0, integer=True)
        price = r.num("price", lo=0.01)
        closing = r.num("closing_stock", required=False, lo=0, integer=True)
        flag = r.num("stocked_out", required=False, lo=0, hi=1, integer=True, default=0)
        promo = r.num("promo", required=False, lo=0, hi=1, integer=True, default=0)
        if None in (sku, d, units, price):
            continue
        so = int(bool(flag) or closing == 0)
        close = -1 if closing is None else int(closing)            # −1 = not reported
        opening = -1 if closing is None else int(closing) + int(units)
        out_stock.append((sku, d, opening, 0, int(units), 0, close, round(units * price, 2), 0.0, so))
        out_price.append((sku, d, price, int(promo), "import"))
    if errors:
        return 0
    conn.executemany("INSERT OR REPLACE INTO stock_daily VALUES (?,?,?,?,?,?,?,?,?,?)", out_stock)
    conn.executemany("INSERT OR REPLACE INTO price_book VALUES (?,?,?,?,?)", out_price)
    return len(out_stock)


def _stock(conn, rows, errors, append: bool = False) -> int:
    skus = _known_skus(conn)
    shelf = {r[0]: r[1] for r in conn.execute("SELECT sku, shelf_life_days FROM products")}
    today = date.today().isoformat()
    lots = []
    for r in rows:
        sku = _check_sku(r, skus)
        qty = r.num("qty_on_hand", lo=0, integer=True)
        cost = r.num("unit_cost", lo=0.01)
        rec = r.day("received_on", required=False) or today
        exp = r.day("expires_on", required=False)
        if None in (sku, qty, cost):
            continue
        if exp is None:
            exp = (date.fromisoformat(rec) + timedelta(days=int(shelf[sku]) - 1)).isoformat()
        if exp < rec:
            r._err("expires_on", "is before received_on")
            continue
        lots.append((sku, rec, exp, int(qty), cost))
    if errors:
        return 0
    touched = [] if append else sorted({l[0] for l in lots})  # append = add a batch, keep existing stock
    conn.executemany("UPDATE lots SET qty_on_hand = 0 WHERE sku = ? AND qty_on_hand > 0", [(s,) for s in touched])
    conn.executemany("INSERT INTO lots(sku, po_id, received_on, expires_on, qty_received, qty_on_hand, unit_cost) "
                     "VALUES (?, NULL, ?, ?, ?, ?, ?)", [(s, rc, ex, q, q, c) for s, rc, ex, q, c in lots])
    return len(lots)


def _competitor(conn, rows, errors) -> int:
    skus = _known_skus(conn)
    out = []
    for r in rows:
        sku, d, price = _check_sku(r, skus), r.day("date"), r.num("price", lo=0.01)
        if None not in (sku, d, price):
            out.append((sku, d, price))
    if errors:
        return 0
    conn.executemany("INSERT OR REPLACE INTO competitor_prices VALUES (?,?,?)", out)
    return len(out)


def _write_offs(conn, rows, errors) -> int:
    skus = _known_skus(conn)
    out = []
    for r in rows:
        sku, d, qty = _check_sku(r, skus), r.day("date"), r.num("qty", lo=1, integer=True)
        reason = (r.text("reason", required=False) or "spoiled").lower()
        if reason not in ("expired", "spoiled", "damaged"):
            r._err("reason", "must be expired, spoiled or damaged")
        cost = r.num("unit_cost", required=False, lo=0, default=0.0)
        if None not in (sku, d, qty):
            out.append((sku, 0, d, qty, "spoiled" if reason == "damaged" else reason, cost))
    if errors:
        return 0
    conn.executemany("INSERT INTO write_offs(sku, lot_id, date, qty, reason, unit_cost) VALUES (?,?,?,?,?,?)", out)
    return len(out)


IMPORTERS: dict[str, Callable] = {"products": _products, "sales": _sales, "stock": _stock,
                                  "competitor": _competitor, "write_offs": _write_offs}


def run_import(conn: sqlite3.Connection, kind: str, text: str, append: bool = False) -> dict:
    if kind not in IMPORTERS:
        raise KeyError(kind)
    rows, errors = _read(kind, text)
    with transaction(conn):
        n = IMPORTERS[kind](conn, rows, errors, append) if kind == "stock" else IMPORTERS[kind](conn, rows, errors)
        if errors:
            raise ImportError_(errors, len(rows))
        meta_set(conn, "data_version", str(int(meta_get(conn, "data_version", "0")) + 1))
        meta_set(conn, f"last_import_{kind}", date.today().isoformat())
    return {"kind": kind, "rows": len(rows), "written": n}
