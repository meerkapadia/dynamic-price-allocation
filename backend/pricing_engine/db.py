"""SQLite store: schema and connection helpers.

One file holds the whole store: master data, sales/stock/competitor history, pricing runs, recommendations and
approvals. WAL mode lets the API read while a pricing run writes. The schema is plain SQL and ports to PostgreSQL.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from collections.abc import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS categories (
    name TEXT PRIMARY KEY,
    storage_class TEXT NOT NULL,
    perishable INTEGER NOT NULL,
    min_margin_pct REAL NOT NULL,         -- pricing policy: floor margin over full cost
    max_daily_change_pct REAL NOT NULL    -- pricing policy: largest price move per day
);
CREATE TABLE IF NOT EXISTS suppliers (
    supplier_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    city TEXT NOT NULL,
    lead_time_days INTEGER NOT NULL,
    order_cost REAL NOT NULL              -- fixed cost per purchase order (freight + admin)
);
CREATE TABLE IF NOT EXISTS products (
    sku TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL REFERENCES categories(name),
    brand TEXT NOT NULL,
    mrp REAL NOT NULL,                    -- printed maximum retail price (legal ceiling)
    shelf_life_days INTEGER NOT NULL,
    storage_class TEXT NOT NULL,
    handling_cost REAL NOT NULL,          -- pick, pack and payment cost per unit sold
    storage_cost_per_day REAL NOT NULL,   -- per unit per day
    salvage_pct REAL NOT NULL,            -- value of an expired/unsold unit as a share of cost (negative = disposal fee)
    supplier_id TEXT NOT NULL REFERENCES suppliers(supplier_id),
    case_pack INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS market_truth (  -- simulator only; never read by the estimator or optimiser
    sku TEXT PRIMARY KEY,
    params TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS calendar_events (
    event TEXT NOT NULL,
    label TEXT NOT NULL,
    starts_on TEXT NOT NULL,
    peak_on TEXT NOT NULL,
    ends_on TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS purchase_orders (
    po_id TEXT PRIMARY KEY,
    sku TEXT NOT NULL,
    supplier_id TEXT NOT NULL,
    ordered_on TEXT NOT NULL,
    expected_on TEXT NOT NULL,
    received_on TEXT,
    qty_ordered INTEGER NOT NULL,
    qty_received INTEGER,
    unit_cost REAL NOT NULL,
    status TEXT NOT NULL,                 -- open | received
    raised_by TEXT NOT NULL               -- planner | buyer
);
CREATE INDEX IF NOT EXISTS ix_po_sku ON purchase_orders(sku, ordered_on);
CREATE INDEX IF NOT EXISTS ix_po_open ON purchase_orders(status, expected_on);
CREATE TABLE IF NOT EXISTS lots (
    lot_id INTEGER PRIMARY KEY,
    sku TEXT NOT NULL,
    po_id TEXT,
    received_on TEXT NOT NULL,
    expires_on TEXT NOT NULL,             -- last sellable day
    qty_received INTEGER NOT NULL,
    qty_on_hand INTEGER NOT NULL,
    unit_cost REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_lots_sku ON lots(sku, qty_on_hand);
CREATE TABLE IF NOT EXISTS sales (
    order_id TEXT NOT NULL,
    line INTEGER NOT NULL,
    sku TEXT NOT NULL,
    sale_date TEXT NOT NULL,
    sold_at TEXT NOT NULL,
    qty INTEGER NOT NULL,
    unit_price REAL NOT NULL,
    city TEXT NOT NULL,
    channel TEXT NOT NULL,
    PRIMARY KEY (order_id, line)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS ix_sales_date ON sales(sale_date);
CREATE INDEX IF NOT EXISTS ix_sales_sku_date ON sales(sku, sale_date);
CREATE TABLE IF NOT EXISTS write_offs (
    id INTEGER PRIMARY KEY,
    sku TEXT NOT NULL,
    lot_id INTEGER NOT NULL,
    date TEXT NOT NULL,
    qty INTEGER NOT NULL,
    reason TEXT NOT NULL,                 -- expired | spoiled
    unit_cost REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_wo_sku ON write_offs(sku, date);
CREATE TABLE IF NOT EXISTS price_book (
    sku TEXT NOT NULL,
    date TEXT NOT NULL,
    price REAL NOT NULL,
    promo INTEGER NOT NULL,
    set_by TEXT NOT NULL,                 -- pricing team | import | <approver name>
    PRIMARY KEY (sku, date)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS competitor_prices (
    sku TEXT NOT NULL,
    date TEXT NOT NULL,
    price REAL NOT NULL,
    PRIMARY KEY (sku, date)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS stock_daily (
    sku TEXT NOT NULL,
    date TEXT NOT NULL,
    opening INTEGER NOT NULL,
    received INTEGER NOT NULL,
    sold INTEGER NOT NULL,
    written_off INTEGER NOT NULL,
    closing INTEGER NOT NULL,
    revenue REAL NOT NULL,
    cogs REAL NOT NULL,
    stocked_out INTEGER NOT NULL,         -- 1 when demand exceeded stock (sales are a censored observation)
    PRIMARY KEY (sku, date)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS demand_models (
    sku TEXT PRIMARY KEY,
    fitted_on TEXT NOT NULL,
    n_obs INTEGER NOT NULL,
    n_censored INTEGER NOT NULL,
    coef TEXT NOT NULL,
    se TEXT NOT NULL,
    holdout_wape REAL NOT NULL,
    dispersion REAL NOT NULL,
    newton_iters INTEGER NOT NULL,
    competitor_reaction REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS pricing_runs (
    run_id INTEGER PRIMARY KEY,
    price_date TEXT NOT NULL,             -- the day the recommended prices are for
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,                 -- running | done | failed
    products INTEGER NOT NULL DEFAULT 0,
    priced INTEGER NOT NULL DEFAULT 0,
    errors TEXT NOT NULL DEFAULT '[]',
    model_fitted_on TEXT,
    requested_by TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS recommendations (
    run_id INTEGER NOT NULL,
    sku TEXT NOT NULL,
    price_date TEXT NOT NULL,
    current_price REAL NOT NULL,
    recommended_price REAL NOT NULL,
    floor REAL NOT NULL,
    ceiling REAL NOT NULL,
    competitor_price REAL NOT NULL,
    action TEXT NOT NULL,                 -- increase | decrease | hold
    change_pct REAL NOT NULL,
    profit_gain REAL NOT NULL,
    binding TEXT NOT NULL,
    horizon_days INTEGER NOT NULL,
    payload TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',   -- pending | approved | overridden | rejected
    final_price REAL,
    decided_by TEXT,
    decided_at TEXT,
    note TEXT,
    PRIMARY KEY (run_id, sku)
);
CREATE INDEX IF NOT EXISTS ix_rec_sku ON recommendations(sku, run_id);
CREATE TABLE IF NOT EXISTS store_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def connect(path: Path | str, readonly: bool = False) -> sqlite3.Connection:
    path = Path(path)
    if readonly:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30, check_same_thread=False)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, timeout=30, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=OFF")
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    conn.execute("BEGIN")
    try:
        yield conn
    except BaseException:
        conn.rollback()
        raise
    else:
        conn.commit()


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def meta_get(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM store_meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else default


def meta_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT INTO store_meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                 (key, value))
