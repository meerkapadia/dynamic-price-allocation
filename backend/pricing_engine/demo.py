"""Demo store tools (sample data only; a real store imports its own CSV files instead).

    python -m pricing_engine.demo build              build the two-year demo store if it is missing
    python -m pricing_engine.demo next-day [--days N] trade the next day(s) at the prices in the price book

`next-day` lets you try the full loop on the demo store: run pricing, approve prices, play out the day with the
simulated shoppers (sales, competitor moves, deliveries, expiry), then run pricing again on the new data.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date

import numpy as np

from .config import get_settings
from .data.generate import Planner, PricingTeam, _TeamMemory, generate_store
from .data.market import Sink
from .data.store import flush, load_market
from .db import connect, meta_get, meta_set, transaction

log = logging.getLogger(__name__)


def next_day(days: int = 1) -> list[dict]:
    s = get_settings()
    conn = connect(s.db_path)
    out = []
    try:
        if conn.execute("SELECT COUNT(*) FROM market_truth").fetchone()[0] == 0:
            raise SystemExit("This database holds imported store data; next-day only works on the demo store.")
        last = conn.execute("SELECT MAX(date) FROM stock_daily").fetchone()[0]
        o0 = date.fromisoformat(last).toordinal() + 1
        rng = np.random.default_rng([s.random_seed, 777, o0])
        team, planner = PricingTeam(rng), Planner()
        hist: dict[str, list] = {}
        for r in conn.execute("SELECT sku, sold, stocked_out FROM stock_daily WHERE date >= ? ORDER BY date",
                              (date.fromordinal(o0 - 14).isoformat(),)):
            hist.setdefault(r["sku"], []).append((r["sold"], bool(r["stocked_out"])))
        for sku, rows in hist.items():
            team.mem[sku] = _TeamMemory(next_review=10**9)
            for sold, so in rows:
                team.observe(sku, sold, so)
        for o in range(o0, o0 + days):
            d = date.fromordinal(o).isoformat()
            market, states = load_market(conn, s.random_seed, o)
            sink = Sink()
            with transaction(conn):
                received = market.morning(sink, states, o)
                prices = {}
                for sku in states:
                    row = conn.execute("SELECT price FROM price_book WHERE sku = ? AND date <= ? ORDER BY date DESC LIMIT 1",
                                       (sku, d)).fetchone()
                    prices[sku] = float(row[0]) if row else states[sku].truth.usual_price
                approved = {r[0]: r[1] for r in conn.execute("SELECT sku, set_by FROM price_book WHERE date = ?", (d,))}
                res = market.close(sink, states, o, prices, {}, received, "store")
                # keep the approver's name on prices that were approved for this day
                sink.price_book = [(k, dd, p, pr, approved.get(k, "carried over")) for k, dd, p, pr, _ in sink.price_book]
                for sku, (sold, _dem, so) in res.by_sku.items():
                    team.observe(sku, sold, so)
                planner.review(market, sink, states, team, o, rng)
                flush(conn, sink)
                meta_set(conn, "data_version", str(int(meta_get(conn, "data_version", "0")) + 1))
            out.append({"date": d, "revenue": res.revenue, "units": res.units, "written_off": res.written_off})
            log.info("traded %s: %d units, revenue ₹%.0f", d, res.units, res.revenue)
    finally:
        conn.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Demo store tools")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    nd = sub.add_parser("next-day")
    nd.add_argument("--days", type=int, default=1)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.cmd == "build":
        print(generate_store(get_settings()))
    else:
        for r in next_day(args.days):
            print(f"{r['date']}: {r['units']} units, revenue ₹{r['revenue']:,.0f}, {r['written_off']} written off")


if __name__ == "__main__":
    main()
