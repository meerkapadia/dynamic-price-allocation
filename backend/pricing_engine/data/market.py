"""Market simulator: the ground truth that produces every sale, write-off and competitor quote in the database.

It is used to generate the demo store's two years of history (driven by a human-like pricing team and planner)
and by `python -m pricing_engine.demo next-day` to play out a further day at approved prices. The optimiser
never calls into this module; it only sees the rows this module writes.

Daily cycle for one store day d
    morning(d)   receive purchase orders due today (supplier fill rate applies) and read competitor quotes
    <prices>     set by the pricing team (history) or approved recommendations (demo next-day)
    close(d)     draw demand, sell first-expiry-first-out, spoil, expire, snapshot stock, write order lines
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache

import numpy as np

from .calendar import EVENT_KEYS, day_of_year, event_weights, weekday
from .catalog import (
    CATEGORY_SUPPLIER,
    CATEGORY_WEEKDAY,
    COMPETITOR_REACTION,
    PRODUCTS,
    SUPPLIERS,
    WEEKDAY,
    Spec,
    handling_cost,
    is_perishable,
    storage_cost_per_day,
)

INFLATION = 0.03           # annual supplier cost inflation
COST_REF = date(2026, 9, 1).toordinal()  # catalogue cost ratios describe supplier prices on this date
PROMO_LIFT = 0.08          # demand lift from the "offer" badge itself, on top of the price effect
DISPERSION = 10.0          # gamma shape of day-level demand shocks (negative binomial overdispersion)
COMP_RHO = 0.85
COMP_SIGMA = 0.025
CITIES = ("Mumbai", "Pune", "Bengaluru", "Hyderabad", "Ahmedabad", "Nashik", "Surat", "Thane")
CITY_W = np.array([0.24, 0.17, 0.18, 0.13, 0.09, 0.06, 0.06, 0.07])
HOUR_W = np.array([1, 0.5, 0.3, 0.2, 0.2, 0.5, 1.5, 3, 4.5, 5.5, 6, 6.2, 5.8, 5, 4.6, 4.4, 4.8, 5.6, 6.8, 7.6, 7.4,
                   6, 4, 2.2], dtype=float)
HOUR_W /= HOUR_W.sum()
EPOCH = date(2024, 1, 1).toordinal()


@lru_cache(maxsize=8192)
def cal_row(o: int) -> tuple[np.ndarray, int, int]:
    """Cached (event weights, day of year, weekday) for one ordinal date."""
    d = np.array([o - 719163], dtype="datetime64[D]")
    return event_weights(d)[0], int(day_of_year(d)[0]), int(weekday(d)[0])


def _circ(doy: np.ndarray, center: int) -> np.ndarray:
    d = np.abs(doy - center)
    return np.minimum(d, 365.25 - d)


def bumps(doy: np.ndarray, spec_bumps) -> np.ndarray:
    out = np.zeros(np.shape(doy), dtype=float)
    for center, width, amp in spec_bumps:
        out += amp * np.exp(-0.5 * (_circ(np.asarray(doy, float), center) / width) ** 2)
    return out


def retail_round(p: float, mrp: float) -> float:
    """How a person prices: whole rupees below ₹300, '…9' endings above. Never above MRP."""
    if p < 300:
        r = float(round(p))
    else:
        r = float(round(p / 10.0) * 10 - 1)
    return float(min(max(r, 1.0), math.floor(mrp)))


@dataclass(frozen=True, slots=True)
class Truth:
    sku: str
    intercept: float
    beta: float
    gamma: float
    usual_price: float
    comp_usual_price: float
    comp_reaction: float
    weekday: tuple[float, ...]
    season: tuple[tuple[int, int, float], ...]
    events: dict[str, float]
    trend: float
    spoil: float
    cost_ratio: float
    cost_season: tuple[tuple[int, int, float], ...]
    mrp: float

    def log_demand(self, ords: np.ndarray, price, comp, promo) -> np.ndarray:
        dates = np.asarray(ords, dtype=np.int64) - 719163  # ordinal → days since 1970-01-01
        dts = dates.astype("datetime64[D]")
        ev = event_weights(dts)
        ev_eff = sum(self.events.get(k, 0.0) * ev[:, j] for j, k in enumerate(EVENT_KEYS))
        years = (np.asarray(ords) - EPOCH) / 365.25
        return (
            self.intercept
            + self.beta * np.log(np.asarray(price, float) / self.usual_price)
            + self.gamma * np.log(np.asarray(comp, float) / self.comp_usual_price)
            + bumps(day_of_year(dts), self.season)
            + np.asarray(self.weekday)[weekday(dts)]
            + ev_eff
            + math.log1p(self.trend) * years
            + PROMO_LIFT * np.asarray(promo, float)
        )

    def log_demand_at(self, o: int, price: float, comp: float, promo: int) -> float:
        ev, doy, dow = cal_row(o)
        ev_eff = sum(self.events.get(k, 0.0) * ev[j] for j, k in enumerate(EVENT_KEYS))
        return float(
            self.intercept + self.beta * math.log(price / self.usual_price)
            + self.gamma * math.log(comp / self.comp_usual_price)
            + float(bumps(np.array([doy]), self.season)[0]) + self.weekday[dow] + ev_eff
            + math.log1p(self.trend) * (o - EPOCH) / 365.25 + PROMO_LIFT * promo
        )

    def to_json(self) -> str:
        return json.dumps({k: getattr(self, k) for k in self.__slots__})

    @staticmethod
    def from_json(s: str) -> "Truth":
        d = json.loads(s)
        d["weekday"] = tuple(d["weekday"])
        d["season"] = tuple(tuple(x) for x in d["season"])
        d["cost_season"] = tuple(tuple(x) for x in d["cost_season"])
        return Truth(**d)


def build_truth(spec: Spec, rng: np.random.Generator, demand_scale: float) -> Truth:
    usual = spec.usual * spec.mrp
    full_cost = spec.cost_ratio * spec.mrp + handling_cost(spec) + storage_cost_per_day(spec) * min(spec.shelf, 30) / 2
    p_star = max(spec.opt * usual, 1.08 * full_cost)
    # choose β so that the profit optimum *with the rival following our price* sits at p_star:
    # effective elasticity β + γ·κ = −p*/(p* − c)
    reaction = COMPETITOR_REACTION[spec.category]
    beta = float(np.clip(-p_star / (p_star - full_cost) - spec.gamma * reaction, -8.5, -1.3))
    return Truth(
        sku=spec.sku,
        intercept=math.log(spec.base * demand_scale),
        beta=round(beta, 4),
        gamma=spec.gamma,
        usual_price=usual,
        comp_usual_price=usual * float(rng.uniform(0.96, 1.05)),
        comp_reaction=COMPETITOR_REACTION[spec.category],
        weekday=WEEKDAY[CATEGORY_WEEKDAY[spec.category]],
        season=spec.season,
        events=dict(spec.events),
        trend=spec.trend,
        spoil=spec.spoil,
        cost_ratio=spec.cost_ratio,
        cost_season=spec.cost_season,
        mrp=spec.mrp,
    )


def supplier_fill_rate(supplier_id: str, ordinal: int, seed: int) -> float:
    """Deterministic supply-shortage episodes per supplier (about 4% of days run at 50–80% fill)."""
    idx = int(supplier_id.split("-")[1])
    week = ordinal // 7
    r = np.random.default_rng([seed, 7919, idx, week])
    if r.random() < 0.06:
        return float(r.uniform(0.5, 0.8))
    return float(np.random.default_rng([seed, 104729, idx, ordinal]).uniform(0.96, 1.0))


# ====================================================================== mutable store state


@dataclass(slots=True)
class Lot:
    lot_id: int
    sku: str
    po_id: str | None
    received: int
    expires: int
    qty_received: int
    qty: int
    cost: float


@dataclass(slots=True)
class PO:
    po_id: str
    sku: str
    supplier_id: str
    ordered: int
    expected: int
    qty: int
    cost: float
    raised_by: str
    received: int | None = None
    qty_received: int | None = None


@dataclass(slots=True)
class SkuState:
    spec: Spec
    truth: Truth
    supplier_id: str
    lead_time: int
    lots: list[Lot] = field(default_factory=list)
    pos: list[PO] = field(default_factory=list)
    last_price: float = 0.0
    last_comp: float = 0.0
    comp_dev: float = 0.0       # AR(1) state of log competitor price around its target

    @property
    def on_hand(self) -> int:
        return sum(l.qty for l in self.lots)

    @property
    def on_order(self) -> int:
        return sum(p.qty for p in self.pos if p.received is None)


@dataclass(slots=True)
class Sink:
    """Rows produced by the simulator, flushed to SQLite by the caller."""

    sales: list[tuple] = field(default_factory=list)
    write_offs: list[tuple] = field(default_factory=list)
    price_book: list[tuple] = field(default_factory=list)
    competitor: list[tuple] = field(default_factory=list)
    stock: list[tuple] = field(default_factory=list)
    lots: dict[int, Lot] = field(default_factory=dict)
    pos: dict[str, PO] = field(default_factory=dict)


@dataclass(slots=True)
class DayResult:
    revenue: float = 0.0
    cogs: float = 0.0
    units: int = 0
    written_off: int = 0
    by_sku: dict[str, tuple[int, int, bool]] = field(default_factory=dict)  # sold, demand, stocked out


def iso(o: int) -> str:
    return date.fromordinal(o).isoformat()


class Market:
    def __init__(self, truths: dict[str, Truth], seed: int) -> None:
        self.seed = seed
        self.truths = truths
        self.lead = {s[0]: s[3] for s in SUPPLIERS}
        self.next_lot = 1
        self.po_seq: dict[int, int] = {}
        self._sku_index: dict[str, int] = {}

    # ------------------------------------------------------------------ helpers
    def _rng(self, stream: int, o: int, sku: str) -> np.random.Generator:
        idx = self._sku_index.setdefault(sku, int.from_bytes(sku.encode()[:8].ljust(8, b"_"), "little") % 2**31)
        return np.random.default_rng([self.seed, stream, o, idx])

    def new_state(self, spec: Spec) -> SkuState:
        sup = CATEGORY_SUPPLIER[spec.category]
        return SkuState(spec=spec, truth=self.truths[spec.sku], supplier_id=sup, lead_time=self.lead[sup])

    def unit_cost(self, st: SkuState, o: int, rng: np.random.Generator) -> float:
        t = st.truth
        years = (o - COST_REF) / 365.25
        doy = date.fromordinal(o).timetuple().tm_yday
        c = t.mrp * t.cost_ratio * (1 + INFLATION) ** years * math.exp(float(bumps(np.array([doy]), t.cost_season)[0]))
        return round(c * float(np.exp(rng.normal(0, 0.02))), 2)

    def competitor_target(self, st: SkuState, o: int) -> float:
        ev = cal_row(o)[0]
        sale = max(ev[EVENT_KEYS.index("festive_sale")], ev[EVENT_KEYS.index("independence_sale")])
        depth = 0.12 if st.spec.category == "Electronics" else 0.03
        fest = max((st.truth.events.get(k, 0.0) * ev[j] for j, k in enumerate(EVENT_KEYS)), default=0.0)
        return st.truth.comp_usual_price * math.exp(-depth * sale - 0.04 * min(fest, 1.0))

    def raise_po(self, sink: Sink, st: SkuState, o: int, qty: int, cost: float, raised_by: str) -> PO:
        n = self.po_seq.get(o, 0) + 1
        self.po_seq[o] = n
        po = PO(po_id=f"PO-{date.fromordinal(o):%y%m%d}-{n:04d}", sku=st.spec.sku, supplier_id=st.supplier_id,
                ordered=o, expected=o + st.lead_time, qty=int(qty), cost=cost, raised_by=raised_by)
        st.pos.append(po)
        sink.pos[po.po_id] = po
        return po

    def add_lot(self, sink: Sink, st: SkuState, o: int, qty: int, cost: float, po_id: str | None, age: int = 0) -> Lot:
        lot = Lot(lot_id=self.next_lot, sku=st.spec.sku, po_id=po_id, received=o - age,
                  expires=o - age + st.spec.shelf - 1, qty_received=qty, qty=qty, cost=cost)
        self.next_lot += 1
        st.lots.append(lot)
        sink.lots[lot.lot_id] = lot
        return lot

    # ------------------------------------------------------------------ morning
    def morning(self, sink: Sink, states: dict[str, SkuState], o: int) -> dict[str, int]:
        received: dict[str, int] = {}
        for sku, st in states.items():
            rng = self._rng(31, o, sku)
            eps = float(rng.normal(0, COMP_SIGMA))
            got = 0
            for po in st.pos:
                if po.received is None and po.expected <= o:
                    fill = supplier_fill_rate(st.supplier_id, o, self.seed)
                    q = int(math.floor(po.qty * fill)) if fill < 0.999 else po.qty
                    po.received, po.qty_received = o, q
                    sink.pos[po.po_id] = po
                    if q > 0:
                        age = int(rng.integers(0, 2)) if is_perishable(st.spec) and st.spec.shelf > 4 else 0
                        self.add_lot(sink, st, o, q, po.cost, po.po_id, age=age)
                    got += q
            st.pos = [p for p in st.pos if p.received is None]
            received[sku] = got
            # competitor quote for today, reacting (with a lag) to our recent price
            target = self.competitor_target(st, o)
            react = st.truth.comp_reaction * math.log(st.last_price / st.truth.usual_price) if st.last_price else 0.0
            st.comp_dev = COMP_RHO * st.comp_dev + (1 - COMP_RHO) * react + eps
            st.last_comp = retail_round(target * math.exp(st.comp_dev), st.spec.mrp * 1.0)
            sink.competitor.append((sku, iso(o), st.last_comp))
        return received

    # ------------------------------------------------------------------ close of day
    def close(self, sink: Sink, states: dict[str, SkuState], o: int, prices: dict[str, float],
              promos: dict[str, int], received: dict[str, int], set_by: str) -> DayResult:
        lines_rng = np.random.default_rng([self.seed, 98, o])
        res = DayResult()
        day_lines: list[tuple[str, int, float]] = []
        for sku, st in states.items():
            rng = self._rng(97, o, sku)  # one stream per SKU and day: every policy faces the same shoppers
            p = float(prices[sku])
            promo = int(promos.get(sku, 0))
            st.last_price = p
            sink.price_book.append((sku, iso(o), p, promo, set_by))
            opening = st.on_hand - received.get(sku, 0)
            lam = math.exp(st.truth.log_demand_at(o, p, st.last_comp, promo))
            demand = int(rng.poisson(lam * rng.gamma(DISPERSION, 1.0 / DISPERSION)))
            avail = st.on_hand
            sold = min(demand, avail)
            left, cogs = sold, 0.0
            for lot in sorted(st.lots, key=lambda l: (l.expires, l.lot_id)):
                if left == 0:
                    break
                take = min(lot.qty, left)
                if take:
                    lot.qty -= take
                    left -= take
                    cogs += take * lot.cost
                    sink.lots[lot.lot_id] = lot
            wo = 0
            for lot in st.lots:
                if lot.qty <= 0:
                    continue
                if st.truth.spoil > 0:
                    k = int(rng.binomial(lot.qty, st.truth.spoil))
                    if k:
                        lot.qty -= k
                        wo += k
                        sink.write_offs.append((sku, lot.lot_id, iso(o), k, "spoiled", lot.cost))
                        sink.lots[lot.lot_id] = lot
                if lot.expires <= o and lot.qty > 0:
                    sink.write_offs.append((sku, lot.lot_id, iso(o), lot.qty, "expired", lot.cost))
                    wo += lot.qty
                    lot.qty = 0
                    sink.lots[lot.lot_id] = lot
            st.lots = [l for l in st.lots if l.qty > 0]
            revenue = round(p * sold, 2)
            sink.stock.append((sku, iso(o), opening, received.get(sku, 0), sold, wo, st.on_hand, revenue,
                               round(cogs, 2), int(demand > avail)))
            res.revenue += revenue
            res.cogs += cogs
            res.units += sold
            res.written_off += wo
            res.by_sku[sku] = (sold, demand, demand > avail)
            if sold:
                bulky = st.spec.category in ("Electronics", "Home & seasonal")
                probs = (0.95, 0.05, 0, 0) if bulky else (0.6, 0.26, 0.09, 0.05)
                qs = lines_rng.choice(np.array([1, 2, 3, 4]), size=sold, p=probs)
                cs = np.cumsum(qs)
                n = int(np.searchsorted(cs, sold)) + 1
                qs = qs[:n].copy()
                qs[-1] -= int(cs[n - 1] - sold)
                day_lines.extend((sku, int(q), p) for q in qs if q > 0)
        self._order_lines(sink, o, day_lines, lines_rng)
        return res

    def _order_lines(self, sink: Sink, o: int, lines: list[tuple[str, int, float]], rng: np.random.Generator) -> None:
        n = len(lines)
        if not n:
            return
        perm = rng.permutation(n)
        sizes = rng.geometric(0.42, size=n)
        ends = np.cumsum(sizes)
        n_orders = int(np.searchsorted(ends, n)) + 1
        sizes = sizes[:n_orders]
        order_of = np.repeat(np.arange(n_orders), sizes)[:n]
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        line_no = np.arange(n) - starts[order_of] + 1
        hours = rng.choice(24, size=n_orders, p=HOUR_W)
        mins = rng.integers(0, 60, size=n_orders)
        secs = rng.integers(0, 60, size=n_orders)
        cities = rng.choice(len(CITIES), size=n_orders, p=CITY_W)
        app = rng.random(n_orders) < 0.68
        d = date.fromordinal(o)
        ds, prefix = d.isoformat(), f"BL{d:%y%m%d}"
        oid = [f"{prefix}{k + 1:05d}" for k in range(n_orders)]
        ts = [f"{ds} {h:02d}:{m:02d}:{s:02d}" for h, m, s in zip(hours.tolist(), mins.tolist(), secs.tolist())]
        city = [CITIES[c] for c in cities.tolist()]
        chan = ["App" if a else "Web" for a in app.tolist()]
        for pos, (j, k, ln) in enumerate(zip(perm.tolist(), order_of.tolist(), line_no.tolist())):
            sku, q, p = lines[j]
            sink.sales.append((oid[k], ln, sku, ds, ts[k], q, p, city[k], chan[k]))
