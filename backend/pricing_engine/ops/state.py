"""Derives every optimiser input for a SKU from the store database, with a note of where each value came from.

This replaces the old input form. Each field of `ProductState` is a query over master data, purchase lots,
the price book, the competitor feed or write-off history as of the business date.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field
from datetime import date

from ..config import Settings
from ..demand.model import SkuDemand
from ..schemas import CompetitorResponse, ProductState


@dataclass(slots=True)
class DerivedInput:
    key: str
    label: str
    value: float | str
    unit: str
    source: str

    def as_dict(self) -> dict:
        return {"key": self.key, "label": self.label, "value": self.value, "unit": self.unit, "source": self.source}


@dataclass(slots=True)
class SkuContext:
    state: ProductState
    inputs: list[DerivedInput]
    name: str
    category: str
    supplier_id: str
    supplier_name: str
    lead_time: int
    order_cost: float
    case_pack: int
    shelf_life_new: int
    salvage_pct: float
    on_order: int
    usable_before_arrival: int
    fill_rate: float
    sold_28d: int
    lots: list[dict] = field(default_factory=list)


def _iso(o: int) -> str:
    return date.fromordinal(o).isoformat()


def derive(conn: sqlite3.Connection, sku: str, o: int, dm: SkuDemand, settings: Settings) -> SkuContext:
    d = _iso(o)
    p = conn.execute(
        """SELECT p.*, c.perishable, c.min_margin_pct, c.max_daily_change_pct, s.name AS supplier_name,
                  s.lead_time_days, s.order_cost
           FROM products p JOIN categories c ON c.name = p.category JOIN suppliers s ON s.supplier_id = p.supplier_id
           WHERE p.sku = ?""",
        (sku,),
    ).fetchone()
    if p is None:
        raise KeyError(sku)
    lots = [dict(r) for r in conn.execute(
        "SELECT lot_id, po_id, received_on, expires_on, qty_on_hand, unit_cost FROM lots "
        "WHERE sku = ? AND qty_on_hand > 0 ORDER BY expires_on, lot_id", (sku,))]
    qty = sum(l["qty_on_hand"] for l in lots)
    perishable = bool(p["perishable"]) or p["shelf_life_days"] <= 30

    if qty > 0:
        unit_cost = sum(l["qty_on_hand"] * l["unit_cost"] for l in lots) / qty
        age = sum(l["qty_on_hand"] * (o - date.fromisoformat(l["received_on"]).toordinal()) for l in lots) / qty
        remaining = sum(l["qty_on_hand"] * (date.fromisoformat(l["expires_on"]).toordinal() - o + 1) for l in lots) / qty
        cost_src = f"Quantity-weighted cost of {len(lots)} lot{'s' if len(lots) != 1 else ''} on hand"
    else:
        last = conn.execute("SELECT unit_cost FROM purchase_orders WHERE sku = ? ORDER BY ordered_on DESC LIMIT 1",
                            (sku,)).fetchone()
        unit_cost = float(last[0]) if last else 0.6 * p["mrp"]
        age, remaining = 0.0, float(p["shelf_life_days"])
        cost_src = "Last purchase order cost (no stock on hand)"

    price_row = conn.execute("SELECT price, date FROM price_book WHERE sku = ? AND date < ? ORDER BY date DESC LIMIT 1",
                             (sku, d)).fetchone()
    current = float(price_row["price"]) if price_row else float(p["mrp"])
    comp_row = conn.execute("SELECT price, date FROM competitor_prices WHERE sku = ? AND date <= ? ORDER BY date DESC LIMIT 1",
                            (sku, d)).fetchone()
    comp = float(comp_row["price"]) if comp_row else current

    since = _iso(o - 90)
    spoiled = conn.execute("SELECT COALESCE(SUM(qty), 0) FROM write_offs WHERE sku = ? AND reason = 'spoiled' AND date >= ?",
                           (sku, since)).fetchone()[0]
    exposure = conn.execute("SELECT COALESCE(SUM(CASE WHEN opening >= 0 THEN opening + received ELSE sold END), 0) "
                            "FROM stock_daily WHERE sku = ? AND date >= ?", (sku, since)).fetchone()[0]
    spoil = float(spoiled) / exposure if exposure else 0.0

    fill = conn.execute(
        """SELECT COALESCE(SUM(qty_received), 0), COALESCE(SUM(qty_ordered), 0) FROM purchase_orders
           WHERE supplier_id = ? AND status = 'received' AND received_on >= ?""",
        (p["supplier_id"], _iso(o - 60)),
    ).fetchone()
    fill_rate = float(fill[0]) / fill[1] if fill[1] else 1.0
    on_order = conn.execute("SELECT COALESCE(SUM(qty_ordered), 0) FROM purchase_orders WHERE sku = ? AND status = 'open'",
                            (sku,)).fetchone()[0]
    lead = int(p["lead_time_days"])
    usable = sum(l["qty_on_hand"] for l in lots if date.fromisoformat(l["expires_on"]).toordinal() > o + lead)
    sold_28 = conn.execute("SELECT COALESCE(SUM(sold), 0) FROM stock_daily WHERE sku = ? AND date >= ? AND date < ?",
                           (sku, _iso(o - 28), d)).fetchone()[0]

    reaction = dm.competitor_reaction
    response = (CompetitorResponse.STACKELBERG if reaction >= settings.reaction_threshold else CompetitorResponse.STATIC)
    shelf = max(1, int(math.ceil(remaining))) if perishable else int(p["shelf_life_days"])
    salvage = p["salvage_pct"] * unit_cost

    state = ProductState(
        sku=sku, unit_cost=round(unit_cost, 4), handling_cost_per_unit=float(p["handling_cost"]),
        storage_cost_per_unit_day=float(p["storage_cost_per_day"]), days_in_stock=int(round(age)),
        min_margin_pct=float(p["min_margin_pct"]), salvage_value_per_unit=round(salvage, 4),
        current_price=current, mrp=float(p["mrp"]), competitor_price=comp,
        max_price_change_pct=float(p["max_daily_change_pct"]), inventory=int(qty), shelf_life_days=min(shelf, 3650),
        perishable=perishable, daily_spoilage_rate=min(spoil, 0.5), competitor_response=response,
        replenishment_days=lead + 1,
    )

    lo_ci = dm.own_elasticity - 1.645 * dm.se[1]
    hi_ci = dm.own_elasticity + 1.645 * dm.se[1]
    inputs = [
        DerivedInput("unit_cost", "Unit cost", round(unit_cost, 2), "₹", cost_src),
        DerivedInput("handling_cost_per_unit", "Handling per unit", p["handling_cost"], "₹",
                     f"Product master: pick, pack and payment cost for {p['category'].lower()}"),
        DerivedInput("storage_cost_per_unit_day", "Storage per unit per day", p["storage_cost_per_day"], "₹",
                     f"Product master: {p['storage_class']} storage rate"),
        DerivedInput("days_in_stock", "Average age of stock", round(age, 1), "days",
                     "Quantity-weighted days since each lot was received"),
        DerivedInput("inventory", "Units on hand", int(qty), "units", "Sum of open lots after this morning's receipts"),
        DerivedInput("shelf_life_days", "Days left to sell" if perishable else "Planning horizon",
                     shelf if perishable else lead + 1, "days",
                     "Quantity-weighted days until lot expiry" if perishable
                     else f"Supplier lead time {lead} d + 1 review day; stock left after that is valued at cost"),
        DerivedInput("daily_spoilage_rate", "Daily spoilage", round(spoil * 100, 3), "%",
                     "Spoiled write-offs ÷ units held, last 90 days"),
        DerivedInput("current_price", "Yesterday's price", current, "₹",
                     f"Price book, {price_row['date'] if price_row else 'none'}"),
        DerivedInput("competitor_price", "Competitor price", comp, "₹",
                     f"Competitor feed, {comp_row['date'] if comp_row else 'none'}"),
        DerivedInput("mrp", "MRP (ceiling)", p["mrp"], "₹", "Printed maximum retail price, product master"),
        DerivedInput("min_margin_pct", "Minimum margin", round(p["min_margin_pct"] * 100, 1), "%",
                     f"Category policy, {p['category']}"),
        DerivedInput("max_price_change_pct", "Largest daily move", round(p["max_daily_change_pct"] * 100, 1), "%",
                     f"Category policy, {p['category']}"),
        DerivedInput("salvage_value_per_unit", "Value of an unsold unit", round(salvage, 2), "₹",
                     f"{p['salvage_pct'] * 100:+.0f}% of cost (" + ("disposal fee" if p["salvage_pct"] < 0
                                                                      else "liquidation") + ")"),
        DerivedInput("own_elasticity", "Price elasticity", round(dm.own_elasticity, 2), "",
                     f"Demand model, 90% interval {lo_ci:.2f} to {hi_ci:.2f}"),
        DerivedInput("cross_elasticity", "Competitor elasticity", round(dm.cross_elasticity, 2), "",
                     "Demand model, response to the competitor's price"),
        DerivedInput("competitor_response", "Competitor behaviour",
                     "Follows our price" if response == CompetitorResponse.STACKELBERG else "Holds its price", "",
                     f"Estimated long-run reaction slope {reaction:.2f} (threshold {settings.reaction_threshold:.2f})"),
    ]
    return SkuContext(
        state=state, inputs=inputs, name=p["name"], category=p["category"], supplier_id=p["supplier_id"],
        supplier_name=p["supplier_name"], lead_time=lead, order_cost=float(p["order_cost"]), case_pack=int(p["case_pack"]),
        shelf_life_new=int(p["shelf_life_days"]), salvage_pct=float(p["salvage_pct"]), on_order=int(on_order),
        usable_before_arrival=int(usable), fill_rate=fill_rate, sold_28d=int(sold_28), lots=lots,
    )
