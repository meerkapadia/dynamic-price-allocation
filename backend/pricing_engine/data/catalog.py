"""Store master data (categories, suppliers, products) plus the hidden ground-truth market used by the simulator.

The store is a fictional online grocery and general-merchandise retailer shipping across west and south India.
Brand and supplier names are invented. Ground-truth parameters live in `market_truth` and are only read by the
market simulator; the estimator must recover them from sales history alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

STORE_NAME = "Bazaarline"

# name, storage, perishable, min margin, max daily change
CATEGORIES: tuple[tuple[str, str, int, float, float], ...] = (
    ("Fruits & vegetables", "chilled", 1, 0.04, 0.20),
    ("Dairy & bakery", "chilled", 1, 0.04, 0.10),
    ("Staples", "ambient", 0, 0.05, 0.05),
    ("Beverages", "ambient", 0, 0.05, 0.08),
    ("Sweets & dry fruits", "ambient", 0, 0.08, 0.10),
    ("Personal care", "ambient", 0, 0.10, 0.08),
    ("Home & seasonal", "ambient", 0, 0.10, 0.10),
    ("Electronics", "ambient", 0, 0.06, 0.08),
    ("Stationery", "ambient", 0, 0.10, 0.08),
)

# id, name, city, lead time, order cost (₹ per PO)
SUPPLIERS: tuple[tuple[str, str, str, int, float], ...] = (
    ("SUP-01", "Nashik Farm Collective", "Nashik", 1, 350.0),
    ("SUP-02", "Krishna Dairy Distributors", "Pune", 1, 300.0),
    ("SUP-03", "Shree Balaji Traders", "Navi Mumbai", 3, 650.0),
    ("SUP-04", "Western Beverages Pvt Ltd", "Thane", 2, 500.0),
    ("SUP-05", "Mehta Sweets & Dry Fruits", "Ahmedabad", 2, 450.0),
    ("SUP-06", "Unicare Distributors", "Bhiwandi", 4, 500.0),
    ("SUP-07", "HomeLine Wholesale", "Surat", 6, 900.0),
    ("SUP-08", "Techport Imports", "Chennai", 10, 1600.0),
    ("SUP-09", "Saraswati Stationers", "Indore", 5, 400.0),
)

CATEGORY_SUPPLIER = {
    "Fruits & vegetables": "SUP-01", "Dairy & bakery": "SUP-02", "Staples": "SUP-03", "Beverages": "SUP-04",
    "Sweets & dry fruits": "SUP-05", "Personal care": "SUP-06", "Home & seasonal": "SUP-07",
    "Electronics": "SUP-08", "Stationery": "SUP-09",
}
CATEGORY_HANDLING = {  # ₹ per unit: pick, pack, payment gateway
    "Fruits & vegetables": 4.0, "Dairy & bakery": 3.0, "Staples": 9.0, "Beverages": 6.0, "Sweets & dry fruits": 12.0,
    "Personal care": 7.0, "Home & seasonal": 28.0, "Electronics": 35.0, "Stationery": 6.0,
}
STORAGE_RATE = {"ambient": 0.0004, "chilled": 0.0025, "frozen": 0.0035}  # share of MRP per unit-day

# log demand multiplier by weekday, Monday first
WEEKDAY = {
    "grocery": (-0.06, -0.08, -0.04, 0.0, 0.05, 0.14, 0.11),
    "general": (-0.10, -0.07, -0.05, -0.02, 0.04, 0.12, 0.16),
}
CATEGORY_WEEKDAY = {c[0]: ("grocery" if c[0] in {"Fruits & vegetables", "Dairy & bakery", "Staples", "Beverages"}
                           else "general") for c in CATEGORIES}
COMPETITOR_REACTION = {  # how strongly the main rival follows our price (log-log slope, one-day lag)
    "Fruits & vegetables": 0.0, "Dairy & bakery": 0.05, "Staples": 0.35, "Beverages": 0.3, "Sweets & dry fruits": 0.2,
    "Personal care": 0.25, "Home & seasonal": 0.15, "Electronics": 0.5, "Stationery": 0.1,
}


@dataclass(frozen=True, slots=True)
class Spec:
    sku: str
    name: str
    category: str
    brand: str
    mrp: float
    cost_ratio: float           # supplier cost / MRP
    shelf: int                  # shelf life in days from receipt
    base: float                 # units/day at the usual price on an ordinary day
    usual: float                # usual selling price / MRP (pricing team's anchor)
    opt: float                  # where the true profit optimum sits relative to the usual price
    gamma: float                # cross-price elasticity vs the main competitor
    season: tuple[tuple[int, int, float], ...] = ()   # (peak day-of-year, width days, log amplitude)
    events: dict[str, float] = field(default_factory=dict)
    trend: float = 0.05         # annual demand growth
    spoil: float = 0.0          # daily spoilage / damage rate
    storage: str = ""
    case: int = 1
    cost_season: tuple[tuple[int, int, float], ...] = ()


S = Spec
PRODUCTS: tuple[Spec, ...] = (
    # ---------------------------------------------------------------- fruits & vegetables
    S("FV-MANGO-ALP", "Alphonso mangoes, 1 dozen", "Fruits & vegetables", "FarmDay", 1199, 0.60, 6, 7, 0.83, 1.04, 1.4,
      season=((118, 24, 2.1), (300, 70, -0.6)), trend=0.10, spoil=0.04, case=4, cost_season=((118, 30, -0.35),)),
    S("FV-STRAW-200", "Strawberries, 200 g", "Fruits & vegetables", "FarmDay", 149, 0.58, 4, 24, 0.90, 0.97, 1.3,
      season=((20, 30, 1.0), (190, 50, -0.8)), events={"year_end": 0.4}, spoil=0.06, case=10),
    S("FV-TOMATO-1K", "Tomatoes, 1 kg", "Fruits & vegetables", "FarmDay", 58, 0.66, 5, 120, 0.86, 0.98, 1.8,
      season=((200, 30, -0.15),), spoil=0.04, case=20, cost_season=((205, 28, 0.55),)),
    S("FV-BANANA-6", "Robusta bananas, 6 pcs", "Fruits & vegetables", "FarmDay", 60, 0.58, 4, 95, 0.88, 1.03, 1.5,
      events={"diwali": 0.15}, spoil=0.05, case=12),
    S("FV-SPINACH-250", "Spinach, 250 g", "Fruits & vegetables", "FarmDay", 38, 0.52, 3, 48, 0.90, 1.00, 1.2,
      season=((15, 40, 0.5), (160, 30, -0.4)), spoil=0.07, case=20),
    S("FV-MELON-1", "Watermelon, 1 pc", "Fruits & vegetables", "FarmDay", 129, 0.52, 7, 18, 0.86, 1.05, 1.3,
      season=((112, 35, 1.4), (350, 45, -0.7)), spoil=0.03, case=4),
    S("FV-ONION-1K", "Onions, 1 kg", "Fruits & vegetables", "FarmDay", 52, 0.70, 21, 130, 0.88, 0.98, 2.0,
      spoil=0.01, case=25, cost_season=((292, 32, 0.20),)),
    # ---------------------------------------------------------------- dairy & bakery
    S("DB-MILK-1L", "Toned milk, 1 L", "Dairy & bakery", "Dairy Vale", 68, 0.83, 3, 330, 1.00, 1.08, 0.9,
      season=((140, 40, 0.06),), spoil=0.01, case=12),
    S("DB-PANEER-200", "Malai paneer, 200 g", "Dairy & bakery", "Dairy Vale", 99, 0.72, 7, 80, 0.95, 0.97, 1.5,
      events={"diwali": 0.35, "year_end": 0.2}, spoil=0.01, case=10),
    S("DB-CURD-400", "Fresh curd, 400 g", "Dairy & bakery", "Dairy Vale", 52, 0.74, 6, 105, 0.96, 1.00, 1.2,
      season=((130, 45, 0.35),), spoil=0.01, case=12),
    S("DB-BUTTER-100", "Salted butter, 100 g", "Dairy & bakery", "Dairy Vale", 62, 0.80, 60, 60, 1.00, 1.10, 1.0,
      events={"diwali": 0.2}, case=20),
    S("DB-BREAD-400", "Whole wheat bread, 400 g", "Dairy & bakery", "Ovenfresh", 55, 0.68, 4, 125, 0.95, 1.02, 1.4,
      spoil=0.02, case=10),
    S("DB-EGGS-12", "Farm eggs, pack of 12", "Dairy & bakery", "FarmDay", 96, 0.78, 14, 80, 0.94, 0.99, 1.6,
      season=((10, 45, 0.25), (180, 40, -0.12)), events={"year_end": 0.3}, spoil=0.005, case=10),
    # ---------------------------------------------------------------- staples
    S("ST-RICE-5K", "Basmati rice, 5 kg", "Staples", "Harvest Mill", 699, 0.72, 365, 36, 0.86, 0.96, 2.4,
      events={"diwali": 0.3}, trend=0.06, case=4),
    S("ST-ATTA-10K", "Whole wheat atta, 10 kg", "Staples", "Harvest Mill", 529, 0.78, 180, 48, 0.88, 0.98, 2.6,
      case=2),
    S("ST-DAL-1K", "Toor dal, 1 kg", "Staples", "Harvest Mill", 189, 0.76, 300, 62, 0.90, 1.03, 2.0, case=10),
    S("ST-OIL-1L", "Sunflower oil, 1 L", "Staples", "Gharana", 199, 0.80, 270, 70, 0.92, 0.97, 2.8,
      events={"diwali": 0.25}, case=12),
    S("ST-GHEE-1L", "Cow ghee, 1 L", "Staples", "Gharana", 749, 0.74, 270, 17, 0.93, 1.02, 1.6,
      season=((15, 50, 0.2),), events={"diwali": 0.95}, case=6),
    S("ST-SUGAR-1K", "Sugar, 1 kg", "Staples", "Gharana", 54, 0.82, 365, 95, 1.00, 1.06, 1.0,
      events={"diwali": 0.3, "rakhi": 0.15}, case=25),
    # ---------------------------------------------------------------- beverages
    S("BV-COLA-2L", "Cola, 2.25 L", "Beverages", "Coolsip", 105, 0.64, 180, 52, 0.90, 1.02, 1.8,
      season=((125, 45, 0.7), (360, 35, -0.3)), events={"year_end": 0.45}, case=9),
    S("BV-MANGO-1L", "Mango drink, 1.2 L", "Beverages", "Coolsip", 89, 0.60, 180, 34, 0.92, 1.00, 1.5,
      season=((120, 50, 0.6),), case=12),
    S("BV-ICECRM-1L", "Vanilla ice cream, 1 L tub", "Beverages", "Coolsip", 275, 0.58, 180, 22, 0.90, 1.05, 1.4,
      season=((125, 40, 0.95), (5, 40, -0.4)), events={"year_end": 0.3}, storage="frozen", case=6),
    S("BV-TEA-500", "Assam tea, 500 g", "Beverages", "Leafline", 299, 0.68, 365, 32, 0.92, 1.01, 2.0,
      season=((10, 50, 0.22),), events={"diwali": 0.2}, case=12),
    S("BV-COFFEE-200", "Filter coffee, 200 g", "Beverages", "Leafline", 229, 0.66, 270, 16, 0.94, 0.98, 1.4,
      trend=0.16, case=12),
    S("BV-WATER-12", "Mineral water, 12 × 1 L", "Beverages", "Coolsip", 240, 0.62, 365, 26, 0.88, 1.03, 2.2,
      season=((120, 45, 0.6),), case=1),
    # ---------------------------------------------------------------- sweets & dry fruits
    S("SW-KAJU-500", "Kaju katli, 500 g", "Sweets & dry fruits", "Mithaas", 649, 0.60, 15, 6, 0.92, 1.03, 1.6,
      events={"diwali": 2.3, "rakhi": 1.3, "holi": 0.5}, spoil=0.006, case=6),
    S("SW-GIFTBOX-1K", "Dry fruit gift box, 1 kg", "Sweets & dry fruits", "Mithaas", 1599, 0.62, 180, 3, 0.88, 1.05, 1.5,
      events={"diwali": 2.5, "rakhi": 1.0, "year_end": 0.6}, case=4),
    S("SW-ALMOND-500", "California almonds, 500 g", "Sweets & dry fruits", "Mithaas", 579, 0.72, 270, 20, 0.90, 0.98, 2.0,
      season=((10, 45, 0.2),), events={"diwali": 0.9}, case=10),
    S("SW-BHUJIA-400", "Bhujia namkeen, 400 g", "Sweets & dry fruits", "Mithaas", 115, 0.62, 120, 46, 0.93, 1.00, 1.6,
      events={"diwali": 0.55, "holi": 0.4}, case=20),
    S("SW-CHOCO-GIFT", "Chocolate gift pack", "Sweets & dry fruits", "Cocoa Lane", 499, 0.58, 270, 5, 0.90, 1.04, 1.3,
      events={"diwali": 1.6, "year_end": 1.1, "rakhi": 1.1}, case=8),
    # ---------------------------------------------------------------- personal care
    S("PC-SUNSCRN-100", "Sunscreen SPF 50, 100 ml", "Personal care", "Sunova", 399, 0.52, 730, 11, 0.85, 1.05, 1.6,
      season=((118, 50, 1.0), (355, 45, -0.5)), trend=0.12, case=12),
    S("PC-LOTION-400", "Body lotion, 400 ml", "Personal care", "Sunova", 349, 0.54, 730, 12, 0.86, 0.97, 1.6,
      season=((12, 45, 1.0), (170, 45, -0.5)), case=12),
    S("PC-HAIROIL-500", "Coconut hair oil, 500 ml", "Personal care", "Kesh", 215, 0.62, 540, 28, 0.90, 1.02, 2.0,
      case=12),
    S("PC-MOSQ-2", "Mosquito repellent refill, pack of 2", "Personal care", "Kesh", 159, 0.58, 730, 20, 0.90, 1.00, 1.8,
      season=((212, 40, 0.9), (30, 40, -0.3)), case=24),
    S("PC-HANDWASH-750", "Hand wash refill, 750 ml", "Personal care", "Kesh", 135, 0.58, 540, 24, 0.88, 1.04, 1.8,
      trend=0.10, case=12),
    # ---------------------------------------------------------------- home & seasonal
    S("HS-UMBRELLA-3F", "Three-fold umbrella", "Home & seasonal", "Monsoonwear", 499, 0.46, 1000, 1.8, 0.80, 1.06, 1.4,
      season=((190, 30, 2.4),), case=10),
    S("HS-RAINCOAT-L", "Raincoat, size L", "Home & seasonal", "Monsoonwear", 999, 0.50, 1000, 1.0, 0.80, 1.03, 1.4,
      season=((188, 28, 2.2),), case=10),
    S("HS-HEATER-2K", "Room heater, 2000 W", "Home & seasonal", "Warmhaus", 2499, 0.66, 1000, 0.45, 0.84, 1.02, 1.8,
      season=((5, 24, 2.6),), case=2),
    S("HS-FAN-400", "Table fan, 400 mm", "Home & seasonal", "Warmhaus", 2299, 0.68, 1000, 1.2, 0.86, 0.98, 1.9,
      season=((112, 32, 1.7),), case=2),
    S("HS-LIGHTS-10M", "LED string lights, 10 m", "Home & seasonal", "Glowline", 399, 0.42, 1000, 2.0, 0.80, 1.06, 1.3,
      events={"diwali": 2.7, "year_end": 1.5}, case=20),
    S("HS-DIYA-12", "Clay diyas, set of 12", "Home & seasonal", "Glowline", 199, 0.38, 1000, 0.7, 0.85, 1.05, 1.2,
      events={"diwali": 3.3}, case=24),
    S("HS-GULAL-5", "Herbal gulal, 5 colours", "Home & seasonal", "Glowline", 249, 0.40, 365, 0.35, 0.85, 1.04, 1.2,
      events={"holi": 3.7}, case=24),
    # ---------------------------------------------------------------- electronics
    S("EL-EARBUDS-X2", "Wireless earbuds X2", "Electronics", "Volt+", 3999, 0.36, 1000, 10, 0.45, 0.96, 2.2,
      events={"festive_sale": 1.8, "diwali": 0.5, "independence_sale": 0.7, "year_end": 0.3}, trend=0.22, case=10),
    S("EL-POWERBANK-20K", "Power bank, 20000 mAh", "Electronics", "Volt+", 2999, 0.40, 1000, 8, 0.55, 1.02, 2.4,
      events={"festive_sale": 1.5, "independence_sale": 0.6}, trend=0.08, case=10),
    S("EL-CHARGER-65W", "65 W fast charger", "Electronics", "Volt+", 1999, 0.30, 1000, 11, 0.50, 0.95, 2.0,
      events={"festive_sale": 1.1}, trend=0.12, case=20),
    S("EL-WATCH-S1", "Smartwatch S1", "Electronics", "Pulse", 5999, 0.32, 1000, 4, 0.42, 1.03, 2.2,
      events={"festive_sale": 2.0, "diwali": 0.6, "year_end": 0.4}, trend=0.18, case=5),
    S("EL-CABLE-C", "USB-C cable, 1.5 m", "Electronics", "Volt+", 699, 0.18, 1000, 26, 0.43, 0.97, 1.6,
      events={"festive_sale": 0.8}, case=50),
    # ---------------------------------------------------------------- stationery
    S("SN-NOTEBK-6", "Ruled notebooks, pack of 6", "Stationery", "Kaagaz", 240, 0.55, 1000, 14, 0.90, 1.03, 1.6,
      season=((158, 16, 1.7), (20, 30, 0.3)), case=20),
    S("SN-GEOBOX", "Geometry box", "Stationery", "Kaagaz", 165, 0.50, 1000, 4, 0.90, 1.05, 1.4,
      season=((158, 14, 1.9),), case=24),
    S("SN-GELPEN-10", "Gel pens, pack of 10", "Stationery", "Kaagaz", 110, 0.48, 1000, 22, 0.92, 0.99, 1.5,
      season=((158, 18, 0.8),), case=40),
)


def storage_class(spec: Spec) -> str:
    if spec.storage:
        return spec.storage
    return next(c[1] for c in CATEGORIES if c[0] == spec.category)


def is_perishable(spec: Spec) -> bool:
    return spec.shelf <= 30


def handling_cost(spec: Spec) -> float:
    """Category pick-pack-pay cost, but never more than 7% of MRP for cheap staples."""
    return round(max(2.0, min(CATEGORY_HANDLING[spec.category], 0.07 * spec.mrp)), 2)


def storage_cost_per_day(spec: Spec) -> float:
    return round(STORAGE_RATE[storage_class(spec)] * spec.mrp, 4)


def salvage_pct(spec: Spec) -> float:
    """Value of an expired unit as a share of cost: perishables cost a disposal fee, durables are liquidated."""
    if spec.shelf <= 30:
        return -0.05
    if spec.shelf <= 365:
        return 0.10
    return 0.45
