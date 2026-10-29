# Bazaarline pricing

A price recommendation tool for a retail store or a chain's category team, and an optimisation mini project. For
every product it says whether to **raise, cut or hold** the price, **by how much**, the **profit** that adds, and
**why**: unit cost and margin, how strongly demand reacts to price, the season and festivals ahead, stock on hand and
its expiry, the competitor, and the allowed price range (loss floor, MRP, largest daily move). A person approves,
adjusts or rejects each price; only approved prices reach the exported price list.

It ships with a sample store (50 products, two years of simulated history, about a million order lines) so it can be
tried immediately, and imports a real store's data from CSV files.

## How a store uses it

1. **Load data** (Data page, or POST the same CSV files from your systems every night)
   products → daily sales → stock on hand → competitor prices (→ write-offs, optional).
   Each file is validated as a whole; if a row is wrong nothing is imported and the problems are listed by line.
2. **Calculate prices** (Price recommendations page → *Calculate prices*, or set `PRICING_DAILY_RUN_AT=06:00`).
   Demand models are refitted automatically when new data has arrived or they are more than a week old.
3. **Review**: sort and filter by recommendation, open any product to see the reasoning, try other prices on the
   slider, then approve, set your own price (within loss floor and MRP) or keep the current price. Bulk approval works
   on any selection. Every decision is signed with the approver's name and listed on *Price changes*.
4. **Export** the price list (CSV) for the billing system, website or shelf labels.

## The optimisation, step by step

```
your data (SQLite; plain SQL that ports to PostgreSQL)
  ▼
1. demand estimation    Poisson maximum likelihood by damped Newton's method, one model per product:
                        own and competitor price elasticity · yearly seasonality (Fourier) · weekday ·
                        Diwali, Holi, Raksha Bandhan, festive sale, year end, Independence Day sale · trend · offers.
                        Sold-out days are left out (sales were capped); slow sellers borrow strength from their category.
  ▼
2. derived inputs       unit cost from stock batches · age and days to expiry · spoilage rate from write-offs ·
                        current price · competitor price · MRP ceiling · category policy · competitor reaction slope
  ▼
3. competition          two-seller logit market fitted to today's demand and elasticities; if the competitor follows
                        our prices we price as a Stackelberg leader (its best reply in closed form via Lambert W)
  ▼
4. stochastic DP        V_t(x) = max over floor_t ≤ p ≤ MRP of E[profit today] + E[V_{t+1}], demand scaled by the
                        seasonal forecast; horizon = next delivery, or expiry when perishable stock exceeds what can sell
  ▼
5. KKT certificate      first-day price refined on [max(floor, −Δ), min(MRP, +Δ)]; multipliers and complementary
                        slackness checked; rounded to whole rupees without crossing the loss floor
  ▼
6. explanation          the problem is re-solved adding one factor at a time; the four steps
                        (cost & demand, competitor, stock & season, price limits) sum exactly to the recommended change
```

## Run

```bash
# backend (Python 3.10+)
cd backend
pip install -r requirements-dev.txt
python -m pytest                       # 28 tests
uvicorn api.main:app --port 8000       # first start builds the sample store (~15 s) and prices it

# frontend
cd frontend
npm install
npm run dev                            # http://localhost:5173

# or both in containers
docker compose up --build              # http://localhost:8080
```

### Using your own store

```bash
PRICING_DEMO_DATA=0 PRICING_DB_PATH=./data/mystore.db uvicorn api.main:app --port 8000
```

Then import your files on the Data page (templates are downloadable there). At least four months of daily sales are
needed; two years gives the best seasonality. Products without enough history are listed as not priced and the rest of
the run continues.

### Settings (environment)

| Variable | Default | Meaning |
|---|---|---|
| `PRICING_DB_PATH` | `./data/store.db` | Database file |
| `PRICING_DEMO_DATA` | `1` | Build the sample store when the database is missing (never touches a non-demo database) |
| `PRICING_API_KEY` | empty | When set, calculating, approving and importing need the `X-API-Key` header (the UI asks for it) |
| `PRICING_DAILY_RUN_AT` | empty | e.g. `06:00`: calculate new recommendations every morning once new sales data is in |
| `PRICING_WORKERS` | `4` | Products priced in parallel |
| `CORS_ORIGINS` | `http://localhost:5173` | Allowed browser origins for the API |

Sample store only: `python -m pricing_engine.demo next-day` plays out the next trading day at the approved prices
(simulated shoppers, competitor moves, deliveries, expiry) so you can repeat the calculate → approve → trade loop.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/status` | Data coverage, model freshness, latest run, progress |
| POST | `/api/runs` | Calculate recommendations (`{"price_date": "YYYY-MM-DD"?, "refit": bool?}`) |
| GET | `/api/recommendations?run_id=` | All recommendations of a run (latest by default) |
| GET | `/api/products/{sku}` | Full reasoning for one product |
| POST | `/api/products/{sku}/decision` | `{"decision": "approve" \| "override" \| "reject" \| "reset", "price"?, "user", "note"?}` |
| POST | `/api/recommendations/decisions` | Bulk: `{"skus": [...], "decision": "approve" \| "reject", "user"}` |
| GET | `/api/price-changes` | Audit log of decisions |
| GET | `/api/export/prices.csv?run_id=&changed=1` | Price list for the price date |
| POST | `/api/import/{products\|sales\|stock\|competitor\|write_offs}` | CSV body; all-or-nothing validation |
| GET | `/api/import/{kind}/template.csv` | Column layout with example rows |

## Scaling notes

Products are priced independently in a worker pool and a bad product never stops a run, so run time grows linearly
with the catalogue (about 0.1–0.2 s per product per core). The schema is plain SQL; for several stores or a large
catalogue move it to PostgreSQL, run `uvicorn` behind a reverse proxy with SSO in front, and give each store or region
its own database or a `store_id` column.

## Code map

| Area | File |
|---|---|
| Demand estimation | `pricing_engine/demand/model.py` |
| Derived inputs | `pricing_engine/ops/state.py` |
| Optimiser and explanation | `pricing_engine/engine.py`, `optim/sdp.py`, `optim/game.py`, `optim/kkt.py`, `optim/simulation.py` |
| Runs, approvals, schedule | `pricing_engine/ops/service.py` |
| CSV import/export | `pricing_engine/ops/imports.py` |
| Sample store | `pricing_engine/data/*`, `pricing_engine/demo.py` |
| API | `api/main.py` |
| Web app | `frontend/src` (React, TypeScript, Recharts) |
