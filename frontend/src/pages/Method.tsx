import { LoadState, PageHead, Panel, ProductLink, Th } from "../components/ui";
import type { ModelSummary } from "../lib/api";
import { num, pct } from "../lib/format";
import { sortRows, useApi, useSort } from "../lib/hooks";

type ModelRow = ModelSummary["models"][number];

export function MethodPage() {
  const { data, error, loading } = useApi<ModelSummary>("/api/model");
  return (
    <>
      <PageHead
        title="Method"
        desc="How sales history becomes a recommended price, step by step. Every step is an optimisation problem with a check that it was solved."
      />
      <LoadState error={error} loading={loading}>
        {data && (
          <div className="stack">
            <Pipeline settings={data.settings} />
            <Models rows={data.models} />
          </div>
        )}
      </LoadState>
    </>
  );
}

function Pipeline({ settings }: { settings: ModelSummary["settings"] }) {
  return (
    <div className="cols cols--even">
      <Panel title="1. Learn demand from two years of sales">
        <div className="method">
          <p>
            For every product, daily units sold are modelled as Poisson with a log-linear mean in our price, the competitor's price, a three-harmonic yearly cycle, the day of the week, six festival and sale windows, a trend and the offer badge.
          </p>
          <div className="formula">{`log E[q_t] = θ₀ + β·log(p_t/MRP) + γ·log(c_t/MRP)
          + Σₖ aₖ sin(2πk·doy/365) + bₖ cos(2πk·doy/365)
          + δ_weekday + Σₑ λₑ·wₑ(t) + τ·t + π·promo_t`}</div>
          <p>
            The coefficients minimise the negative log-likelihood, which is convex; damped Newton's method reaches the optimum in about five iterations. Days that closed with no stock are left out because sales on those days understate demand. Slow sellers borrow strength from their category (empirical-Bayes shrinkage of β and γ). Models are refitted every {settings.refit_every_days} days on the trailing {settings.fit_window_days} days.
          </p>
        </div>
      </Panel>
      <Panel title="2. Model the competitor">
        <div className="method">
          <p>
            Today's demand and elasticities are matched exactly by a two-seller logit market with an outside option (we and the competitor together hold {pct(settings.market_coverage * 100, 0)} of shoppers). If the competitor's price history shows it following ours (reaction slope above {settings.reaction_threshold}), we price as a Stackelberg leader: the competitor's best reply is solved in closed form with the Lambert W function.
          </p>
          <div className="formula">{`share₀ = e^(a₀−b·p₀) / (1 + e^(a₀−b·p₀) + e^(a₁−b·p₁))
p₁*(p₀) = c₁ + (1 + W(e^z)) / b`}</div>
          <p>Bertrand–Nash prices are found by damped best-response iteration and shown for reference.</p>
        </div>
      </Panel>
      <Panel title="3. Plan prices over stock and time">
        <div className="method">
          <p>
            A finite-horizon stochastic dynamic programme chooses a price for every day and stock level. Demand each day is the market model scaled by the forecast seasonal multiplier, so a coming festival or the end of a season is priced in. The loss floor is enforced inside the maximisation, so no state can ever be priced below cost plus the minimum margin.
          </p>
          <div className="formula">{`V_t(x) = max_{floor_t ≤ p ≤ MRP}  (p − h)·E[min(D,x)] − s·E[x − min(D,x)] + E[V_{t+1}(x')]
D ~ Poisson(λ(p)·m_t),   x' = survivors of (x − sales) after spoilage`}</div>
          <p>
            The horizon is the next possible delivery (leftovers are worth their replacement cost) unless perishable stock exceeds what can sell before expiry, in which case the plan runs to expiry and leftovers are worth only their salvage value. Grid: {settings.price_grid_size} prices, up to {settings.max_inventory_states} stock states, at most {settings.max_horizon_days} days.
          </p>
        </div>
      </Panel>
      <Panel title="4. Check, explain and hand over">
        <div className="method">
          <p>
            The first day's price is refined continuously on the feasible interval [max(floor, current − limit), min(MRP, current + limit)] and checked against the Karush–Kuhn–Tucker conditions: stationarity, the sign of each multiplier and complementary slackness. Prices are rounded to whole rupees without crossing the loss floor.
          </p>
          <div className="formula">{`price change = cost & demand + competitor + stock & season + price limits`}</div>
          <p>
            To explain the result, the problem is solved again adding one factor at a time; the four steps add up exactly to the recommended change. A person then approves, adjusts or rejects each price, and only approved prices reach the exported price list. Products are priced independently on {settings.workers} workers, so a run scales with the size of the catalogue.
          </p>
        </div>
      </Panel>
    </div>
  );
}

function Models({ rows }: { rows: ModelRow[] }) {
  const sort = useSort<keyof ModelRow>("category", "asc");
  const sorted = sortRows(rows, sort.key, sort.dir);
  const wape = rows.reduce((a, r) => a + r.holdout_wape, 0) / Math.max(rows.length, 1);
  const iters = rows.reduce((a, r) => a + r.newton_iters, 0) / Math.max(rows.length, 1);
  return (
    <Panel title="Fitted demand models" note={`${rows.length} products, average holdout error ${pct(wape * 100, 0)}, ${iters.toFixed(1)} Newton iterations on average`} flush>
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              <Th k="name" label="Product" sort={sort} />
              <Th k="category" label="Category" sort={sort} />
              <Th k="own_elasticity" label="Price elasticity" sort={sort} num />
              <Th k="cross_elasticity" label="Competitor elasticity" sort={sort} num />
              <Th k="competitor_reaction" label="Rival reaction" sort={sort} num />
              <Th k="n_obs" label="Days used" sort={sort} num />
              <Th k="n_censored" label="Sold-out days dropped" sort={sort} num />
              <Th k="holdout_wape" label="Holdout error" sort={sort} num />
              <Th k="newton_iters" label="Iterations" sort={sort} num />
            </tr>
          </thead>
          <tbody>
            {sorted.map((m) => (
              <tr key={m.sku}>
                <td><ProductLink sku={m.sku} name={m.name} /></td>
                <td className="muted">{m.category}</td>
                <td className="num">
                  {m.own_elasticity.toFixed(2)} <span className="muted">± {(1.645 * m.own_se).toFixed(2)}</span>
                </td>
                <td className="num">
                  {m.cross_elasticity.toFixed(2)} <span className="muted">± {(1.645 * m.cross_se).toFixed(2)}</span>
                </td>
                <td className="num">{m.competitor_reaction.toFixed(2)}</td>
                <td className="num">{num(m.n_obs)}</td>
                <td className="num">{num(m.n_censored)}</td>
                <td className="num">{pct(m.holdout_wape * 100, 0)}</td>
                <td className="num">{m.newton_iters}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}
