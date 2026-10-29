import { useMemo, useState } from "react";
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { axis, C, Legend, tip, yAxis } from "../components/chart";
import { ActionTag, LoadState, PageHead, Panel, StatusTag } from "../components/ui";
import { ApiError, post, toDecision, type Advice, type Decision, type Explanation, type ModelInfo, type ProductDetail, type Recommendation } from "../lib/api";
import { day, dayShort, MONTH_NAMES, monthYear, num, num1, pct, rupees, rupees2 } from "../lib/format";
import { href, useApi, useRefresh } from "../lib/hooks";
import { getUser } from "../lib/user";

export function ProductPage({ sku }: { sku: string }) {
  const { data, error, loading } = useApi<ProductDetail>(`/api/products/${encodeURIComponent(sku)}`);
  const p = data?.product;
  return (
    <div className={loading && data ? "is-loading" : undefined}>
      <PageHead
        crumbs={[{ label: "Price recommendations", to: href() }]}
        title={p ? p.name : sku}
        desc={
          p ? (
            <>
              {p.sku}, {p.brand}. {p.category}, {p.storage_class} storage, {p.shelf_life_days >= 1000 ? "does not expire" : `${p.shelf_life_days}-day shelf life`}. Supplied by {p.supplier} with a {p.lead_time_days}-day lead time. MRP {rupees(p.mrp)}; minimum margin {pct(p.min_margin_pct * 100, 0)}; prices move at most {pct(p.max_daily_change_pct * 100, 0)} a day.
            </>
          ) : undefined
        }
      />
      <LoadState error={error} loading={loading}>
        {data && <Body d={data} />}
      </LoadState>
    </div>
  );
}

function Body({ d }: { d: ProductDetail }) {
  const rec = d.recommendation;
  const dec = rec ? toDecision(rec) : null;
  const ex = d.explanation;
  const m = d.model;
  const [override, setOverride] = useState<number | null>(null);
  return (
    <div className="stack">
      {!rec && <div className="callout">This product has no recommendation in the latest run (it may have been added after the run, or lacks sales history).</div>}
      {rec && dec && <RecommendationPanel rec={rec} dec={dec} mrp={d.product.mrp} override={override} setOverride={setOverride} />}
      {dec?.advice && (
        <div className="cols cols--wide-left">
          <WhyPanel a={dec.advice} />
          {ex && ex.objective.length > 0 && <WhatIfPanel ex={ex} dec={dec} onUse={rec?.status === "pending" ? setOverride : undefined} />}
        </div>
      )}
      {dec && (
        <div className="cols cols--wide-left">
          <InputsPanel dec={dec} />
          <div className="stack">
            <StockPanel d={d} />
            <KktPanel dec={dec} />
          </div>
        </div>
      )}
      {ex && dec && <ResponsePanel ex={ex} dec={dec} />}
      {ex && ex.plan.length > 0 && dec && (
        <div className="cols cols--even">
          <PlanPanel ex={ex} dec={dec} />
          <StrategyPanel ex={ex} />
        </div>
      )}
      <HistoryPanel d={d} />
      {m && (
        <div className="cols cols--even">
          <SeasonPanel m={m} />
          <ModelPanel m={m} />
        </div>
      )}
      <div className="cols cols--even">
        <RecentPanel d={d} />
        {m && <ForecastPanel m={m} />}
      </div>
      {ex?.policy && <PolicyPanel ex={ex} floor={dec?.floor ?? 0} />}
      <ApprovalsPanel d={d} />
    </div>
  );
}

// ------------------------------------------------------------------ decision

function Ladder({ dec, mrp }: { dec: Decision; mrp: number }) {
  const pts = [dec.floor, dec.competitor_price, dec.prev_price, dec.new_price, mrp, dec.kkt.lower_bound, dec.kkt.upper_bound];
  const lo = Math.min(...pts) * 0.96;
  const hi = Math.max(...pts) * 1.02;
  const x = (v: number) => `${((v - lo) / (hi - lo)) * 100}%`;
  // Tags sit above or below the bar; marks that would collide on the same side share one label.
  const raw = [
    { v: dec.floor, label: "Loss floor", cls: "floor", pos: "bottom" },
    { v: dec.prev_price, label: "Yesterday", cls: "", pos: "bottom" },
    { v: dec.competitor_price, label: "Competitor", cls: "rival", pos: "top" },
    { v: mrp, label: "MRP", cls: "", pos: "top" },
  ];
  const span = hi - lo;
  const tags: { v: number; label: string; cls: string; pos: string }[] = [];
  for (const m of raw) {
    const clash = tags.find((t) => t.pos === m.pos && Math.abs(t.v - m.v) / span < 0.16);
    if (!clash) {
      tags.push({ ...m });
      continue;
    }
    const other = m.pos === "top" ? "bottom" : "top";
    if (!tags.some((t) => t.pos === other && Math.abs(t.v - m.v) / span < 0.16)) tags.push({ ...m, pos: other });
    else if (Math.abs(clash.v - m.v) < 0.5) clash.label = `${clash.label} and ${m.label.toLowerCase()}`;
    else {
      clash.label = `${clash.label} ${rupees(clash.v)}, ${m.label.toLowerCase()}`;
      clash.v = m.v;
    }
  }
  const marks = raw;
  return (
    <>
      <div className="ladder" role="img" aria-label={`Today's price ${rupees(dec.new_price)} between loss floor ${rupees(dec.floor)} and MRP ${rupees(mrp)}`}>
        <div className="ladder__loss" style={{ width: x(dec.floor) }} />
        <div className="ladder__band" style={{ left: x(dec.kkt.lower_bound), width: `calc(${x(dec.kkt.upper_bound)} - ${x(dec.kkt.lower_bound)})` }} />
        {marks.map((mk) => (
          <div key={mk.label} className={`ladder__mark${mk.cls ? ` ladder__mark--${mk.cls}` : ""}`} style={{ left: x(mk.v) }} />
        ))}
        {tags.map((t) => (
          <div key={t.label} className={`ladder__tag ladder__tag--${t.pos}${t.cls ? ` ladder__tag--${t.cls}` : ""}`} style={{ left: `clamp(40px, ${x(t.v)}, calc(100% - 40px))` }}>
            {t.label} <b>{rupees(t.v)}</b>
          </div>
        ))}
        <div className="ladder__mark ladder__mark--now" style={{ left: x(dec.new_price) }} />
      </div>
      <div className="legend">
        <span><i style={{ background: "repeating-linear-gradient(135deg,#e7b7b1 0 2px,transparent 2px 5px)", border: "1px solid #b42318" }} />Would lose money</span>
        <span><i style={{ background: "#cfdcea" }} />Allowed today (move limit and MRP)</span>
        <span><i style={{ background: C.accent, width: 3 }} />Today's price</span>
      </div>
    </>
  );
}

function verdictText(a: Advice): string {
  if (a.action === "hold") return `Keep the price at ${rupees(a.current_price)}`;
  const verb = a.action === "increase" ? "Increase" : "Decrease";
  return `${verb} the price by ${Math.abs(a.change_pct).toFixed(1)}%, from ${rupees(a.current_price)} to ${rupees(a.recommended_price)}`;
}

function holdReason(dec: Decision, a: Advice): string {
  if (a.action !== "hold") return "";
  if (dec.binding === "MRP ceiling") return "It is already at MRP, the highest price allowed, and a higher price would still pay.";
  if (dec.binding === "Loss floor") return "It is already at the loss floor; any lower price would sell below cost plus the minimum margin.";
  return "This is already the profit-maximising price given today's demand, stock and competitor.";
}

function DecideBar({ rec, override, setOverride }: { rec: Recommendation; override: number | null; setOverride: (v: number | null) => void }) {
  const refresh = useRefresh();
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [note, setNote] = useState("");
  const send = async (decision: string, price?: number) => {
    const user = getUser();
    if (!user) {
      setMsg("Enter your name at the top right first; every approval is signed.");
      return;
    }
    setBusy(true);
    setMsg(null);
    try {
      await post(`/api/products/${encodeURIComponent(rec.sku)}/decision`, { decision, price, user, note, run_id: rec.run_id });
      setOverride(null);
      refresh();
    } catch (e) {
      setMsg(e instanceof ApiError ? e.message : "Could not save the decision.");
    } finally {
      setBusy(false);
    }
  };
  if (rec.status !== "pending") {
    return (
      <div className="decide">
        <StatusTag status={rec.status} price={rec.final_price} by={rec.decided_by} />
        <span className="muted">
          {rec.status === "rejected" ? "Price stays as it was." : `Price list for ${day(rec.price_date)}: ${rupees(rec.final_price)}.`}
          {rec.decided_at ? ` ${day(rec.decided_at)} ${rec.decided_at.slice(11, 16)}.` : ""}
          {rec.note ? ` Note: ${rec.note}` : ""}
        </span>
        <button className="linkbtn" disabled={busy} onClick={() => send("reset")}>
          Undo
        </button>
      </div>
    );
  }
  return (
    <>
      <div className="decide">
        <button className="btn btn--primary" disabled={busy} onClick={() => send("approve")}>
          Approve {rupees(rec.recommended_price)}
        </button>
        <span className="muted">or set</span>
        <input
          type="number"
          aria-label="Your price"
          min={Math.ceil(rec.floor)}
          max={rec.ceiling}
          step={1}
          value={override ?? ""}
          placeholder={String(Math.round(rec.recommended_price))}
          onChange={(e) => setOverride(e.target.value === "" ? null : Number(e.target.value))}
        />
        <button className="btn" disabled={busy || override === null} onClick={() => override !== null && send("override", override)}>
          Use my price
        </button>
        <button className="btn" disabled={busy} onClick={() => send("reject")}>
          Keep current price
        </button>
        <input className="userbox__input" style={{ width: 200, marginTop: 0 }} placeholder="Note (optional)" value={note} onChange={(e) => setNote(e.target.value)} aria-label="Note" />
      </div>
      <div className="muted" style={{ fontSize: 12.5, marginTop: 6 }}>
        Allowed: {rupees(Math.ceil(rec.floor))} (loss floor) to {rupees(rec.ceiling)} (MRP).
      </div>
      {msg && <div className="callout callout--bad" style={{ marginTop: 10 }}>{msg}</div>}
    </>
  );
}

function RecommendationPanel({ rec, dec, mrp, override, setOverride }: { rec: Recommendation; dec: Decision; mrp: number; override: number | null; setOverride: (v: number | null) => void }) {
  const a = dec.advice;
  if (!a) return null;
  const h = a.horizon_days;
  const heading = Math.abs(a.target_price - a.recommended_price) >= 1;
  return (
    <section className="panel decision" aria-label="Recommendation">
      <div className="decision__main">
        <div className="decision__label">Recommendation for {day(dec.price_date, true)}</div>
        <div style={{ margin: "6px 0 2px" }}>
          <ActionTag action={a.action} pct={a.action === "hold" ? null : a.change_pct} />
        </div>
        <div className="verdict">{verdictText(a)}</div>
        <div className="verdict__sub">
          {a.action === "hold"
            ? holdReason(dec, a)
            : a.current_price < dec.floor - 0.5 && a.profit_gain < 0
              ? `Today's price is below the loss floor of ${rupees(dec.floor)}, so it has to rise. Selling below cost would have earned ${rupees(-a.profit_gain)} more over the next ${h} ${h === 1 ? "day" : "days"}, but the store never sells at a loss.`
              : `Expected to add ${rupees(Math.max(a.profit_gain, 0))} of profit over the next ${h} ${h === 1 ? "day" : "days"} compared with keeping today's price.`}
          {heading && ` The daily move limit allows only this step today; the price is heading to ${rupees(a.target_price)} (${pct(a.target_change_pct, 1, true)} in total).`}
        </div>
        <div className="impact">
          <div>
            <div className="impact__label">Units sold today</div>
            <div className="impact__value">{num1(a.units_today_new)}</div>
            <div className="impact__was">{num1(a.units_today_current)} at the current price</div>
          </div>
          <div>
            <div className="impact__label">Margin per unit</div>
            <div className={`impact__value${a.unit_margin_new < 0 ? " down" : ""}`}>{rupees2(a.unit_margin_new)}</div>
            <div className="impact__was">{rupees2(a.unit_margin_current)} now; full cost {rupees2(a.full_unit_cost)}</div>
          </div>
          <div>
            <div className="impact__label">Profit over {h} {h === 1 ? "day" : "days"}</div>
            <div className="impact__value">{rupees(a.profit_horizon_new ?? a.profit_today_new)}</div>
            <div className="impact__was">{rupees(a.profit_horizon_hold ?? a.profit_today_current)} if the price is kept</div>
          </div>
        </div>
        <DecideBar rec={rec} override={override} setOverride={setOverride} />
      </div>
      <div className="decision__side">
        <div className="decision__label" style={{ marginBottom: 4 }}>Where the price sits</div>
        <Ladder dec={dec} mrp={mrp} />
        <dl className="facts" style={{ marginTop: 14 }}>
          <dt>Competitor</dt>
          <dd>{dec.response === "stackelberg" ? "Follows our price" : "Does not react to our price"}</dd>
          <dt>Planning horizon</dt>
          <dd>
            {h} {h === 1 ? "day" : "days"}, {a.horizon_mode === "expiry" ? "until the stock expires" : "until the next delivery can arrive"}
          </dd>
        </dl>
        {dec.warnings.length > 0 && (
          <ul className="notes">
            {dec.warnings.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
        )}
      </div>
    </section>
  );
}

function WhyPanel({ a }: { a: Advice }) {
  const start = a.current_price;
  const pts = [start, ...a.factors.map((f) => f.price_after)];
  const lo = Math.min(...pts);
  const hi = Math.max(...pts);
  const pad = (hi - lo) * 0.08 || start * 0.05;
  const x = (v: number) => ((v - (lo - pad)) / (hi - lo + 2 * pad)) * 100;
  let prev = start;
  return (
    <Panel title="Why this price" note="Each factor's push, in rupees" flush>
      <div className="why">
        <div className="why__row why__row--total">
          <div>Current price</div>
          <div className="why__delta">{rupees2(start)}</div>
          <div className="why__track">
            <div className="why__axis" style={{ left: `${x(start)}%` }} />
          </div>
        </div>
        {a.factors.map((f) => {
          const from = prev;
          prev = f.price_after;
          const left = Math.min(from, f.price_after);
          const width = Math.abs(f.price_after - from);
          const up = f.delta > 0;
          return (
            <div className="why__row" key={f.key}>
              <div>
                <div className="why__label">{f.label}</div>
                <div className="why__detail">{f.detail}</div>
              </div>
              <div className={`why__delta ${Math.abs(f.delta) < 0.5 ? "flat" : up ? "up" : "down"}`}>
                {Math.abs(f.delta) < 0.005 ? "±₹0" : `${up ? "+" : "−"}${rupees2(Math.abs(f.delta))}`}
              </div>
              <div className="why__track">
                <div className="why__bar" style={{ left: `${x(left)}%`, width: `max(2px, ${x(left + width) - x(left)}%)`, background: up ? C.good : C.rival }} />
              </div>
            </div>
          );
        })}
        <div className="why__row why__row--total">
          <div>Recommended price</div>
          <div className="why__delta">{rupees2(a.recommended_price)}</div>
          <div className="why__track">
            <div className="why__axis" style={{ left: `${x(a.recommended_price)}%`, background: C.accent, width: 3 }} />
          </div>
        </div>
      </div>
    </Panel>
  );
}

function WhatIfPanel({ ex, dec, onUse }: { ex: Explanation; dec: Decision; onUse?: (price: number) => void }) {
  const pts = ex.objective;
  const rec = pts.reduce((b, p, i) => (Math.abs(p.price - dec.new_price) < Math.abs(pts[b]!.price - dec.new_price) ? i : b), 0);
  const [i, setI] = useState(rec);
  const p = pts[Math.min(i, pts.length - 1)];
  const r = pts[rec];
  if (!p || !r) return null;
  const hasPlan = p.profit_horizon !== null;
  const diff = hasPlan ? (p.profit_horizon ?? 0) - (r.profit_horizon ?? 0) : p.profit_today - r.profit_today;
  return (
    <Panel title="Try another price" note={hasPlan ? `Profit over the ${dec.horizon_days}-day plan` : "Profit today"}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <span className="whatif__price">{rupees(p.price)}</span>
        <span className="rowbtns">
          {onUse && !p.below_floor && i !== rec && (
            <button className="btn" onClick={() => { onUse(p.price); window.scrollTo({ top: 0, behavior: "smooth" }); }}>
              Use this price
            </button>
          )}
          <button className="btn" onClick={() => setI(rec)} disabled={i === rec}>
            Back to recommended
          </button>
        </span>
      </div>
      <input
        className="whatif__slider"
        type="range"
        min={0}
        max={pts.length - 1}
        value={i}
        onChange={(e) => setI(Number(e.target.value))}
        aria-label="Price to test"
      />
      <div className="whatif__scale">
        <span>{rupees(pts[0]!.price)}</span>
        <span>MRP {rupees(pts.at(-1)!.price)}</span>
      </div>
      <dl className="facts" style={{ marginTop: 14 }}>
        <dt>Units sold today</dt>
        <dd>{num1(p.units_today)}</dd>
        <dt>Profit today</dt>
        <dd className={p.profit_today < 0 ? "down" : undefined}>{rupees(p.profit_today)}</dd>
        {hasPlan && (
          <>
            <dt>Profit over the plan</dt>
            <dd>{rupees(p.profit_horizon)}</dd>
          </>
        )}
        <dt>Against the recommendation</dt>
        <dd className={i === rec ? "flat" : diff >= 0 ? "up" : "down"}>{i === rec ? "This is the recommended price" : `${diff >= 0 ? "+" : "−"}${rupees(Math.abs(diff))}`}</dd>
      </dl>
      {p.below_floor && <div className="callout callout--bad" style={{ marginTop: 12 }}>Below the loss floor of {rupees(dec.floor)}: this price would lose money and cannot be approved.</div>}
      <p style={{ fontSize: 12.5, color: "var(--muted)", marginTop: 12 }}>
        {hasPlan
          ? "Plan profit assumes today's price is the one shown and later days follow the optimal plan. It includes expiry losses and storage, and stock left at the end is valued at cost or salvage."
          : "No stock on hand, so only today's sales are counted."}
      </p>
    </Panel>
  );
}

function fmtInput(v: number | string, unit: string) {
  if (typeof v === "string") return v;
  if (unit === "₹") return rupees2(v);
  if (unit === "%") return `${v}%`;
  if (unit === "") return String(v);
  return `${Number.isInteger(v) ? num(v) : num1(v)} ${unit}`;
}

function InputsPanel({ dec }: { dec: Decision }) {
  return (
    <Panel title="Where the inputs came from" note="Read from your data when the prices were calculated" flush>
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              <th>Input</th>
              <th className="num">Value</th>
              <th>Source</th>
            </tr>
          </thead>
          <tbody>
            {dec.inputs.map((i) => (
              <tr key={i.key}>
                <td>{i.label}</td>
                <td className="num strong">{fmtInput(i.value, i.unit)}</td>
                <td className="wrap muted">{i.source}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

const CONSTRAINT_TEXT: Record<string, string> = {
  profit_floor: "loss floor",
  rate_limit_lower: "daily move limit (down)",
  ceiling_or_rate_limit_upper: "upper limit (MRP or daily move)",
};

function KktPanel({ dec }: { dec: Decision }) {
  const k = dec.kkt;
  const active = k.active_constraints.map((a) => CONSTRAINT_TEXT[a] ?? a);
  return (
    <Panel title="Optimality check" note={k.primal_feasible && k.dual_feasible ? "KKT conditions hold" : "Check failed"}>
      <dl className="facts">
        <dt>Feasible range</dt>
        <dd>
          {rupees2(k.lower_bound)} to {rupees2(k.upper_bound)}
        </dd>
        <dt>Active constraints</dt>
        <dd>{active.length ? active.join(", ") : "None, interior optimum"}</dd>
        <dt>Value gradient</dt>
        <dd>{k.gradient.toFixed(3)} ₹ per ₹</dd>
        <dt>Multipliers</dt>
        <dd>
          μ floor {k.mu_floor.toFixed(3)}, μ upper {k.mu_ceiling.toFixed(3)}
        </dd>
        <dt>Stationarity residual</dt>
        <dd>{k.stationarity_residual.toExponential(1)}</dd>
      </dl>
    </Panel>
  );
}

// ------------------------------------------------------------------ price response & plan

function ResponsePanel({ ex, dec }: { ex: Explanation; dec: Decision }) {
  const g = ex.game;
  return (
    <Panel title="How demand and profit respond to price today" note={`Competitor at ${rupees(dec.competitor_price)}; its estimated cost ${rupees(g.competitor_cost_estimate)}`} flush>
      <div className="cols cols--even" style={{ gap: 0 }}>
        <div style={{ padding: "14px 8px 0 0" }}>
          <div style={{ height: 240 }}>
          <ResponsiveContainer>
            <LineChart data={ex.curve} margin={{ top: 8, right: 12, left: 4, bottom: 0 }}>
              <CartesianGrid stroke={C.grid} vertical={false} />
              <XAxis dataKey="price" type="number" domain={["dataMin", "dataMax"]} {...axis} tickFormatter={(v: number) => `₹${Math.round(v)}`} />
              <YAxis {...yAxis} width={44} tickFormatter={(v: number) => num(v)} />
              <ReferenceLine x={dec.floor} stroke={C.bad} strokeDasharray="3 3" />
              <ReferenceLine x={dec.new_price} stroke={C.accent} strokeWidth={2} />
              <Tooltip
                content={tip<Explanation["curve"][number]>(
                  (r) => rupees2(r.price),
                  (r) => [
                    { label: "Competitor follows", value: num1(r.demand_with_response), color: C.accent },
                    { label: "Competitor holds", value: num1(r.demand_static), color: C.grey },
                    { label: "Demand model", value: num1(r.demand_model), color: C.rival },
                    { label: "Competitor's reply", value: rupees(r.competitor_response_price) },
                  ],
                )}
              />
              <Line dataKey="demand_with_response" stroke={C.accent} strokeWidth={1.75} dot={false} isAnimationActive={false} />
              <Line dataKey="demand_static" stroke={C.grey} strokeWidth={1.5} dot={false} isAnimationActive={false} />
              <Line dataKey="demand_model" stroke={C.rival} strokeWidth={1.25} strokeDasharray="4 3" dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
          </div>
          <Legend items={[{ label: "Units/day if competitor follows", color: C.accent }, { label: "If it holds", color: C.grey }, { label: "Fitted demand model", color: C.rival, dashed: true }]} />
        </div>
        <div style={{ padding: "14px 8px 0 0", borderLeft: "1px solid var(--line)" }}>
          <div style={{ height: 240 }}>
          <ResponsiveContainer>
            <LineChart data={ex.curve} margin={{ top: 8, right: 12, left: 4, bottom: 0 }}>
              <CartesianGrid stroke={C.grid} vertical={false} />
              <XAxis dataKey="price" type="number" domain={["dataMin", "dataMax"]} {...axis} tickFormatter={(v: number) => `₹${Math.round(v)}`} />
              <YAxis {...yAxis} tickFormatter={(v: number) => `₹${num(v)}`} />
              <ReferenceLine y={0} stroke={C.greyLight} />
              <ReferenceLine x={dec.floor} stroke={C.bad} strokeDasharray="3 3" label={{ value: "floor", position: "insideTopLeft", fontSize: 11, fill: C.bad }} />
              <ReferenceLine x={g.stackelberg[0]} stroke={C.grey} strokeDasharray="2 3" label={{ value: "leader", position: "insideTopRight", fontSize: 11, fill: C.grey }} />
              <ReferenceLine x={dec.new_price} stroke={C.accent} strokeWidth={2} />
              <Tooltip content={tip<Explanation["curve"][number]>((r) => rupees2(r.price), (r) => [{ label: "Profit today", value: rupees(r.daily_profit), color: C.good }])} />
              <Line dataKey="daily_profit" stroke={C.good} strokeWidth={1.75} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
          </div>
          <Legend items={[{ label: "Profit for one day", color: C.good }, { label: "Today's price", color: C.accent }, { label: "Loss floor", color: C.bad, dashed: true }]} />
        </div>
      </div>
      <div style={{ padding: "4px 16px 14px", fontSize: 13, color: "var(--ink-2)" }}>
        Logit market fitted to today's elasticities: we hold {pct(g.our_share * 100, 0)} of shoppers, the competitor {pct(g.competitor_share * 100, 0)}. Simultaneous (Nash) prices would be {rupees(g.nash[0])} vs {rupees(g.nash[1])}; moving first (Stackelberg) we would charge {rupees(g.stackelberg[0])} and expect the reply {rupees(g.stackelberg[1])}. At today's price the competitor's best reply is {rupees(g.predicted_competitor_reply)}. The final price differs from the one-day optimum because the stock plan looks ahead {dec.horizon_days} {dec.horizon_days === 1 ? "day" : "days"}.
      </div>
    </Panel>
  );
}

function PlanPanel({ ex, dec }: { ex: Explanation; dec: Decision }) {
  const data = ex.plan.map((p) => ({ ...p, band: [p.stock_p10, p.stock_p90] as [number, number] }));
  return (
    <Panel title={`Plan for the next ${dec.horizon_days} ${dec.horizon_days === 1 ? "day" : "days"}`} note="Mean of simulated paths under the optimal policy" flush>
      <div style={{ height: 250, padding: "14px 8px 0 0" }}>
        <ResponsiveContainer>
          <ComposedChart data={data} margin={{ top: 8, right: 8, left: 4, bottom: 0 }}>
            <CartesianGrid stroke={C.grid} vertical={false} />
            <XAxis dataKey="date" {...axis} tickFormatter={dayShort} minTickGap={24} />
            <YAxis yAxisId="u" {...yAxis} width={44} tickFormatter={(v: number) => num(v)} />
            <YAxis yAxisId="p" orientation="right" {...yAxis} tickFormatter={(v: number) => `₹${num(v)}`} domain={["auto", "auto"]} />
            <Tooltip
              content={tip<(typeof data)[number]>(
                (r) => day(r.date, true),
                (r) => [
                  { label: "Expected price", value: r.price_mean === null ? "sold out" : rupees2(r.price_mean), color: C.accent },
                  { label: "Loss floor", value: rupees2(r.floor), color: C.bad },
                  { label: "Stock at opening", value: num1(r.stock_mean), color: C.grey },
                  { label: "If price held", value: num1(r.stock_hold) },
                  { label: "Demand vs today", value: `×${r.multiplier.toFixed(2)}` },
                ],
              )}
            />
            <Area yAxisId="u" dataKey="band" stroke="none" fill={C.greyLight} fillOpacity={0.6} isAnimationActive={false} />
            <Line yAxisId="u" dataKey="stock_mean" stroke={C.grey} strokeWidth={1.5} dot={false} isAnimationActive={false} />
            <Line yAxisId="u" dataKey="stock_hold" stroke={C.grey} strokeDasharray="4 3" strokeWidth={1} dot={false} isAnimationActive={false} />
            <Line yAxisId="p" dataKey="price_mean" stroke={C.accent} strokeWidth={1.75} dot={false} connectNulls={false} isAnimationActive={false} type="stepAfter" />
            <Line yAxisId="p" dataKey="floor" stroke={C.bad} strokeDasharray="3 3" strokeWidth={1} dot={false} isAnimationActive={false} type="stepAfter" />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <Legend
        items={[
          { label: "Price (right axis)", color: C.accent },
          { label: "Loss floor", color: C.bad, dashed: true },
          { label: "Stock, 10–90% range", color: C.greyLight, box: true },
          { label: "Stock if price held", color: C.grey, dashed: true },
        ]}
      />
    </Panel>
  );
}

function StrategyPanel({ ex }: { ex: Explanation }) {
  const best = Math.max(...ex.strategies.map((s) => s.profit));
  return (
    <Panel title="Compared with simple rules" note="Same simulated shoppers for every rule" flush>
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              <th>Rule</th>
              <th className="num">Profit</th>
              <th className="num">5th–95th pct</th>
              <th className="num">Sold</th>
              <th className="num">Wasted</th>
            </tr>
          </thead>
          <tbody>
            {ex.strategies.map((s) => (
              <tr key={s.name}>
                <td className={s.profit === best ? "strong" : undefined}>{s.name}</td>
                <td className={`num${s.profit < 0 ? " down" : ""}${s.profit === best ? " strong" : ""}`}>{rupees(s.profit)}</td>
                <td className="num muted">
                  {rupees(s.profit_p05)} to {rupees(s.profit_p95)}
                </td>
                <td className="num">{num1(s.units_sold)}</td>
                <td className={`num${s.units_wasted > 0.5 ? " down" : " muted"}`}>{num1(s.units_wasted)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p style={{ padding: "10px 16px 14px", fontSize: 12.5, color: "var(--muted)" }}>
        Profit over the planning horizon on the stock held this morning, net of its cost. Rules that ignore the loss floor can sell below cost.
      </p>
    </Panel>
  );
}

// ------------------------------------------------------------------ history & model

function HistoryPanel({ d }: { d: ProductDetail }) {
  const units = d.weekly.reduce((a, w) => a + w.units, 0);
  return (
    <Panel title="Sales history" note={`${num(units)} units over ${num(d.weekly.length)} weeks`} flush>
      <div style={{ height: 270, padding: "14px 8px 0 0" }}>
        <ResponsiveContainer>
          <ComposedChart data={d.weekly} margin={{ top: 8, right: 8, left: 4, bottom: 0 }}>
            <CartesianGrid stroke={C.grid} vertical={false} />
            <XAxis dataKey="week" {...axis} tickFormatter={monthYear} minTickGap={40} />
            <YAxis yAxisId="u" {...yAxis} width={44} tickFormatter={(v: number) => num(v)} />
            <YAxis yAxisId="p" orientation="right" {...yAxis} tickFormatter={(v: number) => `₹${num(v)}`} domain={["auto", "auto"]} />
            <Tooltip
              content={tip<ProductDetail["weekly"][number]>(
                (r) => `Week of ${day(r.week)}`,
                (r) => [
                  { label: "Units", value: num(r.units), color: C.accentLight },
                  { label: "Our average price", value: rupees2(r.avg_price), color: C.accent },
                  { label: "Competitor", value: rupees2(r.competitor), color: C.rival },
                  { label: "Days sold out", value: num(r.stockout_days) },
                ],
              )}
            />
            <Bar yAxisId="u" dataKey="units" fill={C.accentLight} maxBarSize={32} isAnimationActive={false} />
            <Line yAxisId="p" dataKey="avg_price" stroke={C.accent} strokeWidth={1.5} dot={false} isAnimationActive={false} connectNulls />
            <Line yAxisId="p" dataKey="competitor" stroke={C.rival} strokeWidth={1.25} dot={false} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <Legend items={[{ label: "Units per week", color: C.accentLight, box: true }, { label: "Our average price", color: C.accent }, { label: "Competitor price", color: C.rival }]} />
    </Panel>
  );
}

function heat(v: number, lo: number, hi: number): string {
  const t = Math.max(0, Math.min(1, (v - lo) / Math.max(hi - lo, 1e-9)));
  // interpolate #edf2f8 → #1f4e79
  const a = [0xed, 0xf2, 0xf8];
  const b = [0x1f, 0x4e, 0x79];
  return `rgb(${a.map((x, i) => Math.round(x + (b[i]! - x) * t)).join(",")})`;
}

function SeasonPanel({ m }: { m: ModelInfo }) {
  const weeks = useMemo(() => {
    const out: number[] = [];
    for (let w = 0; w < 52; w++) {
      const s = m.yearly_index.slice(w * 7, w * 7 + 7);
      out.push(s.reduce((a, b) => a + b, 0) / Math.max(s.length, 1));
    }
    return out;
  }, [m.yearly_index]);
  const lo = Math.min(...weeks);
  const hi = Math.max(...weeks);
  const peak = weeks.indexOf(hi);
  const trough = weeks.indexOf(lo);
  const wd = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].map((name, i) => ({ name, v: m.weekday_index[i] ?? 1 }));
  const weekName = (w: number) => `${dayShort(new Date(Date.UTC(2026, 0, 1 + w * 7)).toISOString())}`;
  return (
    <Panel title="Seasonality learned from history" note="Relative to an average day">
      <div style={{ fontSize: 13, marginBottom: 8 }}>
        Busiest around <strong>{weekName(peak)}</strong> ({(hi).toFixed(2)}× average), quietest around <strong>{weekName(trough)}</strong> ({lo.toFixed(2)}×).
      </div>
      <div className="yearstrip" role="img" aria-label="Weekly demand index across the year">
        {weeks.map((v, i) => (
          <span key={i} title={`Week of ${weekName(i)}: ${v.toFixed(2)}×`} style={{ background: heat(v, lo, hi) }} />
        ))}
      </div>
      <div className="yearstrip__months">
        {MONTH_NAMES.map((mn) => (
          <span key={mn}>{mn}</span>
        ))}
      </div>
      <div style={{ height: 130, marginTop: 16 }}>
        <ResponsiveContainer>
          <BarChart data={wd} margin={{ top: 4, right: 4, left: 0, bottom: 0 }}>
            <CartesianGrid stroke={C.grid} vertical={false} />
            <XAxis dataKey="name" {...axis} />
            <YAxis {...yAxis} width={36} domain={[0, "auto"]} tickFormatter={(v: number) => `${v.toFixed(1)}×`} />
            <ReferenceLine y={1} stroke={C.grey} strokeDasharray="3 3" />
            <Tooltip content={tip<(typeof wd)[number]>((r) => r.name, (r) => [{ label: "Demand vs average", value: `${r.v.toFixed(2)}×` }])} />
            <Bar dataKey="v" fill={C.accent} maxBarSize={32} isAnimationActive={false} barSize={22} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p style={{ fontSize: 12.5, color: "var(--muted)", marginTop: 4 }}>Day-of-week pattern. The year strip shows the smooth yearly cycle; festival effects are listed separately.</p>
    </Panel>
  );
}

function ModelPanel({ m }: { m: ModelInfo }) {
  const events = [...m.events].sort((a, b) => b.lift - a.lift);
  return (
    <Panel title="Demand model" note={`Fitted ${day(m.fitted_on)}`} flush>
      <div style={{ padding: "12px 16px" }}>
        <dl className="facts">
          <dt>Price elasticity</dt>
          <dd>
            {m.own_elasticity.toFixed(2)} ± {(1.645 * m.own_se).toFixed(2)} <span className="muted">(1% higher price, {Math.abs(m.own_elasticity).toFixed(1)}% fewer units)</span>
          </dd>
          <dt>Competitor elasticity</dt>
          <dd>
            {m.cross_elasticity.toFixed(2)} ± {(1.645 * m.cross_se).toFixed(2)}
          </dd>
          <dt>Competitor reaction</dt>
          <dd>{m.competitor_reaction.toFixed(2)} (long-run share of our price moves it copies)</dd>
          <dt>Underlying growth</dt>
          <dd>{pct(m.annual_growth * 100, 1, true)} a year</dd>
          <dt>Fit</dt>
          <dd>
            {num(m.n_obs)} days used, {num(m.n_censored)} sold-out days left out; holdout error {pct(m.holdout_wape * 100, 0)} (WAPE)
          </dd>
          <dt>Solver</dt>
          <dd>Newton's method, {m.iterations} iterations; dispersion {m.dispersion.toFixed(2)}</dd>
        </dl>
      </div>
      <table className="data">
        <thead>
          <tr>
            <th>Festival or sale</th>
            <th className="num">Peak-day lift</th>
            <th className="num">90% interval</th>
          </tr>
        </thead>
        <tbody>
          {events.map((e) => (
            <tr key={e.event}>
              <td>{e.label}</td>
              <td className={`num ${e.lo > 1 ? "strong" : "muted"}`}>×{e.lift.toFixed(2)}</td>
              <td className="num muted">
                ×{e.lo.toFixed(2)} to ×{e.hi.toFixed(2)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Panel>
  );
}

function RecentPanel({ d }: { d: ProductDetail }) {
  return (
    <Panel title="Last 120 days: sales against the model" flush>
      <div style={{ height: 240, padding: "14px 8px 0 0" }}>
        <ResponsiveContainer>
          <ComposedChart data={d.daily} margin={{ top: 8, right: 8, left: 4, bottom: 0 }}>
            <CartesianGrid stroke={C.grid} vertical={false} />
            <XAxis dataKey="date" {...axis} tickFormatter={dayShort} minTickGap={30} />
            <YAxis yAxisId="u" {...yAxis} width={44} tickFormatter={(v: number) => num(v)} />
            <YAxis yAxisId="p" orientation="right" {...yAxis} tickFormatter={(v: number) => `₹${num(v)}`} domain={["auto", "auto"]} />
            <Tooltip
              content={tip<ProductDetail["daily"][number]>(
                (r) => day(r.date, true),
                (r) => [
                  { label: "Sold", value: `${num(r.sold)}${r.stocked_out ? " (sold out)" : ""}`, color: C.accentLight },
                  { label: "Model expected", value: num1(r.fitted), color: C.accent },
                  { label: "Our price", value: rupees(r.price), color: C.good },
                  { label: "Competitor", value: rupees(r.competitor), color: C.rival },
                ],
              )}
            />
            <Bar yAxisId="u" dataKey="sold" fill={C.accentLight} maxBarSize={32} isAnimationActive={false} />
            <Line yAxisId="u" dataKey="fitted" stroke={C.accent} strokeWidth={1.5} dot={false} isAnimationActive={false} />
            <Line yAxisId="p" dataKey="price" stroke={C.good} strokeWidth={1.25} dot={false} type="stepAfter" isAnimationActive={false} />
            <Line yAxisId="p" dataKey="competitor" stroke={C.rival} strokeWidth={1} dot={false} type="stepAfter" isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <Legend items={[{ label: "Units sold", color: C.accentLight, box: true }, { label: "Model expectation", color: C.accent }, { label: "Our price", color: C.good }, { label: "Competitor", color: C.rival }]} />
    </Panel>
  );
}

function ForecastPanel({ m }: { m: ModelInfo }) {
  const total = m.forecast.reduce((a, b) => a + b.units, 0);
  return (
    <Panel title="Next eight weeks" note={`${num(total)} units forecast at ${rupees(m.forecast_price)}`} flush>
      <div style={{ height: 240, padding: "14px 8px 0 0" }}>
        <ResponsiveContainer>
          <BarChart data={m.forecast} margin={{ top: 8, right: 8, left: 4, bottom: 0 }}>
            <CartesianGrid stroke={C.grid} vertical={false} />
            <XAxis dataKey="date" {...axis} tickFormatter={dayShort} minTickGap={30} />
            <YAxis {...yAxis} width={44} tickFormatter={(v: number) => num(v)} />
            <Tooltip content={tip<ModelInfo["forecast"][number]>((r) => day(r.date, true), (r) => [{ label: "Expected units", value: num1(r.units) }])} />
            <Bar dataKey="units" fill={C.accent} maxBarSize={32} isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p style={{ padding: "6px 16px 14px", fontSize: 12.5, color: "var(--muted)" }}>Daily expected demand from the fitted seasonality, weekday pattern, festivals and trend, holding today's prices.</p>
    </Panel>
  );
}

// ------------------------------------------------------------------ stock & orders

function PolicyPanel({ ex, floor }: { ex: Explanation; floor: number }) {
  const pol = ex.policy!;
  const all = pol.prices.flat();
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const cols = pol.inventory.length;
  return (
    <Panel title="Pricing policy by day and stock level" note={`Solved in ${pol.solve_ms.toFixed(0)} ms${pol.lot_size > 1 ? `, stock in lots of ${pol.lot_size}` : ""}`}>
      <p style={{ fontSize: 13, color: "var(--ink-2)", marginBottom: 12 }}>
        The dynamic programme gives a price for every day of the horizon and every stock level. Darker cells are higher prices; outlined cells sit on the loss floor. More stock than the season can absorb pushes prices down; scarcity pushes them up.
      </p>
      <div className="heat" style={{ gridTemplateColumns: `56px repeat(${cols}, minmax(0, 1fr))` }}>
        {pol.days.map((dd, i) => (
          <div key={dd} style={{ display: "contents" }}>
            <div className="heat__row-label">Day {dd + 1}</div>
            {pol.prices[i]!.map((p, j) => (
              <div
                key={j}
                className="heat__cell"
                title={`Day ${dd + 1}, ${num(pol.inventory[j])} units: ${rupees2(p)}`}
                style={{ background: heat(p, lo, hi), boxShadow: p <= floor * 1.0001 ? "inset 0 0 0 1px #b42318" : undefined }}
              />
            ))}
          </div>
        ))}
      </div>
      <div className="heat__axis" style={{ paddingLeft: 56 }}>
        <span>{num(pol.inventory[0])} units</span>
        <span>Stock on hand</span>
        <span>{num(pol.inventory.at(-1))} units</span>
      </div>
      <div className="legend" style={{ marginTop: 10, alignItems: "center" }}>
        <span>{rupees(lo)}</span>
        <span className="ramp" />
        <span>{rupees(hi)}</span>
        <span>Marginal value of one more unit today: {rupees2(pol.marginal_value_of_stock)}</span>
      </div>
    </Panel>
  );
}


function StockPanel({ d }: { d: ProductDetail }) {
  const total = d.lots.reduce((a, l) => a + l.qty_on_hand, 0);
  const ref = d.recommendation?.price_date ?? d.last_sales_date ?? "";
  return (
    <Panel title="Stock on hand" note={`${num(total)} units`} flush>
      {d.lots.length === 0 ? (
        <div className="empty">No stock on hand.</div>
      ) : (
        <table className="data">
          <thead>
            <tr>
              <th>Received</th>
              <th>Expires</th>
              <th className="num">Units</th>
              <th className="num">Unit cost</th>
            </tr>
          </thead>
          <tbody>
            {d.lots.slice(0, 8).map((l) => {
              const left = ref ? Math.round((Date.parse(l.expires_on) - Date.parse(ref)) / 86_400_000) : 0;
              return (
                <tr key={l.lot_id}>
                  <td>{day(l.received_on)}</td>
                  <td className={left <= 2 && d.product.shelf_life_days < 1000 ? "down" : undefined}>{d.product.shelf_life_days >= 1000 ? "No expiry" : `${day(l.expires_on)} (${left} d)`}</td>
                  <td className="num">{num(l.qty_on_hand)}</td>
                  <td className="num">{rupees2(l.unit_cost)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </Panel>
  );
}

function ApprovalsPanel({ d }: { d: ProductDetail }) {
  if (d.history.length === 0) return null;
  return (
    <Panel title="Recommendations and decisions for this product" note={`Last ${d.history.length}`} flush>
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              <th>Prices for</th>
              <th className="num">Current</th>
              <th>Recommended</th>
              <th className="num">Price set</th>
              <th>Decision</th>
              <th>Note</th>
            </tr>
          </thead>
          <tbody>
            {d.history.map((h) => (
              <tr key={h.run_id}>
                <td>{day(h.price_date, true)}</td>
                <td className="num muted">{rupees(h.current_price)}</td>
                <td>
                  <ActionTag action={h.action} pct={h.action === "hold" ? null : (h.recommended_price / h.current_price - 1) * 100} /> {rupees(h.recommended_price)}
                </td>
                <td className="num strong">{h.final_price ? rupees(h.final_price) : "–"}</td>
                <td><StatusTag status={h.status} price={h.final_price} by={h.decided_by} /></td>
                <td className="wrap muted">{h.note ?? ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}
