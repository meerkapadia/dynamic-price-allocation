import { Fragment, useMemo, useState } from "react";
import { ActionTag, BindingTag, Figures, LoadState, PageHead, Panel, ProductLink, StatusTag, Th } from "../components/ui";
import { ApiError, BASE, post, type RecList, type RecRow, type Status } from "../lib/api";
import { day, num, num1, rupees, rupeesShort } from "../lib/format";
import { href, sortRows, useApi, useRefresh, useSort } from "../lib/hooks";
import { getUser } from "../lib/user";

type Key = keyof RecRow;
export const DRIVER: Record<string, string> = {
  economics: "Cost and demand",
  competitor: "Competitor",
  stock: "Stock and season",
  range: "Price limits",
};
const FILTERS = [
  { key: "all", label: "All" },
  { key: "increase", label: "Increase" },
  { key: "decrease", label: "Decrease" },
  { key: "hold", label: "Hold" },
  { key: "pending", label: "To review" },
  { key: "decided", label: "Decided" },
] as const;

function initialCategory(): string {
  const q = window.location.hash.split("?")[1];
  return q ? new URLSearchParams(q).get("category") ?? "" : "";
}

function time(iso: string | null | undefined): string {
  return iso ? iso.slice(11, 16) : "";
}

export function PricingPage({ status }: { status: Status }) {
  const { data, error, loading } = useApi<RecList>("/api/recommendations");
  const refresh = useRefresh();
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState(initialCategory);
  const [filter, setFilter] = useState<(typeof FILTERS)[number]["key"]>("all");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [busy, setBusy] = useState<string | null>(null);
  const [message, setMessage] = useState<{ kind: "ok" | "bad"; text: string } | null>(null);
  const sort = useSort<Key>("category", "asc");

  const all = data?.rows ?? [];
  const run = data?.run ?? null;
  const categories = useMemo(() => [...new Set(all.map((r) => r.category))], [all]);
  const rows = useMemo(() => {
    const q = query.trim().toLowerCase();
    const r = all.filter((p) => {
      if (category && p.category !== category) return false;
      if (q && !`${p.name} ${p.sku} ${p.brand}`.toLowerCase().includes(q)) return false;
      if (filter === "pending") return p.status === "pending";
      if (filter === "decided") return p.status !== "pending";
      if (filter !== "all") return p.action === filter;
      return true;
    });
    return sortRows(r, sort.key, sort.dir);
  }, [all, query, category, filter, sort.key, sort.dir]);

  const running = status.progress?.running;
  const count = (a: string) => all.filter((r) => r.action === a).length;
  const pending = all.filter((r) => r.status === "pending");
  const decided = all.length - pending.length;
  const gain = all.reduce((a, r) => a + Math.max(r.profit_gain, 0), 0);
  const newData = status.next_price_date && run && status.next_price_date > run.price_date;

  const act = async (skus: string[], decision: "approve" | "reject" | "reset") => {
    const user = getUser();
    if (!user) {
      setMessage({ kind: "bad", text: "Enter your name at the top right first; every approval is signed." });
      return;
    }
    setBusy(decision);
    setMessage(null);
    try {
      const res = await post<{ updated: number; failed: { sku: string; error: string }[] }>("/api/recommendations/decisions", {
        skus, decision, user, run_id: run?.run_id,
      });
      const verb = decision === "approve" ? "Approved" : decision === "reject" ? "Rejected" : "Reset";
      setMessage({
        kind: res.failed.length ? "bad" : "ok",
        text: `${verb} ${res.updated} ${res.updated === 1 ? "price" : "prices"}.` + (res.failed.length ? ` ${res.failed.length} failed: ${res.failed[0]?.error}` : ""),
      });
      setSelected(new Set());
      refresh();
    } catch (e) {
      setMessage({ kind: "bad", text: e instanceof ApiError ? e.message : "Could not save the decision." });
    } finally {
      setBusy(null);
    }
  };

  const recalc = async () => {
    setMessage(null);
    try {
      await post("/api/runs", { user: getUser() || "web" });
      refresh();
    } catch (e) {
      setMessage({ kind: "bad", text: e instanceof ApiError ? e.message : "Could not start the calculation." });
    }
  };

  const toggle = (sku: string) =>
    setSelected((s) => {
      const n = new Set(s);
      if (n.has(sku)) n.delete(sku);
      else n.add(sku);
      return n;
    });
  const visiblePending = rows.filter((r) => r.status === "pending").map((r) => r.sku);
  const allVisibleSelected = visiblePending.length > 0 && visiblePending.every((s) => selected.has(s));

  return (
    <>
      <PageHead
        title="Price recommendations"
        desc={
          run ? (
            <>
              Prices for <strong>{day(run.price_date, true)}</strong>, calculated {day(run.started_at)} at {time(run.finished_at ?? run.started_at)} from sales up to {day(status.sales_to)}. Each
              recommendation weighs cost and margin, how demand reacts to price, the season and festivals ahead, stock and expiry, the competitor and the allowed price range. Nothing changes until you approve it.
            </>
          ) : (
            "No prices calculated yet."
          )
        }
        actions={
          <>
            {run && (
              <a className="btn" href={`${BASE}/api/export/prices.csv?run_id=${run.run_id}`} download>
                Export price list
              </a>
            )}
            <button className="btn btn--primary" onClick={recalc} disabled={Boolean(running)}>
              {running ? `Calculating… ${status.progress?.done ?? 0}/${status.progress?.total ?? 0}` : newData ? `Calculate prices for ${day(status.next_price_date)}` : "Recalculate"}
            </button>
          </>
        }
      />
      {running && <div className="callout" style={{ marginBottom: 16 }}>{status.progress?.phase}: {status.progress?.done ?? 0} of {status.progress?.total ?? 0} products.</div>}
      {newData && !running && (
        <div className="callout callout--warn" style={{ marginBottom: 16 }}>
          New sales data has arrived since these prices were calculated. Calculate prices for {day(status.next_price_date)} to use it.
        </div>
      )}
      {message && <div className={`callout${message.kind === "bad" ? " callout--bad" : " callout--good"}`} style={{ marginBottom: 16 }}>{message.text}</div>}
      <LoadState error={error} loading={loading}>
        {data && !run && (
          <Panel>
            <div className="empty">
              {status.products ? (
                <>No prices calculated yet. Choose <strong>Recalculate</strong> above to price all {status.products} products.</>
              ) : (
                <>
                  There is no data yet. Start on the <a href={href("data")}>Data</a> page: import your product list, then at least a few months of daily sales.
                </>
              )}
            </div>
          </Panel>
        )}
        {run && (
          <div className="stack">
            <Figures
              items={[
                { label: "Raise the price", value: num(count("increase")), sub: "products" },
                { label: "Cut the price", value: num(count("decrease")), sub: "products" },
                { label: "Keep the price", value: num(count("hold")), sub: "already at the best allowed price" },
                { label: "Expected extra profit", value: rupeesShort(gain), sub: "if all are approved, over each plan period" },
                { label: "Reviewed", value: `${num(decided)} of ${num(all.length)}`, sub: pending.length ? `${num(pending.length)} waiting for a decision` : "all decided" },
              ]}
            />
            {typeof run.errors !== "string" && run.errors.length > 0 && (
              <div className="callout callout--warn">
                {run.errors.length} {run.errors.length === 1 ? "product" : "products"} could not be priced: {run.errors.slice(0, 3).map((e) => `${e.sku} (${e.error})`).join("; ")}
              </div>
            )}
            <Panel
              flush
              toolbar={
                <div className="toolbar">
                  <input type="search" placeholder="Search name, SKU or brand" value={query} onChange={(e) => setQuery(e.target.value)} aria-label="Search products" />
                  <label>
                    Category
                    <select value={category} onChange={(e) => setCategory(e.target.value)}>
                      <option value="">All categories</option>
                      {categories.map((c) => (
                        <option key={c}>{c}</option>
                      ))}
                    </select>
                  </label>
                  <div className="seg" role="group" aria-label="Show">
                    {FILTERS.map((f) => (
                      <button key={f.key} aria-pressed={filter === f.key} onClick={() => setFilter(f.key)}>
                        {f.label}
                      </button>
                    ))}
                  </div>
                  <span className="toolbar__end">
                    {selected.size > 0 ? (
                      <>
                        <span className="panel__note">{selected.size} selected</span>
                        <button className="btn btn--primary" disabled={busy !== null} onClick={() => act([...selected], "approve")}>
                          Approve {selected.size}
                        </button>
                        <button className="btn" disabled={busy !== null} onClick={() => act([...selected], "reject")}>
                          Reject
                        </button>
                        <button className="btn" onClick={() => setSelected(new Set())}>
                          Clear
                        </button>
                      </>
                    ) : (
                      <span className="panel__note">
                        {rows.length} of {all.length} products
                      </span>
                    )}
                  </span>
                </div>
              }
            >
              <div className="table-wrap">
                <table className="data">
                  <thead>
                    <tr>
                      <th style={{ width: 28 }}>
                        <input
                          type="checkbox"
                          aria-label="Select all products waiting for review"
                          checked={allVisibleSelected}
                          disabled={visiblePending.length === 0}
                          onChange={() => setSelected(allVisibleSelected ? new Set() : new Set(visiblePending))}
                        />
                      </th>
                      <Th k="name" label="Product" sort={sort} />
                      <Th k="current_price" label="Current" sort={sort} num />
                      <Th k="change_pct" label="Recommendation" sort={sort} />
                      <Th k="recommended_price" label="New price" sort={sort} num />
                      <Th k="profit_gain" label="Extra profit" sort={sort} num />
                      <Th k="main_driver" label="Main reason" sort={sort} />
                      <Th k="on_hand" label="Stock" sort={sort} num />
                      <Th k="status" label="Decision" sort={sort} />
                      <th className="num">
                        <span className="sr-only">Actions</span>
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((r, i) => (
                      <Fragment key={r.sku}>
                        {sort.key === "category" && r.category !== rows[i - 1]?.category && (
                          <tr className="group">
                            <td colSpan={10}>{r.category}</td>
                          </tr>
                        )}
                        <tr className={selected.has(r.sku) ? "is-selected" : undefined}>
                          <td>
                            <input type="checkbox" aria-label={`Select ${r.name}`} checked={selected.has(r.sku)} disabled={r.status !== "pending"} onChange={() => toggle(r.sku)} />
                          </td>
                          <td><ProductLink sku={r.sku} name={r.name} /></td>
                          <td className="num">{rupees(r.current_price)}</td>
                          <td><ActionTag action={r.action} pct={r.change_pct} /></td>
                          <td className="num strong">
                            {rupees(r.recommended_price)}
                            {r.target_price !== null && Math.abs(r.target_price - r.recommended_price) >= 1 && (
                              <div className="sku" title="Where the price is heading once the daily move limit allows">heading to {rupees(r.target_price)}</div>
                            )}
                          </td>
                          <td className={`num${r.profit_gain >= 1 ? " up" : r.profit_gain <= -1 ? " down" : " muted"}`} title={r.profit_gain <= -1 ? "The current price is below the loss floor; selling at a loss is not allowed" : undefined}>
                            {r.profit_gain >= 1 ? `+${rupees(r.profit_gain)}` : r.profit_gain <= -1 ? `−${rupees(-r.profit_gain)}` : "–"}
                            <div className="sku">{r.horizon_days}-day plan</div>
                          </td>
                          <td className="muted">
                            {r.main_driver ? `${DRIVER[r.main_driver] ?? r.main_driver} ${r.main_driver_up ? "push up" : "push down"}` : "–"}
                            {r.binding !== "None" && (
                              <div className="sku">
                                held at <BindingTag value={r.binding} />
                              </div>
                            )}
                          </td>
                          <td className="num">
                            {num(r.on_hand)}
                            {r.expiring_3d > 0 ? (
                              <div className="sku down">{num(r.expiring_3d)} expiring</div>
                            ) : r.days_of_cover !== null ? (
                              <div className="sku">{num1(r.days_of_cover)} days</div>
                            ) : null}
                          </td>
                          <td>
                            <StatusTag status={r.status} price={r.final_price} by={r.decided_by} />
                          </td>
                          <td className="num">
                            {r.status === "pending" ? (
                              <span className="rowbtns">
                                <button className="btn btn--sm btn--primary" disabled={busy !== null} onClick={() => act([r.sku], "approve")}>
                                  Approve
                                </button>
                                <button className="btn btn--sm" disabled={busy !== null} onClick={() => act([r.sku], "reject")}>
                                  Reject
                                </button>
                              </span>
                            ) : (
                              <button className="linkbtn" disabled={busy !== null} onClick={() => act([r.sku], "reset")}>
                                Undo
                              </button>
                            )}
                          </td>
                        </tr>
                      </Fragment>
                    ))}
                  </tbody>
                </table>
                {rows.length === 0 && <div className="empty">No products match these filters.</div>}
              </div>
            </Panel>
          </div>
        )}
      </LoadState>
    </>
  );
}
