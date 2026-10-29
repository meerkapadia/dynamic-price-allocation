import { useState } from "react";
import { LoadState, PageHead, Pager, Panel } from "../components/ui";
import { ApiError, postCsv, type DataKind, type ImportProblem, type Paged } from "../lib/api";
import { num } from "../lib/format";
import { useApi, useRefresh } from "../lib/hooks";

const TABS: { kind: DataKind["kind"]; label: string; add: string }[] = [
  { kind: "products", label: "Products", add: "Add or update a product" },
  { kind: "stock", label: "Stock on hand", add: "Add a stock batch (goods received)" },
  { kind: "sales", label: "Daily sales", add: "Record a day's sales" },
  { kind: "competitor", label: "Competitor prices", add: "Record a competitor price" },
  { kind: "write_offs", label: "Write-offs", add: "Record a write-off" },
];

const HINT: Record<string, string> = {
  date: "YYYY-MM-DD", received_on: "YYYY-MM-DD (default today)", expires_on: "YYYY-MM-DD (default from shelf life)",
  storage_class: "ambient, chilled or frozen", reason: "expired, spoiled or damaged", stocked_out: "1 if it sold out",
  promo: "1 if on offer", salvage_pct: "e.g. -0.05 (disposal fee) or 0.1", min_margin_pct: "e.g. 0.05 = 5%",
  max_daily_change_pct: "e.g. 0.10 = 10%",
};

function csvCell(v: string): string {
  return /[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v;
}

function AddForm({ spec, onDone }: { spec: DataKind; onDone: () => void }) {
  const cols = [...spec.required, ...spec.optional];
  const [vals, setVals] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ ok: boolean; text: string; errors?: ImportProblem[] } | null>(null);
  const save = async () => {
    setBusy(true);
    setMsg(null);
    const csv = `${cols.join(",")}\n${cols.map((c) => csvCell((vals[c] ?? "").trim())).join(",")}\n`;
    try {
      await postCsv(`/api/import/${spec.kind}${spec.kind === "stock" ? "?mode=append" : ""}`, csv);
      setMsg({ ok: true, text: "Saved." });
      setVals({});
      onDone();
    } catch (e) {
      setMsg({ ok: false, text: e instanceof ApiError ? e.message : "Could not save.", errors: e instanceof ApiError ? e.errors : undefined });
    } finally {
      setBusy(false);
    }
  };
  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        void save();
      }}
    >
      <div className="formgrid">
        {cols.map((c) => (
          <label key={c} className="formfield">
            <span>
              {c.replace(/_/g, " ")}
              {spec.required.includes(c) ? " *" : ""}
            </span>
            <input value={vals[c] ?? ""} placeholder={HINT[c] ?? ""} onChange={(e) => setVals((v) => ({ ...v, [c]: e.target.value }))} />
          </label>
        ))}
      </div>
      <div className="decide" style={{ borderTop: 0, paddingTop: 0 }}>
        <button className="btn btn--primary" type="submit" disabled={busy}>
          {busy ? "Saving…" : "Save"}
        </button>
        <span className="muted" style={{ fontSize: 12.5 }}>* required. Saved records are used the next time prices are calculated.</span>
      </div>
      {msg && (
        <div className={`callout ${msg.ok ? "callout--good" : "callout--bad"}`} style={{ marginTop: 10 }}>
          {msg.text}
          {msg.errors && (
            <ul className="errors">
              {msg.errors.map((er, i) => (
                <li key={i}>
                  {er.column ?? "row"}: {er.message}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </form>
  );
}

export function DatabasePage() {
  const [tab, setTab] = useState<DataKind["kind"]>("products");
  const [q, setQ] = useState("");
  const [page, setPage] = useState(1);
  const [adding, setAdding] = useState(false);
  const refresh = useRefresh();
  const specs = useApi<DataKind[]>("/api/data");
  const qs = new URLSearchParams({ page: String(page), page_size: "50", ...(q.trim() && { q: q.trim() }) });
  const { data, error, loading } = useApi<Paged<Record<string, string | number | null>>>(`/api/records/${tab}?${qs}`);
  const spec = specs.data?.find((s) => s.kind === tab);
  const t = TABS.find((x) => x.kind === tab)!;
  const cols = data?.rows[0] ? Object.keys(data.rows[0]) : [];

  return (
    <>
      <PageHead
        title="Database"
        desc="Everything the prices are worked out from, table by table. Add records by hand here, or load whole files under Import &amp; export."
        actions={
          <button className="btn btn--primary" onClick={() => setAdding((a) => !a)}>
            {adding ? "Close form" : t.add}
          </button>
        }
      />
      <div className="seg" role="tablist" style={{ marginBottom: 14 }}>
        {TABS.map((x) => (
          <button
            key={x.kind}
            aria-pressed={tab === x.kind}
            onClick={() => {
              setTab(x.kind);
              setPage(1);
              setAdding(false);
            }}
          >
            {x.label}
          </button>
        ))}
      </div>
      <div className="stack">
        {adding && spec && (
          <Panel title={t.add}>
            <AddForm spec={spec} onDone={refresh} />
          </Panel>
        )}
        <LoadState error={error} loading={loading && !data}>
          <Panel
            flush
            toolbar={
              <div className="toolbar">
                <input type="search" placeholder="Search SKU or name" value={q} onChange={(e) => { setQ(e.target.value); setPage(1); }} aria-label="Search" />
                {data && <span className="toolbar__end panel__note">{num(data.total)} records</span>}
              </div>
            }
          >
            {data && data.total === 0 ? (
              <div className="empty">No records yet. Use “{t.add}” above or import a file under Import &amp; export.</div>
            ) : (
              data && (
                <div className={loading ? "is-loading" : undefined}>
                  <div className="table-wrap">
                    <table className="data">
                      <thead>
                        <tr>
                          {cols.map((c) => (
                            <th key={c} className={typeof data.rows[0]?.[c] === "number" ? "num" : undefined}>
                              {c.replace(/_/g, " ")}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {data.rows.map((r, i) => (
                          <tr key={i}>
                            {cols.map((c) => (
                              <td key={c} className={typeof r[c] === "number" ? "num" : undefined}>
                                {r[c] === null ? <span className="muted">–</span> : typeof r[c] === "number" ? (Number.isInteger(r[c]) ? num(r[c] as number) : (r[c] as number).toFixed(2)) : r[c]}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <Pager page={data.page} pageSize={data.page_size} total={data.total} onPage={setPage} unit="records" />
                </div>
              )
            )}
          </Panel>
        </LoadState>
      </div>
    </>
  );
}
