import { useState } from "react";
import { ActionTag, LoadState, PageHead, Pager, Panel, ProductLink, StatusTag } from "../components/ui";
import type { Paged, PriceChange } from "../lib/api";
import { day, num, rupees } from "../lib/format";
import { useApi } from "../lib/hooks";

const STATUSES: [string, string][] = [
  ["", "All decisions"],
  ["approved", "Approved"],
  ["overridden", "Changed by a person"],
  ["rejected", "Rejected"],
];

export function ChangesPage() {
  const [status, setStatus] = useState("");
  const [page, setPage] = useState(1);
  const qs = new URLSearchParams({ page: String(page), page_size: "50", ...(status && { status }) });
  const { data, error, loading } = useApi<Paged<PriceChange>>(`/api/price-changes?${qs}`);
  return (
    <>
      <PageHead title="Price changes" desc="Every decision on a recommendation: who made it, when, and the price that went to the price list." />
      <LoadState error={error} loading={loading && !data}>
        <Panel
          flush
          toolbar={
            <div className="toolbar">
              <div className="seg" role="group" aria-label="Decision">
                {STATUSES.map(([k, l]) => (
                  <button
                    key={k}
                    aria-pressed={status === k}
                    onClick={() => {
                      setStatus(k);
                      setPage(1);
                    }}
                  >
                    {l}
                  </button>
                ))}
              </div>
              {data && <span className="toolbar__end panel__note">{num(data.total)} decisions</span>}
            </div>
          }
        >
          {data && data.total === 0 && <div className="empty">No decisions yet. Approve or reject recommendations on the Price recommendations page.</div>}
          {data && data.total > 0 && (
            <div className={loading ? "is-loading" : undefined}>
              <div className="table-wrap">
                <table className="data">
                  <thead>
                    <tr>
                      <th>Decided</th>
                      <th>Product</th>
                      <th>Prices for</th>
                      <th className="num">Was</th>
                      <th>Recommended</th>
                      <th className="num">Set to</th>
                      <th>Decision</th>
                      <th>Note</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.rows.map((r) => (
                      <tr key={`${r.run_id}-${r.sku}`}>
                        <td>
                          {day(r.decided_at)} <span className="muted">{r.decided_at?.slice(11, 16)}</span>
                        </td>
                        <td><ProductLink sku={r.sku} name={r.name} /></td>
                        <td>{day(r.price_date)}</td>
                        <td className="num muted">{rupees(r.current_price)}</td>
                        <td>
                          <ActionTag action={r.action} pct={r.action === "hold" ? null : (r.recommended_price / r.current_price - 1) * 100} /> {rupees(r.recommended_price)}
                        </td>
                        <td className="num strong">{r.final_price ? rupees(r.final_price) : <span className="muted">unchanged</span>}</td>
                        <td><StatusTag status={r.status} price={r.final_price} by={r.decided_by} /></td>
                        <td className="wrap muted">{r.note ?? ""}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <Pager page={data.page} pageSize={data.page_size} total={data.total} onPage={setPage} unit="decisions" />
            </div>
          )}
        </Panel>
      </LoadState>
    </>
  );
}
