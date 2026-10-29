import type { ReactNode } from "react";
import { href } from "../lib/hooks";

export function PageHead({ title, desc, crumbs, actions }: { title: ReactNode; desc?: ReactNode; crumbs?: { label: string; to: string }[]; actions?: ReactNode }) {
  return (
    <header className="page-head">
      <div>
        {crumbs && crumbs.length > 0 && (
          <nav className="crumbs" aria-label="Breadcrumb">
            {crumbs.map((c, i) => (
              <span key={c.to}>
                {i > 0 && " / "}
                <a href={c.to}>{c.label}</a>
              </span>
            ))}
          </nav>
        )}
        <h1 className="page-title">{title}</h1>
        {desc && <p className="page-desc">{desc}</p>}
      </div>
      {actions && <div className="page-actions">{actions}</div>}
    </header>
  );
}

export function Panel({ title, note, children, flush, toolbar, id }: { title?: ReactNode; note?: ReactNode; children: ReactNode; flush?: boolean; toolbar?: ReactNode; id?: string }) {
  return (
    <section className="panel" id={id} aria-label={typeof title === "string" ? title : undefined}>
      {(title || note) && (
        <div className="panel__head">
          {title && <h2 className="panel__title">{title}</h2>}
          {note && <span className="panel__note">{note}</span>}
        </div>
      )}
      {toolbar}
      <div className={flush ? "panel__body panel__body--flush" : "panel__body"}>{children}</div>
    </section>
  );
}

export function Figures({ items }: { items: { label: string; value: ReactNode; sub?: ReactNode }[] }) {
  return (
    <div className="figures">
      {items.map((it) => (
        <div className="figure" key={it.label}>
          <div className="figure__label">{it.label}</div>
          <div className="figure__value">{it.value}</div>
          {it.sub !== undefined && <div className="figure__sub">{it.sub}</div>}
        </div>
      ))}
    </div>
  );
}

const BINDING_CLASS: Record<string, string> = {
  "Loss floor": "tag tag--floor",
  "MRP ceiling": "tag tag--mrp",
  "Daily move limit (up)": "tag tag--limit",
  "Daily move limit (down)": "tag tag--limit",
};

export function BindingTag({ value }: { value: string | null | undefined }) {
  if (!value) return <span className="muted">–</span>;
  if (value === "None") return <span className="muted">Unconstrained</span>;
  return <span className={BINDING_CLASS[value] ?? "tag"}>{value}</span>;
}

export function ProductLink({ sku, name }: { sku: string; name: string }) {
  return (
    <a href={href("products", sku)} className="name-cell">
      <span>{name}</span>
      <span className="sku">{sku}</span>
    </a>
  );
}

export function Pager({ page, pageSize, total, onPage, unit = "rows" }: { page: number; pageSize: number; total: number; onPage: (p: number) => void; unit?: string }) {
  const pages = Math.max(1, Math.ceil(total / pageSize));
  const from = total === 0 ? 0 : (page - 1) * pageSize + 1;
  const to = Math.min(total, page * pageSize);
  return (
    <div className="pager">
      <span>
        {from.toLocaleString("en-IN")}–{to.toLocaleString("en-IN")} of {total.toLocaleString("en-IN")} {unit}
      </span>
      <span className="pager__btns">
        <button className="btn" disabled={page <= 1} onClick={() => onPage(page - 1)}>
          Previous
        </button>
        <button className="btn" disabled={page >= pages} onClick={() => onPage(page + 1)}>
          Next
        </button>
      </span>
    </div>
  );
}

export function LoadState({ error, loading, children }: { error: string | null; loading: boolean; children: ReactNode }) {
  if (error) return <div className="callout callout--bad">{error}</div>;
  if (loading && !children) return <div className="empty">Loading…</div>;
  return <>{children}</>;
}

export function Th<K extends string>({ k, label, sort, num }: { k: K; label: string; sort: { toggle: (k: K) => void; aria: (k: K) => "ascending" | "descending" | undefined }; num?: boolean }) {
  return (
    <th className={`sortable${num ? " num" : ""}`} aria-sort={sort.aria(k)} onClick={() => sort.toggle(k)} scope="col">
      {label}
    </th>
  );
}

export function ActionTag({ action, pct }: { action: "increase" | "decrease" | "hold" | null | undefined; pct?: number | null }) {
  if (!action) return <span className="muted">–</span>;
  const n = pct === null || pct === undefined ? "" : ` ${Math.abs(pct).toFixed(1)}%`;
  if (action === "increase") return <span className="act act--up">Increase{n}</span>;
  if (action === "decrease") return <span className="act act--down">Decrease{n}</span>;
  return <span className="act act--hold">Hold</span>;
}

export function StatusTag({ status, price, by }: { status: string; price?: number | null; by?: string | null }) {
  if (status === "pending") return <span className="muted">To review</span>;
  const label = status === "approved" ? "Approved" : status === "overridden" ? "Changed" : "Rejected";
  const cls = status === "rejected" ? "tag" : "tag tag--good";
  return (
    <span className="name-cell">
      <span>
        <span className={cls}>{label}</span>
        {status === "overridden" && price ? <span className="strong"> ₹{Math.round(price).toLocaleString("en-IN")}</span> : null}
      </span>
      {by && <span className="sku">by {by}</span>}
    </span>
  );
}
