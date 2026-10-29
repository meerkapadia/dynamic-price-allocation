import { useRef, useState } from "react";
import { LoadState, PageHead, Panel } from "../components/ui";
import { ApiError, BASE, postCsv, type DataKind, type ImportProblem, type Status } from "../lib/api";
import { day, num } from "../lib/format";
import { useApi, useRefresh } from "../lib/hooks";
import { getApiKey, setApiKey } from "../lib/user";

const ORDER: DataKind["kind"][] = ["products", "sales", "stock", "competitor", "write_offs"];

function ImportCard({ k }: { k: DataKind }) {
  const refresh = useRefresh();
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<{ ok: boolean; text: string; errors?: ImportProblem[] } | null>(null);

  const upload = async (file: File | undefined) => {
    if (!file) return;
    setBusy(true);
    setResult(null);
    try {
      const text = await file.text();
      const res = await postCsv<{ rows: number; written: number }>(`/api/import/${k.kind}`, text);
      setResult({ ok: true, text: `Imported ${num(res.written)} rows from ${file.name}.` });
      refresh();
    } catch (e) {
      setResult({ ok: false, text: e instanceof ApiError ? e.message : "The upload failed.", errors: e instanceof ApiError ? e.errors : undefined });
    } finally {
      setBusy(false);
      if (input.current) input.current.value = "";
    }
  };

  return (
    <Panel title={k.label} note={k.rows ? `${num(k.rows)} rows${k.from ? `, ${day(k.from)} to ${day(k.to)}` : ""}` : "No data yet"}>
      <p style={{ fontSize: 13, color: "var(--ink-2)", marginBottom: 8 }}>{k.help}</p>
      <p style={{ fontSize: 12.5, color: "var(--muted)", marginBottom: 12 }}>
        Columns: <strong>{k.required.join(", ")}</strong>
        {k.optional.length > 0 && <>; optional {k.optional.join(", ")}</>}.{" "}
        <a href={`${BASE}/api/import/${k.kind}/template.csv`} download>
          Download template
        </a>
      </p>
      <div
        className={`drop${over ? " is-over" : ""}`}
        role="button"
        tabIndex={0}
        onClick={() => input.current?.click()}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          void upload(e.dataTransfer.files[0]);
        }}
      >
        {busy ? "Checking and importing…" : "Drop a CSV file here, or click to choose one"}
      </div>
      <input ref={input} type="file" accept=".csv,text/csv" hidden onChange={(e) => void upload(e.target.files?.[0])} />
      {result && (
        <div className={`callout ${result.ok ? "callout--good" : "callout--bad"}`} style={{ marginTop: 10 }}>
          {result.text}
          {result.errors && result.errors.length > 0 && (
            <ul className="errors">
              {result.errors.map((er, i) => (
                <li key={i}>
                  Line {er.line}
                  {er.column ? `, ${er.column}` : ""}: {er.message}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
      {k.last_import && <p style={{ fontSize: 12, color: "var(--muted)", marginTop: 8 }}>Last imported {day(k.last_import)}.</p>}
    </Panel>
  );
}

function KeyBox() {
  const [key, setKey] = useState(getApiKey());
  const [saved, setSaved] = useState(false);
  return (
    <Panel title="Access key">
      <p style={{ fontSize: 13, color: "var(--ink-2)", marginBottom: 10 }}>This server requires the store's API key to calculate prices, approve them and import data. It is kept in this browser only.</p>
      <form
        className="decide"
        style={{ borderTop: 0, marginTop: 0, paddingTop: 0 }}
        onSubmit={(e) => {
          e.preventDefault();
          setApiKey(key);
          setSaved(true);
        }}
      >
        <input className="userbox__input" type="password" value={key} onChange={(e) => { setKey(e.target.value); setSaved(false); }} aria-label="API key" style={{ width: 260, marginTop: 0 }} />
        <button className="btn btn--primary" type="submit">
          Save key
        </button>
        {saved && <span className="up">Saved</span>}
      </form>
    </Panel>
  );
}

export function DataPage({ status }: { status: Status }) {
  const { data, error, loading } = useApi<DataKind[]>("/api/data");
  const kinds = data ? ORDER.map((o) => data.find((d) => d.kind === o)).filter((x): x is DataKind => Boolean(x)) : [];
  return (
    <>
      <PageHead
        title="Data"
        desc="Everything the recommendations are worked out from. Load your store's files here (or have your systems post them to the same endpoints every night), then calculate prices. Each file is checked as a whole: if any row is wrong, nothing is imported and the problems are listed."
        actions={
          <a className="btn" href={`${BASE}/api/export/prices.csv`} download>
            Export latest price list
          </a>
        }
      />
      {status.demo && (
        <div className="callout" style={{ marginBottom: 16 }}>
          You are looking at the sample store. To price your own products, start the server with an empty database (<code>PRICING_DEMO_DATA=0</code>, <code>PRICING_DB_PATH=./data/mystore.db</code>), then import the files below in order: products, sales, stock, competitor prices.
        </div>
      )}
      <LoadState error={error} loading={loading}>
        <div className="stack">
          {status.api_key_required && <KeyBox />}
          <div className="cols cols--even">
            {kinds.map((k) => (
              <ImportCard key={k.kind} k={k} />
            ))}
          </div>
        </div>
      </LoadState>
    </>
  );
}
