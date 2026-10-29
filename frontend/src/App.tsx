import { useCallback, useEffect, useMemo, useState } from "react";
import { get, type Status } from "./lib/api";
import { day } from "./lib/format";
import { DataVersion, href, useRoute } from "./lib/hooks";
import { getUser, setUser } from "./lib/user";
import { ChangesPage } from "./pages/Changes";
import { DataPage } from "./pages/Data";
import { DatabasePage } from "./pages/Database";
import { MethodPage } from "./pages/Method";
import { PricingPage } from "./pages/Pricing";
import { ProductPage } from "./pages/Product";

const NAV = [
  { key: "", label: "Price recommendations" },
  { key: "changes", label: "Price changes" },
  { key: "database", label: "Database" },
  { key: "data", label: "Import & export" },
  { key: "method", label: "How it works" },
];

function useStatus(bump: number): { status: Status | null; offline: string | null } {
  const [status, setStatus] = useState<Status | null>(null);
  const [offline, setOffline] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    let timer: number | undefined;
    const tick = async () => {
      let next = 15000;
      try {
        const s = await get<Status>("/api/status");
        if (!alive) return;
        setStatus(s);
        setOffline(null);
        if (!s.ready || s.progress?.running) next = 1500;
      } catch (e) {
        if (alive) setOffline((e as Error).message);
        next = 4000;
      }
      if (alive) timer = window.setTimeout(tick, next);
    };
    void tick();
    return () => {
      alive = false;
      window.clearTimeout(timer);
    };
  }, [bump]);
  return { status, offline };
}

function UserBox() {
  const [name, setName] = useState(getUser());
  const [editing, setEditing] = useState(!getUser());
  if (!editing && name) {
    return (
      <div className="userbox">
        <span className="clock__label">Approving as</span>
        <button className="linkbtn" onClick={() => setEditing(true)} title="Change name">
          {name}
        </button>
      </div>
    );
  }
  return (
    <form
      className="userbox"
      onSubmit={(e) => {
        e.preventDefault();
        setUser(name);
        setEditing(!name.trim());
      }}
    >
      <label className="clock__label" htmlFor="user-name">
        Your name, for approvals
      </label>
      <input id="user-name" className="userbox__input" value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Asha Kulkarni" onBlur={() => { setUser(name); setEditing(!name.trim()); }} />
    </form>
  );
}

export default function App() {
  const route = useRoute();
  const [bump, setBump] = useState(0);
  const { status, offline } = useStatus(bump);
  const refresh = useCallback(() => setBump((b) => b + 1), []);
  const runId = status?.last_run?.status === "done" ? status.last_run.run_id : 0;
  const ctx = useMemo(() => ({ version: runId * 100_000 + bump, refresh }), [runId, bump, refresh]);
  const page = route[0] ?? "";

  if (!status?.ready) {
    return (
      <div className="boot">
        <h1>{status?.store ?? "Bazaarline"} pricing</h1>
        {offline ? (
          <p>{offline}</p>
        ) : status?.error ? (
          <p>Startup failed: {status.error}</p>
        ) : (
          <>
            <p>{status?.phase ?? "Connecting to the pricing service"}…</p>
            <div className="progress" aria-hidden="true">
              <span />
            </div>
          </>
        )}
      </div>
    );
  }

  let content;
  switch (page) {
    case "products":
      content = route[1] ? <ProductPage sku={route[1]} /> : <PricingPage status={status} />;
      break;
    case "changes":
      content = <ChangesPage />;
      break;
    case "database":
      content = <DatabasePage />;
      break;
    case "data":
      content = <DataPage status={status} />;
      break;
    case "method":
      content = <MethodPage />;
      break;
    default:
      content = <PricingPage status={status} />;
  }
  const navKey = page === "products" ? "" : page;

  return (
    <DataVersion.Provider value={ctx}>
      <header className="topbar">
        <a className="brand" href={href()} style={{ textDecoration: "none", color: "inherit" }}>
          <span className="brand__mark" aria-hidden="true">B</span>
          <span className="brand__name">{status.store}</span>
          <span className="brand__sub">Pricing</span>
        </a>
        <nav className="topnav" aria-label="Sections">
          {NAV.map((n) => (
            <a key={n.key} className="topnav__link" href={href(...(n.key ? [n.key] : []))} aria-current={navKey === n.key ? "page" : undefined}>
              {n.label}
            </a>
          ))}
        </nav>
        <div className="topbar__spacer" />
        <div className="clock">
          <div className="clock__item clock__item--hide">
            <span className="clock__label">Sales data to</span>
            <strong>{day(status.sales_to)}</strong>
          </div>
          <UserBox />
        </div>
      </header>
      <main className="main main--wide">
        {status.demo && page !== "data" && (
          <div className="demo-note">
            Sample store with two years of simulated history. Add records on the <a href={href("database")}>Database</a> page or import files under <a href={href("data")}>Import &amp; export</a>.
          </div>
        )}
        {content}
      </main>
    </DataVersion.Provider>
  );
}
