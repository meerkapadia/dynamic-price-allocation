import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { ApiError, get } from "./api";

// ------------------------------------------------------------------ hash router

export function useRoute(): string[] {
  const read = () => (window.location.hash.replace(/^#\/?/, "").split("?")[0] ?? "").split("/").filter(Boolean).map(decodeURIComponent);
  const [parts, setParts] = useState<string[]>(read);
  useEffect(() => {
    const on = () => {
      setParts(read());
      window.scrollTo(0, 0);
    };
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return parts;
}

export const href = (...parts: (string | number)[]) => `#/${parts.map((p) => encodeURIComponent(String(p))).join("/")}`;

// ------------------------------------------------------------------ data version (bumps after each run, approval or import)

export const DataVersion = createContext<{ version: number; refresh: () => void }>({ version: 0, refresh: () => undefined });

/** Call after a change (approval, import) so every open view reloads its data. */
export function useRefresh(): () => void {
  return useContext(DataVersion).refresh;
}

export interface Loaded<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
}

/** Fetches `path`, refetching whenever it changes or the data changes. Keeps old data while reloading. */
export function useApi<T>(path: string | null): Loaded<T> {
  const { version } = useContext(DataVersion);
  const [state, setState] = useState<Loaded<T>>({ data: null, error: null, loading: path !== null });
  useEffect(() => {
    if (path === null) return;
    const ctl = new AbortController();
    setState((s) => ({ ...s, loading: true }));
    get<T>(path, ctl.signal)
      .then((data) => setState({ data, error: null, loading: false }))
      .catch((e: unknown) => {
        if ((e as Error).name === "AbortError") return;
        setState((s) => ({ data: s.data, error: e instanceof ApiError ? e.message : "Could not load this page.", loading: false }));
      });
    return () => ctl.abort();
  }, [path, version]);
  return state;
}

export function useSort<K extends string>(initial: K, initialDir: "asc" | "desc" = "desc") {
  const [key, setKey] = useState<K>(initial);
  const [dir, setDir] = useState<"asc" | "desc">(initialDir);
  const toggle = useCallback(
    (k: K) => {
      if (k === key) setDir((d) => (d === "asc" ? "desc" : "asc"));
      else {
        setKey(k);
        setDir("desc");
      }
    },
    [key],
  );
  const aria = (k: K) => (k === key ? (dir === "asc" ? "ascending" : "descending") : undefined);
  return { key, dir, toggle, aria };
}

export function sortRows<T>(rows: T[], key: keyof T, dir: "asc" | "desc"): T[] {
  const m = dir === "asc" ? 1 : -1;
  return [...rows].sort((a, b) => {
    const x = a[key] as unknown;
    const y = b[key] as unknown;
    if (x === y) return 0;
    if (x === null || x === undefined) return 1;
    if (y === null || y === undefined) return -1;
    if (typeof x === "number" && typeof y === "number") return (x - y) * m;
    return String(x).localeCompare(String(y)) * m;
  });
}
