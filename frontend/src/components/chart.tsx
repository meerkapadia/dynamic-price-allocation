// Shared Recharts styling: quiet grid, small muted ticks, a plain white tooltip.
import type { ReactNode } from "react";
import type { TooltipProps } from "recharts";

export const C = {
  accent: "#1f4e79",
  accentLight: "#a9c0d8",
  good: "#1e7b4f",
  bad: "#b42318",
  rival: "#a15c07",
  grey: "#8a93a3",
  greyLight: "#d5dae1",
  grid: "#eceef2",
};

export const axis = {
  tick: { fontSize: 11.5, fill: "#5d6676" },
  tickLine: false,
  axisLine: { stroke: "#c8cdd5" },
} as const;

export const yAxis = { ...axis, axisLine: false, width: 56 } as const;

export function TipBox({ title, rows }: { title: ReactNode; rows: { label: string; value: ReactNode; color?: string }[] }) {
  return (
    <div style={{ background: "#fff", border: "1px solid #c8cdd5", borderRadius: 4, padding: "7px 10px", fontSize: 12.5, boxShadow: "0 2px 6px rgba(27,34,48,.08)" }}>
      <div style={{ fontWeight: 600, marginBottom: 3 }}>{title}</div>
      {rows.map((r) => (
        <div key={r.label} style={{ display: "flex", justifyContent: "space-between", gap: 16 }}>
          <span style={{ color: "#5d6676" }}>
            {r.color && <i style={{ display: "inline-block", width: 8, height: 8, background: r.color, marginRight: 6, borderRadius: 1 }} />}
            {r.label}
          </span>
          <span style={{ fontVariantNumeric: "tabular-nums" }}>{r.value}</span>
        </div>
      ))}
    </div>
  );
}

/** Tooltip content builder: map a payload row to labelled values. */
export function tip<T>(title: (row: T) => ReactNode, rows: (row: T) => { label: string; value: ReactNode; color?: string }[]) {
  return function Content(props: TooltipProps<number, string>) {
    const row = props.payload?.[0]?.payload as T | undefined;
    if (!props.active || !row) return null;
    return <TipBox title={title(row)} rows={rows(row)} />;
  };
}

export function Legend({ items }: { items: { label: string; color: string; box?: boolean; dashed?: boolean }[] }) {
  return (
    <div className="chart-legend">
      {items.map((i) => (
        <span key={i.label}>
          <i
            className={i.box ? "box" : undefined}
            style={i.dashed ? { borderTop: `2px dashed ${i.color}`, height: 0 } : { background: i.color }}
          />
          {i.label}
        </span>
      ))}
    </div>
  );
}
