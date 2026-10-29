// Indian number formatting (lakh / crore grouping) and short date helpers.

const inr0 = new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 });
const inr2 = new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", minimumFractionDigits: 2, maximumFractionDigits: 2 });
const int = new Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 });
const dec1 = new Intl.NumberFormat("en-IN", { minimumFractionDigits: 1, maximumFractionDigits: 1 });

const ok = (x: number | null | undefined): x is number => typeof x === "number" && Number.isFinite(x);

export const rupees = (x: number | null | undefined) => (ok(x) ? inr0.format(x) : "–");
export const rupees2 = (x: number | null | undefined) => (ok(x) ? inr2.format(x) : "–");
export const num = (x: number | null | undefined) => (ok(x) ? int.format(x) : "–");
export const num1 = (x: number | null | undefined) => (ok(x) ? dec1.format(x) : "–");

/** ₹4.2 L, ₹2.11 Cr — how Indian business reports large sums. */
export function rupeesShort(x: number | null | undefined): string {
  if (!ok(x)) return "–";
  const a = Math.abs(x);
  const sign = x < 0 ? "−" : "";
  if (a >= 1e7) return `${sign}₹${(a / 1e7).toFixed(a >= 1e8 ? 1 : 2)} Cr`;
  if (a >= 1e5) return `${sign}₹${(a / 1e5).toFixed(a >= 1e6 ? 1 : 2)} L`;
  if (a >= 1e3) return `${sign}₹${(a / 1e3).toFixed(1)}k`;
  return `${sign}₹${a.toFixed(0)}`;
}

export function pct(x: number | null | undefined, digits = 1, signed = false): string {
  if (!ok(x)) return "–";
  const s = x.toFixed(digits);
  if (!signed) return `${s}%`;
  return x > 0.0001 ? `+${s}%` : x < -0.0001 ? `−${Math.abs(x).toFixed(digits)}%` : `${(0).toFixed(digits)}%`;
}

export function deltaClass(x: number | null | undefined, goodWhenUp = true): string {
  if (!ok(x) || Math.abs(x) < 0.05) return "flat";
  return (x > 0) === goodWhenUp ? "up" : "down";
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const DAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

function parse(iso: string): Date {
  const [y, m, d] = iso.slice(0, 10).split("-").map(Number);
  return new Date(Date.UTC(y ?? 1970, (m ?? 1) - 1, d ?? 1));
}

export function day(iso: string | null | undefined, withWeekday = false): string {
  if (!iso) return "–";
  const d = parse(iso);
  const s = `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
  return withWeekday ? `${DAYS[d.getUTCDay()]} ${s}` : s;
}

export function dayShort(iso: string): string {
  const d = parse(iso);
  return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]}`;
}

export function monthYear(iso: string): string {
  const d = parse(iso);
  return `${MONTHS[d.getUTCMonth()]} ${String(d.getUTCFullYear()).slice(2)}`;
}

export function daysBetween(a: string, b: string): number {
  return Math.round((parse(b).getTime() - parse(a).getTime()) / 86_400_000);
}

export function duration(seconds: number | null | undefined): string {
  if (!ok(seconds)) return "–";
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${String(s % 60).padStart(2, "0")}s`;
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, "0")}m`;
}

export const MONTH_NAMES = MONTHS;
