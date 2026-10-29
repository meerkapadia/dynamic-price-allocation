// Client for the pricing service API. Types mirror backend/api/main.py responses.
import { getApiKey } from "./user";

export class ApiError extends Error {
  constructor(public readonly status: number, message: string, public readonly errors?: ImportProblem[]) {
    super(message);
  }
}

export interface ImportProblem {
  line: number;
  column: string | null;
  message: string;
}

export const BASE = ((import.meta as unknown as { env?: Record<string, string> }).env?.VITE_API_BASE ?? "").replace(/\/$/, "");

async function send<T>(path: string, init: RequestInit, signal?: AbortSignal): Promise<T> {
  let res: Response;
  const key = getApiKey();
  try {
    res = await fetch(`${BASE}${path}`, {
      ...init,
      signal,
      headers: { Accept: "application/json", ...(key ? { "X-API-Key": key } : {}), ...(init.headers ?? {}) },
    });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new ApiError(0, "The pricing service is not reachable. Start the backend on port 8000.");
  }
  if (!res.ok) {
    let msg = `Request failed (${res.status})`;
    let errors: ImportProblem[] | undefined;
    try {
      const body = (await res.json()) as { detail?: unknown; errors?: ImportProblem[] };
      if (typeof body.detail === "string") msg = body.detail;
      errors = body.errors;
    } catch {
      /* keep generic message */
    }
    throw new ApiError(res.status, msg, errors);
  }
  return (await res.json()) as T;
}

export const get = <T,>(path: string, signal?: AbortSignal) => send<T>(path, { method: "GET" }, signal);
export const post = <T,>(path: string, body: unknown) =>
  send<T>(path, { method: "POST", body: JSON.stringify(body), headers: { "Content-Type": "application/json" } });
export const postCsv = <T,>(path: string, text: string) =>
  send<T>(path, { method: "POST", body: text, headers: { "Content-Type": "text/csv" } });

// ------------------------------------------------------------------ shapes

export type Action = "increase" | "decrease" | "hold";
export type RecStatus = "pending" | "approved" | "overridden" | "rejected";

export interface Run {
  run_id: number;
  price_date: string;
  started_at: string;
  finished_at: string | null;
  status: "running" | "done" | "failed";
  products: number;
  priced: number;
  errors: { sku: string | null; error: string }[] | string;
  model_fitted_on: string | null;
  requested_by: string;
  decided?: number;
}

export interface Status {
  store: string;
  version: string;
  ready: boolean;
  phase: string;
  error: string | null;
  api_key_required: boolean;
  daily_run_at: string | null;
  demo?: boolean;
  products?: number;
  sales_from?: string | null;
  sales_to?: string | null;
  sales_rows?: number;
  next_price_date?: string;
  model_fitted_on?: string | null;
  models_stale?: boolean;
  last_run?: Run | null;
  progress?: { running: boolean; run_id?: number; done?: number; total?: number; phase?: string };
}

export interface RecRow {
  sku: string;
  name: string;
  category: string;
  brand: string;
  mrp: number;
  shelf_life_days: number;
  run_id: number;
  price_date: string;
  current_price: number;
  recommended_price: number;
  floor: number;
  ceiling: number;
  competitor_price: number;
  action: Action;
  change_pct: number;
  profit_gain: number;
  binding: string;
  horizon_days: number;
  status: RecStatus;
  final_price: number | null;
  decided_by: string | null;
  decided_at: string | null;
  target_price: number | null;
  expected_units: number;
  on_hand: number;
  expiring_3d: number;
  sold_28d: number;
  main_driver: string | null;
  main_driver_up: boolean;
  days_of_cover: number | null;
}

export interface RecList {
  run: Run | null;
  rows: RecRow[];
}

export interface Factor {
  key: "economics" | "competitor" | "stock" | "range";
  label: string;
  delta: number;
  price_after: number;
  detail: string;
}

export interface Advice {
  action: "increase" | "decrease" | "hold";
  change_pct: number;
  change_abs: number;
  current_price: number;
  recommended_price: number;
  target_price: number;
  target_change_pct: number;
  factors: Factor[];
  units_today_current: number;
  units_today_new: number;
  profit_today_current: number;
  profit_today_new: number;
  horizon_days: number;
  horizon_mode: string;
  profit_horizon_hold: number | null;
  profit_horizon_new: number | null;
  profit_gain: number;
  unit_margin_new: number;
  unit_margin_current: number;
  full_unit_cost: number;
}

export interface DerivedInput {
  key: string;
  label: string;
  value: number | string;
  unit: string;
  source: string;
}

export interface KKT {
  price: number;
  lower_bound: number;
  upper_bound: number;
  gradient: number;
  mu_floor: number;
  mu_ceiling: number;
  active_constraints: string[];
  stationarity_residual: number;
  complementary_slackness: number;
  dual_feasible: boolean;
  primal_feasible: boolean;
}

export interface Explanation {
  curve: { price: number; demand_static: number; demand_with_response: number; demand_model: number; competitor_response_price: number; daily_profit: number }[];
  game: {
    our_share: number;
    competitor_share: number;
    competitor_cost_estimate: number;
    nash: [number, number];
    nash_profit: number;
    nash_iterations: number;
    stackelberg: [number, number];
    stackelberg_profit: number;
    predicted_competitor_reply: number;
    logit_price_sensitivity: number;
    market_size: number;
  };
  policy: { days: number[]; inventory: number[]; prices: number[][]; lot_size: number; solve_ms: number; expected_value: number; marginal_value_of_stock: number } | null;
  strategies: { name: string; revenue: number; profit: number; profit_p05: number; profit_p95: number; units_sold: number; units_wasted: number; sell_through_pct: number; avg_price: number; loss_making_units: number }[];
  objective: { price: number; units_today: number; profit_today: number; profit_horizon: number | null; below_floor: boolean }[];
  plan: { day: number; date: string; stock_mean: number; stock_p10: number; stock_p90: number; price_mean: number | null; floor: number; multiplier: number; stock_hold: number }[];
}

export interface ModelInfo {
  fitted_on: string;
  n_obs: number;
  n_censored: number;
  holdout_wape: number;
  dispersion: number;
  iterations: number;
  own_elasticity: number;
  own_se: number;
  cross_elasticity: number;
  cross_se: number;
  competitor_reaction: number;
  annual_growth: number;
  yearly_index: number[];
  weekday_index: number[];
  events: { event: string; label: string; lift: number; lo: number; hi: number }[];
  forecast: { date: string; units: number }[];
  forecast_price: number;
}

/** A recommendation as stored by a pricing run (row columns + the solver's payload). */
export interface Recommendation {
  run_id: number;
  sku: string;
  price_date: string;
  current_price: number;
  recommended_price: number;
  floor: number;
  ceiling: number;
  competitor_price: number;
  action: Action;
  binding: string;
  horizon_days: number;
  status: RecStatus;
  final_price: number | null;
  decided_by: string | null;
  decided_at: string | null;
  note: string | null;
  state: Record<string, number | string | boolean>;
  inputs: DerivedInput[];
  advice: Advice;
  kkt: KKT;
  warnings: string[];
  expected_units: number;
  expected_units_at_prev: number;
  unit_margin: number;
  response: string;
  horizon_mode: string;
}

/** Shape used by the product page components. */
export interface Decision {
  price_date: string;
  prev_price: number;
  new_price: number;
  floor: number;
  ceiling: number;
  competitor_price: number;
  binding: string;
  horizon_days: number;
  inputs: DerivedInput[];
  kkt: KKT;
  warnings: string[];
  response: string;
  advice: Advice;
}

export function toDecision(r: Recommendation): Decision {
  return {
    price_date: r.price_date, prev_price: r.current_price, new_price: r.recommended_price, floor: r.floor,
    ceiling: r.ceiling, competitor_price: r.competitor_price, binding: r.binding, horizon_days: r.horizon_days,
    inputs: r.inputs, kkt: r.kkt, warnings: r.warnings, response: r.response, advice: r.advice,
  };
}

export interface HistoryRow {
  run_id: number;
  price_date: string;
  current_price: number;
  recommended_price: number;
  action: Action;
  status: RecStatus;
  final_price: number | null;
  decided_by: string | null;
  decided_at: string | null;
  note: string | null;
}

export interface ProductDetail {
  product: Record<string, string | number> & { sku: string; name: string; category: string; brand: string; mrp: number; supplier: string; lead_time_days: number; shelf_life_days: number; storage_class: string; min_margin_pct: number; max_daily_change_pct: number };
  recommendation: Recommendation | null;
  explanation: Explanation | null;
  model: ModelInfo | null;
  weekly: { week: string; units: number; avg_price: number | null; competitor: number | null; stockout_days: number }[];
  daily: { date: string; sold: number; stocked_out: number; price: number; competitor: number | null; fitted?: number }[];
  lots: { lot_id: number; received_on: string; expires_on: string; qty_on_hand: number; unit_cost: number }[];
  history: HistoryRow[];
  last_sales_date: string | null;
}

export interface PriceChange {
  run_id: number;
  sku: string;
  name: string;
  price_date: string;
  current_price: number;
  recommended_price: number;
  final_price: number | null;
  status: RecStatus;
  decided_by: string;
  decided_at: string;
  note: string | null;
  action: Action;
  profit_gain: number;
}

export interface Paged<T> {
  total: number;
  page: number;
  page_size: number;
  rows: T[];
}

export interface DataKind {
  kind: "products" | "sales" | "stock" | "competitor" | "write_offs";
  label: string;
  help: string;
  required: string[];
  optional: string[];
  rows: number;
  from: string | null;
  to: string | null;
  last_import: string | null;
}

export interface ModelSummary {
  models: { sku: string; name: string; category: string; fitted_on: string; n_obs: number; n_censored: number; holdout_wape: number; newton_iters: number; competitor_reaction: number; own_elasticity: number; own_se: number; cross_elasticity: number; cross_se: number }[];
  settings: { max_horizon_days: number; price_grid_size: number; max_inventory_states: number; refit_every_days: number; fit_window_days: number; market_coverage: number; reaction_threshold: number; workers: number };
}
