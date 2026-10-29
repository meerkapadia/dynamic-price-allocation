"""Retail calendar: festival and sale windows that shift demand (public knowledge, so a valid model feature).

Each event contributes a weight w ∈ [0, 1] per day: a linear ramp up to the peak day and a short tail after it
(or a flat window for sales). Weights are shared by the market simulator and the demand estimator; the
*size* of each event's effect per product is unknown to the estimator and has to be learned from sales.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np


@dataclass(frozen=True, slots=True)
class Event:
    key: str
    label: str
    peaks: tuple[str, ...]      # ISO peak dates (or window starts when flat=True)
    pre_days: int               # ramp length before the peak
    post_days: int              # tail after the peak (window length when flat=True)
    flat: bool = False


EVENTS: tuple[Event, ...] = (
    Event("diwali", "Diwali", ("2023-11-12", "2024-11-01", "2025-10-20", "2026-11-08", "2027-10-29", "2028-10-17"), 16, 2),
    Event("holi", "Holi", ("2024-03-25", "2025-03-14", "2026-03-04", "2027-03-22", "2028-03-11"), 7, 1),
    Event("rakhi", "Raksha Bandhan", ("2024-08-19", "2025-08-09", "2026-08-28", "2027-08-17", "2028-08-05"), 8, 0),
    Event("festive_sale", "Festive sale", ("2024-09-27", "2025-09-22", "2026-10-14", "2027-10-02", "2028-09-20"), 0, 9, True),
    Event("year_end", "Christmas and New Year", ("2023-12-31", "2024-12-31", "2025-12-31", "2026-12-31", "2027-12-31"), 10, 1),
    Event("independence_sale", "Independence Day sale", ("2024-08-08", "2025-08-08", "2026-08-08", "2027-08-08", "2028-08-08"), 0, 8, True),
)
EVENT_KEYS: tuple[str, ...] = tuple(e.key for e in EVENTS)


def _as_days(dates) -> np.ndarray:
    return np.asarray(dates, dtype="datetime64[D]")


def event_weights(dates) -> np.ndarray:
    """(n, E) matrix of event intensities for the given dates."""
    d = _as_days(dates)
    out = np.zeros((d.size, len(EVENTS)))
    for j, ev in enumerate(EVENTS):
        for peak in ev.peaks:
            k = (d - np.datetime64(peak)).astype(np.int64)  # days after peak (or after window start)
            if ev.flat:
                w = ((k >= 0) & (k < ev.post_days)).astype(float)
            else:
                pre = np.where((k <= 0) & (k >= -ev.pre_days), 1.0 - np.abs(k) / (ev.pre_days + 1.0), 0.0)
                post = np.where((k > 0) & (k <= ev.post_days), 1.0 - k / (ev.post_days + 1.0), 0.0)
                w = pre + post
            out[:, j] = np.maximum(out[:, j], w)
    return out


def day_of_year(dates) -> np.ndarray:
    d = _as_days(dates)
    years = d.astype("datetime64[Y]")
    return (d - years.astype("datetime64[D]")).astype(np.int64) + 1


def weekday(dates) -> np.ndarray:
    """Monday = 0 … Sunday = 6."""
    return ((_as_days(dates).astype(np.int64) + 3) % 7).astype(np.int64)  # 1970-01-01 was a Thursday


def calendar_rows(first: date, last: date) -> list[tuple[str, str, str, str, str]]:
    rows = []
    for ev in EVENTS:
        for peak in ev.peaks:
            p = date.fromisoformat(peak)
            if ev.flat:
                start, end, mark = p, p + timedelta(days=ev.post_days - 1), p
            else:
                start, end, mark = p - timedelta(days=ev.pre_days), p + timedelta(days=ev.post_days), p
            if end >= first and start <= last:
                rows.append((ev.key, ev.label, start.isoformat(), mark.isoformat(), end.isoformat()))
    return sorted(rows, key=lambda r: r[3])
