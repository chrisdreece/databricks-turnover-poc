"""Seeded mock HR data: monthly employee snapshots for a turnover model.

Simulates a workforce of about 8,000 employees through 40 years of burn-in, so
the age and tenure mix comes from the hazards themselves, then writes the last
25 month-end snapshots: 24 months of history plus the current month.

One row per employee per month while employed. A leaver appears for the last
time in the month they left, with turnover = 1 or retirement = 1 and a
term_reason. Same seed -> identical file.

Built-in signal
---------------
- Turnover risk is highest in the first year of tenure and falls with tenure.
- Younger employees leave more often, over and above tenure.
- Part-time staff and some segments (plant operations, logistics) leave more.
- Pay below the segment norm raises risk, and each employee has an unobserved
  propensity to leave, so age and tenure explain only part of it.
- Retirement starts at 55 and climbs at 62 and 65.
- Quits peak in early summer, and the labor market cools across the window,
  so the most recent year runs below the year before (about 13% vs 16%).
- A reduction in force hits Logistics & Transportation in the West region in
  January 2026. The source system flags those as turnover = 1; the modeling
  step should reset them to 0.

Usage:
    python src/adm_toy/datagen.py --out data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Fixed rather than "this month" so the output is reproducible.
CURRENT_MONTH = "2026-09"
SNAPSHOT_MONTHS = 25  # 24 months of history + the current month
BURN_IN_YEARS = 40
HEADCOUNT = 8_000
EPOCH = pd.Timestamp("1900-01-01")

SEGMENTS = pd.DataFrame(
    [
        # name, share of hires, quit multiplier, median salary, hire age = min + gamma(shape, scale), part-time share
        ("Plant Operations", 0.34, 1.30, 46_000, 19, 2.0, 6.0, 0.06),
        ("Maintenance & Trades", 0.12, 0.80, 64_000, 21, 2.5, 5.0, 0.02),
        ("Logistics & Transportation", 0.16, 1.40, 50_000, 21, 2.0, 6.0, 0.08),
        ("Administrative Support", 0.10, 0.95, 44_000, 20, 2.0, 7.0, 0.20),
        ("Professional", 0.15, 0.80, 88_000, 23, 2.0, 5.0, 0.03),
        ("Sales", 0.06, 1.15, 75_000, 23, 2.0, 5.0, 0.02),
        ("Management", 0.07, 0.60, 120_000, 30, 3.0, 4.0, 0.00),
    ],
    columns=["name", "hire_share", "quit_mult", "median_salary", "hire_age_min", "hire_age_shape", "hire_age_scale", "part_time_share"],
)
REGIONS = np.array(["Central", "East", "South", "West"])
REGION_SHARES = [0.4, 0.2, 0.2, 0.2]
STATUSES = np.array(["Full-Time", "Part-Time", "Leave"])
REASONS = np.array([None, "Voluntary", "Involuntary", "Retirement", "Reduction in Force"], dtype=object)

BASE_QUIT = 0.020  # monthly voluntary hazard before multipliers
BASE_INVOLUNTARY = 0.0022
RIF = {"month": "2026-01", "segment": "Logistics & Transportation", "region": "West", "share": 0.15}
MANDATORY_RETIREMENT_AGE = 72


def _choose(rng, weights):
    """Draw one category index per row from an (n_rows, n_categories) weight matrix."""
    cum = np.cumsum(weights, axis=1)
    u = rng.random(len(weights)) * cum[:, -1]
    return (u[:, None] > cum).sum(axis=1)


def _retirement_hazard(age):
    return np.select(
        [age < 55, age < 60, age < 62, age < 65, age < 67],
        [0.0, 0.002, 0.004, 0.010, 0.030],
        0.025,
    )


def _hire(rng, n, first_day, last_day, first_id):
    seg = rng.choice(len(SEGMENTS), n, p=SEGMENTS["hire_share"])
    s = SEGMENTS.iloc[seg]
    hire_age = np.minimum(s["hire_age_min"].to_numpy() + rng.gamma(s["hire_age_shape"].to_numpy(), s["hire_age_scale"].to_numpy()), 64)
    hire_day = rng.integers(first_day, last_day + 1, n)
    part_time = (rng.random(n) < s["part_time_share"].to_numpy()).astype(int)
    return {
        "id": np.arange(first_id, first_id + n),
        "segment": seg,
        "region": rng.choice(len(REGIONS), n, p=REGION_SHARES),
        "status": part_time,
        "home_status": part_time,  # status to return to after a leave of absence
        "hire_day": hire_day,
        "birth_day": hire_day - np.round(hire_age * 365.25).astype(int),
        "salary": s["median_salary"].to_numpy() * rng.lognormal(0, 0.12, n) * (1 + 0.006 * (hire_age - 25)),
        "propensity": rng.gamma(2.0, 0.5, n),  # unobserved, mean 1
    }


def _separations(rng, staff, day, calendar_month, market):
    """Status changes, April raises and this month's separations for existing staff. Returns reason codes."""
    n = len(staff["id"])
    u = rng.random(n)
    status, home = staff["status"], staff["home_status"]
    on_leave = status == 2
    to_leave = ~on_leave & (u < 0.004)
    switch = ~on_leave & (u >= 0.004) & (u < 0.004 + np.where(status == 0, 0.003, 0.010))
    back = on_leave & (u < 0.35)
    status[switch] = 1 - status[switch]
    home[switch] = status[switch]
    status[to_leave] = 2
    status[back] = home[back]
    if calendar_month == 4:
        staff["salary"] *= 1.03 * rng.lognormal(0, 0.02, n)

    age = (day - staff["birth_day"]) / 365.25
    tenure = (day - staff["hire_day"]) / 365.25
    s = SEGMENTS.iloc[staff["segment"]]
    expected_pay = s["median_salary"].to_numpy() * 1.03 ** np.minimum(tenure, 25)
    quit = (
        BASE_QUIT
        * s["quit_mult"].to_numpy()
        * (0.45 + 2.4 * np.exp(-tenure / 1.1))
        * (0.75 + 1.1 * np.exp(-(age - 18) / 7)) * np.where(age >= 58, 0.8, 1.0)
        * np.array([1.0, 1.5, 1.3])[status]
        * (staff["salary"] / expected_pay) ** -1.2
        * staff["propensity"]
        * (1 + 0.15 * np.cos(2 * np.pi * (calendar_month - 6) / 12))
        * market
    )
    involuntary = BASE_INVOLUNTARY * (1 + 1.5 * np.exp(-tenure / 0.75)) * np.where(status == 1, 1.3, 1.0)
    hazards = np.column_stack([quit, involuntary, _retirement_hazard(age)])
    leaves = rng.random(n) < 1 - np.exp(-hazards.sum(axis=1))
    reason = np.where(leaves, _choose(rng, hazards) + 1, 0)
    reason[age >= MANDATORY_RETIREMENT_AGE] = 3
    return reason, age, tenure


def simulate(seed=42, current_month=CURRENT_MONTH, n_months=SNAPSHOT_MONTHS, burn_in_years=BURN_IN_YEARS, headcount=HEADCOUNT):
    rng = np.random.default_rng(seed)
    total = burn_in_years * 12 + n_months
    months = pd.period_range(end=current_month, periods=total, freq="M")
    month_end = months.to_timestamp(how="end").normalize()
    end_day = (month_end - EPOCH).days.to_numpy()
    first_snapshot = total - n_months
    rif_k = months.get_loc(pd.Period(RIF["month"], "M"))
    rif_segment = SEGMENTS.index[SEGMENTS["name"] == RIF["segment"]][0]
    rif_region = list(REGIONS).index(RIF["region"])

    staff = {k: v[:0] for k, v in _hire(rng, 0, 0, 1, 0).items()}
    next_id, target, snapshots = 1, headcount, []
    for k in range(total):
        w = k - first_snapshot
        market = np.interp(w, [-24, 0, n_months - 1], [1.0, 1.20, 0.85])
        reason, age, tenure = _separations(rng, staff, end_day[k], months[k].month, market)
        if k == rif_k:
            rif = (reason == 0) & (staff["segment"] == rif_segment) & (staff["region"] == rif_region)
            rif &= rng.random(len(rif)) < RIF["share"]
            reason[rif] = 4
            target -= rif.sum()  # positions eliminated, not backfilled

        n_hire = max(0, round(target - (reason == 0).sum() + rng.normal(0, 15)))
        first_day = end_day[k - 1] + 1 if k else end_day[k] - 30
        hires = _hire(rng, n_hire, first_day, end_day[k], next_id)
        next_id += n_hire

        if w >= 0:
            snapshots.append(_snapshot(staff, reason, age, tenure, hires, end_day[k], month_end[k]))
        keep = reason == 0
        staff = {key: np.concatenate([staff[key][keep], hires[key]]) for key in staff}

    df = pd.concat(snapshots, ignore_index=True)
    df["employee_id"] = "E" + df["employee_id"].astype(str).str.zfill(6)
    return df


def _snapshot(staff, reason, age, tenure, hires, day, date):
    """Month-end rows: existing staff (with this month's separations flagged) plus this month's hires."""
    rows = {key: np.concatenate([staff[key], hires[key]]) for key in ("id", "segment", "region", "status", "hire_day", "salary")}
    hire_age = (day - hires["birth_day"]) / 365.25
    hire_tenure = (day - hires["hire_day"]) / 365.25
    reason = np.concatenate([reason, np.zeros(len(hires["id"]), dtype=int)])
    return pd.DataFrame(
        {
            "employee_id": rows["id"],
            "snapshot_date": date,
            "segment": SEGMENTS["name"].to_numpy()[rows["segment"]],
            "region": REGIONS[rows["region"]],
            "employment_status": STATUSES[rows["status"]],
            "hire_date": EPOCH + pd.to_timedelta(rows["hire_day"], unit="D"),
            "age": np.concatenate([age, hire_age]).round(2),
            "tenure": np.concatenate([tenure, hire_tenure]).round(2),
            "annual_salary": np.round(rows["salary"], -2),
            "turnover": np.isin(reason, [1, 2, 4]).astype(int),
            "retirement": (reason == 3).astype(int),
            "term_reason": REASONS[reason],
        }
    )


def summarize(df):
    months = df.groupby("snapshot_date").agg(
        headcount=("employee_id", "size"),
        turnover=("turnover", "sum"),
        rif=("term_reason", lambda r: (r == "Reduction in Force").sum()),
        retirement=("retirement", "sum"),
    )
    print(f"rows {len(df):,}  employees {df['employee_id'].nunique():,}  months {len(months)}")
    print(f"headcount first/last month: {months['headcount'].iloc[0]:,} / {months['headcount'].iloc[-1]:,}")
    last_year = months.iloc[-13:-1]
    year_before = months.iloc[-25:-13]
    for label, m in [("year before", year_before), ("last 12 months", last_year)]:
        hc = m["headcount"].mean()
        print(
            f"{label}: turnover excl. RIF {(m['turnover'] - m['rif']).sum() / hc:.3f}, "
            f"retirement {m['retirement'].sum() / hc:.3f}, RIF {m['rif'].sum()}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--current-month", default=CURRENT_MONTH)
    parser.add_argument("--headcount", type=int, default=HEADCOUNT)
    parser.add_argument("--format", choices=["csv", "parquet"], default="csv")
    args = parser.parse_args()

    df = simulate(args.seed, args.current_month, headcount=args.headcount)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"employee_snapshots.{args.format}"
    if args.format == "csv":
        df.to_csv(path, index=False, date_format="%Y-%m-%d")
    else:
        df.to_parquet(path, index=False)
    print(f"wrote {path}")
    summarize(df)


if __name__ == "__main__":
    main()
