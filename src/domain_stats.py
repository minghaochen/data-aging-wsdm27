"""Catalog-dynamics statistics per domain, computed from the raw log alone
(no model training). These are the candidate predictors of the drift
exponent delta.

- pop_half_life_days: smallest lag L where the Spearman correlation between
  item interaction counts in a base window and in the window lagged by L
  drops below 0.5 (averaged over base windows in the last 2 years).
- new_item_share: fraction of events landing on items first seen within the
  preceding 30 days (computed over the last year of the log).
- item_median_age_days: median over items (>=20 events) of the median event
  age relative to the item's first event ("engagement half-life").
- user_median_age_days: same for users.

Usage: python src/domain_stats.py [DOMAIN ...]
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
DAY = 86400


def pop_half_life(item, ts, win_days, lags_days, n_base=6):
    t_end = ts[-1]
    hi = t_end - (max(lags_days) + win_days) * DAY
    lo = max(ts[0], hi - 2 * 365 * DAY)
    if hi <= lo:
        lo = ts[0]
    base_starts = np.linspace(lo, hi, n_base)
    rows = []
    for L in lags_days:
        cors = []
        for b in base_starts:
            a1, b1 = np.searchsorted(ts, [b, b + win_days * DAY])
            a2, b2 = np.searchsorted(ts, [b + L * DAY,
                                          b + (L + win_days) * DAY])
            if b1 - a1 < 1000 or b2 - a2 < 1000:
                continue
            base_items, base_cnt = np.unique(item[a1:b1], return_counts=True)
            if len(base_items) < 50:
                continue
            lag_items, lag_cnt = np.unique(item[a2:b2], return_counts=True)
            lag_map = dict(zip(lag_items.tolist(), lag_cnt.tolist()))
            lagged = np.array([lag_map.get(i, 0) for i in base_items.tolist()])
            cors.append(spearmanr(base_cnt, lagged).statistic)
        if cors:
            rows.append((L, float(np.mean(cors))))
    # first lag with rho < 0.5, log-interpolated
    prev_l, prev_r = None, None
    for L, r in rows:
        if r < 0.5 and prev_l is not None:
            frac = (prev_r - 0.5) / (prev_r - r)
            return float(np.exp(np.log(prev_l)
                                + frac * (np.log(L) - np.log(prev_l)))), rows
        prev_l, prev_r = L, r
    return float(rows[-1][0]) if rows else np.nan, rows  # censored


def median_entity_age(codes, ts, min_events=20):
    order = np.argsort(codes, kind="stable")
    c_sorted, t_sorted = codes[order], ts[order]
    bounds = np.flatnonzero(np.diff(c_sorted)) + 1
    ages = []
    for seg in np.split(t_sorted, bounds):
        if len(seg) >= min_events:
            ages.append(np.median(seg - seg.min()) / DAY)
    return float(np.median(ages)), len(ages)


def new_item_share(item, ts, recent_days=30):
    first_seen = {}
    order = np.arange(len(ts))
    shares_num, shares_den = 0, 0
    year_start = ts[-1] - 365 * DAY
    for i in order:
        it = item[i]
        if it not in first_seen:
            first_seen[it] = ts[i]
        if ts[i] >= year_start:
            shares_den += 1
            if ts[i] - first_seen[it] <= recent_days * DAY:
                shares_num += 1
    return shares_num / max(shares_den, 1)


def main(domains):
    if not domains:
        raise SystemExit(
            "No preprocessed data/*.npz files found; build them first via src/prep_*.py "
            "or pass domain names explicitly."
        )
    out_fp = ROOT / "results" / "domain_stats.json"
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    out = json.loads(out_fp.read_text()) if out_fp.exists() else {}
    for dom in domains:
        npz = ROOT / "data" / f"{dom}.npz"
        if not npz.exists():
            raise SystemExit(
                f"Missing {npz}; build preprocessed .npz first via src/prep_*.py."
            )
        with np.load(npz) as z:
            item, user, ts = z["item"], z["user"], z["ts"]
        span_days = (ts[-1] - ts[0]) / DAY

        if span_days > 3000:
            win, lags = 30, [30, 60, 90, 180, 365, 730, 1460, 2920]
        elif span_days > 400:
            win, lags = 30, [30, 60, 90, 180, 365, 730]
        else:
            win, lags = 2, [1, 2, 4, 7, 14, 21]
        phl, rho_curve = pop_half_life(item, ts, win, lags)
        # dimensionless, censoring-free drift stat: mean d(rho)/d(ln L)
        Ls = np.array([r[0] for r in rho_curve], float)
        rhos = np.array([r[1] for r in rho_curve], float)
        decay_slope = (-np.polyfit(np.log(Ls), rhos, 1)[0]
                       if len(Ls) >= 3 else np.nan)
        i_age, _ = median_entity_age(item, ts)
        u_age, _ = median_entity_age(user, ts)
        nis = new_item_share(item, ts, recent_days=30 if span_days > 400 else 3)
        out[dom] = dict(span_days=float(span_days),
                        pop_half_life_days=phl,
                        rho_decay_slope=float(decay_slope),
                        rho_curve=rho_curve,
                        item_median_age_days=i_age,
                        user_median_age_days=u_age,
                        new_item_share=nis,
                        n_events=int(len(ts)))
        print(f"{dom}: pop_half_life={phl:.1f}d  item_age={i_age:.1f}d  "
              f"user_age={u_age:.1f}d  new_share={nis:.3f}  "
              f"span={span_days:.0f}d")
    out_fp.write_text(json.dumps(out, indent=2))
    print(f"saved {out_fp}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if any(a in ("-h", "--help") for a in args):
        print(__doc__.strip())
        sys.exit(0)
    doms = args or sorted(Path(f).stem for f in
                          glob.glob(str(ROOT / "data" / "*.npz")))
    main(doms)
