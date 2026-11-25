"""How much of tau_half depends on the cohort subsample draw?

The composition-fixed cohort is the set of test events whose user and item
appear in the seed-0 subsample of EVERY training window. Fixing that draw makes
cohort membership well defined and removes subsample noise from the intervals
by construction -- but it also ties membership to one realization. This script
rebuilds the cohort on subsample seed 1 and compares tau_half across draws.

Usage: python src/cohort_seed_analysis.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_SRC = str(Path(__file__).resolve().parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from collapse import tau_half_obs

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def curve(fp):
    rows = [json.loads(line) for line in fp.read_text().splitlines() if line]
    c = defaultdict(list)
    for r in rows:
        c[round(r["age_days"], 1)].append(r["loss_cohort"])
    ages = np.array(sorted(c))
    y = np.array([np.mean(c[a]) for a in ages])
    keep = y <= 1.0
    return ages[keep], y[keep], {a: c[a] for a in ages[keep]}, rows[0]


def boot(cells, ages, n=200, seed=0):
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        yb = np.array([np.mean(rng.choice(cells[a], size=len(cells[a]),
                                          replace=True)) for a in ages])
        v = tau_half_obs(ages, yb)
        if np.isfinite(v):
            bs.append(v)
    return np.percentile(bs, [5, 95]) if len(bs) > 20 else (np.nan, np.nan)


def main():
    print("=" * 78)
    print("COHORT SUBSAMPLE-SEED SENSITIVITY")
    print(f"{'domain':10s} {'sub_seed':>9s} {'n_cohort':>9s} {'tau_half':>9s} "
          f"{'[90% CI]':>18s} {'rate':>7s} {'stable':>7s}")
    out = {}
    for dom in ("kuairand", "recflow"):
        rec = {}
        for tag, lab in ((f"domain_{dom}__cohort.jsonl", "0 (baseline)"),
                         (f"domain_{dom}__cohort__ss1.jsonl", "1 (new)")):
            fp = RES / tag
            if not fp.exists():
                print(f"{dom:10s} {lab:>9s}   not run yet")
                continue
            ages, y, cells, r0 = curve(fp)
            t = tau_half_obs(ages, y)
            lo, hi = boot(cells, ages)
            tr = {}
            for cap in ages[3:]:
                m = ages <= cap
                v = tau_half_obs(ages[m], y[m])
                if np.isfinite(v):
                    tr[float(cap)] = float(v)
            vals = list(tr.values())
            stable = (max(vals) / min(vals) < 1.5) if len(vals) > 2 else None
            rate = 1000 * np.polyfit(np.log(ages), y, 1)[0]
            rec[lab] = dict(tau_half=float(t), lo=float(lo), hi=float(hi),
                            n_cohort=int(r0.get("n_cohort", 0)),
                            rate=float(rate), stable=stable,
                            truncations=tr)
            print(f"{dom:10s} {lab:>9s} {r0.get('n_cohort', 0):9,d} "
                  f"{t:9.1f} [{lo:8.1f},{hi:8.1f}] {rate:7.2f} "
                  f"{str(stable):>7s}")
        if len(rec) == 2:
            a, b = rec["0 (baseline)"], rec["1 (new)"]
            ratio = max(a["tau_half"], b["tau_half"]) / min(a["tau_half"],
                                                            b["tau_half"])
            ci_w = a["hi"] - a["lo"]
            print(f"  -> {dom}: the two draws differ by "
                  f"{ratio:.2f}x in tau_half "
                  f"({abs(a['tau_half']-b['tau_half'])/max(ci_w,1e-9):.2f} x "
                  f"the baseline seed-only CI width), and share "
                  f"{100*min(a['n_cohort'], b['n_cohort'])/max(a['n_cohort'], b['n_cohort']):.0f}% "
                  f"of their cohort size")
        out[dom] = rec
    (RES / "cohort_seed_sensitivity.json").write_text(
        json.dumps(out, indent=2, default=str))
    print("\nsaved results/cohort_seed_sensitivity.json")


if __name__ == "__main__":
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__.strip())
        sys.exit(0)
    main()
