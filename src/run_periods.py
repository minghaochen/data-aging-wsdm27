"""Age sweep repeated over several test periods, with a prior-shift control.

Two limitations of the main study are addressed here, and both need the
same runs:

1. TEST-PERIOD VARIANCE.  Every age curve in the main study is measured
   against a single fixed test period, so period difficulty is an
   uncontrolled additive term -- and the rolling backtest suggests it is
   worth ~4 mnat, the same order as the whole feed-log age effect.  Here the
   identical sweep is repeated with the test period slid back by k of its
   own lengths (k = 0, 1, 2), giving three independent measurements of the
   same domain and turning period difficulty into a variance component we
   can report.

2. TRAINING-WINDOW PRIOR SHIFT.  An older window has a different positive
   rate, so the probe learns a different global bias and the test BCE moves
   with zero preference drift.  Every run here also fits a single scalar
   logit offset on a small fresh slice ending at the test boundary and
   reports the recalibrated loss.  Refitting the age law on `loss_recal`
   asks whether the ageing survives when that channel is closed.

Windows, subsampling, cohort construction and the optimisation budget are
otherwise identical to run_domain.py.

Usage: python src/run_periods.py <domain> [--periods 0,1,2] [--cohort] [--item-only] [--device cuda|cpu] [--seeds N]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SRC = str(Path(__file__).resolve().parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from train import run
from run_domain import DOMAINS, D_MIN, D_CAP

ROOT = Path(__file__).resolve().parent.parent
DAY = 86400
# 3 seeds is enough for the apparent curves but NOT for tau_half: resampling
# the published 9-seed KuaiRand cohort down to 3 seeds puts tau_half anywhere
# in [4.3, 13.4] d on one and the same test period, so cohort sweeps should
# be run with --seeds 9 before any timescale is quoted.
SEEDS = [0, 1, 2]


def build(spec, ts, test_t0, test_t1):
    u = spec["unit"]

    def volume(t0, t1):
        a, b = np.searchsorted(ts, [t0, t1])
        return int(b - a)

    cands = []
    for g in spec["gaps"]:
        t1 = test_t0 - g * u
        t0 = t1 - spec["wA"] * u
        if t0 < ts[0]:
            continue
        cands.append({"exp": "A", "gap": g, "train_t0": int(t0),
                      "train_t1": int(t1), "vol": volume(t0, t1),
                      "age_days": (g + spec["wA"] / 2 + spec["test"] / 2)
                      * u / DAY})
    for s in spec["spans"]:
        t0 = test_t0 - s * u
        if t0 < ts[0] - 2 * u:
            continue
        t0 = max(t0, int(ts[0]))
        cands.append({"exp": "C", "span": s, "train_t0": int(t0),
                      "train_t1": int(test_t0), "vol": volume(t0, test_t0),
                      "age_days": (s / 2 + spec["test"] / 2) * u / DAY})
    return [c for c in cands if c["vol"] >= D_MIN]


def cohort_of(z, ts, wins, test_t0, test_t1, d_fix, item_only=False):
    user_all, item_all = z["user"], z["item"]
    c0, e0 = np.searchsorted(ts, [test_t0, test_t1])
    te_u, te_i = user_all[c0:e0], item_all[c0:e0]
    mask = np.ones(e0 - c0, dtype=bool)
    for c in wins:
        a, b = np.searchsorted(ts, [c["train_t0"], c["train_t1"]])
        if d_fix < b - a:
            rng = np.random.default_rng(0)
            sel = a + np.sort(rng.choice(b - a, size=int(d_fix),
                                         replace=False))
        else:
            sel = np.arange(a, b)
        su = np.zeros(user_all.max() + 1, dtype=bool)
        si = np.zeros(item_all.max() + 1, dtype=bool)
        su[user_all[sel]] = True
        si[item_all[sel]] = True
        mask &= si[te_i] if item_only else (su[te_u] & si[te_i])
    return np.flatnonzero(mask)


def main(domain, periods, cohort, item_only, device, seeds=SEEDS):
    if domain not in DOMAINS:
        raise SystemExit(
            f"Unknown domain {domain!r}. Known domains: {sorted(DOMAINS)}"
        )
    spec = DOMAINS[domain]
    npz = ROOT / "data" / spec["npz"]
    if not npz.exists():
        raise SystemExit(
            f"Missing {npz}; build preprocessed .npz first via src/prep_*.py."
        )
    with np.load(npz) as z:
        ts = z["ts"]
        z_ui = {"user": z["user"], "item": z["item"]} if cohort else None
    u = spec["unit"]
    log_end = int(ts[-1]) + 1
    test_len = spec["test"] * u

    for k in periods:
        test_t1 = log_end - k * test_len
        test_t0 = test_t1 - test_len
        # fresh calibration slice: one unit of data ending at the test
        # boundary, i.e. the newest data a serving system would have.
        calib_t1 = test_t0
        calib_t0 = test_t0 - u
        wins = build(spec, ts, test_t0, test_t1)
        if len(wins) < 4:
            print(f"period {k}: only {len(wins)} configs, skipping")
            continue
        d_fix = min(D_CAP, min(c["vol"] for c in wins))
        cohort_idx = None
        if cohort:
            wins = [c for c in wins if c["exp"] == "A"]
            cohort_idx = cohort_of(z_ui, ts, wins, test_t0, test_t1, d_fix,
                                   item_only)
            print(f"period {k}: cohort {len(cohort_idx)} test events")
        # the tag must distinguish the two cohort definitions: they are
        # different measurements on the same windows, and a shared filename
        # would both mix them and make the resume key skip the second one.
        tag = f"__p{k}" + ("icohort" if (cohort and item_only)
                           else "cohort" if cohort else "")
        out = ROOT / "results" / f"domain_{domain}{tag}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        # NB: the key must carry `exp`. When a C span equals the A window
        # length the two manipulations share the identical window, and a key
        # without `exp` silently drops the C cell -- which is a real cell in
        # the protocol (the two arms get separate intercepts in the fit).
        done = set()
        if out.exists():
            for line in out.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    done.add((r["exp"], r["train_t0"], r["train_t1"],
                              r["seed"]))
        print(f"{domain} period {k}: {len(wins)} configs, D_fix={d_fix}, "
              f"test=[{test_t0},{test_t1})")
        with out.open("a") as fh:
            for c in wins:
                for seed in seeds:
                    if (c["exp"], c["train_t0"], c["train_t1"],
                            seed) in done:
                        continue
                    res = run(c["train_t0"], c["train_t1"], test_t0, test_t1,
                              seed=seed, subsample=int(d_fix),
                              subsample_seed=0 if cohort else None,
                              data_path=str(npz),
                              eval_every=spec.get("eval_every", 200),
                              cohort_idx=cohort_idx, device=device,
                              calib_t0=int(calib_t0), calib_t1=int(calib_t1))
                    res.update(exp=c["exp"], vol=c["vol"], period=k,
                               age_days=c["age_days"], domain=f"{domain}{tag}",
                               gap=c.get("gap", -1), span=c.get("span", -1))
                    fh.write(json.dumps(res) + "\n")
                    fh.flush()
                    print(f"  p{k} {c['exp']} age={c['age_days']:7.1f} "
                          f"s={seed} loss={res['loss']:.5f} "
                          f"recal={res.get('loss_recal', float('nan')):.5f} "
                          f"dmu={res.get('delta_mu', float('nan')):+.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Run age sweep across multiple test periods with prior-shift calibration."
    )
    ap.add_argument("domain")
    ap.add_argument("--periods", default="0,1,2")
    ap.add_argument("--cohort", action="store_true")
    ap.add_argument("--item-only", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", type=int, default=len(SEEDS),
                    help="number of model seeds per cell (use 9 for cohort "
                         "runs: tau_half is unstable at 3)")
    a = ap.parse_args()
    main(a.domain, [int(x) for x in a.periods.split(",") if x.strip()], a.cohort,
         a.item_only, a.device, list(range(a.seeds)))
