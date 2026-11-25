"""Test-period variance and the prior-shift control.

Consumes the sweeps produced by run_periods.py and answers two questions
the single-period study could not:

1. How much of the measured exponent / timescale is the domain, and how
   much is the particular test period we happened to pick?  For each domain
   we refit the same law against k independent test periods and report the
   spread next to the seed-only bootstrap CI.  If the spread is larger, the
   published intervals were the wrong error bar.

2. Does the age effect survive removing training-window prior shift?  Every
   run carries `loss_recal`: the same predictions after a single scalar
   logit offset fitted on a small fresh slice ending at the test boundary.
   Refitting on `loss_recal` closes the base-rate channel and nothing else.

Outputs results/period_analysis.json.

Usage: python src/period_analysis.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_SRC = str(Path(__file__).resolve().parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from fit_domains import fit_joint
from collapse import tau_half_obs

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
DOMS = ["kuairand", "recflow", "ml25m", "netflix", "kuairec", "yelp"]
N_BOOT = 200


def load(fp):
    return [json.loads(l) for l in fp.read_text().splitlines() if l]


def delta_of(rows, metric):
    if not any(metric in r for r in rows):
        return None
    base = fit_joint(rows, metric=metric)
    boots = []
    for b in range(N_BOOT):
        try:
            boots.append(fit_joint(rows, seed=b, metric=metric)["delta"])
        except Exception:
            pass
    lo, hi = np.percentile(boots, [5, 95]) if boots else (np.nan, np.nan)
    return dict(delta=base["delta"], lo=float(lo), hi=float(hi),
                r2=base["r2"], at_bound=base["at_bound"])


def tau_of(rows, metric, with_ci=False):
    if not any(metric in r for r in rows):
        return None
    from collections import defaultdict
    c = defaultdict(list)
    for r in rows:
        if metric in r:
            c[round(r["age_days"], 1)].append(r[metric])
    ages = np.array(sorted(c))
    y = np.array([np.mean(c[a]) for a in ages])
    keep = y <= 1.0
    if keep.sum() < 4:
        return None
    ages, y = ages[keep], y[keep]
    t = tau_half_obs(ages, y)
    if not np.isfinite(t):
        return None
    if not with_ci:
        return float(t)
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(400):
        yb = np.array([np.mean(rng.choice(c[a], size=len(c[a]),
                                          replace=True)) for a in ages])
        v = tau_half_obs(ages, yb)
        if np.isfinite(v):
            bs.append(v)
    lo, hi = (np.percentile(bs, [5, 95]) if len(bs) > 20
              else (np.nan, np.nan))
    return dict(tau=float(t), lo=float(lo), hi=float(hi),
                n_seeds=int(min(len(v) for v in c.values())),
                n_ages=len(ages), age_max=float(ages.max()),
                amp_mnat=float(1000 * (y.max() - y.min())))


def seed_subset_spread(fp, metric, k=3, n_draw=200):
    """How much does the statistic move if you keep only k of the seeds?

    tau_half is an interpolation on a noisy curve, and on KuaiRand the
    published 9-seed cohort gives anywhere in [4.3, 13.4] d when resampled
    down to 3 seeds. Without this null, an across-period spread measured at
    3 seeds cannot be attributed to the period at all.
    """
    from collections import defaultdict
    rows = load(fp)
    c = defaultdict(dict)
    for r in rows:
        if metric in r:
            c[round(r["age_days"], 1)][r["seed"]] = r[metric]
    ages = np.array(sorted(c))
    seeds = sorted(set().union(*[set(v) for v in c.values()]))
    if len(seeds) <= k or len(ages) < 4:
        return None
    import itertools
    combos = list(itertools.combinations(seeds, k))
    if len(combos) > n_draw:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(combos), size=n_draw, replace=False)
        combos = [combos[i] for i in idx]
    vals = []
    for tri in combos:            # exhaustive when C(n,k) is small enough
        y = np.array([np.mean([c[a][s] for s in tri if s in c[a]])
                      for a in ages])
        t = tau_half_obs(ages, y)
        if np.isfinite(t):
            vals.append(t)
    if len(vals) < 20:
        return None
    return dict(k=k, n=len(vals), lo=float(np.percentile(vals, 5)),
                hi=float(np.percentile(vals, 95)),
                median=float(np.median(vals)))


def spread(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if len(v) < 2:
        return None
    return dict(values=[float(x) for x in v], n=len(v),
                min=float(min(v)), max=float(max(v)),
                mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)),
                range=float(max(v) - min(v)))


def main():
    out = {}
    for dom in DOMS:
        rec = {"apparent": {}, "cohort": {}, "prior_shift": {}}
        for k in (0, 1, 2):
            fp = RES / f"domain_{dom}__p{k}.jsonl"
            if fp.exists():
                rows = load(fp)
                if len(rows) >= 12:
                    rec["apparent"][k] = {
                        "loss": delta_of(rows, "loss"),
                        "loss_recal": delta_of(rows, "loss_recal"),
                        "n_runs": len(rows)}
                    pr = [r["train_pos_rate"] for r in rows
                          if "train_pos_rate" in r]
                    dmu = [abs(r["delta_mu"]) for r in rows
                           if "delta_mu" in r]
                    if pr:
                        rec["prior_shift"][k] = dict(
                            train_pos_rate_min=float(min(pr)),
                            train_pos_rate_max=float(max(pr)),
                            test_pos_rate=float(rows[0]["test_pos_rate"]),
                            abs_delta_mu_max=float(max(dmu)) if dmu else None,
                            abs_delta_mu_mean=(float(np.mean(dmu))
                                               if dmu else None))
            # review platforms carry item cohorts (__icohort), production
            # logs the user-and-item cohort (__cohort); they are different
            # measurements and live in different files
            fpc = RES / f"domain_{dom}__p{k}cohort.jsonl"
            if not fpc.exists():
                fpc = RES / f"domain_{dom}__p{k}icohort.jsonl"
            if fpc.exists():
                rows = load(fpc)
                if len(rows) >= 12:
                    rec["cohort_kind"] = ("item" if "icohort" in fpc.name
                                          else "user_and_item")
                    rec["cohort"][k] = {
                        "delta_cohort": delta_of(rows, "loss_cohort"),
                        "delta_cohort_recal": delta_of(
                            rows, "loss_cohort_recal"),
                        "tau_half": tau_of(rows, "loss_cohort"),
                        "tau_half_ci": tau_of(rows, "loss_cohort",
                                              with_ci=True),
                        "tau_half_recal": tau_of(rows, "loss_cohort_recal"),
                        "n_runs": len(rows)}
        rec["summary"] = {
            "delta_across_periods": spread(
                [v["loss"]["delta"] if v.get("loss") else None
                 for v in rec["apparent"].values()]),
            "delta_recal_across_periods": spread(
                [v["loss_recal"]["delta"] if v.get("loss_recal") else None
                 for v in rec["apparent"].values()]),
            "tau_across_periods": spread(
                [v.get("tau_half") for v in rec["cohort"].values()]),
            "tau_recal_across_periods": spread(
                [v.get("tau_half_recal") for v in rec["cohort"].values()]),
            "mean_seed_ci_width": float(np.mean(
                [v["loss"]["hi"] - v["loss"]["lo"]
                 for v in rec["apparent"].values() if v.get("loss")]))
            if rec["apparent"] else None,
            # null for the tau_half spread: same test period, fewer seeds
            "tau_seed_subset_null": seed_subset_spread(
                RES / f"domain_{dom}__cohort.jsonl", "loss_cohort")
            if (RES / f"domain_{dom}__cohort.jsonl").exists() else None,
        }
        out[dom] = rec

    (RES / "period_analysis.json").write_text(json.dumps(out, indent=2))

    print("=" * 78)
    print("TEST-PERIOD VARIANCE (apparent delta, joint A+C)")
    print(f"{'domain':10s} {'per-period delta':>26s} {'range':>7s} "
          f"{'seed CI':>8s}  verdict")
    for dom, rec in out.items():
        s = rec["summary"]["delta_across_periods"]
        if not s:
            print(f"{dom:10s} (insufficient periods)")
            continue
        ci = rec["summary"]["mean_seed_ci_width"]
        vals = " ".join(f"{v:+.2f}" for v in s["values"])
        verdict = ("period dominates" if s["range"] > ci
                   else "seed noise dominates")
        print(f"{dom:10s} {vals:>26s} {s['range']:7.2f} {ci:8.2f}  {verdict}")

    print("\n" + "=" * 78)
    print("TEST-PERIOD VARIANCE (cohort tau_half, days)")
    print("A period effect is claimed only when two periods with the SAME "
          "age grid have disjoint within-period bootstrap CIs.")
    for dom, rec in out.items():
        cis = {k: v["tau_half_ci"] for k, v in rec["cohort"].items()
               if v.get("tau_half_ci")}
        if len(cis) < 2:
            continue
        for k, v in sorted(cis.items()):
            print(f"{dom:10s} p{k}  tau={v['tau']:7.1f} "
                  f"[{v['lo']:7.1f},{v['hi']:7.1f}]  "
                  f"{v['n_ages']} ages to {v['age_max']:.0f} d, "
                  f"{v['n_seeds']} seeds, amp {v['amp_mnat']:.1f} mnat")
        # compare only periods sharing an age grid
        by_grid = defaultdict(list)
        for k, v in cis.items():
            by_grid[(v["n_ages"], v["age_max"])].append((k, v))
        for grid, items in by_grid.items():
            if len(items) < 2:
                continue
            items.sort(key=lambda kv: kv[1]["tau"])
            lo_k, lo_v = items[0]
            hi_k, hi_v = items[-1]
            disjoint = lo_v["hi"] < hi_v["lo"]
            print(f"{'':10s}  matched grid {grid[0]} ages/{grid[1]:.0f} d: "
                  f"p{lo_k} {lo_v['tau']:.1f} vs p{hi_k} {hi_v['tau']:.1f} "
                  f"= {hi_v['tau']/lo_v['tau']:.2f}x, CIs "
                  f"{'disjoint -> period effect real' if disjoint else 'overlap -> not resolved'}")

    print("\n" + "=" * 78)
    print("PRIOR-SHIFT CONTROL (delta on loss vs loss_recal)")
    for dom, rec in out.items():
        for k, v in rec["apparent"].items():
            a, b = v.get("loss"), v.get("loss_recal")
            if not (a and b):
                continue
            ps = rec["prior_shift"].get(k, {})
            print(f"{dom:10s} p{k}  delta {a['delta']:+.3f} -> "
                  f"{b['delta']:+.3f}   train pos rate "
                  f"[{ps.get('train_pos_rate_min', float('nan')):.3f},"
                  f"{ps.get('train_pos_rate_max', float('nan')):.3f}] "
                  f"vs test {ps.get('test_pos_rate', float('nan')):.3f}  "
                  f"|dmu|max={ps.get('abs_delta_mu_max', float('nan')):.3f}")
        for k, v in rec["cohort"].items():
            if v.get("tau_half") and v.get("tau_half_recal"):
                print(f"{dom:10s} p{k}  tau_half {v['tau_half']:.1f} -> "
                      f"{v['tau_half_recal']:.1f} d (recalibrated)")

    print("\nsaved results/period_analysis.json")


if __name__ == "__main__":
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__.strip())
        sys.exit(0)
    main()
