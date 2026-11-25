"""Held-out prediction test, step 1 (TRAINING-FREE).

For each held-out domain: replicate run_domain.py's Exp-A window protocol
exactly (same gaps, same D_fix rule, same subsample rng), compute seen_frac
by pure counting, derive the coverage elasticity e_cov, and register a
blind prediction delta_hat from the regression line fitted on the six
original drift domains. Written to results/blind_predictions.json BEFORE
any training run on these domains.

Usage: python src/blind_predict.py amz_Toys_and_Games amz_Office_Products ...
"""
import json
import sys
from pathlib import Path

import numpy as np

_SRC = str(Path(__file__).resolve().parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from run_domain import DOMAINS, D_MIN, D_CAP, SEEDS, DAY

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

# original drift domains used to fit the line (CDs excluded: beta~0)
LINE_DOMAINS = ["kuairand", "ml25m", "amz_Appliances", "amz_Software",
                "amz_Video_Games", "amz_Baby_Products"]


def ecov_for(domain):
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
    z = np.load(npz)
    ts, user, item = z["ts"], z["user"], z["item"]
    u = spec["unit"]
    test_t1 = int(ts[-1]) + 1
    test_t0 = test_t1 - spec["test"] * u
    c, e = np.searchsorted(ts, [test_t0, test_t1])
    te_u, te_i = user[c:e], item[c:e]

    # candidate volumes (A gaps and C spans) -> same D_fix rule as run_domain
    vols = []
    windows = []
    for g in spec["gaps"]:
        t1 = test_t0 - g * u
        t0 = t1 - spec["wA"] * u
        if t0 < ts[0]:
            continue
        a, b = np.searchsorted(ts, [t0, t1])
        if b - a >= D_MIN:
            vols.append(b - a)
            windows.append((g, a, b))
    for s in spec["spans"]:
        t0 = test_t0 - s * u
        if t0 < ts[0] - 2 * u:
            continue
        t0 = max(t0, int(ts[0]))
        a, b = np.searchsorted(ts, [t0, test_t0])
        if b - a >= D_MIN:
            vols.append(b - a)
    d_fix = int(min(D_CAP, min(vols)))

    ages, sfs = [], []
    for g, a, b in windows:
        age_days = (g + spec["wA"] / 2 + spec["test"] / 2) * u / DAY
        vals = []
        for seed in SEEDS:
            if d_fix < b - a:
                rng = np.random.default_rng(seed)
                sel = a + np.sort(rng.choice(b - a, size=d_fix,
                                             replace=False))
            else:
                sel = np.arange(a, b)
            su = np.zeros(user.max() + 1, dtype=bool)
            si = np.zeros(item.max() + 1, dtype=bool)
            su[user[sel]] = True
            si[item[sel]] = True
            vals.append(float((su[te_u] & si[te_i]).mean()))
        ages.append(age_days)
        sfs.append(float(np.mean(vals)))
    ages, sfs = np.array(ages), np.array(sfs)
    m = sfs > 1e-4
    e_cov = float(np.polyfit(np.log(ages[m]), np.log(sfs[m]), 1)[0])
    return e_cov, d_fix, list(zip(ages.tolist(), sfs.tolist()))


def main(new_domains):
    if not new_domains:
        raise SystemExit(
            "Usage: python src/blind_predict.py DOMAIN [DOMAIN ...]\n"
            "Refusing to run without target domains so results/blind_predictions.json is not overwritten."
        )
    dd = json.loads((RES / "delta_dictionary.json").read_text())
    # e_cov of line domains comes from dictionary_analysis (A-runs, cached
    # in results by rerunning its helper here to stay self-contained)
    from dictionary_analysis import coverage_elasticity
    xs, ys = [], []
    for d in LINE_DOMAINS:
        e, _, _ = coverage_elasticity(d)
        xs.append(e)
        ys.append(dd[d]["delta"])
    b, a = np.polyfit(xs, ys, 1)
    resid = np.array(ys) - (a + b * np.array(xs))
    sd = float(np.std(resid))
    print(f"line: delta = {a:.3f} + {b:.3f} * e_cov   "
          f"(residual sd={sd:.3f}, n={len(xs)})")

    fp = RES / "blind_predictions.json"
    out = json.loads(fp.read_text()) if fp.exists() else {}
    out.setdefault("line", {"intercept": a, "slope": b, "resid_sd": sd,
                            "fitted_on": LINE_DOMAINS})
    for dom in new_domains:
        e_cov, d_fix, curve = ecov_for(dom)
        pred = a + b * e_cov
        out[dom] = dict(e_cov=e_cov, delta_hat=float(pred), d_fix=d_fix,
                        seen_curve=curve,
                        criteria="success if delta_hat in measured 90% CI "
                                 "or |delta_hat - delta| < 0.35; "
                                 "rank order preserved")
        print(f"{dom}: e_cov={e_cov:.3f}  ->  BLIND delta_hat={pred:.2f}  "
              f"(D_fix={d_fix})")
    fp.write_text(json.dumps(out, indent=2))
    print(f"registered in {fp}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if any(a in ("-h", "--help") for a in args):
        print(__doc__.strip())
        sys.exit(0)
    main(args)
