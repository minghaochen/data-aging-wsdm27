"""Round-2 stratified blind prediction using the two-term law (training-free
for the new domains).

M1 baseline: delta_hat = a + b * e_cov, line refit on all 10 valid drift
domains measured prior to round 2.

M2 two-term engine: for a new domain, measure sf(tau) by counting (exact
Exp-A and Exp-C protocol replication), then simulate
    L_mix(tau) = sf(tau) * (bw * ln(tau/tau0) - gap - sc * ln(tau/tau0))
                 + sc * ln(tau/tau0) + const
with (bw, gap, sc) taken from each of the 8 measured Amazon domains'
warm-slope / cold-warm-gap / cold-slope calibration; joint Box-Cox fit of
each simulated curve gives a delta_hat sample; report the median and spread.

Predictions are registered to results/blind_predictions_round2.json BEFORE
any training on the new domains.

Usage: python src/blind_predict2.py amz_Musical_Instruments amz_Kindle_Store ...
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

_SRC = str(Path(__file__).resolve().parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from blind_predict import ecov_for
from two_term_law import cells_for
from dictionary_analysis import coverage_elasticity

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

M1_DOMAINS = ["kuairand", "ml25m", "amz_Appliances", "amz_Software",
              "amz_Video_Games", "amz_Baby_Products", "amz_Toys_and_Games",
              "amz_Office_Products", "amz_Pet_Supplies", "beer"]
CALIB_AMZ = ["amz_Video_Games", "amz_CDs_and_Vinyl", "amz_Software",
             "amz_Appliances", "amz_Baby_Products", "amz_Toys_and_Games",
             "amz_Office_Products", "amz_Pet_Supplies"]


def boxcox_delta(ages, y):
    t0 = ages.min()

    def bc(t, delta):
        r = t / t0
        return np.log(r) if abs(delta) < 1e-6 else (r ** delta - 1) / delta

    def f(t, c, beta, delta):
        return c + beta * np.array([bc(x, delta) for x in t])

    p, _ = curve_fit(f, ages, y, p0=[y.min(), 0.005, 0.3],
                     bounds=([y.min() - 0.05, 0, -1],
                             [y.min() + 0.03, 10, 4]), maxfev=100000)
    return float(p[2])


def calibrate():
    """(b_w, gap, s_c) per measured Amazon domain from Exp-A curves.
    s_c is the small but nonzero cold-loss slope vs ln(age) — H3 showed it
    carries the mid-range curvature once sf has collapsed."""
    pairs = []
    for dom in CALIB_AMZ:
        ages, agg = cells_for(dom)
        m = ~np.isnan(agg["unseen"])
        lnr = np.log(ages / ages.min())
        bw, cw = np.polyfit(lnr, agg["seen"], 1)
        sc, cc = np.polyfit(lnr[m], agg["unseen"][m], 1)
        gap = float(cc) - float(cw)
        pairs.append((dom, float(bw), gap, float(sc)))
    return pairs


def sf_curve(domain):
    """Training-free seen_frac at the Exp-A protocol ages."""
    _, _, curve = ecov_for(domain)
    ages = np.array([c[0] for c in curve])
    sf = np.array([c[1] for c in curve])
    return ages, sf


def sf_curve_c(domain):
    """Training-free seen_frac for the Exp-C widening windows (needed to
    simulate the joint A+C fit, which is what the dictionary delta is)."""
    from run_domain import DOMAINS, D_MIN, D_CAP, SEEDS, DAY
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

    vols = []
    windows = []
    for g in spec["gaps"]:
        t1 = test_t0 - g * u
        t0 = t1 - spec["wA"] * u
        if t0 >= ts[0]:
            a, b = np.searchsorted(ts, [t0, t1])
            if b - a >= D_MIN:
                vols.append(b - a)
    for s in spec["spans"]:
        t0 = test_t0 - s * u
        if t0 < ts[0] - 2 * u:
            continue
        t0 = max(t0, int(ts[0]))
        a, b = np.searchsorted(ts, [t0, test_t0])
        if b - a >= D_MIN:
            vols.append(b - a)
            windows.append((s, a, b))
    d_fix = int(min(D_CAP, min(vols)))

    ages, sfs = [], []
    for s, a, b in windows:
        age_days = (s / 2 + spec["test"] / 2) * u / DAY
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
    return np.array(ages), np.array(sfs)


def joint_boxcox_delta(ages_a, y_a, ages_c, y_c):
    """Shared beta/delta, separate intercepts — mirrors fit_domains."""
    xs = np.concatenate([ages_a, ages_c])
    ys = np.concatenate([y_a, y_c])
    gs = np.concatenate([np.zeros(len(ages_a)), np.ones(len(ages_c))])
    t0 = xs.min()

    def bc(t, delta):
        r = t / t0
        return np.log(r) if abs(delta) < 1e-6 else (r ** delta - 1) / delta

    def f(x, cA, cC, beta, delta):
        t, gi = x
        return (np.where(gi == 0, cA, cC)
                + beta * np.array([bc(v, delta) for v in t]))

    ymin = ys.min()
    p, _ = curve_fit(f, (xs, gs), ys, p0=[ymin, ymin, 0.005, 0.5],
                     bounds=([ymin - 0.05, ymin - 0.05, 0, -1.0],
                             [ymin + 0.03, ymin + 0.03, 10, 4]),
                     maxfev=100000)
    return float(p[3])


def main(new_domains):
    if not new_domains:
        raise SystemExit(
            "Usage: python src/blind_predict2.py DOMAIN [DOMAIN ...]\n"
            "Refusing to run without target domains so results/blind_predictions_round2.json is not overwritten."
        )
    dd = json.loads((RES / "delta_dictionary.json").read_text())
    xs = np.array([coverage_elasticity(d)[0] for d in M1_DOMAINS])
    ys = np.array([dd[d]["delta"] for d in M1_DOMAINS])
    b1, a1 = np.polyfit(xs, ys, 1)
    print(f"M1 line (n={len(xs)}): delta = {a1:.3f} + {b1:.3f} * e_cov")

    pairs = calibrate()
    print("M2 calibration (b_w, gap, s_c) from measured Amazon domains:")
    for dom, bw, gap, sc in pairs:
        print(f"  {dom:26s} b_w={bw:+.3f}  gap={gap:+.3f}  s_c={sc:+.4f}")

    fp = RES / "blind_predictions_round2.json"
    out = json.loads(fp.read_text()) if fp.exists() else {}
    out.setdefault("M1_line", {"intercept": a1, "slope": b1,
                               "fitted_on": M1_DOMAINS})
    out["M2_calibration"] = [
        {"dom": d, "b_w": bw, "gap": g, "s_c": sc} for d, bw, g, sc in pairs]
    for dom in new_domains:
        e_cov, _, _ = ecov_for(dom)
        m1 = a1 + b1 * e_cov

        ages_a, sf_a = sf_curve(dom)
        ages_c, sf_c = sf_curve_c(dom)
        t0 = min(ages_a.min(), ages_c.min())
        samples = []
        for _, bw, gap, sc in pairs:
            la, lc = np.log(ages_a / t0), np.log(ages_c / t0)
            ya = sf_a * (bw * la - gap - sc * la) + sc * la
            yc = sf_c * (bw * lc - gap - sc * lc) + sc * lc
            off = 0.5 - min(ya.min(), yc.min())
            try:
                samples.append(joint_boxcox_delta(ages_a, ya + off,
                                                  ages_c, yc + off))
            except RuntimeError:
                pass
        m2 = float(np.median(samples))
        lo, hi = np.percentile(samples, [10, 90])
        out[dom] = dict(
            e_cov=float(e_cov), M1_delta_hat=float(m1),
            M2_delta_hat=m2, M2_spread=[float(lo), float(hi)],
            M2_samples=[float(s) for s in samples],
            note="M2 revised pre-training: v1 simulated only the A arm and "
                 "saturated at the bound; v2 simulates the joint A+C fit "
                 "(the actual dictionary target)",
            criteria="per-domain success: |delta_hat - delta| < 0.35 or "
                     "in measured 90% CI; M2 beats M1 if lower MAE; "
                     "rank order preserved")
        print(f"{dom}: e_cov={e_cov:.3f}  M1={m1:+.2f}  "
              f"M2={m2:+.2f} [{lo:+.2f},{hi:+.2f}]")
    fp.write_text(json.dumps(out, indent=2))
    print(f"registered in {fp}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if any(a in ("-h", "--help") for a in args):
        print(__doc__.strip())
        sys.exit(0)
    main(args)
