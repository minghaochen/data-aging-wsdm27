"""Does the two-term simulator transfer ACROSS platforms?

Both pre-registered blind rounds held out Amazon categories and calibrated on
Amazon categories, so what they tested was transfer within one platform. This
script runs the complementary test the panel already contains the data for:
leave-one-PLATFORM-out. For each target domain we recalibrate the simulator's
(b_w, gap, s_c) triplets on the measured domains of *every other platform*,
feed it only the target's training-free coverage curves, and compare the
simulated Box-Cox exponent with the measured one.

This is a RETRODICTION, not a blind test: these domains were measured before
the simulator was written, and only the calibration set is held out. It
cannot establish predictive accuracy. What it can do -- and what neither
blind round could -- is show whether the mechanism survives being carried
from one platform to a different one.

Models compared:
  M1: linear elasticity baseline delta_hat = a + b * e_cov (refit on the
      same leave-one-platform-out calibration set).
  M2: two-term simulator transplanting (b_w, gap, s_c) triplets from measured
      domains on other platforms into the target's coverage curve sf(tau).
  const: mean measured exponent of the leave-one-platform-out calibration set.

Usage: python src/cross_platform_test.py
"""
import json
import sys
from pathlib import Path

import numpy as np

_SRC = str(Path(__file__).resolve().parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from blind_predict2 import joint_boxcox_delta, sf_curve, sf_curve_c
from two_term_law import cells_for
from dictionary_analysis import coverage_elasticity

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

# platform of each measured domain: the three Kuaishou logs are one platform,
# the twelve Amazon categories are one platform, game-UGC's two labels are one
PLATFORM = {
    "kuairand": "kuaishou", "kuairec": "kuaishou", "recflow": "kuaishou",
    "ml25m": "movielens", "netflix": "netflix", "yelp": "yelp",
    "steam": "steam", "beer": "beeradvocate", "epinions": "epinions",
    "gameugc_play": "gameugc", "gameugc_finish": "gameugc",
}


def platform_of(dom):
    return "amazon" if dom.startswith("amz_") else PLATFORM.get(dom, dom)


def triplet(dom):
    """(b_w, gap, s_c) for one measured domain -- same estimator as
    blind_predict2.calibrate(), which only ever ran it on Amazon."""
    ages, agg = cells_for(dom)
    m = ~np.isnan(agg["unseen"])
    if m.sum() < 3 or len(ages) < 3:
        return None
    lnr = np.log(ages / ages.min())
    bw, cw = np.polyfit(lnr, agg["seen"], 1)
    sc, cc = np.polyfit(lnr[m], agg["unseen"][m], 1)
    return float(bw), float(cc) - float(cw), float(sc)


def simulate(dom, triplets):
    """Simulated joint-arm Box-Cox delta for `dom`, using only its
    training-free coverage curves plus transplanted triplets."""
    ages_a, sf_a = sf_curve(dom)
    ages_c, sf_c = sf_curve_c(dom)
    t0 = min(ages_a.min(), ages_c.min())
    la, lc = np.log(ages_a / t0), np.log(ages_c / t0)
    samples = []
    for bw, gap, sc in triplets:
        ya = sf_a * (bw * la - gap - sc * la) + sc * la
        yc = sf_c * (bw * lc - gap - sc * lc) + sc * lc
        off = 0.5 - min(ya.min(), yc.min())
        try:
            samples.append(joint_boxcox_delta(ages_a, ya + off,
                                              ages_c, yc + off))
        except (RuntimeError, ValueError):
            pass
    if not samples:
        return None, None
    return float(np.median(samples)), samples


def check_reimplementation():
    """The cross-platform arm must use the SAME simulator as round two.

    Rebuild the registered round-2 calibration set and re-predict the four
    hold-outs; the values have to match the registration file exactly, or
    any difference below is an artifact of this script rather than of the
    platform swap.
    """
    from blind_predict2 import CALIB_AMZ
    reg = json.loads((RES / "blind_predictions_round2.json").read_text())
    trip = [t for t in (triplet(d) for d in CALIB_AMZ) if t is not None]
    ok = True
    for dom in [d for d in reg if d.startswith("amz_")]:
        m2, _ = simulate(dom, trip)
        got, want = round(m2, 4), round(reg[dom]["M2_delta_hat"], 4)
        ok &= got == want
        print(f"  {dom:28s} registered {want:+.4f}  reimplemented {got:+.4f}"
              f"{'' if got == want else '   <-- MISMATCH'}")
    print(f"  reimplementation matches the registration file: {ok}")
    return ok


def main():
    print("Reproducing the registered round-2 predictions with this code:")
    check_reimplementation()
    print()
    dd = json.loads((RES / "delta_dictionary.json").read_text())
    # targets: every measured domain whose delta is an actual measurement
    targets = [d for d, v in dd.items()
               if "__" not in d and not v["at_bound"] and not v["no_drift"]]
    # calibration pool: any domain we can build a triplet for
    pool = {}
    for d in dd:
        if "__" in d:
            continue
        t = triplet(d)
        if t is not None:
            pool[d] = t

    print(f"{len(targets)} interior-fit targets, "
          f"{len(pool)} domains in the calibration pool")
    print(f"platforms represented: "
          f"{sorted({platform_of(d) for d in pool})}\n")

    rows = []
    for dom in sorted(targets):
        plat = platform_of(dom)
        # every calibration domain from a DIFFERENT platform
        src = [d for d in pool if platform_of(d) != plat]
        trip = [pool[d] for d in src]
        m2, samples = simulate(dom, trip)
        if m2 is None:
            continue
        # M1 line and constant baseline, refit on the same held-out-platform
        # calibration set (interior fits only, as in the registered protocol)
        fit_on = [d for d in src
                  if not dd[d]["at_bound"] and not dd[d]["no_drift"]]
        xs = np.array([coverage_elasticity(d)[0] for d in fit_on])
        ys = np.array([dd[d]["delta"] for d in fit_on])
        keep = np.isfinite(xs)
        b1, a1 = np.polyfit(xs[keep], ys[keep], 1)
        e_cov = coverage_elasticity(dom)[0]
        m1 = a1 + b1 * e_cov
        const = float(ys[keep].mean())
        meas = dd[dom]["delta"]
        lo, hi = np.percentile(samples, [10, 90])
        rows.append(dict(dom=dom, platform=plat, measured=meas,
                         M2=m2, M2_spread=[float(lo), float(hi)],
                         M1=float(m1), const=const,
                         n_calib_domains=len(trip),
                         n_calib_platforms=len({platform_of(d)
                                                for d in src}),
                         err_M2=abs(m2 - meas), err_M1=abs(m1 - meas),
                         err_const=abs(const - meas)))
        print(f"{dom:20s} ({plat:12s})  measured {meas:+.2f}   "
              f"M2 {m2:+.2f} [{lo:+.2f},{hi:+.2f}] err {abs(m2-meas):.2f}   "
              f"M1 {m1:+.2f} err {abs(m1-meas):.2f}   "
              f"const {const:+.2f} err {abs(const-meas):.2f}")

    # ---- control: the SAME simulator, calibrated within the platform -----
    # This is the arm the blind rounds actually ran. Comparing it with the
    # cross-platform arm above separates "the transplant fails" from "the
    # simulator fails".
    print("\nWithin-platform control (leave-one-domain-out inside the "
          "target's own platform, where the platform has >1 measured domain):")
    within = []
    for dom in sorted(targets):
        plat = platform_of(dom)
        src = [d for d in pool if platform_of(d) == plat and d != dom]
        if len(src) < 2:
            continue
        m2, samples = simulate(dom, [pool[d] for d in src])
        if m2 is None:
            continue
        meas = dd[dom]["delta"]
        within.append(dict(dom=dom, platform=plat, measured=meas, M2=m2,
                           n_calib_domains=len(src), err_M2=abs(m2 - meas)))
        print(f"{dom:20s} ({plat:12s})  measured {meas:+.2f}   "
              f"M2 {m2:+.2f} err {abs(m2-meas):.2f}   "
              f"(calibrated on {len(src)} same-platform domains)")
    if within:
        ew = np.array([r["err_M2"] for r in within])
        print(f"  within-platform: {int((ew<0.35).sum())}/{len(ew)} hits, "
              f"MAE {ew.mean():.2f}")
        # paired comparison on the same targets
        cross = {r["dom"]: r["err_M2"] for r in rows}
        pair = [(r["dom"], r["err_M2"], cross[r["dom"]]) for r in within
                if r["dom"] in cross]
        if pair:
            wi = np.mean([p[1] for p in pair])
            cr = np.mean([p[2] for p in pair])
            print(f"  on the {len(pair)} targets scored both ways: "
                  f"within-platform MAE {wi:.2f} vs cross-platform "
                  f"MAE {cr:.2f}")

    # ---- dynamic range: can the simulator produce a positive exponent? ---
    from scipy import stats
    pm = np.array([r["M2"] for r in rows])
    me = np.array([r["measured"] for r in rows])
    sp = stats.spearmanr(pm, me)
    print(f"\nDynamic range over the {len(rows)} cross-platform targets:")
    print(f"  simulated delta spans [{pm.min():+.2f}, {pm.max():+.2f}]  "
          f"vs measured [{me.min():+.2f}, {me.max():+.2f}]")
    print(f"  Spearman(simulated, measured) = {sp.statistic:+.3f} "
          f"(p={sp.pvalue:.3f})")
    coll = [r for r in rows if r["measured"] <= 0]
    rise = [r for r in rows if r["measured"] > 0]
    for name, grp in (("measured delta <= 0", coll), ("measured delta > 0",
                                                      rise)):
        if grp:
            e = np.array([r["err_M2"] for r in grp])
            print(f"  {name:22s} n={len(grp):2d}  M2 MAE {e.mean():.2f}  "
                  f"hits {int((e<0.35).sum())}/{len(e)}")

    # ---- why: the simulator cannot express an accelerating exponent ------
    # Its warm term is log-linear in age by construction, so all curvature
    # comes from sf(tau), and a decaying sf can only bend the curve DOWN.
    # Sweep synthetic coverage curves to show the ceiling is structural, not
    # a property of any particular calibration.
    print("\nStructural range of the simulator (synthetic coverage curves):")
    print(f"{'sf(tau0)':>9s} {'coverage decay':>15s} {'simulated delta':>16s}")
    ceiling = []
    aa = np.array([2.5, 3.5, 4.5, 6.5, 10.5, 18.5, 26.5])
    cc = np.array([2.5, 3.5, 5.5, 9.5, 14.5])
    t0 = 2.5
    la_, lc_ = np.log(aa / t0), np.log(cc / t0)
    bw, gap, sc = 0.119, -0.129, 0.0041          # a mid-panel triplet
    for sf0 in (0.62, 0.30, 0.10):
        for dec in (0.0, 0.3, 0.8, 0.99):
            sfa, sfc = sf0 * (aa / t0) ** -dec, sf0 * (cc / t0) ** -dec
            ya = sfa * (bw * la_ - gap - sc * la_) + sc * la_
            yc = sfc * (bw * lc_ - gap - sc * lc_) + sc * lc_
            off = 0.5 - min(ya.min(), yc.min())
            try:
                dsim = joint_boxcox_delta(aa, ya + off, cc, yc + off)
            except (RuntimeError, ValueError):
                continue
            ceiling.append(dsim)
            print(f"{sf0:9.2f} {dec:15.2f} {dsim:+16.3f}")
    print(f"  max simulated delta over the sweep: {max(ceiling):+.3f} "
          f"-- the model cannot reach the accelerating regime, where "
          f"{sum(1 for r in rows if r['measured'] > 0)} of the "
          f"{len(rows)} measured targets live.")

    tol = 0.35
    out = dict(rows=rows, within_platform=within, tolerance=tol,
               structural_ceiling=float(max(ceiling)),
               dynamic_range=dict(
                   sim_min=float(pm.min()), sim_max=float(pm.max()),
                   meas_min=float(me.min()), meas_max=float(me.max()),
                   spearman=float(sp.statistic), p=float(sp.pvalue)))
    print(f"\n{'predictor':12s} {'hits':>8s} {'MAE':>7s} {'median AE':>10s}")
    for k in ("M2", "M1", "const"):
        e = np.array([r[f"err_{k}"] for r in rows])
        out[k] = dict(hits=int((e < tol).sum()), n=len(e),
                      mae=float(e.mean()), median_ae=float(np.median(e)))
        print(f"{k:12s} {int((e<tol).sum()):4d}/{len(e):<3d} "
              f"{e.mean():7.2f} {np.median(e):10.2f}")

    # how the same simulator scored on the within-platform blind rounds, for
    # contrast (registration-box numbers, from paper_numbers.py)
    print("\nFor contrast, the registered within-Amazon round-2 scoring was "
          "M2 3/3 hits, MAE 0.11.")

    (RES / "cross_platform_test.json").write_text(json.dumps(out, indent=2))
    print("saved results/cross_platform_test.json")


if __name__ == "__main__":
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__.strip())
        sys.exit(0)
    main()
