"""Two-mechanism decomposition of the data-age law (pure re-analysis).

Every run recorded (loss, loss_seen, seen_frac), and the identity
    loss = sf * loss_seen + (1 - sf) * loss_unseen
lets us recover loss_unseen algebraically per run. Tests:

  H1 (cold plateau): loss_unseen is ~flat in window age.
  H2 (shared warm drift): loss_seen follows c + beta*tau^delta_w with
     delta_w shared across Amazon categories.
  H3 (generative): the apparent overall-loss delta spectrum is reproduced
     by mixing the fitted warm law with the MEASURED sf(tau).

Exp A rows only (translated windows; C mixes ages inside the window).

Usage: python src/two_term_law.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

AMZ = ["amz_Video_Games", "amz_CDs_and_Vinyl", "amz_Software",
       "amz_Appliances", "amz_Baby_Products", "amz_Toys_and_Games",
       "amz_Office_Products", "amz_Pet_Supplies"]
ALL = ["kuairand", "ml25m"] + AMZ


def cells_for(dom):
    fp = RES / f"domain_{dom}.jsonl"
    rows = [json.loads(l) for l in fp.read_text().splitlines() if l]
    cells = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["exp"] != "A":
            continue
        sf, ls, lo = r["seen_frac"], r["loss_seen"], r["loss"]
        lu = (lo - sf * ls) / (1 - sf) if sf < 0.999 else np.nan
        c = cells[round(r["age_days"], 2)]
        c["sf"].append(sf)
        c["seen"].append(ls)
        c["unseen"].append(lu)
        c["loss"].append(lo)
    ages = np.array(sorted(cells))
    agg = {k: np.array([np.mean(cells[a][k]) for a in ages])
           for k in ("sf", "seen", "unseen", "loss")}
    return ages, agg


def main():
    print("=== H1: is the cold loss flat in age? "
          "(range of L_unseen vs range of L_overall) ===")
    data = {}
    for dom in ALL:
        ages, agg = cells_for(dom)
        data[dom] = (ages, agg)
        ru = np.nanmax(agg["unseen"]) - np.nanmin(agg["unseen"])
        ro = agg["loss"].max() - agg["loss"].min()
        rs = np.nanmax(agg["seen"]) - np.nanmin(agg["seen"])
        # correlation of unseen loss with log-age
        m = ~np.isnan(agg["unseen"])
        slope_u = np.polyfit(np.log(ages[m]), agg["unseen"][m], 1)[0]
        slope_s = np.polyfit(np.log(ages[m]), agg["seen"][m], 1)[0]
        print(f"{dom:26s} range: unseen={ru:.4f} seen={rs:.4f} "
              f"overall={ro:.4f} | slope/ln(age): unseen={slope_u:+.4f} "
              f"seen={slope_s:+.4f}")

    print("\n=== H2: warm drift exponent shared across Amazon? ===")
    # per-domain warm Box-Cox fit and a joint fit with shared delta_w
    def bc(t, t0, delta):
        r = t / t0
        return np.log(r) if abs(delta) < 1e-6 else (r ** delta - 1) / delta

    per = {}
    for dom in AMZ + ["ml25m", "kuairand"]:
        ages, agg = data[dom]
        y = agg["seen"]
        t0 = ages.min()

        def f(t, c, beta, delta):
            return c + beta * bc(t, t0, delta)

        try:
            p, _ = curve_fit(f, ages, y,
                             p0=[y.min(), 0.005, 0.3],
                             bounds=([y.min() - 0.1, 0, -1],
                                     [y.min() + 0.05, 10, 4]),
                             maxfev=100000)
            pred = f(ages, *p)
            r2 = 1 - np.sum((y - pred) ** 2) / max(np.sum(
                (y - y.mean()) ** 2), 1e-12)
            per[dom] = dict(delta_w=float(p[2]), beta_w=float(p[1]),
                            c_w=float(p[0]), r2=float(r2))
            print(f"{dom:26s} delta_w={p[2]:+.3f}  beta_w={p[1]:.3e}  "
                  f"R2={r2:.3f}")
        except RuntimeError as ex:
            print(f"{dom:26s} fit failed: {ex}")

    # joint Amazon fit: shared delta_w, per-domain c and beta
    doms = [d for d in AMZ if d in per]
    def pack(delta_w, cs, betas):
        sses = 0.0
        for d, c, b in zip(doms, cs, betas):
            ages, agg = data[d]
            pred = c + b * np.array([bc(t, ages.min(), delta_w)
                                     for t in ages])
            sses += np.sum((agg["seen"] - pred) ** 2)
        return sses

    from scipy.optimize import minimize
    def obj(theta):
        return pack(theta[0], theta[1:1 + len(doms)], theta[1 + len(doms):])

    x0 = [0.3] + [per[d]["c_w"] for d in doms] + [per[d]["beta_w"]
                                                  for d in doms]
    r = minimize(obj, x0, method="Nelder-Mead",
                 options={"maxiter": 20000, "xatol": 1e-6, "fatol": 1e-10})
    sse_shared = r.fun
    sse_free = sum(
        np.sum((data[d][1]["seen"]
                - (per[d]["c_w"] + per[d]["beta_w"]
                   * np.array([bc(t, data[d][0].min(), per[d]["delta_w"])
                               for t in data[d][0]]))) ** 2) for d in doms)
    n_pts = sum(len(data[d][0]) for d in doms)
    k_shared = 1 + 2 * len(doms)
    k_free = 3 * len(doms)
    aic_shared = n_pts * np.log(sse_shared / n_pts) + 2 * k_shared
    aic_free = n_pts * np.log(sse_free / n_pts) + 2 * k_free
    print(f"\nshared delta_w = {r.x[0]:.3f}   "
          f"AIC shared={aic_shared:.1f} vs free={aic_free:.1f}  "
          f"(shared wins if lower; dAIC={aic_shared - aic_free:+.1f})")

    print("\n=== H3: does [measured sf x warm + constant cold] reproduce "
          "the apparent delta? ===")

    def apparent_delta(ages, y):
        t0 = ages.min()

        def f(t, c, beta, delta):
            return c + beta * np.array([bc(x, t0, delta) for x in t])

        p, _ = curve_fit(f, ages, y, p0=[y.min(), 0.005, 0.3],
                         bounds=([y.min() - 0.05, 0, -1],
                                 [y.min() + 0.03, 10, 4]), maxfev=100000)
        return p[2]

    h3 = {}
    for dom in ALL:
        ages, agg = data[dom]
        m = ~np.isnan(agg["unseen"])
        c_cold = float(np.nanmean(agg["unseen"]))
        y_mix = agg["sf"] * agg["seen"] + (1 - agg["sf"]) * c_cold
        try:
            d_true = apparent_delta(ages, agg["loss"])
            d_mix = apparent_delta(ages, y_mix)
            h3[dom] = (float(d_true), float(d_mix))
            print(f"{dom:26s} apparent delta: measured={d_true:+.3f}  "
                  f"mixture-implied={d_mix:+.3f}  diff={d_mix - d_true:+.3f}")
        except RuntimeError:
            print(f"{dom:26s} H3 fit failed")

    if len(h3) >= 4:
        a = np.array([v[0] for v in h3.values()])
        b_ = np.array([v[1] for v in h3.values()])
        from scipy import stats as sst
        rr = sst.pearsonr(a, b_)
        print(f"\nmeasured vs mixture-implied apparent delta: "
              f"pearson={rr.statistic:.3f} (p={rr.pvalue:.4f})  "
              f"mean|diff|={np.mean(np.abs(a - b_)):.3f}")

    # figure: decomposition for each domain
    ncol = 5
    nrow = 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow),
                             squeeze=False)
    for i, dom in enumerate(ALL):
        ages, agg = data[dom]
        ax = axes[i // ncol][i % ncol]
        ax.semilogx(ages, agg["loss"], "o-", ms=4, label="overall")
        ax.semilogx(ages, agg["seen"], "s-", ms=4, label="warm (seen)")
        m = ~np.isnan(agg["unseen"])
        ax.semilogx(ages[m], agg["unseen"][m], "^-", ms=4, label="cold")
        ax2 = ax.twinx()
        ax2.semilogx(ages, agg["sf"], ":", color="gray", lw=1)
        ax2.set_ylabel("seen_frac", fontsize=7, color="gray")
        ax2.tick_params(labelsize=6, colors="gray")
        ax.set_title(dom.replace("amz_", "AMZ "), fontsize=9)
        ax.set_xlabel("age (days)", fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)
    fig.suptitle("Two-mechanism decomposition: overall = sf·warm + (1−sf)·cold")
    fig.tight_layout()
    fig.savefig(RES / "two_term_decomposition.png", dpi=150)
    (RES / "two_term_fits.json").write_text(json.dumps(
        {"per_domain": per, "shared_delta_w": float(r.x[0]),
         "aic_shared": float(aic_shared), "aic_free": float(aic_free)},
        indent=2))
    print(f"saved figure and two_term_fits.json")


if __name__ == "__main__":
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__.strip())
        sys.exit(0)
    main()
