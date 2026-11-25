"""LEGACY -- not used by the paper. Superseded by fit_domains.py.

This is the first-generation MovieLens-only pipeline, with a volume term
alpha * D^-gamma that the published protocol removes by subsampling every
window to a common D. No number in the manuscript comes from this file, and
paper_numbers.py never imports it; it is kept only so the earlier
exploratory runs in results/results*.jsonl remain readable.

Fit power laws to experiment results, following the decomposition in
kexue.fm/archives/11833:

  Exp A (fixed W, sliding gap):  L(tau) = c + beta * tau^delta
  Exp B (window ends at test):   L(W)   = c + alpha * D(W)^-gamma + beta * age(W)^delta

where tau/age is the mean age (months) of training data relative to the test
midpoint. For a window of length W ending gap months before a test window of
length T_te: mean_age = gap + W/2 + T_te/2.

Usage: python src/fit.py
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
RES = ROOT / "results" / "results.jsonl"
FIG = ROOT / "results"

W_A = 24.0      # Exp A window length, months
T_TE = 6.0      # test window length, months


def load(metric="loss"):
    rows = [json.loads(l) for l in RES.read_text().splitlines() if l]
    a, b, c = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in rows:
        if r["exp"] == "A":
            a[r["gap_months"]].append(r)
        elif r["exp"] == "B":
            b[r["w_months"]].append(r)
        elif r["exp"] == "C":
            c[r["w_months"]].append(r)

    def agg(d):
        xs = sorted(d)
        mean = np.array([np.mean([r[metric] for r in d[x]]) for x in xs])
        std = np.array([np.std([r[metric] for r in d[x]]) for x in xs])
        ntr = np.array([d[x][0]["n_train"] for x in xs])
        return np.array(xs, float), mean, std, ntr

    return agg(a), agg(b), agg(c)


def fit_all(metric="loss"):
    (ga, la, sa, _), (wb, lb, sb, nb), (wc, lc, sc, nc) = load(metric)

    # ---- Exp A: L = c + beta * age^delta
    age_a = ga + W_A / 2 + T_TE / 2

    def f_age(t, c, beta, delta):
        return c + beta * t ** delta

    p0 = [la.min() - 0.01, 1e-3, 1.0]
    bounds = ([0, 1e-8, 0.005], [la.min(), 10, 4])
    pa, _ = curve_fit(f_age, age_a, la, p0=p0, bounds=bounds, maxfev=50000)
    c_a, beta_a, delta_a = pa
    pred_a = f_age(age_a, *pa)
    r2_a = 1 - np.sum((la - pred_a) ** 2) / np.sum((la - la.mean()) ** 2)

    # ---- Exp B: L = c + alpha * D^-gamma + beta * age^delta  (delta fixed from A)
    age_b = wb / 2 + T_TE / 2

    def f_b(x, c, alpha, gamma, beta):
        D, t = x
        return c + alpha * D ** (-gamma) + beta * t ** delta_a

    p0 = [c_a, 10.0, 0.3, beta_a]
    bounds = ([0, 1e-8, 0.01, 1e-8], [lb.min(), 1e6, 2, 10])
    pb, _ = curve_fit(f_b, (nb.astype(float), age_b), lb, p0=p0,
                      bounds=bounds, maxfev=100000)
    c_b, alpha_b, gamma_b, beta_b = pb
    pred_b = f_b((nb.astype(float), age_b), *pb)
    r2_b = 1 - np.sum((lb - pred_b) ** 2) / np.sum((lb - lb.mean()) ** 2)

    # analytic optimum over continuous W: D(W) ~ rate*W (use mean rate)
    rate = np.median(nb / wb)  # events per month
    Wgrid = np.geomspace(wb.min(), wb.max(), 400)
    Lgrid = f_b((rate * Wgrid, Wgrid / 2 + T_TE / 2), *pb)
    W_star = Wgrid[np.argmin(Lgrid)]

    # ---- Exp C: fixed D, widening span -> pure age effect at constant volume
    # mean age of a uniform window [0, W] before a test window: W/2 + T_te/2
    res_c = None
    if len(wc):
        age_c = wc / 2 + T_TE / 2
        pc, _ = curve_fit(f_age, age_c, lc,
                          p0=[lc.min() - 0.01, 1e-3, 1.0],
                          bounds=([0, 1e-8, 0.005], [lc.min(), 10, 4]),
                          maxfev=50000)
        pred_c = f_age(age_c, *pc)
        r2_c = 1 - np.sum((lc - pred_c) ** 2) / np.sum((lc - lc.mean()) ** 2)
        res_c = {"w": wc, "age": age_c, "loss": lc, "std": sc, "n": nc,
                 "c": pc[0], "beta": pc[1], "delta": pc[2], "r2": r2_c}

    # ---- Joint A+C fit: shared beta/delta, separate intercepts (D differs).
    # Tests whether mean age is a sufficient statistic across two very
    # different age manipulations (translated window vs widening window).
    joint = None
    if len(wc):
        age_all = np.concatenate([age_a, age_c])
        loss_all = np.concatenate([la, lc])
        grp = np.concatenate([np.zeros(len(age_a)), np.ones(len(age_c))])

        def f_joint(x, cA, cC, beta, delta):
            t, gi = x
            return np.where(gi == 0, cA, cC) + beta * t ** delta

        pj, _ = curve_fit(f_joint, (age_all, grp), loss_all,
                          p0=[la.min() - 0.01, lc.min() - 0.01, 1e-3, 1.0],
                          bounds=([0, 0, 1e-8, 0.05],
                                  [la.min(), lc.min(), 10, 4]),
                          maxfev=100000)
        pred_j = f_joint((age_all, grp), *pj)
        r2_j = 1 - (np.sum((loss_all - pred_j) ** 2)
                    / np.sum((loss_all - loss_all.mean()) ** 2))
        joint = {"cA": pj[0], "cC": pj[1], "beta": pj[2], "delta": pj[3],
                 "r2": r2_j}

    return {
        "metric": metric,
        "joint": joint,
        "A": {"gap": ga, "age": age_a, "loss": la, "std": sa,
              "c": c_a, "beta": beta_a, "delta": delta_a, "r2": r2_a},
        "B": {"w": wb, "age": age_b, "loss": lb, "std": sb, "n": nb,
              "c": c_b, "alpha": alpha_b, "gamma": gamma_b, "beta": beta_b,
              "delta": delta_a, "r2": r2_b,
              "W_star": W_star, "Wgrid": Wgrid, "Lgrid": Lgrid},
        "C": res_c,
    }


def make_fig(res, fname):
    A, B, C = res["A"], res["B"], res.get("C")
    ncols = 3 if C else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 4.2))

    ax = axes[0]
    ax.errorbar(A["age"], A["loss"], yerr=A["std"], fmt="o", capsize=3,
                label="observed")
    tg = np.geomspace(A["age"].min(), A["age"].max(), 200)
    ax.plot(tg, A["c"] + A["beta"] * tg ** A["delta"], "--",
            label=(f"fit: {A['c']:.3f} + {A['beta']:.2e}"
                   f"·τ^{A['delta']:.2f}  (R²={A['r2']:.3f})"))
    ax.set_xscale("log")
    ax.set_xlabel("mean data age τ (months, log)")
    ax.set_ylabel(res["metric"])
    ax.set_title("Exp A: fixed 2y window, increasing age")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.errorbar(B["w"], B["loss"], yerr=B["std"], fmt="o", capsize=3,
                label="observed")
    ax.plot(B["Wgrid"], B["Lgrid"], "--",
            label=(f"fit: c+α·D^-{B['gamma']:.2f}+β·τ^{B['delta']:.2f}"
                   f"  (R²={B['r2']:.3f})\nW*≈{B['W_star']:.0f} months"))
    ax.axvline(B["W_star"], color="gray", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("training window length W (months, log)")
    ax.set_ylabel(res["metric"])
    ax.set_title("Exp B: window ends at test start, varying length")
    ax.legend(fontsize=8)

    if C:
        ax = axes[2]
        ax.errorbar(C["age"], C["loss"], yerr=C["std"], fmt="o", capsize=3,
                    label="observed (D=500k fixed)")
        tg = np.geomspace(C["age"].min(), C["age"].max(), 200)
        ax.plot(tg, C["c"] + C["beta"] * tg ** C["delta"], "--",
                label=(f"fit: {C['c']:.3f} + {C['beta']:.2e}"
                       f"·τ^{C['delta']:.2f}  (R²={C['r2']:.3f})"))
        ax.set_xscale("log")
        ax.set_xlabel("mean data age τ (months, log)")
        ax.set_ylabel(res["metric"])
        ax.set_title("Exp C: fixed D, widening span")
        ax.legend(fontsize=8)

    fig.suptitle(f"Data-age scaling law on MovieLens-25M ({res['metric']})")
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=150)
    print(f"saved {FIG / fname}")


def main():
    for metric in ("loss", "loss_seen"):
        res = fit_all(metric)
        A, B = res["A"], res["B"]
        print(f"\n=== metric: {metric} ===")
        print(f"Exp A: L = {A['c']:.4f} + {A['beta']:.3e} * tau^{A['delta']:.3f}"
              f"   R2={A['r2']:.4f}")
        print(f"Exp B: L = {B['c']:.4f} + {B['alpha']:.3g} * D^-{B['gamma']:.3f}"
              f" + {B['beta']:.3e} * tau^{B['delta']:.3f}   R2={B['r2']:.4f}")
        print(f"       optimal window W* ~ {B['W_star']:.1f} months "
              f"(empirical argmin: {B['w'][np.argmin(B['loss'])]:.0f} months)")
        if res.get("C"):
            Cx = res["C"]
            print(f"Exp C: L = {Cx['c']:.4f} + {Cx['beta']:.3e} * "
                  f"tau^{Cx['delta']:.3f}   R2={Cx['r2']:.4f}"
                  f"   [delta A vs C: {res['A']['delta']:.3f} vs {Cx['delta']:.3f}]")
        if res.get("joint"):
            J = res["joint"]
            print(f"Joint A+C (shared beta,delta): cA={J['cA']:.4f} "
                  f"cC={J['cC']:.4f} beta={J['beta']:.3e} "
                  f"delta={J['delta']:.3f}   R2={J['r2']:.4f}")
        make_fig(res, f"scaling_{metric}.png")


if __name__ == "__main__":
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__.strip())
        sys.exit(0)
    main()
