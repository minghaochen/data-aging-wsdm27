"""LEGACY EXPLORATORY -- not used by the paper. Superseded by src/fit_domains.py.

Power-law fits for early KuaiRand-Pure exploratory runs in
results/results_kuairand.jsonl (day-scale analog of src/fit.py). Exp A here is
fixed-D (subsampled), so A and C are both pure age effects; Exp B carries the
volume term.

Usage: python src/fit_kuairand.py
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
RES = ROOT / "results" / "results_kuairand.jsonl"
FIG = ROOT / "results"

W_A = 2.0     # Exp A window length, days
T_TE = 3.0    # test window length, days


def load(metric="loss"):
    rows = [json.loads(l) for l in RES.read_text().splitlines() if l]
    a, b, c = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in rows:
        {"A": a, "B": b, "C": c}[r["exp"]][
            r["gap_days"] if r["exp"] == "A" else r["w_days"]].append(r)

    def agg(d):
        xs = sorted(d)
        mean = np.array([np.mean([r[metric] for r in d[x]]) for x in xs])
        std = np.array([np.std([r[metric] for r in d[x]]) for x in xs])
        ntr = np.array([d[x][0]["n_train"] for x in xs])
        return np.array(xs, float), mean, std, ntr

    return agg(a), agg(b), agg(c)


def f_age(t, c, beta, delta):
    return c + beta * t ** delta


def fit_all(metric="loss"):
    (ga, la, sa, _), (wb, lb, sb, nb), (wc, lc, sc, nc) = load(metric)

    age_a = ga + W_A / 2 + T_TE / 2
    pa, _ = curve_fit(f_age, age_a, la, p0=[la.min() - 0.01, 1e-3, 1.0],
                      bounds=([0, 1e-8, 0.005], [la.min(), 10, 4]),
                      maxfev=50000)
    r2_a = 1 - np.sum((la - f_age(age_a, *pa)) ** 2) / np.sum(
        (la - la.mean()) ** 2)

    age_c = wc / 2 + T_TE / 2
    pc, _ = curve_fit(f_age, age_c, lc, p0=[lc.min() - 0.01, 1e-3, 1.0],
                      bounds=([0, 1e-8, 0.005], [lc.min(), 10, 4]),
                      maxfev=50000)
    r2_c = 1 - np.sum((lc - f_age(age_c, *pc)) ** 2) / np.sum(
        (lc - lc.mean()) ** 2)

    # joint A+C: shared beta/delta, separate intercepts
    age_all = np.concatenate([age_a, age_c])
    loss_all = np.concatenate([la, lc])
    grp = np.concatenate([np.zeros(len(age_a)), np.ones(len(age_c))])

    def f_joint(x, cA, cC, beta, delta):
        t, gi = x
        return np.where(gi == 0, cA, cC) + beta * t ** delta

    pj, _ = curve_fit(f_joint, (age_all, grp), loss_all,
                      p0=[la.min() - 0.01, lc.min() - 0.01, 1e-3, 1.0],
                      bounds=([0, 0, 1e-8, 0.05], [la.min(), lc.min(), 10, 4]),
                      maxfev=100000)
    r2_j = 1 - np.sum((loss_all - f_joint((age_all, grp), *pj)) ** 2) / np.sum(
        (loss_all - loss_all.mean()) ** 2)

    # Exp B combined model, delta from joint fit
    delta_j = pj[3]
    age_b = wb / 2 + T_TE / 2

    def f_b(x, c, alpha, gamma, beta):
        D, t = x
        return c + alpha * D ** (-gamma) + beta * t ** delta_j

    pb, _ = curve_fit(f_b, (nb.astype(float), age_b), lb,
                      p0=[lb.min() - 0.02, 10.0, 0.3, pj[2]],
                      bounds=([0, 1e-8, 0.01, 1e-8], [lb.min(), 1e6, 2, 10]),
                      maxfev=100000)
    r2_b = 1 - np.sum((lb - f_b((nb.astype(float), age_b), *pb)) ** 2) / np.sum(
        (lb - lb.mean()) ** 2)

    rate = np.median(nb / wb)
    Wgrid = np.geomspace(wb.min(), wb.max(), 400)
    Lgrid = f_b((rate * Wgrid, Wgrid / 2 + T_TE / 2), *pb)
    W_star = Wgrid[np.argmin(Lgrid)]

    return {
        "metric": metric,
        "A": {"x": ga, "age": age_a, "loss": la, "std": sa,
              "c": pa[0], "beta": pa[1], "delta": pa[2], "r2": r2_a},
        "C": {"x": wc, "age": age_c, "loss": lc, "std": sc,
              "c": pc[0], "beta": pc[1], "delta": pc[2], "r2": r2_c},
        "joint": {"cA": pj[0], "cC": pj[1], "beta": pj[2], "delta": pj[3],
                  "r2": r2_j},
        "B": {"w": wb, "loss": lb, "std": sb, "n": nb,
              "c": pb[0], "alpha": pb[1], "gamma": pb[2], "beta": pb[3],
              "delta": delta_j, "r2": r2_b,
              "W_star": W_star, "Wgrid": Wgrid, "Lgrid": Lgrid},
    }


def make_fig(res, fname):
    A, B, C, J = res["A"], res["B"], res["C"], res["joint"]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.2))

    ax = axes[0]
    ax.errorbar(A["age"], A["loss"], yerr=A["std"], fmt="o", capsize=3,
                label="Exp A (translated, D=25k)")
    ax.errorbar(C["age"], C["loss"], yerr=C["std"], fmt="s", capsize=3,
                label="Exp C (widening, D=25k)")
    tg = np.geomspace(min(A["age"].min(), C["age"].min()),
                      max(A["age"].max(), C["age"].max()), 200)
    ax.plot(tg, J["cA"] + J["beta"] * tg ** J["delta"], "--", color="C0",
            label=(f"joint fit A: {J['cA']:.3f}+{J['beta']:.2e}"
                   f"·τ^{J['delta']:.2f}"))
    ax.plot(tg, J["cC"] + J["beta"] * tg ** J["delta"], "--", color="C1",
            label=(f"joint fit C: {J['cC']:.3f}+... (R²={J['r2']:.3f})"))
    ax.set_xscale("log")
    ax.set_xlabel("mean data age τ (days, log)")
    ax.set_ylabel(res["metric"])
    ax.set_title("Age effect at fixed D (A: translated, C: widening)")
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.errorbar(B["w"], B["loss"], yerr=B["std"], fmt="o", capsize=3,
                label="observed")
    ax.plot(B["Wgrid"], B["Lgrid"], "--",
            label=(f"fit: c+α·D^-{B['gamma']:.2f}+β·τ^{B['delta']:.2f}"
                   f" (R²={B['r2']:.3f})\nW*≈{B['W_star']:.1f} days"))
    ax.axvline(B["W_star"], color="gray", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("training window length W (days, log)")
    ax.set_ylabel(res["metric"])
    ax.set_title("Exp B: window ends at test start, natural volume")
    ax.legend(fontsize=8)

    ax = axes[2]
    resid_a = A["loss"] - J["cA"]
    resid_c = C["loss"] - J["cC"]
    ax.loglog(A["age"], resid_a, "o", label="A: L - cA")
    ax.loglog(C["age"], resid_c, "s", label="C: L - cC")
    ax.loglog(tg, J["beta"] * tg ** J["delta"], "--",
              label=f"β·τ^{J['delta']:.2f}")
    ax.set_xlabel("mean data age τ (days, log)")
    ax.set_ylabel("age penalty (log)")
    ax.set_title("log-log check of the age power law")
    ax.legend(fontsize=8)

    fig.suptitle(f"Data-age scaling law on KuaiRand-Pure ({res['metric']})")
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=150)
    print(f"saved {FIG / fname}")


def main():
    for metric in ("loss", "loss_seen"):
        res = fit_all(metric)
        A, B, C, J = res["A"], res["B"], res["C"], res["joint"]
        print(f"\n=== metric: {metric} ===")
        print(f"Exp A: L = {A['c']:.4f} + {A['beta']:.3e}*tau^{A['delta']:.3f}"
              f"  R2={A['r2']:.4f}")
        print(f"Exp C: L = {C['c']:.4f} + {C['beta']:.3e}*tau^{C['delta']:.3f}"
              f"  R2={C['r2']:.4f}")
        print(f"Joint: cA={J['cA']:.4f} cC={J['cC']:.4f} "
              f"beta={J['beta']:.3e} delta={J['delta']:.3f}  R2={J['r2']:.4f}")
        print(f"Exp B: L = {B['c']:.4f} + {B['alpha']:.3g}*D^-{B['gamma']:.3f}"
              f" + {B['beta']:.3e}*tau^{B['delta']:.3f}  R2={B['r2']:.4f}")
        print(f"       W* ~ {B['W_star']:.1f} days "
              f"(empirical argmin: {B['w'][np.argmin(B['loss'])]:.0f} days)")
        make_fig(res, f"kuairand_{metric}.png")


if __name__ == "__main__":
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__.strip())
        sys.exit(0)
    main()
