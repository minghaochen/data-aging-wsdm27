"""Does the static age law (Exp A) predict rolling-retrain serving loss?

Prediction: a model retrained with window W and served at offset u days has
mean data age u + W/2 (+0.5 for the daily midpoint). So
    Lbar_pred(P) = mean_{u=0..P-1} L_A(age = u + 0.5 + W/2)
with L_A interpolated (log-age linear) from the Exp-A curve of the SAME
domain -- measured on a single fixed test period, so agreement is an
out-of-period transfer test.

Two things this script is now careful about, because they were overstated:

* THE TWO BETAS.  The age law is L(tau) = c + beta*((tau/tau0)^delta - 1)/delta,
  i.e. L = L0 + b*tau^delta with b = beta / (delta * tau0^delta).  The
  serving-time average over a period P is Lbar(P) = L0 + b/(delta+1) * P^delta.
  The closed form P* = [lamC*(delta+1) / (b*delta)]^(1/(1+delta)) is stated in
  terms of b, NOT of the serving-average coefficient b/(delta+1); mixing the
  two shifts P* by (delta+1)^(1/(1+delta)) (~1.4x here).  We report the
  out-of-sample P* (b, delta taken from the STATIC age curve) as the headline
  and the in-sample P* (b, delta refit on the backtest curve itself) only as
  a diagnostic -- the latter cannot validate the age law, since both sides
  then come from the same data.

* MONOTONE NULLS.  The shape correlation between measured and predicted
  staleness penalty is computed over a handful of periods, and any two
  increasing curves correlate near 1 there.  We therefore report the same
  correlation for two null shapes (log P and linear P) so the reader can see
  how much of it is information and how much is monotonicity.

Usage: python src/backtest_analysis.py <domain> [suffix]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

W = {"recflow": 2, "kuairec": 3}
TEST_HALF = {"recflow": 1.5, "kuairec": 2.5}


def a_curve(domain, metric="loss"):
    rows = [json.loads(l) for l in
            (RES / f"domain_{domain}.jsonl").read_text().splitlines() if l]
    cells = defaultdict(list)
    for r in rows:
        if r["exp"] == "A":
            cells[round(r["age_days"], 2)].append(r[metric])
    ages = np.array(sorted(cells))
    return ages, np.array([np.mean(cells[a]) for a in ages])


def fit_power(ages, y):
    """L = L0 + b * age^delta, by grid over delta with exact linear solve.
    Returns (L0, b, delta) -- b is the coefficient of tau^delta, the
    quantity Eq. (P*) expects."""
    best = None
    for d in np.linspace(-0.9, 4.0, 981):
        X = np.column_stack([np.ones_like(ages), ages ** d])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        if coef[1] <= 0:
            continue
        ss = float(np.sum((X @ coef - y) ** 2))
        if best is None or ss < best[0]:
            best = (ss, coef[0], coef[1], d)
    _, L0, b, d = best
    return L0, b, d


def backtest_curve(domain, suffix=""):
    rows = [json.loads(l) for l in
            (RES / f"backtest_{domain}{suffix}.jsonl").read_text().splitlines()
            if l]
    agg = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    for r in rows:
        a = agg[r["P"]]
        a[0] += r["loss"] * r["n_test"]
        a[1] += r["n_test"]
        a[2] += r["auc"] * r["n_test"]
        a[3] += 1
    Ps = np.array(sorted(agg))
    lbar = np.array([agg[p][0] / agg[p][1] for p in Ps])
    auc = np.array([agg[p][2] / agg[p][1] for p in Ps])
    nev = np.array([agg[p][3] for p in Ps])
    return Ps, lbar, auc, nev


def main(domain, suffix=""):
    ages, la = a_curve(domain)
    Ps, lbar, auc, nev = backtest_curve(domain, suffix)
    # Note: the default RecFlow backtest uses D=217,575 vs. D=300,000 in the
    # static sweep; suffix="__d300k" evaluates the volume-matched backtest.
    if suffix:
        print(f"[variant {suffix}]")

    def L_A(age):
        return np.interp(np.log(age), np.log(ages), la)

    pred = np.array([np.mean(L_A(np.arange(P) + 0.5 + W[domain] / 2))
                     for P in Ps])

    print(f"{domain}: rolling-retrain backtest")
    print(f"  periods probed: {list(Ps)}  "
          f"(model evaluations per period: {list(nev)})")
    print(f"  {'P':>3s} {'measured':>9s} {'predicted':>9s} {'diff':>9s} "
          f"{'AUC':>8s}")
    for P, m, p, a, n in zip(Ps, lbar, pred, auc, nev):
        print(f"  {P:3d} {m:9.5f} {p:9.5f} {1000*(m-p):+7.2f}mn {a:8.5f}")
    print(f"  total staleness penalty over the probed range: "
          f"{1000*(lbar[-1]-lbar[0]):+.1f} mnat "
          f"(AUC {auc[0]:.4f} -> {auc[-1]:.4f}, "
          f"{100*(auc[-1]-auc[0]):+.2f} points)")

    dm, dp = lbar - lbar[0], pred - pred[0]
    n_corr = len(Ps) - 1
    rr = np.corrcoef(dm[1:], dp[1:])[0, 1]
    ratio = np.polyfit(dp[1:], dm[1:], 1)[0]
    # monotone nulls on the same points
    null_log = np.log(Ps) - np.log(Ps[0])
    null_lin = Ps - Ps[0]
    r_log = np.corrcoef(dm[1:], null_log[1:])[0, 1]
    r_lin = np.corrcoef(dm[1:], null_lin[1:])[0, 1]
    print(f"  shape corr (age law)      = {rr:.4f}   (n={n_corr} points), "
          f"measured/predicted slope = {ratio:.2f}")
    print(f"  shape corr (log-P null)   = {r_log:.4f}")
    print(f"  shape corr (linear-P null)= {r_lin:.4f}")
    print("  -> the correlation alone does not discriminate; the slope ratio "
          "and the absolute scale do")

    # ---- optimal period -------------------------------------------------
    L0_s, b_s, d_s = fit_power(ages, la)          # STATIC age curve
    print(f"  static age curve : L = {L0_s:.5f} + {b_s:.3e} * age^{d_s:.3f}")
    pen = lbar - lbar[0]
    m = pen > 0
    d_i, lb = np.polyfit(np.log(Ps[m]), np.log(pen[m]), 1)
    b_serve_i = np.exp(lb)                        # serving-average coeff
    b_i = b_serve_i * (d_i + 1)
    print(f"  backtest curve   : penalty = {b_serve_i:.3e} * P^{d_i:.3f} "
          f"(serving-average coefficient)")

    Pg = np.geomspace(Ps.min(), Ps.max(), 400)
    Lg = np.interp(np.log(Pg), np.log(Ps), lbar)
    print(f"\n  {'lam*C':>9s} {'P* out-of-sample':>17s} {'P* in-sample':>13s} "
          f"{'P* empirical':>13s}")
    for lamC in np.geomspace(1e-5, 1e-2, 7):
        p_oos = (lamC * (d_s + 1) / (b_s * d_s)) ** (1 / (1 + d_s))
        p_ins = (lamC * (d_i + 1) / (b_i * d_i)) ** (1 / (1 + d_i))
        j = Lg + lamC / Pg
        k = int(np.argmin(j))
        p_emp = Pg[k]
        edge = " (at grid edge)" if k in (0, len(Pg) - 1) else ""
        print(f"  {lamC:9.1e} {p_oos:17.2f} {p_ins:13.2f} "
              f"{p_emp:13.2f}{edge}")
    print("  P* empirical pinned at the grid edge means the optimum lies "
          "outside the probed period range and the comparison is vacuous "
          "there.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    ax.plot(Ps, lbar, "o-", label="measured (rolling retrain backtest)")
    ax.plot(Ps, pred, "s--", label="predicted from static age law (Exp A)")
    ax.set_xscale("log")
    ax.set_xlabel("retraining period P (days)")
    ax.set_ylabel("volume-weighted serving loss")
    ax.set_title(f"{domain}: serving loss vs cadence")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(Ps, auc, "o-", color="C3")
    ax.set_xscale("log")
    ax.set_xlabel("retraining period P (days)")
    ax.set_ylabel("volume-weighted serving AUC")
    ax.set_title("Ranking quality vs cadence")
    fig.tight_layout()
    fig.savefig(RES / f"backtest_{domain}{suffix}.png", dpi=150)
    print(f"\nsaved backtest_{domain}{suffix}.png")


if __name__ == "__main__":
    args = sys.argv[1:]
    if any(a in ("-h", "--help") for a in args):
        print(__doc__.strip())
        sys.exit(0)
    if not args:
        raise SystemExit(
            "Usage: python src/backtest_analysis.py <domain> [suffix]"
        )
    main(args[0], args[1] if len(args) > 1 else "")
