"""Fit the age law per domain with bootstrap CIs.

    L_X(tau) = c_X + beta * ((tau/tau0)^delta - 1)/delta        (Box-Cox)

Usage:  python src/fit_domains.py [METRIC] [DELTA_LO DELTA_HI]

METRIC selects the loss column: "loss" (overall, default), "loss_seen"
(warm subset) or "loss_cohort" (composition-fixed cohort).  Outputs go to
results/delta_dictionary[_METRIC][_boxLO_HI].json.

Three things this fitter is explicit about, because they changed conclusions:

1. ARMS.  The main runs carry both manipulations (A' translated, C' widened)
   and are fit jointly with a shared (beta, delta) and one intercept per arm.
   The cohort runs only ever carried the A' arm, so their fit is A'-only.
   The two are different estimators and the output records which was used
   (field "arms"), so A'-only and joint numbers are never silently compared.

2. THE DELTA BOX.  delta is searched on a bounded grid.  A fit that lands on
   a bound is not a measurement of delta, it is a statement that the data
   want to leave the box; such fits are flagged ("at_bound") and their
   bootstrap CI collapses to the bound.  Run with a wider box to see whether
   a boundary value is data or box.

3. beta >= 0 is enforced.  A curve that *falls* with age cannot be described
   by this family; it is flagged ("descending") instead of being fit with a
   meaningless delta.
"""
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "results"
METRIC = "loss"
DBOX = (-1.0, 4.0)
N_BOOT = 300
N_GRID = 241             # coarse delta grid; golden-section refined after
N_REFINE = 60            # golden-section iterations (delta precision ~1e-6)
BOUND_TOL = 1e-3


def load_rows(domain):
    fp = ROOT / "results" / f"domain_{domain}.jsonl"
    rows = [json.loads(l) for l in fp.read_text().splitlines() if l]
    return [r for r in rows if METRIC in r]


def _cells(rows, metric=None):
    metric = metric or METRIC
    cells = defaultdict(list)
    for r in rows:
        if metric in r:
            cells[(r["exp"], round(r["age_days"], 3))].append(r[metric])
    return cells


def _boxcox(t, t0, delta):
    r = t / t0
    if abs(delta) < 1e-9:
        return np.log(r)
    return (r ** delta - 1) / delta


def fit_joint(rows, seed=None, dbox=None, metric=None):
    """rows -> dict(delta, beta, intercepts, r2, ...).

    For a fixed delta the model is linear in (intercepts, beta), so we scan
    delta on a grid and solve the linear part exactly.  This is a global
    search over the box: unlike a local optimiser it cannot stall in a bad
    basin, which matters because several domains sit near a bound.
    """
    metric = metric or METRIC
    if dbox is None:
        dbox = DBOX
    cells = _cells(rows, metric)
    if seed is not None:
        rng = np.random.default_rng(seed)
        cells = {k: list(rng.choice(v, size=len(v), replace=True))
                 for k, v in cells.items()}

    arms = sorted({k[0] for k in cells})
    xs, ys, gs = [], [], []
    for (exp, age), vals in sorted(cells.items()):
        xs.append(age)
        ys.append(np.mean(vals))
        gs.append(arms.index(exp))
    xs, ys, gs = np.asarray(xs), np.asarray(ys), np.asarray(gs)
    t0 = xs.min()
    # one indicator column per arm, plus the shared age term
    D = np.column_stack([(gs == a).astype(float) for a in range(len(arms))])
    sstot = float(np.sum((ys - ys.mean()) ** 2))

    # beta = 0 fallback (intercepts only), precomputed once
    c0 = np.linalg.solve(D.T @ D, D.T @ ys)
    ss0 = float(np.sum((D @ c0 - ys) ** 2))

    def solve(delta):
        X = np.column_stack([D, _boxcox(xs, t0, delta)])
        G = X.T @ X
        try:
            coef = np.linalg.solve(G + 1e-12 * np.eye(len(G)), X.T @ ys)
        except np.linalg.LinAlgError:
            return np.append(c0, 0.0), ss0
        if coef[-1] < 0:                     # beta >= 0
            return np.append(c0, 0.0), ss0
        return coef, float(np.sum((X @ coef - ys) ** 2))

    grid = np.linspace(dbox[0], dbox[1], N_GRID)
    ssq = np.array([solve(d)[1] for d in grid])
    k = int(np.argmin(ssq))
    d_best, ss_best = grid[k], ssq[k]
    # golden-section refinement inside the winning bracket
    a = grid[max(k - 1, 0)]
    b = grid[min(k + 1, N_GRID - 1)]
    gr = (np.sqrt(5) - 1) / 2
    c_, d_ = b - gr * (b - a), a + gr * (b - a)
    fc, fd = solve(c_)[1], solve(d_)[1]
    for _ in range(N_REFINE):
        if fc < fd:
            b, d_, fd = d_, c_, fc
            c_ = b - gr * (b - a)
            fc = solve(c_)[1]
        else:
            a, c_, fc = c_, d_, fd
            d_ = a + gr * (b - a)
            fd = solve(d_)[1]
        if b - a < 1e-7:
            break
    for cand in (c_, d_, grid[k]):
        coef, s = solve(cand)
        if s <= ss_best:
            d_best, ss_best, coef_best = cand, s, coef
    coef_best = solve(d_best)[0]

    beta = float(coef_best[-1])
    r2 = 1 - ss_best / sstot if sstot > 0 else float("nan")

    # robust drift intensity: linear fit vs ln(tau) -> nats per e-fold
    A = np.column_stack([D, np.log(xs)])
    b_log = float(np.linalg.lstsq(A, ys, rcond=None)[0][-1])

    at_bound = (abs(d_best - dbox[0]) < BOUND_TOL
                or abs(d_best - dbox[1]) < BOUND_TOL)
    return dict(delta=float(d_best), beta=beta, r2=float(r2), b_log=b_log,
                t0=float(t0), arms=arms, at_bound=bool(at_bound),
                intercepts=[float(c) for c in coef_best[:-1]],
                descending=bool(beta <= 0 and b_log < 0),
                xs=xs, ys=ys, gs=gs, n_cells=len(xs))


def main():
    # Exclude auxiliary files (validation-stop robustness, alternate cohort
    # subsample seeds, multi-period sweeps __p0..__p2, and weekday control __w7)
    # so the dictionary contains the exact 68 primary and cohort settings.
    SKIP = ("valstop", "__ss", "__p0", "__p1", "__p2", "__w7")
    domains = sorted(
        Path(f).stem.replace("domain_", "")
        for f in glob.glob(str(ROOT / "results" / "domain_*.jsonl"))
        if not any(t in Path(f).stem for t in SKIP))
    out, panels = {}, []
    for dom in domains:
        rows = load_rows(dom)
        if len(rows) < 8:
            continue
        base = fit_joint(rows)
        if base["n_cells"] < 4:
            continue
        deltas, blogs = [], []
        for b in range(N_BOOT):
            try:
                fb = fit_joint(rows, seed=b)
                deltas.append(fb["delta"])
                blogs.append(fb["b_log"])
            except Exception:
                pass
        lo_d, hi_d = np.percentile(deltas, [5, 95])
        lo_b, hi_b = np.percentile(blogs, [5, 95])
        no_drift = base["beta"] < 1e-6 or base["r2"] < 0.2
        out[dom] = dict(
            delta=base["delta"], delta_lo=float(lo_d), delta_hi=float(hi_d),
            b_log=base["b_log"], b_log_lo=float(lo_b), b_log_hi=float(hi_b),
            beta=base["beta"], r2=base["r2"], t0=base["t0"],
            intercepts=base["intercepts"], arms=base["arms"],
            n_cells=base["n_cells"], at_bound=base["at_bound"],
            descending=base["descending"], no_drift=bool(no_drift),
            delta_box=list(DBOX), metric=METRIC, n_boot=len(deltas))
        flags = "".join([" BOUND" if base["at_bound"] else "",
                         " DESC" if base["descending"] else "",
                         " NODRIFT" if no_drift else ""])
        print(f"{dom:40s} arms={'+'.join(base['arms']):3s} "
              f"delta={base['delta']:+.3f} [{lo_d:+.3f},{hi_d:+.3f}] "
              f"beta={base['beta']:.2e} R2={base['r2']:+.3f}{flags}")
        panels.append((dom, base, lo_d, hi_d))

    ncol = 4
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.8 * nrow),
                             squeeze=False)
    for i, (dom, base, lo_d, hi_d) in enumerate(panels):
        ax = axes[i // ncol][i % ncol]
        for gi, arm in enumerate(base["arms"]):
            m = base["gs"] == gi
            ax.semilogx(base["xs"][m], base["ys"][m], "os"[gi % 2], ms=5,
                        label=f"{arm}' arm")
        tg = np.geomspace(base["xs"].min(), base["xs"].max(), 200)
        bc_ = _boxcox(tg, base["t0"], base["delta"])
        for gi, c in enumerate(base["intercepts"]):
            ax.semilogx(tg, c + base["beta"] * bc_, "--", color=f"C{gi}")
        flag = " (at bound)" if base["at_bound"] else ""
        ax.set_title(f"{dom}\nδ={base['delta']:.2f} "
                     f"[{lo_d:.2f},{hi_d:.2f}]{flag}  R²={base['r2']:.2f}",
                     fontsize=9)
        ax.set_xlabel("mean data age (days, log)", fontsize=8)
        ax.set_ylabel(METRIC, fontsize=8)
        ax.legend(fontsize=7)
    for j in range(len(panels), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"Data-age law across domains (metric={METRIC}, "
                 f"delta box={DBOX})")
    fig.tight_layout()

    suffix = "" if METRIC == "loss" else f"_{METRIC}"
    if DBOX != (-1.0, 4.0):
        suffix += f"_box{DBOX[0]:g}_{DBOX[1]:g}"
    fig.savefig(FIG / f"delta_dictionary_panels{suffix}.png", dpi=150)
    (ROOT / "results" / f"delta_dictionary{suffix}.json").write_text(
        json.dumps(out, indent=2))
    n_bound = sum(v["at_bound"] for v in out.values())
    print(f"\n{len(out)} fits, {n_bound} at a delta bound "
          f"({100*n_bound/max(len(out),1):.0f}%)")
    print(f"saved delta_dictionary{suffix}.json and panels figure")


if __name__ == "__main__":
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__.strip())
        sys.exit(0)
    if len(sys.argv) > 1:
        METRIC = sys.argv[1]
    if len(sys.argv) > 3:
        DBOX = (float(sys.argv[2]), float(sys.argv[3]))
    main()
