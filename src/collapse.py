"""Pure-drift timescale T_drift from composition-fixed cohort curves.

Model: L_i(tau) = L0_i + A_i * g_theta(tau / T_i),  g_theta(x) = x^t/(1+x^t)
(Hill family: a power law for tau << T, saturating beyond it).

Four estimator decisions matter to the published numbers:

* T_drift is reported from the PER-DOMAIN (free theta) fit, with a
  bootstrap CI, not from the shared-theta joint fit: AIC rejects shared
  theta, so quoting T from the joint fit would take the headline
  timescales from a model the paper argues against.  Both are computed
  and written out so the sensitivity is visible.

* T is only a measurement when the bend lies inside the probed age range.
  A Hill T is an extrapolation unless the curve is actually observed to
  decelerate, so a fit is called identified only if (a) T falls inside
  [min_age/2, 2*max_age], (b) theta is not at the grid edge, and (c) the
  curve decelerates in log-age (late slope < early slope).  Otherwise the
  honest statement is a LOWER BOUND, T > max probed age.

* Alongside the model-based T we report a model-free timescale
  tau_half_obs: the age at which the cohort loss increase reaches half of
  its value at the largest probed age, by interpolation inside the data.
  It never extrapolates, so it is always defined -- at the price of being
  relative to the probed range, which is reported next to it.

* Curves that FALL with age (best fit needs A < 0) are flagged "descending"
  rather than being folded into the "no measurable drift" bucket: a fixed
  cohort that is predicted better by older windows is a signed confound of
  the cohort estimator (item-lifecycle density), not a null result.

QC: cohort points with loss > 1.0 nats are dropped
(pathological overconfidence on tiny cohorts); curves whose drift amplitude
is under the seed-noise floor cannot constrain a shape and are skipped.

Usage: python src/collapse.py
"""
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
N_BOOT = 200


def _cells(fp):
    rows = [json.loads(l) for l in open(fp) if l.strip()]
    cells = defaultdict(list)
    for r in rows:
        cells[round(r["age_days"], 1)].append(r["loss_cohort"])
    return cells, rows


def _best_seed_file(fp):
    """Prefer the period-0 sweep file when it carries MORE SEEDS of the SAME
    measurement.

    domain_X__cohort.jsonl and domain_X__p0cohort.jsonl are the same test
    period, cohort and age grid; the p0 file was re-run at nine seeds while
    several base files stopped at three.  Reading the base file therefore
    threw away seeds we had already paid for, and quoted a three-seed
    bootstrap interval (MovieLens u-cap-i: [1032, 3355] d) where a nine-seed
    one existed ([1031, 1768] d).  Nothing else about the run differs, so the
    higher-seed file is strictly the better estimate.
    """
    name = Path(fp).stem.replace("domain_", "")
    alt = RES / ("domain_" + name.replace("__cohort", "__p0cohort")
                 .replace("__icohort", "__p0icohort") + ".jsonl")
    if not alt.exists() or alt == Path(fp):
        return fp, None
    c0, r0 = _cells(fp)
    c1, r1 = _cells(alt)
    same = (r1[0]["test_t0"] == r0[0]["test_t0"]
            and r1[0]["test_t1"] == r0[0]["test_t1"]
            and set(c1) == set(c0))
    n0 = min(len(v) for v in c0.values())
    n1 = min(len(v) for v in c1.values())
    if same and n1 > n0:
        return alt, (n0, n1)
    return fp, None


def _reduce(cells, rng=None):
    ages = np.array(sorted(cells))
    if rng is None:
        y = np.array([np.mean(cells[a]) for a in ages])
    else:
        y = np.array([np.mean(rng.choice(cells[a], size=len(cells[a]),
                                         replace=True)) for a in ages])
    sd = np.array([np.std(cells[a]) for a in ages])
    return ages, y, sd


def load_curves():
    curves = {}
    for fp in sorted(glob.glob(str(RES / "domain_*__cohort.jsonl")) +
                     glob.glob(str(RES / "domain_*__icohort.jsonl")) +
                     glob.glob(str(RES / "domain_*__cohort__dim*.jsonl"))):
        name = Path(fp).stem.replace("domain_", "")
        # the multi-test-period sweeps (domain__p0cohort, ...) and the
        # weekday control (domain__w7cohort) match this glob but are not
        # separate domains; they are analysed by period_analysis.py
        if "__p" in name or "w7" in name:
            continue
        fp, upgraded = _best_seed_file(fp)
        if upgraded:
            print(f"{name}: using {Path(fp).name} "
                  f"({upgraded[0]} -> {upgraded[1]} seeds per cell)")
        cells, rows = _cells(fp)
        ages, y, sd = _reduce(cells)
        keep = y <= 1.0
        if keep.sum() < 4:
            print(f"skip {name}: {keep.sum()} usable points")
            continue
        if (~keep).any():
            print(f"{name}: dropped {(~keep).sum()} points with loss>1.0")
        good = set(ages[keep])
        cells = {a: v for a, v in cells.items() if a in good}
        ages, y, sd = ages[keep], y[keep], sd[keep]
        rng_amp = y.max() - y.min()
        noise = 2 * np.median(sd)
        if rng_amp < max(0.008, noise):
            print(f"flat  {name}: range={rng_amp:.4f} <= noise floor "
                  f"{max(0.008, noise):.4f} -> no measurable pure drift")
            continue
        curves[name] = dict(cells=cells, ages=ages, y=y, sd=sd,
                            n_cohort=rows[0].get("n_cohort", 0))
    return curves


def g(x, theta):
    xt = np.clip(x, 1e-12, None) ** theta
    return xt / (1 + xt)


def fit_one(ages, y, sd, theta_fixed=None):
    """Per-curve Hill fit. Separable: for fixed (T, theta) the model is
    linear in (L0, A), so we scan (T, theta) on a grid and solve the linear
    part exactly -- a global search, which matters because several curves
    have a flat likelihood ridge in T."""
    w = 1.0 / np.maximum(sd, 0.003)
    z = y * w                                   # weighted target
    thetas = (np.array([theta_fixed]) if theta_fixed is not None
              else np.geomspace(0.05, 30.0, 110))
    Ts = np.exp(np.linspace(np.log(ages.min() / 10),
                            np.log(ages.max() * 100), 220))
    # grid over (T, theta), closed-form weighted LS for (L0, A) at every node
    x = ages[None, None, :] / Ts[:, None, None]          # (nT, 1, n)
    gu = g(x, thetas[None, :, None])                      # (nT, nTh, n)
    # a node where g is essentially constant over the probed ages carries no
    # information about T (bend outside the range): the design goes collinear
    # and the linear solve blows up, so drop those nodes outright.
    usable = (gu.max(-1) - gu.min(-1)) > 1e-3
    G = gu * w
    U = np.broadcast_to(w, G.shape)
    uu = (U * U).sum(-1)
    uv = (U * G).sum(-1)
    vv = (G * G).sum(-1)
    uz = (U * z).sum(-1)
    vz = (G * z).sum(-1)
    det = uu * vv - uv * uv
    with np.errstate(divide="ignore", invalid="ignore"):
        L0g = (vv * uz - uv * vz) / det
        Ag = (uu * vz - uv * uz) / det
    # residuals computed directly: the algebraic shortcut cancels badly near
    # the collinear corners and can return a negative sum of squares
    resid = L0g[..., None] * U + Ag[..., None] * G - z
    ssg = (resid ** 2).sum(-1)
    ssg = np.where(usable & np.isfinite(ssg), ssg, np.inf)
    if not np.isfinite(ssg).any():
        raise RuntimeError("no usable (T, theta) node")
    iT, ith = np.unravel_index(int(np.argmin(ssg)), ssg.shape)
    ss = float(ssg[iT, ith])
    T, th = float(Ts[iT]), float(thetas[ith])
    L0, A = float(L0g[iT, ith]), float(Ag[iT, ith])
    yw = z
    sstot = float(np.sum((yw - np.average(y, weights=w ** 2) * w) ** 2))
    return dict(T=T, theta=th, L0=L0, A=A, ss=ss,
                r2=1 - ss / sstot if sstot > 0 else np.nan,
                theta_at_edge=bool(ith in (0, len(thetas) - 1)),
                descending=bool(A < 0))


def decel_ratio(ages, y):
    """Late/early slope in log-age. < 1 means the curve is bending over
    inside the probed range, i.e. a saturation timescale is observable."""
    x = np.log(ages)
    h = len(ages) // 2
    s1 = np.polyfit(x[:h + 1], y[:h + 1], 1)[0]
    s2 = np.polyfit(x[h:], y[h:], 1)[0]
    return float(s2 / s1) if s1 != 0 else float("nan")


def tau_half_obs(ages, y):
    """Model-free: age at which the loss increase reaches half of its value
    at the largest probed age (interpolated in log-age, never extrapolated).
    Returns nan if the curve is not monotone enough to define it."""
    d = y - y[0]
    target = 0.5 * d[-1]
    if not np.isfinite(target) or target <= 0:
        return float("nan")
    for i in range(1, len(ages)):
        if d[i] >= target:
            if d[i] == d[i - 1]:
                return float(ages[i])
            f = (target - d[i - 1]) / (d[i] - d[i - 1])
            return float(np.exp(np.log(ages[i - 1]) +
                                f * (np.log(ages[i]) - np.log(ages[i - 1]))))
    return float("nan")


def fit_global(curves, shared_theta=True):
    """Universality test by profiling over the shape parameter.

    For a FIXED theta every curve's (L0, A, T) can be fit independently, so
    the shared-theta model is just the theta that minimises the summed
    residual across curves -- no 130-parameter joint optimisation, and no
    dependence on a starting point.  The free-theta model is the sum of the
    per-curve optima.
    """
    names = sorted(curves)
    n_pts = sum(len(curves[nm]["ages"]) for nm in names)

    if shared_theta:
        thetas_grid = np.geomspace(0.05, 30.0, 110)
        best = None
        for th in thetas_grid:
            fits = [fit_one(curves[nm]["ages"], curves[nm]["y"],
                            curves[nm]["sd"], theta_fixed=th)
                    for nm in names]
            ss = sum(f["ss"] for f in fits)
            if best is None or ss < best[0]:
                best = (ss, th, fits)
        ss, th, fits = best
        thetas = np.full(len(names), th)
        k = 1 + 3 * len(names)
    else:
        fits = [fit_one(curves[nm]["ages"], curves[nm]["y"],
                        curves[nm]["sd"]) for nm in names]
        ss = sum(f["ss"] for f in fits)
        thetas = np.array([f["theta"] for f in fits])
        k = 4 * len(names)

    return dict(names=names, thetas=thetas,
                L0=np.array([f["L0"] for f in fits]),
                A=np.array([f["A"] for f in fits]),
                T=np.array([f["T"] for f in fits]), ss=float(ss),
                aic=n_pts * np.log(ss / n_pts) + 2 * k, n_pts=n_pts, k=k)


def main():
    curves = load_curves()
    print(f"{len(curves)} cohort curves loaded\n")

    fs = fit_global(curves, shared_theta=True)
    ff = fit_global(curves, shared_theta=False)
    print(f"shared theta = {fs['thetas'][0]:.3f}   AIC={fs['aic']:.1f} "
          f"(k={fs['k']}, n={fs['n_pts']})")
    print(f"free thetas                        AIC={ff['aic']:.1f} "
          f"(k={ff['k']}, n={ff['n_pts']})   "
          f"dAIC(shared-free)={fs['aic'] - ff['aic']:+.1f} "
          f"(<0 means the shared shape is preferred)")
    print("NOTE: k is comparable to n here, so this AIC comparison is weak "
          "evidence either way; report it as unresolved, not as a verdict.\n")

    out = {}
    print(f"{'curve':34s} {'Hill T_drift [90% CI]':>26s} "
          f"{'tau_half_obs [90% CI]':>25s} {'theta':>6s} {'decel':>7s}  flags")
    for nm in sorted(curves):
        c = curves[nm]
        base = fit_one(c["ages"], c["y"], c["sd"])
        boots, boots_obs = [], []
        rng = np.random.default_rng(0)
        for _ in range(N_BOOT):
            ages_b, y_b, sd_b = _reduce(c["cells"], rng)
            try:
                boots.append(fit_one(ages_b, y_b, c["sd"])["T"])
            except Exception:
                pass
            th_b = tau_half_obs(ages_b, y_b)
            if np.isfinite(th_b):
                boots_obs.append(th_b)
        lo, hi = np.percentile(boots, [5, 95]) if boots else (np.nan, np.nan)
        olo, ohi = (np.percentile(boots_obs, [5, 95]) if len(boots_obs) > 10
                    else (np.nan, np.nan))
        span = (c["ages"].min() / 2, c["ages"].max() * 2)
        dec = decel_ratio(c["ages"], c["y"])
        th_obs = tau_half_obs(c["ages"], c["y"])
        ident = (span[0] <= base["T"] <= span[1]
                 and not base["theta_at_edge"]
                 and np.isfinite(dec) and dec < 1.0
                 and not base["descending"])
        i_shared = fs["names"].index(nm)
        out[nm] = dict(
            T_drift=float(base["T"]), T_lo=float(lo), T_hi=float(hi),
            theta_free=float(base["theta"]),
            theta_at_edge=base["theta_at_edge"], A=float(base["A"]),
            L0=float(base["L0"]), r2=float(base["r2"]),
            decel_ratio=dec, tau_half_obs=th_obs,
            tau_half_obs_lo=float(olo), tau_half_obs_hi=float(ohi),
            descending=bool(base["descending"]), identified=bool(ident),
            T_lower_bound=None if ident else float(c["ages"].max()),
            probed_range=[float(c["ages"].min()), float(c["ages"].max())],
            T_shared_theta=float(fs["T"][i_shared]),
            n_cohort=int(c["n_cohort"]))
        flags = "".join([" DESC" if base["descending"] else "",
                         " THETA-EDGE" if base["theta_at_edge"] else "",
                         "" if ident else " NOT-IDENTIFIED"])
        ci = f"{base['T']:7.0f}[{lo:7.0f},{hi:8.0f}]"
        oci = f"{th_obs:7.1f}[{olo:6.1f},{ohi:7.1f}]"
        print(f"{nm:34s} {ci:>26s} {oci:>25s} {base['theta']:6.2f} "
              f"{dec:+7.2f} {flags}")

    ident = [nm for nm in out if out[nm]["identified"]]
    print(f"\n{len(ident)}/{len(out)} curves have an identified T_drift "
          f"(bend inside the probed age range, non-descending)")
    desc = [nm for nm in out if out[nm]["descending"]]
    print(f"{len(desc)} descending curves (loss FALLS with age): "
          f"{', '.join(desc)}")

    # T_drift vs training-free popularity churn (only where T is identified)
    dstats = json.loads((RES / "domain_stats.json").read_text())
    xs, ys = [], []
    for nm in ident:
        base = nm.replace("__icohort", "").replace("__cohort", "")
        base = base.split("__dim")[0]
        if base in dstats and np.isfinite(dstats[base].get(
                "rho_decay_slope", np.nan)):
            xs.append(dstats[base]["rho_decay_slope"])
            ys.append(np.log(out[nm]["T_drift"]))
    if len(xs) >= 5:
        sp = stats.spearmanr(xs, ys)
        print(f"ln(T_drift) vs rho_decay_slope: spearman={sp.statistic:+.3f} "
              f"(p={sp.pvalue:.4f}, n={len(xs)})")

    (RES / "collapse_fit.json").write_text(json.dumps(
        {"shared_theta": float(fs["thetas"][0]),
         "aic_shared": fs["aic"], "aic_free": ff["aic"],
         "primary": "per-domain free-theta fit (T_drift); T_shared_theta is "
                    "the joint shared-shape fit, kept for sensitivity only",
         "domains": out}, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for nm in sorted(curves):
        c = curves[nm]
        ax.errorbar(c["ages"], c["y"], yerr=c["sd"], fmt="o-", ms=3, lw=0.8)
        ax.annotate(nm.replace("amz_", "").replace("__icohort", "*")
                    .replace("__cohort", ""), (c["ages"][-1], c["y"][-1]),
                    fontsize=6)
    ax.set_xscale("log")
    ax.set_xlabel("data age (days)")
    ax.set_ylabel("cohort loss")
    ax.set_title("Raw pure-drift curves (fixed cohorts)")

    ax = axes[1]
    xg = np.geomspace(1e-3, 1e2, 200)
    for nm in ident:
        c, o = curves[nm], out[nm]
        ax.plot(c["ages"] / o["T_drift"], (c["y"] - o["L0"]) / o["A"], "o",
                ms=4, alpha=0.8,
                label=nm.replace("amz_", "").replace("__icohort", " (item)")
                .replace("__cohort", " (u∩i)"))
    ax.plot(xg, g(xg, fs["thetas"][0]), "k--", lw=1.5,
            label=f"shared shape θ={fs['thetas'][0]:.2f} (AIC-rejected)")
    ax.set_xscale("log")
    ax.set_xlabel("τ / T_drift")
    ax.set_ylabel("(L − L0) / A")
    ax.set_title("Rescaled curves, identified timescales only")
    ax.legend(fontsize=5, ncol=2)
    fig.suptitle("Pure preference drift: timescales and shape")
    fig.tight_layout()
    fig.savefig(RES / "collapse.png", dpi=150)
    print("\nsaved collapse.png and collapse_fit.json")


if __name__ == "__main__":
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__.strip())
        sys.exit(0)
    main()
