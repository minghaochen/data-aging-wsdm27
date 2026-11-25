"""Publication-quality figures for the WSDM 2027 paper (vector PDF, column-fit).

Paper (single column, \\linewidth = 3.33 in):
fig1_dictionary.pdf  : blind round 1 -- registered line and the three
                       hold-outs, all of which fall below it
fig2_twoterm.pdf     : two-mechanism decomposition, two contrasting domains
fig3_horizon.pdf     : (a) tau_half against probing horizon, whole panel
                       (b) truncation traces: what a timescale would look like
fig4_backtest.pdf    : rolling-retrain backtest vs static age law

Supplement (full text width):
fig1_mechanism.pdf   : (a) delta vs e_cov with holdouts  (b) coverage decay
fig3_spectrum.pdf    : (a) cohort drift curves (signal domains)
                       (b) tau_half spectrum dot plot by content type

Usage: python src/paper_figs.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SRC = str(Path(__file__).resolve().parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = ROOT / "paper" / "figs"
# Create paper/figs/ on demand so offline runs work out of the box.
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 7.5,
    "axes.titlesize": 8,
    "axes.labelsize": 7.5,
    "legend.fontsize": 6.2,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.5,
    "figure.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "mathtext.fontset": "cm",
})
COL = 3.33   # \linewidth of one acmart column, inches
DBL = 7.0


def load_jsonl(fp):
    return [json.loads(l) for l in Path(fp).read_text().splitlines() if l]


def cells(fp, exp=None, metric="loss"):
    rows = load_jsonl(fp)
    c = defaultdict(list)
    for r in rows:
        if exp and r["exp"] != exp:
            continue
        c[round(r["age_days"], 2)].append(r[metric])
    ages = np.array(sorted(c))
    return ages, np.array([np.mean(c[a]) for a in ages]), \
        np.array([np.std(c[a]) for a in ages])


# ---------------------------------------------------------------- fig 1
def fig1():
    dd = json.loads((RES / "delta_dictionary.json").read_text())
    from dictionary_analysis import coverage_elasticity, LABELS, HOLDOUT

    doms = ["kuairand", "ml25m", "amz_Appliances", "amz_Software",
            "amz_Video_Games", "amz_Baby_Products", "amz_Toys_and_Games",
            "amz_Office_Products", "amz_Pet_Supplies", "beer",
            "amz_CDs_and_Vinyl"]
    fig, axes = plt.subplots(1, 2, figsize=(DBL, 2.1))

    ax = axes[0]
    xs_in, ys_in = [], []
    for d in doms:
        e, ages, sf = coverage_elasticity(d)
        v = dd[d]
        no_drift = v["beta"] < 1e-6 or v["r2"] < 0.2
        hold = d in HOLDOUT
        if no_drift:
            # Flagged "no measurable drift": delta is pinned to a box bound
            # and is not a measurement, so plotting it on the delta axis
            # would show CDs & Vinyl as the fastest-drifting domain here.
            # It is excluded from every fit and correlation, and from this
            # panel; the caption says so.
            continue
        color = "#dd8452" if hold else "#4c72b0"
        marker = "^" if hold else "o"
        ax.errorbar([e], [v["delta"]],
                    yerr=[[max(0, v["delta"] - v["delta_lo"])],
                          [max(0, v["delta_hi"] - v["delta"])]],
                    fmt=marker, color=color, capsize=2, ms=4, lw=0.8)
        lab = LABELS.get(d, d).replace("AMZ ", "").replace("*", "") \
            .replace("†", "")
        off = (4, 3)
        ax.annotate(lab, (e, v["delta"]), fontsize=5.5,
                    textcoords="offset points", xytext=off)
        if not hold:
            xs_in.append(e)
            ys_in.append(v["delta"])
    b, a = np.polyfit(xs_in, ys_in, 1)
    xg = np.linspace(-2.4, 0.1, 50)
    ax.plot(xg, a + b * xg, "--", color="gray", lw=0.8,
            label="in-sample fit ($\\rho=0.96$)")
    ax.axhline(0, color="k", lw=0.4, ls=":")
    ax.set_xlabel(r"coverage elasticity  $d\ln \omega\,/\,d\ln\tau$")
    ax.set_ylabel(r"apparent exponent $\delta$")
    ax.set_title("(a) Blind round 1: hold-outs fall below the line")
    ax.legend(loc="upper left", frameon=False)

    ax = axes[1]
    show = ["kuairand", "ml25m", "amz_Video_Games", "amz_Software",
            "beer", "amz_Pet_Supplies"]
    for d in show:
        e, ages, sf = coverage_elasticity(d)
        sf = np.array(sf)
        ages = np.array(ages)
        m = sf > 1e-4
        lab = LABELS.get(d, d).replace("AMZ ", "Amazon ").replace("*", "") \
            .replace("†", "")
        ax.loglog(ages[m], sf[m], "o-", ms=2.5, lw=0.9, label=lab)
    ax.set_xlabel(r"training-window mean age $\tau$ (days)")
    ax.set_ylabel(r"warm coverage $\omega$", labelpad=1.5)
    ax.set_title("(b) Training-free coverage decay", pad=4)
    ax.legend(frameon=False, ncol=2, columnspacing=0.8, loc="lower left")
    fig.savefig(OUT / "fig1_mechanism.pdf")
    print("fig1 saved")


# ------------------------------------------------- fig 1 (paper, 1 column)
def fig1_paper():
    """Blind round 1, in the paper: the registered line and its three
    hold-outs.

    Everything plotted is read from the registration file, so the line and
    the predicted points are the ones that were frozen, not a refit. Domains
    flagged "no measurable drift" are omitted rather than drawn at the box
    bound they are pinned to (drawn at the bound, CDs & Vinyl would sit at
    delta=4 and read as the fastest-drifting domain in the panel).
    """
    dd = json.loads((RES / "delta_dictionary.json").read_text())
    reg = json.loads((RES / "blind_predictions.json").read_text())
    from dictionary_analysis import coverage_elasticity, LABELS

    a, b = reg["line"]["intercept"], reg["line"]["slope"]
    ins = reg["line"]["fitted_on"]
    holds = [d for d in reg if d != "line"]
    # label offsets, in points, for the crowded corner at e_cov ~ -2.2
    OFF = {"amz_Video_Games": (-4, 2), "amz_Baby_Products": (-5, -4),
           "amz_Software": (-4, 4), "amz_Appliances": (5, -2)}
    HOFF = {"amz_Toys_and_Games": (5, -7, "left"),
            "amz_Office_Products": (0, -9, "center"),
            "amz_Pet_Supplies": (5, -3, "left")}
    RIGHT = {"amz_Video_Games", "amz_Appliances"}

    fig, ax = plt.subplots(figsize=(COL, 1.72))
    xg = np.linspace(-2.45, 0.15, 50)
    ax.plot(xg, a + b * xg, "--", color="gray", lw=0.8, zorder=1,
            label=r"registered line ($\rho = 1.00$ in sample)")

    for d in ins:
        e, _, _ = coverage_elasticity(d)
        v = dd[d]
        ax.errorbar([e], [v["delta"]],
                    yerr=[[max(0, v["delta"] - v["delta_lo"])],
                          [max(0, v["delta_hi"] - v["delta"])]],
                    fmt="o", color="#4c72b0", capsize=1.5, ms=3.5, lw=0.8,
                    zorder=3, label="dictionary domain" if d == ins[0] else None)
        lab = LABELS.get(d, d).replace("AMZ ", "").replace("*", "") \
            .replace("†", "")
        ax.annotate(lab, (e, v["delta"]), fontsize=5.2, color="#4c72b0",
                    textcoords="offset points", xytext=OFF.get(d, (5, 2)),
                    ha="left" if d in RIGHT or OFF.get(d, (5, 0))[0] > 0
                    else "right")

    for i, d in enumerate(holds):
        e = reg[d]["e_cov"]
        pred = reg[d]["delta_hat"]
        v = dd[d]
        # the registered prediction, the measurement, and the gap between
        ax.plot([e], [pred], "x", color="gray", ms=4, mew=1.0, zorder=3,
                label="registered prediction" if i == 0 else None)
        ax.annotate("", xy=(e, v["delta"]), xytext=(e, pred),
                    arrowprops=dict(arrowstyle="->", color="#c44e52",
                                    lw=0.7, shrinkA=1.5, shrinkB=1.5),
                    zorder=2)
        ax.errorbar([e], [v["delta"]],
                    yerr=[[max(0, v["delta"] - v["delta_lo"])],
                          [max(0, v["delta_hi"] - v["delta"])]],
                    fmt="^", color="#c44e52", capsize=1.5, ms=4.5, lw=0.8,
                    zorder=3, label="hold-out, measured" if i == 0 else None)
        lab = LABELS.get(d, d).replace("AMZ ", "").replace("*", "") \
            .replace("†", "")
        dx, dy, ha = HOFF.get(d, (5, -3, "left"))
        ax.annotate(f"{lab} ({abs(pred - v['delta']):.2f})",
                    (e, v["delta"]), fontsize=5.2, color="#c44e52",
                    textcoords="offset points", xytext=(dx, dy), ha=ha)

    ax.axhline(0, color="k", lw=0.4, ls=":", zorder=0)
    ax.set_xlabel(r"coverage elasticity  $e_{\mathrm{cov}} = "
                  r"d\ln \omega\,/\,d\ln\tau$")
    ax.set_ylabel(r"apparent exponent $\delta$")
    ax.set_xlim(-2.55, 0.5)
    ax.set_ylim(-1.5, 3.1)
    ax.legend(loc="upper left", frameon=False, fontsize=5.6,
              handletextpad=0.4, borderpad=0.2, labelspacing=0.3)
    fig.savefig(OUT / "fig1_dictionary.pdf")
    print("fig1_dictionary saved")


# ------------------------------------------------- fig 3 (paper, 1 column)
def fig3_paper():
    """The section-5 claim, drawn: tau_half tracks the probing horizon.

    (a) every cohort curve in the panel, tau_half against the horizon it was
    measured over. Points hug the diagonal; the one log that outruns its own
    timescale sits two decades below it.
    (b) the same statement per curve: refit after truncating the curve at a
    smaller maximum age. A timescale is a horizontal trace.
    """
    from paper_numbers import horizon_sensitivity
    hs = horizon_sensitivity()

    fam = [("short video", ("kuairand", "kuairec", "recflow"), "#4c72b0"),
           ("movies", ("ml25m", "netflix"), "#dd8452"),
           ("reviews / POI", ("yelp", "beer", "amz_"), "#55a868"),
           ("game UGC", ("gameugc",), "#8172b3")]

    def family(name):
        for lab, keys, colr in fam:
            if any(name.startswith(k) for k in keys):
                return lab, colr
        return "other", "gray"

    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.85),
                            gridspec_kw={"width_ratios": [1, 1],
                                         "wspace": 0.58})

    # ---- (a) tau_half against the horizon it was measured over -----------
    ax = axes[0]
    seen = set()
    ratios = []
    for name, v in sorted(hs.items()):
        lab, colr = family(name)
        ratios.append(v["frac_of_horizon"])
        ax.plot([v["tau_max"]], [v["tau_half"]], "o", ms=3.2, color=colr,
                label=lab if lab not in seen else None, zorder=3)
        seen.add(lab)
    med = float(np.median(ratios))
    xg = np.array([3.0, 8e3])
    ax.plot(xg, med * xg, "--", color="gray", lw=0.8, zorder=1,
            label=rf"$\tau_{{1/2}} = {med:.2f}\,H$")
    ax.annotate("game UGC", (514.5, 4.6), fontsize=5.2, color="#8172b3",
                textcoords="offset points", xytext=(-2, -9), ha="right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"probing horizon $H$ (days)")
    ax.set_ylabel(r"$\tau_{1/2}$ (days)")
    ax.set_title("(a) An estimate, or a horizon?", fontsize=7)
    ax.legend(frameon=False, fontsize=5.2, loc="upper left",
              handletextpad=0.3, borderpad=0.15, labelspacing=0.25)

    # ---- (b) truncation traces -------------------------------------------
    ax = axes[1]
    # markers separate the two curves that share a family colour
    show = [("kuairand__cohort", "KuaiRand", "o-"),
            ("recflow__cohort", "RecFlow", "s--"),
            ("ml25m__cohort", "MovieLens", "o-"),
            ("yelp__icohort", "Yelp*", "s--")]
    for name, lab, style in show:
        t = hs[name]["truncations"]
        caps = sorted(t, key=float)
        ax.plot([float(k) for k in caps], [t[k] for k in caps], style,
                ms=2.6, lw=0.9, color=family(name)[1], label=lab)
    from age_diagnostics import game_curve
    for tag, n, label, style in [
        ("domain_gameugc_finish__icohort.jsonl", 9, "UGC primary", "o-"),
        ("domain_gameugc_finish__icohort__ss1.jsonl", 3, "UGC redraw", "^--"),
    ]:
        t = game_curve(tag, n)["truncations"]
        ax.plot([float(k) for k in t], [np.nan if v is None else v for v in t.values()],
                style, ms=2.5, lw=0.9, color="#8172b3", label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"truncation horizon (days)")
    ax.set_ylabel(r"refitted $\tau_{1/2}$ (days)")
    ax.set_title("(b) Truncate, then refit", fontsize=7)
    ax.legend(frameon=False, fontsize=5.2, loc="upper left",
              handletextpad=0.3, borderpad=0.15, labelspacing=0.25)
    fig.savefig(OUT / "fig3_horizon.pdf")
    print("fig3_horizon saved")


# ---------------------------------------------------------------- fig 2
def fig2():
    # Two panels, not four: the contrast that carries the argument is
    # coverage-collapsed (Amazon) against coverage-stable (KuaiRand), where
    # the warm/cold ordering reverses. Four panels are illegible in one
    # column and the other two domains add no distinct regime.
    picks = [
        ("amz_Video_Games", "Amazon Video Games"),
        ("kuairand", "KuaiRand (feed)"),
    ]
    from matplotlib.ticker import NullFormatter
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.65),
                             gridspec_kw={"wspace": 0.42})
    for i, (dom, title) in enumerate(picks):
        fp = RES / f"domain_{dom}.jsonl"
        rows = [r for r in load_jsonl(fp) if r["exp"] == "A"]
        c = defaultdict(lambda: defaultdict(list))
        for r in rows:
            sf, ls, lo = r["seen_frac"], r["loss_seen"], r["loss"]
            lu = (lo - sf * ls) / (1 - sf) if sf < 0.999 else np.nan
            cc = c[round(r["age_days"], 1)]
            cc["o"].append(lo)
            cc["w"].append(ls)
            cc["c"].append(lu)
            cc["s"].append(sf)
        ages = np.array(sorted(c))
        g = {k: np.array([np.mean(c[a][k]) for a in ages])
             for k in "owcs"}
        # seed spread, drawn: the age effect here is a few mnat, which is the
        # order of the seed noise itself, so a bare mean would overstate it
        e = {k: np.array([np.std(c[a][k]) for a in ages]) for k in "owc"}
        ax = axes[i]
        ax.set_xscale("log")
        ax.errorbar(ages, g["o"], yerr=e["o"], fmt="o-", ms=2.5, lw=0.9,
                    capsize=1.2, elinewidth=0.5, color="#4c72b0",
                    label="overall")
        ax.errorbar(ages, g["w"], yerr=e["w"], fmt="s-", ms=2.5, lw=0.9,
                    capsize=1.2, elinewidth=0.5, color="#dd8452",
                    label="warm")
        m = ~np.isnan(g["c"])
        ax.errorbar(ages[m], g["c"][m], yerr=e["c"][m], fmt="^-", ms=2.5,
                    lw=0.9, capsize=1.2, elinewidth=0.5, color="#55a868",
                    label="cold")
        ax2 = ax.twinx()
        ax2.semilogx(ages, g["s"], ":", color="gray", lw=0.8)
        ax2.set_ylim(0, 1)
        if i == len(picks) - 1:
            ax2.set_ylabel(r"$\omega$ (dotted, 0--1)", fontsize=6.5,
                           color="gray")
        else:
            ax2.set_yticklabels([])
        ax2.tick_params(labelsize=5.5, colors="gray", width=0.4)
        ax.set_title(title, fontsize=7)
        ax.set_xlabel(r"$\tau$ (days)")
        ax.xaxis.set_minor_formatter(NullFormatter())
        if i == 0:
            ax.set_ylabel("loss (nats)")
            ax.legend(frameon=False, loc="center left", fontsize=5.8)
    fig.savefig(OUT / "fig2_twoterm.pdf")
    print("fig2 saved")


# ---------------------------------------------------------------- fig 3
def fig3():
    cf = json.loads((RES / "collapse_fit.json").read_text())["domains"]
    fig, axes = plt.subplots(1, 2, figsize=(DBL, 2.15),
                             gridspec_kw={"width_ratios": [1, 1.05],
                                          "wspace": 0.42})

    ax = axes[0]
    # Production logs use the user-and-item cohort; review platforms use the
    # item cohort throughout, because their user-side cohort is empty or
    # descending. Mixing the two estimators across platforms (as an earlier
    # version did) is what made the cross-platform agreement look tighter
    # than it is.
    show = [("kuairand__cohort", "KuaiRand (u$\\cap$i)", "#4c72b0"),
            ("recflow__cohort", "RecFlow (u$\\cap$i)", "#55a868"),
            ("kuairec__cohort", "KuaiRec (u$\\cap$i)", "#8172b3"),
            ("ml25m__icohort", "MovieLens (item)", "#dd8452"),
            ("yelp__icohort", "Yelp (item)", "#c44e52"),
            ("netflix__icohort", "Netflix (item)", "#937860")]
    for name, lab, colr in show:
        fp = RES / f"domain_{name}.jsonl"
        if not fp.exists():
            continue
        ages, y, sd = cells(fp, metric="loss_cohort")
        keep = y <= 1.0
        ages, y, sd = ages[keep], y[keep], sd[keep]
        ax.errorbar(ages, y - y.min(), yerr=sd, fmt="o-", ms=2.5, lw=0.9,
                    capsize=1.5, color=colr, label=lab)
    ax.set_xscale("log")
    # symlog: the KuaiRec curve is an order of magnitude larger than the
    # rest, and on a linear axis it flattens every other domain to a line.
    ax.set_yscale("symlog", linthresh=0.004, linscale=0.4)
    ax.set_ylim(-0.001, 0.25)
    ax.set_xlabel(r"training-window mean age $\tau$ (days)")
    ax.set_ylabel(r"cohort loss increase (nats)")
    ax.set_title("(a) Fixed-cohort loss curves")
    # upper left: the curves rise with age, so that corner is the empty one
    # (at lower right the legend sat on top of the KuaiRec and Yelp points)
    ax.legend(frameon=False, fontsize=5.5, ncol=2, loc="upper left",
              columnspacing=0.8, handletextpad=0.4)

    ax = axes[1]
    # Every point below is read from the fit output; nothing is typed
    # here, so the figure cannot drift away from the measurements.
    spec = [
        ("Short video", [("KuaiRand", "kuairand__cohort"),
                         ("KuaiRec", "kuairec__cohort"),
                         ("RecFlow", "recflow__cohort")]),
        ("Movies", [("Netflix (item)", "netflix__icohort"),
                    ("MovieLens (item)", "ml25m__icohort"),
                    ("MovieLens (u∩i)", "ml25m__cohort")]),
        ("POI", [("Yelp (item)", "yelp__icohort"),
                 ("Yelp (u∩i)", "yelp__cohort")]),
        ("Beer", [("BeerAdvocate (item)", "beer__icohort")]),
        ("E-commerce", [("Amazon " + n.replace("amz_", "")
                         .replace("__icohort", ""), n)
                        for n in sorted(cf)
                        if n.startswith("amz_")
                        and np.isfinite(cf[n]["tau_half_obs"])]),
        ("Game UGC", [("Game UGC (item)", "gameugc_finish__icohort")]),
    ]
    colors = {"Short video": "#4c72b0", "Movies": "#dd8452",
              "POI": "#c44e52", "Beer": "#937860",
              "E-commerce": "#55a868", "Game UGC": "#8172b3"}
    rows = []
    for group, pts in spec:
        entries = [(n, k) for n, k in pts
                   if k in cf and np.isfinite(cf[k]["tau_half_obs"])]
        entries.sort(key=lambda e: cf[e[1]]["tau_half_obs"])
        for name, key in entries:
            rows.append((group, name, key))
        if entries:
            rows.append(None)              # spacer between groups
    if rows and rows[-1] is None:
        rows.pop()
    ypos, yticks, ylabels = 0, [], []
    for row in rows:
        if row is None:
            ypos += 0.7
            continue
        group, name, key = row
        t = cf[key]["tau_half_obs"]
        lo, hi = cf[key]["tau_half_obs_lo"], cf[key]["tau_half_obs_hi"]
        ax.errorbar([t], [ypos], xerr=[[max(t - lo, 0)], [max(hi - t, 0)]],
                    fmt="o", ms=3.5, capsize=2, color=colors[group])
        # probed horizon: tau_half is bounded by it by construction
        ax.plot([cf[key]["probed_range"][1]], [ypos], "|", ms=7,
                color=colors[group], alpha=0.6)
        if key == "gameugc_finish__icohort":
            # the one truncation-invariant timescale in the panel
            ax.annotate("primary draw", (t, ypos),
                        fontsize=5.5, color=colors[group],
                        textcoords="offset points", xytext=(6, -2),
                        fontweight="bold")
        yticks.append(ypos)
        ylabels.append(name.replace("Amazon ", "AMZ ")
                       .replace("_", " ").replace(" (item)", "*"))
        ypos += 1
    for x, lab in [(7, "week"), (30, "month"), (365, "year"),
                   (3650, "decade")]:
        ax.axvline(x, color="gray", lw=0.4, ls=":")
        ax.annotate(lab, (x, ypos - 0.5), fontsize=5, color="gray",
                    ha="center")
    ax.set_xscale("log")
    ax.set_ylim(-0.8, ypos)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(r"drift front-loading time $\tau_{1/2}$ (days); "
                  r"bar = probed horizon; * = item cohort")
    ax.set_title("(b) Half-rise times on primary cohorts")
    ax.set_xlim(2, 3e4)
    fig.savefig(OUT / "fig3_spectrum.pdf")
    print("fig3 saved")


# ---------------------------------------------------------------- fig 4
def fig4():
    # The published figure must use the volume-matched backtest: the
    # original runs trained on 217,575 events against the static sweep's
    # 300,000, which is the mismatch S9 documents. Reading the released
    # 217,575 records here instead reproduces the mismatched slope ratio
    # of 1.07 and contradicts the caption.
    rows = load_jsonl(RES / "backtest_recflow__d300k.jsonl")
    agg = defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        agg[r["P"]][0] += r["loss"] * r["n_test"]
        agg[r["P"]][1] += r["n_test"]
    Ps = np.array(sorted(agg))
    lbar = np.array([agg[p][0] / agg[p][1] for p in Ps])

    a_rows = [r for r in load_jsonl(RES / "domain_recflow.jsonl")
              if r["exp"] == "A"]
    c = defaultdict(list)
    for r in a_rows:
        c[round(r["age_days"], 2)].append(r["loss"])
    ages = np.array(sorted(c))
    la = np.array([np.mean(c[a]) for a in ages])

    def L_A(age):
        return np.interp(np.log(age), np.log(ages), la)

    # W/2 = 1 day of window offset, as in backtest_analysis.py
    pred = np.array([np.mean(L_A(np.arange(P) + 1.5)) for P in Ps])

    dm, dp = 1000 * (lbar - lbar[0]), 1000 * (pred - pred[0])
    ratio = np.polyfit(dp[1:], dm[1:], 1)[0]     # never typed by hand

    fig, ax = plt.subplots(figsize=(COL, 1.85))
    ax.plot(Ps, dm, "o-", color="#4c72b0", label="measured (rolling backtest)")
    ax.plot(Ps, dp, "s--", color="#dd8452", label="predicted (static age law)")
    ax.set_xscale("log")
    ax.set_xlabel("retraining period $P$ (days)")
    ax.set_ylabel("staleness penalty (mnat)")
    ax.legend(frameon=False)
    ax.set_title(f"RecFlow: measured vs predicted, slope ratio {ratio:.2f}",
                 fontsize=7.5)
    fig.savefig(OUT / "fig4_backtest.pdf")
    print("fig4 saved")


if __name__ == "__main__":
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__.strip())
        sys.exit(0)
    fig1()
    fig1_paper()
    fig2()
    fig3()
    fig3_paper()
    fig4()
