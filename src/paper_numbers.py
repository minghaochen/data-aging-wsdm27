"""Single source of truth for every number quoted in the paper.

Recomputes the panel table, the two-term slopes, the coverage-elasticity
correlations, the probe/capacity variation, the boundary-saturation census,
the descending-cohort census and the blind-round scoring, and writes them to
results/paper_numbers.json (plus a readable dump on stdout).

Anything the manuscript states should be traceable to this file.  It exists
because several published numbers could not be reproduced from the released
artifacts: figure literals and table entries had drifted away from the fits.

Usage: python src/paper_numbers.py
"""
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

_SRC = str(Path(__file__).resolve().parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
DATA = ROOT / "data"

PANEL = ["kuairand", "recflow", "kuairec", "ml25m", "netflix", "yelp",
         "steam", "beer", "epinions", "gameugc_play", "gameugc_finish"]
AMZ = sorted(Path(f).stem.replace("domain_", "")
             for f in glob.glob(str(RES / "domain_amz_*.jsonl"))
             if "__" not in Path(f).stem.replace("domain_", ""))


def rows_of(dom):
    fp = RES / f"domain_{dom}.jsonl"
    return [json.loads(l) for l in fp.read_text().splitlines() if l]


def cells_of(rows, metric, exp=None):
    c = defaultdict(list)
    for r in rows:
        if exp is not None and r["exp"] != exp:
            continue
        if metric in r:
            c[round(r["age_days"], 2)].append(r[metric])
    ages = np.array(sorted(c))
    return ages, np.array([np.mean(c[a]) for a in ages]), \
        np.array([np.std(c[a]) for a in ages])


# --------------------------------------------------------------- panel
def panel_stats():
    cache = RES / "panel_stats.json"
    known = json.loads(cache.read_text()) if cache.exists() else {}
    out = {}
    for dom in PANEL + AMZ:
        fp = RES / f"domain_{dom}.jsonl"
        if not fp.exists():
            continue
        rows = rows_of(dom)
        a_rows = [r for r in rows if r["exp"] == "A"]
        fresh = min(a_rows, key=lambda r: r["age_days"])
        test_days = (fresh["test_t1"] - fresh["test_t0"]) / 86400
        rec = dict(sf_freshest=round(float(np.mean(
            [r["seen_frac"] for r in a_rows
             if abs(r["age_days"] - fresh["age_days"]) < 1e-6])), 4),
            age_min=fresh["age_days"],
            age_max=max(r["age_days"] for r in a_rows),
            test_days=round(test_days, 1),
            subsample=fresh["subsample"], n_test=fresh["n_test"])
        if dom in known:
            rec.update(known[dom])
        else:
            npz = DATA / f"{dom}.npz"
            if npz.exists():
                z = np.load(npz)
                rec["n_events"] = int(len(z["ts"]))
                rec["n_items"] = int(z["item"].max()) + 1
                rec["n_users"] = int(z["user"].max()) + 1
                rec["span_days"] = round(
                    float((z["ts"].max() - z["ts"].min()) / 86400), 1)
                rec["pos_rate"] = round(float(z["label"].mean()), 4)
                known[dom] = {k: rec[k] for k in
                              ("n_events", "n_items", "n_users", "span_days",
                               "pos_rate")}
        out[dom] = rec
    cache.write_text(json.dumps(known, indent=2))
    return out


# ------------------------------------------------------- two-term slopes
def two_term():
    out = {}
    for dom in PANEL + AMZ:
        fp = RES / f"domain_{dom}.jsonl"
        if not fp.exists():
            continue
        rows = [r for r in rows_of(dom) if r["exp"] == "A"]
        ages, L, _ = cells_of(rows, "loss", "A")
        _, Lw, _ = cells_of(rows, "loss_seen", "A")
        _, sf, _ = cells_of(rows, "seen_frac", "A")
        if len(ages) < 3:
            continue
        Lc = (L - sf * Lw) / np.maximum(1 - sf, 1e-9)
        x = np.log(ages)
        sl = lambda y: float(np.polyfit(x, y, 1)[0])
        out[dom] = dict(sf_hi=float(sf.max()), sf_lo=float(sf.min()),
                        slope_overall=sl(L), slope_warm=sl(Lw),
                        slope_cold=sl(Lc),
                        amp_overall=float(L.max() - L.min()),
                        amp_cold=float(Lc.max() - Lc.min()))
    return out


# ------------------------------------------------------ coverage elasticity
def e_cov(dom):
    rows = [r for r in rows_of(dom) if r["exp"] == "A"]
    ages, sf, _ = cells_of(rows, "seen_frac", "A")
    m = sf > 1e-4
    if m.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log(ages[m]), np.log(sf[m]), 1)[0])


def dictionary_corr():
    dd = json.loads((RES / "delta_dictionary.json").read_text())
    recs = []
    for dom in PANEL + AMZ:
        if dom not in dd:
            continue
        v = dd[dom]
        recs.append(dict(dom=dom, delta=v["delta"], r2=v["r2"],
                         at_bound=v["at_bound"], no_drift=v["no_drift"],
                         e_cov=e_cov(dom)))
    use = [r for r in recs if not r["no_drift"] and np.isfinite(r["e_cov"])]
    x = np.array([r["e_cov"] for r in use])
    y = np.array([r["delta"] for r in use])
    sp, pe = stats.spearmanr(x, y), stats.pearsonr(x, y)
    # excluding fits pinned at a delta bound (delta is not measured there)
    fre = [r for r in use if not r["at_bound"]]
    xf = np.array([r["e_cov"] for r in fre])
    yf = np.array([r["delta"] for r in fre])
    spf, pef = stats.spearmanr(xf, yf), stats.pearsonr(xf, yf)
    return dict(domains=recs,
                n_all=len(use), spearman_all=float(sp.statistic),
                p_all=float(sp.pvalue), pearson_all=float(pe.statistic),
                n_interior=len(fre), spearman_interior=float(spf.statistic),
                p_interior=float(spf.pvalue),
                pearson_interior=float(pef.statistic))


# -------------------------------------------------- probe / capacity spread
def probe_spread():
    dd = json.loads((RES / "delta_dictionary.json").read_text())
    dc_fp = RES / "delta_dictionary_loss_cohort.json"
    dc = json.loads(dc_fp.read_text()) if dc_fp.exists() else {}
    out = {}
    for base in ["kuairand", "recflow", "ml25m"]:
        for kind, src in (("apparent", dd), ("cohort_loss", dd),
                          ("cohort_true", dc)):
            keys = []
            for k in src:
                stem = k.split("__dim")[0]
                if kind == "apparent" and stem == base and "cohort" not in k \
                        and "mlp" not in k:
                    keys.append(k)
                if kind != "apparent" and stem == f"{base}__cohort":
                    keys.append(k)
            if not keys:
                continue
            vals = {k: src[k]["delta"] for k in sorted(keys)}
            cis = {k: [src[k]["delta_lo"], src[k]["delta_hi"]]
                   for k in sorted(keys)}
            v = np.array(list(vals.values()))
            out[f"{base}:{kind}"] = dict(
                values=vals, cis=cis, spread=float(v.max() - v.min()),
                seed_ci_width=float(np.mean(
                    [c[1] - c[0] for c in cis.values()])))
    return out


# ------------------------------------------------------- boundary census
def boundary_census():
    out = {}
    for tag, fp in [("loss", RES / "delta_dictionary.json"),
                    ("loss_cohort", RES / "delta_dictionary_loss_cohort.json"),
                    ("loss_widebox", RES / "delta_dictionary_box-6_4.json")]:
        if not fp.exists():
            continue
        d = json.loads(fp.read_text())
        out[tag] = dict(n=len(d),
                        n_at_bound=sum(v["at_bound"] for v in d.values()),
                        at_bound=[k for k, v in d.items() if v["at_bound"]],
                        n_descending=sum(v["descending"] for v in d.values()),
                        n_no_drift=sum(v["no_drift"] for v in d.values()))
    return out


# --------------------------------------------------------- blind rescoring
def blind_rescore():
    r2 = json.loads((RES / "blind_predictions_round2.json").read_text())
    dd = json.loads((RES / "delta_dictionary.json").read_text())
    wide_fp = RES / "delta_dictionary_box-6_4.json"
    wide = json.loads(wide_fp.read_text()) if wide_fp.exists() else {}
    doms = [k for k in r2 if k.startswith("amz_")]
    rows = []
    for d in doms:
        m = dd.get(d, {})
        w = wide.get(d, {})
        rows.append(dict(
            dom=d, M1=r2[d]["M1_delta_hat"], M2=r2[d]["M2_delta_hat"],
            measured=m.get("delta"), measured_ci=[m.get("delta_lo"),
                                                  m.get("delta_hi")],
            r2=m.get("r2"), at_bound=m.get("at_bound"),
            no_drift=m.get("no_drift"),
            measured_widebox=w.get("delta"),
            widebox_at_bound=w.get("at_bound")))
    tol = 0.35

    def score(key, pred_of):
        errs, hits, n = [], 0, 0
        for r in rows:
            truth = r[key]
            if truth is None or r["no_drift"]:
                continue
            n += 1
            e = abs(pred_of(r) - truth)
            errs.append(e)
            hits += e < tol
        return dict(n=n, hits=hits, mae=float(np.mean(errs)) if errs else None)

    preds = {"M1": lambda r: r["M1"], "M2": lambda r: r["M2"],
             "const_-1": lambda r: -1.0,
             "const_mean_measured": lambda r: float(np.mean(
                 [x["measured"] for x in rows
                  if x["measured"] is not None and not x["no_drift"]]))}
    return dict(rows=rows, tolerance=tol,
                scored_in_registration_box={
                    k: score("measured", f) for k, f in preds.items()},
                scored_in_wide_box={
                    k: score("measured_widebox", f) for k, f in preds.items()})


def weekday_control():
    """W=7d control (weekday-balanced windows) vs the main W=2d design.

    Reported as a pair per domain so the comparison cannot drift apart in
    the manuscript; the control is deliberately weaker (fewer seeds, shorter
    age range) and its amplitude is reported next to the timescale so that
    is visible.
    """
    from collapse import tau_half_obs
    out = {}
    for dom in ("kuairand", "recflow"):
        rec = {}
        for tag, lab in (("__cohort", "main_W2"), ("__w7cohort", "ctrl_W7")):
            fp = RES / f"domain_{dom}{tag}.jsonl"
            if not fp.exists():
                continue
            rows = [json.loads(l) for l in fp.read_text().splitlines() if l]
            c = defaultdict(list)
            for r in rows:
                c[round(r["age_days"], 1)].append(r["loss_cohort"])
            ages = np.array(sorted(c))
            y = np.array([np.mean(c[a]) for a in ages])
            rng = np.random.default_rng(0)
            bs = []
            for _ in range(200):
                yb = np.array([np.mean(rng.choice(c[a], size=len(c[a]),
                                                  replace=True))
                               for a in ages])
                t = tau_half_obs(ages, yb)
                if np.isfinite(t):
                    bs.append(t)
            lo, hi = (np.percentile(bs, [5, 95]) if len(bs) > 20
                      else (np.nan, np.nan))
            rec[lab] = dict(
                tau_half=tau_half_obs(ages, y), lo=float(lo), hi=float(hi),
                amp_mnat=float(1000 * (y.max() - y.min())),
                seeds_per_age=int(min(len(c[a]) for a in ages)),
                n_ages=len(ages), age_min=float(ages.min()),
                age_max=float(ages.max()),
                n_cohort=int(rows[0].get("n_cohort", 0)))
        if rec:
            out[dom] = rec
    return out


def horizon_sensitivity():
    """Is tau_half a property of the domain or of the probing horizon?

    tau_half is defined relative to the loss increase at the largest probed
    age, so on a curve that has not saturated it moves with the horizon.
    Here we truncate each cohort curve at successively smaller maximum ages
    and refit. A curve whose drift really saturates inside the window gives
    the same answer at every truncation; a curve that is still rising gives
    roughly a fixed fraction of whatever horizon it was given.

    Also reported: the drift rate in nats per e-fold of age, which is
    horizon-free and therefore comparable across domains whose absolute age
    ranges do not overlap at all (feed logs 2.5-36 d, review corpora
    270-4770 d).
    """
    from collapse import tau_half_obs, _best_seed_file
    out = {}
    for fp in sorted(glob.glob(str(RES / "domain_*__cohort.jsonl"))) + \
            sorted(glob.glob(str(RES / "domain_*__icohort.jsonl"))):
        nm = Path(fp).stem.replace("domain_", "")
        if "__dim" in nm or "__p" in nm or "w7" in nm:
            continue
        # use every seed we have for this measurement (see collapse.py)
        fp, _ = _best_seed_file(fp)
        rows = [json.loads(l) for l in open(fp, encoding="utf-8") if l.strip()]
        c = defaultdict(list)
        for r in rows:
            c[round(r["age_days"], 1)].append(r["loss_cohort"])
        ages = np.array(sorted(c))
        y = np.array([np.mean(c[a]) for a in ages])
        keep = y <= 1.0
        ages, y = ages[keep], y[keep]
        if len(ages) < 5:
            continue
        full = tau_half_obs(ages, y)
        if not np.isfinite(full):
            continue
        trunc = {}
        for cap in ages[3:]:
            m = ages <= cap
            t = tau_half_obs(ages[m], y[m])
            if np.isfinite(t):
                trunc[float(cap)] = float(t)
        vals = list(trunc.values())
        # a saturating curve keeps the same tau_half at every truncation
        stable = (max(vals) / min(vals) < 1.5) if len(vals) > 2 else None
        out[nm] = dict(
            seeds_per_cell=int(min(len(v) for v in c.values())),
            source=Path(fp).stem.replace("domain_", ""),
            tau_half=float(full), tau_max=float(ages.max()),
            tau_min=float(ages.min()),
            frac_of_horizon=float(full / ages.max()),
            decades=float(np.log10(ages.max() / ages.min())),
            truncations=trunc, saturating=stable,
            rate_mnat_per_efold=float(
                1000 * np.polyfit(np.log(ages), y, 1)[0]))
    return out


def matched_horizon_movies():
    """The one cross-platform comparison this panel can actually support.

    Netflix (270-1710 d) and MovieLens (450-4770 d) are the only pair whose
    probed age ranges overlap, so they can be compared at a common horizon
    with a common cohort definition. Everything else in the spectrum table
    compares domains whose age ranges are disjoint, where tau_half is
    largely a restatement of the probing horizon.
    """
    from collapse import tau_half_obs
    out = {}
    cap = 1710.0
    for dom in ("netflix", "ml25m"):
        fp = RES / f"domain_{dom}__p0icohort.jsonl"
        if not fp.exists():
            continue
        rows = [json.loads(l) for l in fp.read_text().splitlines() if l]
        c = defaultdict(list)
        for r in rows:
            c[round(r["age_days"], 1)].append(r["loss_cohort"])
        ages = np.array(sorted(c))
        m = ages <= cap
        aa = ages[m]
        yy = np.array([np.mean(c[a]) for a in aa])
        rng = np.random.default_rng(0)
        bs = []
        for _ in range(500):
            yb = np.array([np.mean(rng.choice(c[a], size=len(c[a]),
                                              replace=True)) for a in aa])
            v = tau_half_obs(aa, yb)
            if np.isfinite(v):
                bs.append(v)
        lo, hi = np.percentile(bs, [5, 95])
        out[dom] = dict(horizon=cap, n_ages=int(m.sum()),
                        tau_half=float(tau_half_obs(aa, yy)),
                        lo=float(lo), hi=float(hi),
                        rate_mnat_per_efold=float(
                            1000 * np.polyfit(np.log(aa), yy, 1)[0]),
                        n_seeds=int(min(len(v) for v in c.values())))
    if len(out) == 2:
        a, b = out["netflix"], out["ml25m"]
        out["comparison"] = dict(
            tau_ratio=a["tau_half"] / b["tau_half"],
            rate_ratio=(a["rate_mnat_per_efold"]
                        / b["rate_mnat_per_efold"]),
            ci_disjoint=bool(b["hi"] < a["lo"] or a["hi"] < b["lo"]))
    return out


def matched_horizon_periods():
    """The matched-horizon movie test, repeated on all three test periods.

    matched_horizon_movies() runs it on period 0 only, where the two
    platforms' seed-only intervals are disjoint.  Section 3 of the paper
    shows the test period moves timescales by 2-3x, so the honest version of
    this test slides it: the same item cohorts, the same 1,710-day cap, nine
    seeds, three test periods per platform.
    """
    from collapse import tau_half_obs
    cap = 1710.0
    out = {}
    for dom in ("netflix", "ml25m"):
        recs = {}
        for p in (0, 1, 2):
            fp = RES / f"domain_{dom}__p{p}icohort.jsonl"
            if not fp.exists():
                continue
            rows = [json.loads(line) for line in fp.read_text().splitlines()
                    if line]
            c = defaultdict(list)
            for r in rows:
                c[round(r["age_days"], 1)].append(r["loss_cohort"])
            ages = np.array([a for a in sorted(c) if a <= cap])
            if len(ages) < 4:
                continue
            y = np.array([np.mean(c[a]) for a in ages])
            rng = np.random.default_rng(0)
            bs = []
            for _ in range(300):
                yb = np.array([np.mean(rng.choice(c[a], size=len(c[a]),
                                                  replace=True))
                               for a in ages])
                v = tau_half_obs(ages, yb)
                if np.isfinite(v):
                    bs.append(v)
            lo, hi = np.percentile(bs, [5, 95])
            recs[f"p{p}"] = dict(
                tau_half=float(tau_half_obs(ages, y)), lo=float(lo),
                hi=float(hi), n_ages=len(ages),
                n_seeds=int(min(len(c[a]) for a in ages)),
                rate_mnat_per_efold=float(
                    1000 * np.polyfit(np.log(ages), y, 1)[0]))
        out[dom] = recs
    if len(out) == 2 and out["netflix"] and out["ml25m"]:
        nt = [v["tau_half"] for v in out["netflix"].values()]
        mt = [v["tau_half"] for v in out["ml25m"].values()]
        nr = [v["rate_mnat_per_efold"] for v in out["netflix"].values()]
        mr = [v["rate_mnat_per_efold"] for v in out["ml25m"].values()]
        out["comparison"] = dict(
            netflix_tau_range=[min(nt), max(nt)],
            ml25m_tau_range=[min(mt), max(mt)],
            tau_ranges_overlap=bool(min(nt) <= max(mt) and min(mt) <= max(nt)),
            netflix_rate_range=[min(nr), max(nr)],
            ml25m_rate_range=[min(mr), max(mr)],
            rate_ranges_overlap=bool(min(nr) <= max(mr) and min(mr) <= max(nr)),
            worst_case_rate_ratio=min(nr) / max(mr))
    return out


# ------------------------------------------------------------ run census
def run_census():
    """Count released training runs under an explicit inclusion rule.

    The rule below is stated and mechanical, so the manuscript's run
    accounting is reproducible from the released records alone.
    """
    dd = set(json.loads((RES / "delta_dictionary.json").read_text()))
    cr_strict = {
        "backtest_kuairand_strict",
        "backtest_recflow_nooverlap19d",
        "backtest_gameugc_strict",
    }
    cr_gameugc = {
        "gameugc_finish__icohort_draws",
        "gameugc_finish__valstop",
        "gameugc_finish__icohort_valstop",
        "gameugc_finish__icohort_finegrid",
    }
    cat = defaultdict(int)
    files = defaultdict(list)
    cr_cat = defaultdict(int)
    cr_files = defaultdict(list)
    for fp in sorted(glob.glob(str(RES / "*.jsonl"))):
        nm = Path(fp).stem
        n = sum(1 for line in open(fp, encoding="utf-8") if line.strip())
        if nm.startswith("fourway_"):
            cr_cat["fourway_warm_cold"] += n
            cr_files["fourway_warm_cold"].append(nm)
            continue
        if nm in cr_strict:
            cr_cat["strict_nooverlap_backtests"] += n
            cr_files["strict_nooverlap_backtests"].append(nm)
            continue
        if nm.startswith("sidefeat_"):
            cr_cat["side_feature_controls"] += n
            cr_files["side_feature_controls"].append(nm)
            continue
        if nm in cr_gameugc:
            cr_cat["gameugc_hygiene"] += n
            cr_files["gameugc_hygiene"].append(nm)
            continue
        key = nm.replace("domain_", "")
        # Robustness variants: the held-out
        # stopping rule, the alternative cohort subsample seed, the
        # volume-matched backtest and the recency half-life sweep. They are
        # reruns of existing settings under a changed convention, not new
        # measurement settings, and are counted separately so that "68
        # settings" keeps meaning what it meant.
        if any(t in nm for t in ("valstop", "__ss", "d300k", "_hl")):
            c = "robustness_variants"
        elif nm.startswith("domain_") and key in dd:
            c = "measurement_settings"
        elif any(t in nm for t in ("__p0", "__p1", "__p2")):
            c = "test_period_sweeps"
        elif "w7" in nm:
            c = "weekday_controls"
        elif nm.startswith("backtest"):
            c = "backtests_and_policies"
        elif nm.startswith("results"):
            c = "legacy_exploratory"
        else:
            c = "other"
        cat[c] += n
        files[c].append(nm)
    total = sum(cat.values())
    cr_total = sum(cr_cat.values())
    excl_legacy = total - cat["legacy_exploratory"]
    return dict(by_category=dict(cat), n_files={k: len(v)
                                                for k, v in files.items()},
                total=total,
                total_excluding_legacy=excl_legacy,
                n_settings=len(dd),
                camera_ready_by_category=dict(cr_cat),
                camera_ready_n_files={k: len(v) for k, v in cr_files.items()},
                camera_ready_total=cr_total,
                total_including_camera_ready=excl_legacy + cr_total)


# ------------------------------------- cold-plateau threshold sensitivity
def cold_threshold_sensitivity(tt=None):
    """How many domains count as a 'near-flat cold plateau' at each cutoff?

    The 12 mnat/e-fold cutoff was chosen after seeing the slopes; this
    reports the whole curve so the count cannot hide in the choice, plus a
    scale-relative reading (cold slope against the warm slope of the same
    domain, which needs no cutoff at all).
    """
    tt = tt or two_term()
    curve = {}
    for thr in (0.006, 0.008, 0.010, 0.012, 0.015, 0.020, 0.025):
        viol = sorted(d for d, v in tt.items() if v["slope_cold"] > thr)
        curve[thr] = dict(n_flat=len(tt) - len(viol), n=len(tt),
                          violators=viol)
    reversed_order = sorted(d for d, v in tt.items()
                            if v["slope_cold"] > v["slope_warm"])
    return dict(curve=curve, cold_exceeds_warm=reversed_order,
                cold_slopes={d: v["slope_cold"] for d, v in tt.items()})


# ------------------------------------------------- age-assignment convention
def age_convention():
    """Refit the dictionary with EVENT-WEIGHTED training-window ages.

    run_domain.py records tau as the geometric midpoint of the training
    window.  The training subsample is a uniform draw from the events in
    that window, so the age of the data the probe actually sees is the
    event-weighted mean age, which on a growing log is much smaller.  The A
    (translated) arm is barely affected; the B (widened) arm can be off by
    up to 2.3x.  Both conventions are reported: the midpoint one is what the
    blind rounds were registered against, the event-weighted one is what the
    data are.
    """
    import fit_domains as FD
    dd = json.loads((RES / "delta_dictionary.json").read_text())
    ts_cache = {}
    out = {}
    for dom in PANEL + AMZ:
        fp = RES / f"domain_{dom}.jsonl"
        npz = DATA / f"{dom}.npz"
        if not fp.exists() or not npz.exists() or dom not in dd:
            continue
        if dom not in ts_cache:
            ts_cache[dom] = np.load(npz)["ts"].astype(np.int64)
        ts = ts_cache[dom]
        rows, cache = [], {}
        for r in rows_of(dom):
            if "loss" not in r:
                continue
            k = (r["train_t0"], r["train_t1"], r["test_t0"], r["test_t1"])
            if k not in cache:
                a, b = np.searchsorted(ts, [r["train_t0"], r["train_t1"]])
                mid = (r["test_t0"] + r["test_t1"]) / 2.0
                cache[k] = float((mid - ts[a:b].mean()) / 86400)
            rr = dict(r)
            rr["age_days_midpoint"] = r["age_days"]
            rr["age_days"] = cache[k]
            rows.append(rr)
        if len(rows) < 8:
            continue
        b = FD.fit_joint(rows)
        v = dd[dom]
        ratios = [rr["age_days_midpoint"] / rr["age_days"] for rr in rows]
        # two-sided: the midpoint overstates the true age on growing logs and
        # understates it where the widest window is truncated at the log start
        worst = max(max(ratios), 1.0 / min(ratios))
        arm_a = [rr["age_days_midpoint"] / rr["age_days"] for rr in rows
                 if rr["exp"] == "A"]
        out[dom] = dict(
            delta_midpoint=v["delta"], delta_eventweighted=b["delta"],
            shift=b["delta"] - v["delta"],
            seed_ci_width=v["delta_hi"] - v["delta_lo"],
            exceeds_seed_ci=bool(abs(b["delta"] - v["delta"])
                                 > v["delta_hi"] - v["delta_lo"]),
            r2_midpoint=v["r2"], r2_eventweighted=b["r2"],
            max_age_discrepancy=float(worst),
            max_age_discrepancy_A_arm=float(
                max(max(arm_a), 1.0 / min(arm_a))) if arm_a else None)
    n = len(out)
    return dict(domains=out, n=n,
                n_exceeding_seed_ci=sum(v["exceeds_seed_ci"]
                                        for v in out.values()),
                n_r2_improved=sum(v["r2_eventweighted"] > v["r2_midpoint"]
                                  for v in out.values()),
                median_abs_shift=float(np.median(
                    [abs(v["shift"]) for v in out.values()])) if n else None)


def main():
    out = dict(panel=panel_stats(), two_term=two_term(),
               dictionary=dictionary_corr(), probe_spread=probe_spread(),
               boundary=boundary_census(), blind=blind_rescore(),
               weekday_control=weekday_control(),
               horizon=horizon_sensitivity(),
               matched_horizon_movies=matched_horizon_movies(),
               run_census=run_census(),
               age_convention=age_convention())
    out["cold_threshold"] = cold_threshold_sensitivity(out["two_term"])
    out["matched_horizon_periods"] = matched_horizon_periods()
    (RES / "paper_numbers.json").write_text(json.dumps(out, indent=2))

    print("=" * 78)
    print("PANEL (Table 1)")
    print(f"{'domain':26s} {'events':>10s} {'items':>9s} {'users':>10s} "
          f"{'span_d':>8s} {'pos':>6s} {'sf@fresh':>9s} {'test_d':>7s}")
    for d, v in out["panel"].items():
        print(f"{d:26s} {v.get('n_events',0):10,d} {v.get('n_items',0):9,d} "
              f"{v.get('n_users',0):10,d} {v.get('span_days',0):8.0f} "
              f"{v.get('pos_rate',float('nan')):6.3f} "
              f"{v['sf_freshest']:9.3f} {v['test_days']:7.1f}")
    print(f"\nAmazon categories in the panel: {len(AMZ)}")

    print("\n" + "=" * 78)
    print("TWO-TERM SLOPES (nats per e-fold of age, A arm)")
    print(f"{'domain':26s} {'sf hi->lo':>16s} {'overall':>8s} {'warm':>8s} "
          f"{'cold':>8s} {'amp_L':>7s}")
    for d, v in out["two_term"].items():
        print(f"{d:26s} {v['sf_hi']:7.3f}->{v['sf_lo']:7.3f} "
              f"{v['slope_overall']:+8.4f} {v['slope_warm']:+8.4f} "
              f"{v['slope_cold']:+8.4f} {v['amp_overall']:7.4f}")
    viol = [d for d, v in out["two_term"].items() if v["slope_cold"] > 0.012]
    print(f"cold slope > 0.012 in: {viol if viol else 'none'}")

    print("\n" + "=" * 78)
    dc = out["dictionary"]
    print(f"delta vs e_cov: all drift domains n={dc['n_all']} "
          f"spearman={dc['spearman_all']:+.3f} (p={dc['p_all']:.3f}) "
          f"pearson={dc['pearson_all']:+.3f}")
    print(f"                interior fits only n={dc['n_interior']} "
          f"spearman={dc['spearman_interior']:+.3f} "
          f"(p={dc['p_interior']:.3f}) "
          f"pearson={dc['pearson_interior']:+.3f}")

    print("\n" + "=" * 78)
    print("PROBE / CAPACITY SPREAD vs seed-only CI width")
    for k, v in out["probe_spread"].items():
        print(f"{k:24s} spread={v['spread']:.3f} "
              f"mean seed CI width={v['seed_ci_width']:.3f}")
        for kk, vv in v["values"].items():
            print(f"    {kk:34s} {vv:+.3f} "
                  f"[{v['cis'][kk][0]:+.3f},{v['cis'][kk][1]:+.3f}]")

    print("\n" + "=" * 78)
    print("BOUNDARY / DESCENDING CENSUS")
    for k, v in out["boundary"].items():
        print(f"{k:14s} n={v['n']:3d} at_bound={v['n_at_bound']:3d} "
              f"descending={v['n_descending']:3d} "
              f"no_drift={v['n_no_drift']:3d}")

    print("\n" + "=" * 78)
    print("BLIND ROUND 2, RESCORED")
    for r in out["blind"]["rows"]:
        print(f"{r['dom']:30s} M1={r['M1']:+.2f} M2={r['M2']:+.2f} "
              f"measured={r['measured']:+.3f}"
              f"{' (BOUND)' if r['at_bound'] else ''} "
              f"R2={r['r2']:.3f} widebox={r['measured_widebox']:+.3f}"
              f"{' (BOUND)' if r['widebox_at_bound'] else ''}"
              f"{'  [no drift: unscoreable]' if r['no_drift'] else ''}")
    for tag in ("scored_in_registration_box", "scored_in_wide_box"):
        print(f"  {tag}:")
        for k, v in out["blind"][tag].items():
            mae = "n/a" if v["mae"] is None else f"{v['mae']:.2f}"
            print(f"    {k:20s} hits {v['hits']}/{v['n']}  MAE {mae}")
    print("\n" + "=" * 78)
    print("WEEKDAY CONTROL (W=7d windows cover all weekdays equally)")
    for dom, rec in out["weekday_control"].items():
        for lab, v in rec.items():
            print(f"{dom:10s} {lab:8s} tau_half={v['tau_half']:6.1f} "
                  f"[{v['lo']:5.1f},{v['hi']:5.1f}] d  "
                  f"amp={v['amp_mnat']:5.1f} mnat  "
                  f"ages {v['age_min']:.0f}-{v['age_max']:.0f} d "
                  f"({v['n_ages']} pts x {v['seeds_per_age']} seeds)  "
                  f"cohort n={v['n_cohort']}")

    print("\n" + "=" * 78)
    print("IS tau_half A DOMAIN PROPERTY OR THE PROBING HORIZON?")
    print(f"{'curve':34s} {'tau12':>7s} {'horizon':>8s} {'frac':>5s} "
          f"{'mnat/e-fold':>11s}  saturating?")
    fr = []
    for nm, v in out["horizon"].items():
        fr.append(v["frac_of_horizon"])
        sat = ("yes" if v["saturating"] else "no"
               if v["saturating"] is not None else "?")
        print(f"{nm:34s} {v['tau_half']:7.1f} {v['tau_max']:8.0f} "
              f"{v['frac_of_horizon']:5.2f} {v['rate_mnat_per_efold']:11.2f}"
              f"  {sat}")
    n_sat = sum(1 for v in out["horizon"].values() if v["saturating"])
    print(f"tau_half / horizon over {len(fr)} curves: median "
          f"{np.median(fr):.2f}, range {min(fr):.2f}-{max(fr):.2f}")
    print(f"curves whose tau_half is stable under truncation (i.e. really "
          f"saturating): {n_sat} of {len(fr)}")

    mh = out.get("matched_horizon_movies", {})
    if "comparison" in mh:
        print("\n" + "=" * 78)
        print("MATCHED-HORIZON MOVIE COMPARISON (item cohorts, period 0, "
              f"horizon {mh['netflix']['horizon']:.0f} d)")
        for k in ("netflix", "ml25m"):
            v = mh[k]
            print(f"  {k:10s} tau_half={v['tau_half']:7.1f} "
                  f"[{v['lo']:.0f},{v['hi']:.0f}]  "
                  f"rate={v['rate_mnat_per_efold']:6.2f} mnat/e-fold  "
                  f"({v['n_ages']} ages, {v['n_seeds']} seeds)")
        c = mh["comparison"]
        print(f"  ratio: tau {c['tau_ratio']:.2f}x, rate "
              f"{c['rate_ratio']:.1f}x, CIs "
              f"{'disjoint' if c['ci_disjoint'] else 'overlapping'}")

    print("\n" + "=" * 78)
    print("RUN CENSUS (explicit inclusion rule)")
    rc = out["run_census"]
    for k, v in sorted(rc["by_category"].items()):
        print(f"  {k:24s} {v:5d} runs in {rc['n_files'][k]:3d} files")
    print(f"  {'TOTAL':24s} {rc['total']:5d}  "
          f"(excluding legacy exploratory: {rc['total_excluding_legacy']})")
    if rc.get("camera_ready_total"):
        print("\n  Extended diagnostic runs (included separately):")
        for k, v in sorted(rc["camera_ready_by_category"].items()):
            print(f"    {k:28s} {v:5d} runs in {rc['camera_ready_n_files'][k]:3d} files")
        print(f"    {'EXTENDED STUDY TOTAL':28s} {rc['camera_ready_total']:5d}  "
              f"(combined study total: {rc['total_including_camera_ready']})")

    print("\n" + "=" * 78)
    print("COLD-PLATEAU THRESHOLD SENSITIVITY")
    for thr, v in out["cold_threshold"]["curve"].items():
        print(f"  cold slope <= {1000*float(thr):5.1f} mnat/e-fold: "
              f"{v['n_flat']}/{v['n']}   violators: {v['violators']}")
    print(f"  domains where the COLD slope exceeds the WARM slope: "
          f"{out['cold_threshold']['cold_exceeds_warm']}")

    print("\n" + "=" * 78)
    print("AGE CONVENTION: window midpoint vs event-weighted mean age")
    ac = out["age_convention"]
    if not ac["domains"]:
        # The one section that cannot run from the run records alone:
        # re-assigning a window's age by event weight needs the timestamps
        # inside the window, i.e. the prepared datasets under data/. Public
        # datasets are not redistributed, so a fresh checkout has to build
        # them with src/prep_*.py first. Everything else here runs from
        # results/ alone, so we report the gap and carry on rather than
        # aborting the whole script.
        print("  [skipped] needs the prepared datasets in data/; build them "
              "with src/prep_*.py first.")
        print("  The published values for this section are in Table S2 of "
              "the supplement.")
    else:
        print(f"{'domain':28s} {'midpoint':>9s} {'evt-wt':>9s} {'shift':>7s} "
              f"{'seedCI':>7s} {'R2 mid':>7s} {'R2 evt':>7s} {'nom/true':>9s}")
        for d, v in ac["domains"].items():
            print(f"{d:28s} {v['delta_midpoint']:+9.3f} "
                  f"{v['delta_eventweighted']:+9.3f} {v['shift']:+7.3f} "
                  f"{v['seed_ci_width']:7.3f} {v['r2_midpoint']:7.3f} "
                  f"{v['r2_eventweighted']:7.3f} "
                  f"{v['max_age_discrepancy']:9.2f}"
                  f"{'  *' if v['exceeds_seed_ci'] else ''}")
        aarm = max(v["max_age_discrepancy_A_arm"]
                   for v in ac["domains"].values())
        print(f"  worst A-arm (translated) age discrepancy anywhere: "
              f"{aarm:.2f}x")
        print(f"  shifts exceeding the seed-only CI: "
              f"{ac['n_exceeding_seed_ci']}/{ac['n']}; "
              f"R2 improves in {ac['n_r2_improved']}/{ac['n']}; "
              f"median |shift| {ac['median_abs_shift']:.3f}")

    mp = out.get("matched_horizon_periods", {})
    if "comparison" in mp:
        print("\n" + "=" * 78)
        print("MATCHED-HORIZON MOVIE TEST ACROSS ALL THREE TEST PERIODS")
        for dom in ("netflix", "ml25m"):
            for p, v in mp[dom].items():
                print(f"  {dom:8s} {p}: tau_half={v['tau_half']:7.1f} "
                      f"[{v['lo']:.0f},{v['hi']:.0f}] d  "
                      f"rate={v['rate_mnat_per_efold']:6.2f} mnat/e-fold "
                      f"({v['n_ages']} ages x {v['n_seeds']} seeds)")
        c = mp["comparison"]
        print(f"  tau ranges: Netflix {c['netflix_tau_range'][0]:.0f}-"
              f"{c['netflix_tau_range'][1]:.0f} d, MovieLens "
              f"{c['ml25m_tau_range'][0]:.0f}-{c['ml25m_tau_range'][1]:.0f} d"
              f" -> {'OVERLAP' if c['tau_ranges_overlap'] else 'disjoint'}")
        print(f"  rate ranges: Netflix {c['netflix_rate_range'][0]:.1f}-"
              f"{c['netflix_rate_range'][1]:.1f}, MovieLens "
              f"{c['ml25m_rate_range'][0]:.1f}-{c['ml25m_rate_range'][1]:.1f}"
              f" -> {'overlap' if c['rate_ranges_overlap'] else 'DISJOINT'}"
              f", worst-case ratio {c['worst_case_rate_ratio']:.1f}x")

    print("\nsaved results/paper_numbers.json")


if __name__ == "__main__":
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__.strip())
        sys.exit(0)
    main()
