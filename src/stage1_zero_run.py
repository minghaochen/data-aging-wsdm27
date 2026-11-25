"""Stage 1 zero-new-training analyses (Exp 2A, Exp 5, Exp 4 Part 1, Exp 1 Part 1).

Reads ONLY from results/ and data/*.npz (pure NumPy, no PyTorch required).
Writes ONLY to results/stage1_zero_run.json.

Usage: python src/stage1_zero_run.py
"""
import argparse
import os

for _k in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_k] = "2"

import glob
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import integrate, optimize, stats

_SRC = str(Path(__file__).resolve().parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import cr_lib as C


def _safe_float(x):
    if x is None:
        return None
    x = float(x)
    return None if not np.isfinite(x) else x


def _load_jsonl(fp):
    return [json.loads(l) for l in Path(fp).read_text().splitlines() if l.strip()]


def _static_recflow_curve(fname, metrics):
    rows = [r for r in _load_jsonl(C.REV_RES / fname) if r["exp"] == "A"]
    by_age = defaultdict(lambda: defaultdict(list))
    for r in rows:
        ak = round(r["age_days"], 2)
        for m in metrics:
            if m in r:
                by_age[ak][m].append(r[m])
    ages = np.array(sorted(by_age), dtype=float)
    out = {}
    for m in metrics:
        out[m] = np.array([np.mean(by_age[a][m]) for a in ages], dtype=float)
    return ages, out


def _backtest_recflow_curve(fname, metrics):
    rows = _load_jsonl(C.REV_RES / fname)
    agg = defaultdict(lambda: {"w": 0.0, "n": 0, "sums": defaultdict(float)})
    for r in rows:
        a = agg[r["P"]]
        w = float(r["n_test"])
        a["w"] += w
        a["n"] += 1
        for m in metrics:
            if m in r:
                a["sums"][m] += r[m] * w
    Ps = np.array(sorted(agg), dtype=int)
    nev = np.array([agg[p]["n"] for p in Ps], dtype=int)
    out = {}
    for m in metrics:
        out[m] = np.array([agg[p]["sums"][m] / agg[p]["w"] for p in Ps], dtype=float)
    return Ps, nev, out


def _transfer_stats(Ps, meas, pred):
    dm = meas - meas[0]
    dp = pred - pred[0]
    null_log = np.log(Ps) - np.log(Ps[0])
    null_lin = Ps - Ps[0]
    return dict(
        measured_total_change=float(meas[-1] - meas[0]),
        predicted_total_change=float(pred[-1] - pred[0]),
        shape_corr=float(np.corrcoef(dm[1:], dp[1:])[0, 1]),
        slope_ratio=float(np.polyfit(dp[1:], dm[1:], 1)[0]),
        r_null_log_p=float(np.corrcoef(dm[1:], null_log[1:])[0, 1]),
        r_null_lin_p=float(np.corrcoef(dm[1:], null_lin[1:])[0, 1]),
    )


def run_exp2a():
    """Exp 2A: RecFlow static-to-rolling AUC transfer & P*_AUC."""
    ages, st = _static_recflow_curve("domain_recflow.jsonl", ("loss", "auc"))
    la = st["loss"]
    auc_a = st["auc"]
    err_a = 1.0 - auc_a

    Ps, nev, bt = _backtest_recflow_curve("backtest_recflow__d300k.jsonl", ("loss", "auc"))
    lbar = bt["loss"]
    auc_bt = bt["auc"]
    err_bt = 1.0 - auc_bt

    W_RECFLOW = 2
    a_offset = W_RECFLOW / 2.0  # 1.0 day

    def _pred_curve(y_static):
        return np.array(
            [
                np.mean(
                    np.interp(
                        np.log(np.arange(P) + 0.5 + a_offset),
                        np.log(ages),
                        y_static,
                    )
                )
                for P in Ps
            ],
            dtype=float,
        )

    pred_loss = _pred_curve(la)
    pred_err = _pred_curve(err_a)
    pred_auc = 1.0 - pred_err

    loss_tr = _transfer_stats(Ps, lbar, pred_loss)
    err_tr = _transfer_stats(Ps, err_bt, pred_err)

    # Power & logarithmic fits on static 1 - AUC
    R0_pow, b_auc, d_auc = C.fit_power(ages, err_a)
    yhat_pow = R0_pow + b_auc * (ages ** d_auc)
    sst = float(np.sum((err_a - err_a.mean()) ** 2))
    r2_pow = float(1.0 - np.sum((err_a - yhat_pow) ** 2) / sst)

    beta_auc, c0_auc = np.polyfit(np.log(ages), err_a, 1)
    yhat_log = c0_auc + beta_auc * np.log(ages)
    r2_log = float(1.0 - np.sum((err_a - yhat_log) ** 2) / sst)

    L0_loss, b_loss, d_loss = C.fit_power(ages, la)

    # Exact analytical serving-average integrals for a = W/2 = 1.0d, verified vs quad
    def _avg_pow(P):
        return R0_pow + b_auc / (P * (d_auc + 1.0)) * (
            (P + a_offset) ** (d_auc + 1.0) - a_offset ** (d_auc + 1.0)
        )

    def _avg_log(P):
        return c0_auc + beta_auc / P * (
            (P + a_offset) * np.log(P + a_offset)
            - a_offset * np.log(a_offset)
            - P
        )

    q_pow, _ = integrate.quad(lambda u: R0_pow + b_auc * (u + a_offset) ** d_auc, 0.0, 5.0)
    q_log, _ = integrate.quad(lambda u: c0_auc + beta_auc * np.log(u + a_offset), 0.0, 5.0)
    assert np.isclose(_avg_pow(5.0), q_pow / 5.0, atol=1e-12)
    assert np.isclose(_avg_log(5.0), q_log / 5.0, atol=1e-12)

    Pg = np.geomspace(Ps.min(), Ps.max(), 400)
    Eg = np.interp(np.log(Pg), np.log(Ps), err_bt)

    lamCs = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    p_star_rows = []
    for lc in lamCs:
        p_pow_a0 = float((lc * (d_auc + 1.0) / (b_auc * d_auc)) ** (1.0 / (1.0 + d_auc)))
        p_log_a0 = float(lc / beta_auc)

        opt_pow_free = optimize.minimize_scalar(
            lambda P: _avg_pow(P) + lc / P,
            bounds=(0.01, 500.0),
            method="bounded",
            options={"xatol": 1e-10},
        )
        opt_log_free = optimize.minimize_scalar(
            lambda P: _avg_log(P) + lc / P,
            bounds=(0.01, 500.0),
            method="bounded",
            options={"xatol": 1e-10},
        )
        assert opt_pow_free.success and opt_log_free.success

        opt_pow_box = optimize.minimize_scalar(
            lambda P: _avg_pow(P) + lc / P,
            bounds=(float(Ps.min()), float(Ps.max())),
            method="bounded",
            options={"xatol": 1e-10},
        )
        opt_log_box = optimize.minimize_scalar(
            lambda P: _avg_log(P) + lc / P,
            bounds=(float(Ps.min()), float(Ps.max())),
            method="bounded",
            options={"xatol": 1e-10},
        )

        k_test = int(np.argmin(err_bt + lc / Ps))
        k_interp = int(np.argmin(Eg + lc / Pg))
        p_star_rows.append(
            dict(
                lamC_auc=float(lc),
                power_a0_closed=p_pow_a0,
                log_a0_closed=p_log_a0,
                power_a1_exact=float(opt_pow_free.x),
                log_a1_exact=float(opt_log_free.x),
                power_a1_box1_22=float(opt_pow_box.x),
                log_a1_box1_22=float(opt_log_box.x),
                tested_opt=int(Ps[k_test]),
                interp_opt=float(Pg[k_interp]),
                interp_at_edge=bool(k_interp in (0, len(Pg) - 1)),
            )
        )

    # Valstop check on backtest_recflow__valstop.jsonl and domain_recflow__valstop.jsonl
    Ps_vs, nev_vs, bt_vs = _backtest_recflow_curve(
        "backtest_recflow__valstop.jsonl",
        ("loss", "loss_valstop", "loss_fixed", "auc", "auc_valstop", "auc_fixed"),
    )
    ages_vs, st_vs = _static_recflow_curve(
        "domain_recflow__valstop.jsonl",
        ("loss", "loss_valstop", "auc", "auc_valstop"),
    )
    err_bt_testargmin = 1.0 - bt_vs["auc"]
    err_bt_valstop = 1.0 - bt_vs["auc_valstop"]
    dm_vs = err_bt_valstop - err_bt_valstop[0]
    dm_ta = err_bt_testargmin - err_bt_testargmin[0]
    pred_err_vs_static = np.array(
        [
            np.mean(
                np.interp(
                    np.log(np.arange(P) + 0.5 + a_offset),
                    np.log(ages_vs),
                    1.0 - st_vs["auc_valstop"],
                )
            )
            for P in Ps_vs
        ],
        dtype=float,
    )

    valstop_summary = dict(
        periods=Ps_vs.tolist(),
        n_evals=nev_vs.tolist(),
        auc_test_argmin=bt_vs["auc"].tolist(),
        auc_valstop=bt_vs["auc_valstop"].tolist(),
        auc_fixed=bt_vs["auc_fixed"].tolist(),
        loss_test_argmin=bt_vs["loss"].tolist(),
        loss_valstop=bt_vs["loss_valstop"].tolist(),
        loss_fixed=bt_vs["loss_fixed"].tolist(),
        auc_drop_test_argmin=float(bt_vs["auc"][0] - bt_vs["auc"][-1]),
        auc_drop_valstop=float(bt_vs["auc_valstop"][0] - bt_vs["auc_valstop"][-1]),
        auc_drop_fixed=float(bt_vs["auc_fixed"][0] - bt_vs["auc_fixed"][-1]),
        mean_auc_gap_valstop_minus_testargmin=float(
            np.mean(bt_vs["auc_valstop"] - bt_vs["auc"])
        ),
        err_corr_valstop_vs_testargmin=float(np.corrcoef(dm_vs[1:], dm_ta[1:])[0, 1]),
        err_slope_valstop_vs_testargmin=float(np.polyfit(dm_ta[1:], dm_vs[1:], 1)[0]),
        valstop_static_to_rolling_transfer=_transfer_stats(
            Ps_vs, err_bt_valstop, pred_err_vs_static
        ),
    )

    return dict(
        static_curve=dict(
            ages_days=ages.tolist(),
            loss=la.tolist(),
            auc=auc_a.tolist(),
            one_minus_auc=err_a.tolist(),
        ),
        backtest_d300k=dict(
            periods=Ps.tolist(),
            n_evals=nev.tolist(),
            loss_measured=lbar.tolist(),
            loss_predicted=pred_loss.tolist(),
            loss_diff_mnat=(1000.0 * (lbar - pred_loss)).tolist(),
            auc_measured=auc_bt.tolist(),
            auc_predicted=pred_auc.tolist(),
            err_measured=err_bt.tolist(),
            err_predicted=pred_err.tolist(),
            loss_transfer=loss_tr,
            one_minus_auc_transfer=err_tr,
        ),
        fits_one_minus_auc=dict(
            power=dict(R0=float(R0_pow), b_auc=float(b_auc), delta_auc=float(d_auc), r2=r2_pow),
            log=dict(c0=float(c0_auc), beta_auc=float(beta_auc), r2=r2_log),
        ),
        fits_loss=dict(
            power=dict(L0=float(L0_loss), b=float(b_loss), delta=float(d_loss)),
        ),
        p_star_auc=p_star_rows,
        backtest_valstop=valstop_summary,
    )


def item_decorrelation_fast_exact(item, ts, y, bin_s, min_ev):
    """Exact empirical item-appeal lag-autocorrelation estimator.

    Exploits sorted `ts` via np.searchsorted slice bounds instead of O(N*nb) full
    mask scans, yielding bit-for-bit identical counts, rates, curve, and half-life.
    """
    assert np.all(ts[:-1] <= ts[1:]), "timestamps must be sorted"
    t0 = int(ts[0])
    bin_s = int(bin_s)
    nb = int((int(ts[-1]) - t0) // bin_s) + 1
    n_items = int(item.max()) + 1
    edges = t0 + np.arange(nb + 1, dtype=np.int64) * bin_s
    bounds = np.searchsorted(ts, edges, side="left")
    assert bounds[-1] == len(ts)

    rate = np.full((n_items, nb), np.nan, dtype=np.float32)
    cnt = np.zeros((n_items, nb), dtype=np.int32)
    y_f = y.astype(np.float64)
    for bi in range(nb):
        lo, hi = int(bounds[bi]), int(bounds[bi + 1])
        if hi == lo:
            continue
        it = item[lo:hi]
        c = np.bincount(it, minlength=n_items)
        s = np.bincount(it, weights=y_f[lo:hi], minlength=n_items)
        ok = c >= min_ev
        cnt[ok, bi] = c[ok]
        rate[ok, bi] = (s[ok] / c[ok]).astype(np.float32)

    max_lag = min(nb - 1, int(np.ceil(nb * 0.7)))
    curve = []
    for L in range(1, max_lag + 1):
        xs, ys = [], []
        for base in range(nb - L):
            m = cnt[:, base] > 0
            m &= cnt[:, base + L] > 0
            if m.sum() < 30:
                continue
            xs.append(rate[m, base])
            ys.append(rate[m, base + L])
        if not xs:
            continue
        x = np.concatenate(xs)
        yv = np.concatenate(ys)
        if len(x) < 50 or x.std() < 1e-6 or yv.std() < 1e-6:
            continue
        r = float(np.corrcoef(x, yv)[0, 1])
        curve.append((float(L * bin_s / 86_400.0), r, int(len(x))))

    half = None
    prev = None
    for lagd, r, _ in curve:
        if r < 0.5 and prev is not None:
            pl, pr = prev
            frac = (pr - 0.5) / (pr - r) if pr != r else 0.0
            half = float(np.exp(np.log(pl) + frac * (np.log(lagd) - np.log(pl))))
            break
        prev = (lagd, r)
    return curve, half


def _verify_decorr_bin_equivalence(item, ts, y, bin_s, min_ev):
    """Spot-check that searchsorted slice bounds match `(ts - t0) // bin_s == bi`."""
    t0 = int(ts[0])
    b = ((ts - t0) // bin_s).astype(np.int64)
    nb = int(b.max()) + 1
    edges = t0 + np.arange(nb + 1, dtype=np.int64) * int(bin_s)
    bounds = np.searchsorted(ts, edges, side="left")
    n_items = int(item.max()) + 1
    for bi in (0, nb // 4, nb // 2, nb - 1):
        lo, hi = int(bounds[bi]), int(bounds[bi + 1])
        m = b == bi
        c_fast = np.bincount(item[lo:hi], minlength=n_items)
        c_slow = np.bincount(item[m], minlength=n_items)
        assert np.array_equal(c_fast, c_slow), f"bin {bi} count mismatch"


def run_domain_analyses():
    """Single-pass per-domain `.npz` loader for Exp 1 Part 1, Exp 5, and Exp 4 Part 1."""
    collapse_domains = json.loads((C.REV_RES / "collapse_fit.json").read_text())["domains"]

    cohort_files_all = sorted(
        glob.glob(str(C.REV_RES / "domain_*__cohort.jsonl"))
        + glob.glob(str(C.REV_RES / "domain_*__icohort.jsonl"))
    )
    cohort_files = [
        f
        for f in cohort_files_all
        if not any(
            t in Path(f).name for t in ("__dim", "__ss1", "__p1", "__p2", "__w7", "__mlp")
        )
    ]
    assert len(cohort_files) == 26, f"Expected 26 cohort files, got {len(cohort_files)}"

    cohorts_by_dom = defaultdict(list)
    for fp in cohort_files:
        stem = Path(fp).stem.replace("domain_", "")
        dom = stem.replace("__icohort", "").replace("__cohort", "")
        cohorts_by_dom[dom].append((stem, Path(fp)))

    exp1_domains = {}
    exp1_checks = {
        "n_domains": 0,
        "n_configs_checked": 0,
        "n_seed_runs_checked": 0,
        "max_sum_omega_err": 0.0,
        "max_abs_diff_vs_seen_frac": 0.0,
    }
    exp5_cohorts = {}
    exp4_curves = {}
    verified_decorr_bins = False

    for dom in C.PANEL + C.AMZ:
        t_dom = time.time()
        npz_path = C.DATA_DIR / C.DOMAINS[dom]["npz"]
        if not npz_path.exists():
            raise SystemExit(
                f"Missing {npz_path}; build preprocessed .npz first via src/prep_*.py."
            )
        with np.load(npz_path) as z:
            ts = z["ts"]
            user_all = z["user"]
            item_all = z["item"]
            y_all = z["label"]
        n_users = int(user_all.max()) + 1
        n_items = int(item_all.max()) + 1
        data_dict = {"ts": ts, "user": user_all, "item": item_all, "label": y_all}

        kept, d_fix, test_t0, test_t1, span_avail, dropped = C.build_window_schedule(
            ts, C.DOMAINS[dom]
        )

        # ---- Exp 1 Part 1: Deterministic 4-way coverage shares -----------
        base_rows = _load_jsonl(C.REV_RES / f"domain_{dom}.jsonl")
        base_idx = {
            (r["exp"], round(r["age_days"], 4), r["seed"]): r for r in base_rows
        }
        assert len(base_rows) == len(kept) * len(C.SEEDS), (
            f"{dom}: base rows {len(base_rows)} != {len(kept) * len(C.SEEDS)}"
        )

        c0, e0 = np.searchsorted(ts, [test_t0, test_t1])
        te_u, te_i = user_all[c0:e0], item_all[c0:e0]
        n_test = int(e0 - c0)
        su = np.zeros(n_users, dtype=bool)
        si = np.zeros(n_items, dtype=bool)

        cfg_records = []
        max_dom_sf_diff = 0.0
        for cfg in kept:
            a, b = np.searchsorted(ts, [cfg["train_t0"], cfg["train_t1"]])
            seed_recs = []
            for seed in C.SEEDS:
                base_row = base_idx[(cfg["exp"], round(cfg["age_days"], 4), seed)]
                assert d_fix == base_row["subsample"], f"{dom} subsample mismatch"
                assert test_t0 == base_row["test_t0"] and test_t1 == base_row["test_t1"]
                assert n_test == base_row["n_test"]

                if d_fix < b - a:
                    rng = np.random.default_rng(seed)
                    sel = a + np.sort(
                        rng.choice(b - a, size=int(d_fix), replace=False)
                    )
                    su_idx, si_idx = user_all[sel], item_all[sel]
                else:
                    su_idx, si_idx = user_all[a:b], item_all[a:b]

                su[su_idx] = True
                si[si_idx] = True
                u_warm = su[te_u]
                i_warm = si[te_i]
                su[su_idx] = False
                si[si_idx] = False

                n11 = int((u_warm & i_warm).sum())
                n10 = int((u_warm & ~i_warm).sum())
                n01 = int((~u_warm & i_warm).sum())
                n00 = int((~u_warm & ~i_warm).sum())
                assert n11 + n10 + n01 + n00 == n_test

                w11 = n11 / n_test
                w10 = n10 / n_test
                w01 = n01 / n_test
                w00 = n00 / n_test
                sum_err = abs((w11 + w10 + w01 + w00) - 1.0)
                assert sum_err < 1e-9

                sf_diff = abs(w11 - base_row["seen_frac"])
                assert sf_diff < 1e-12, (
                    f"{dom} {cfg['exp']} {cfg['age_days']} s{seed}: "
                    f"{w11} vs {base_row['seen_frac']}"
                )

                exp1_checks["n_seed_runs_checked"] += 1
                exp1_checks["max_sum_omega_err"] = max(
                    exp1_checks["max_sum_omega_err"], sum_err
                )
                exp1_checks["max_abs_diff_vs_seen_frac"] = max(
                    exp1_checks["max_abs_diff_vs_seen_frac"], sf_diff
                )
                max_dom_sf_diff = max(max_dom_sf_diff, sf_diff)
                seed_recs.append(
                    dict(
                        seed=seed,
                        omega_11=w11,
                        omega_10=w10,
                        omega_01=w01,
                        omega_00=w00,
                        n_11=n11,
                        n_10=n10,
                        n_01=n01,
                        n_00=n00,
                        shipped_seen_frac=float(base_row["seen_frac"]),
                        shipped_loss=float(base_row["loss"]),
                        shipped_loss_seen=float(base_row["loss_seen"]),
                    )
                )

            exp1_checks["n_configs_checked"] += 1
            cfg_records.append(
                dict(
                    exp=cfg["exp"],
                    age_days=float(cfg["age_days"]),
                    gap=cfg.get("gap"),
                    span=cfg.get("span"),
                    omega_11=float(np.mean([s["omega_11"] for s in seed_recs])),
                    omega_10=float(np.mean([s["omega_10"] for s in seed_recs])),
                    omega_01=float(np.mean([s["omega_01"] for s in seed_recs])),
                    omega_00=float(np.mean([s["omega_00"] for s in seed_recs])),
                    n_11_mean=float(np.mean([s["n_11"] for s in seed_recs])),
                    n_10_mean=float(np.mean([s["n_10"] for s in seed_recs])),
                    n_01_mean=float(np.mean([s["n_01"] for s in seed_recs])),
                    n_00_mean=float(np.mean([s["n_00"] for s in seed_recs])),
                    per_seed=seed_recs,
                )
            )

        arm_a_cfgs = sorted(
            [c for c in cfg_records if c["exp"] == "A"], key=lambda x: x["age_days"]
        )
        ages_a = np.array([c["age_days"] for c in arm_a_cfgs], dtype=float)
        log_ages_a = np.log(ages_a)
        arm_a_slopes = {}
        for k_om in ("omega_11", "omega_10", "omega_01", "omega_00"):
            vals = np.array([c[k_om] for c in arm_a_cfgs], dtype=float)
            arm_a_slopes[k_om] = float(np.polyfit(log_ages_a, vals, 1)[0])

        exp1_domains[dom] = dict(
            n_users=n_users,
            n_items=n_items,
            n_events=int(len(ts)),
            n_test=n_test,
            d_fix=d_fix,
            span_avail=span_avail,
            max_abs_diff_vs_seen_frac=max_dom_sf_diff,
            arm_a_freshest={
                k: arm_a_cfgs[0][k]
                for k in ("age_days", "omega_11", "omega_10", "omega_01", "omega_00")
            },
            arm_a_oldest={
                k: arm_a_cfgs[-1][k]
                for k in ("age_days", "omega_11", "omega_10", "omega_01", "omega_00")
            },
            arm_a_log_age_slopes=arm_a_slopes,
            configs=cfg_records,
        )
        exp1_checks["n_domains"] += 1

        # ---- Exp 5: Cohort item-lifecycle training-density check ----------
        kept_a = sorted(
            [c for c in kept if c["exp"] == "A"], key=lambda x: x["age_days"]
        )
        for cohort_name, fp in cohorts_by_dom.get(dom, []):
            item_only = cohort_name.endswith("__icohort")
            rows = _load_jsonl(fp)
            ci_np = C.build_cohort_idx(
                data_dict,
                kept_a,
                d_fix,
                test_t0,
                test_t1,
                item_only=item_only,
                sub_seed=0,
                as_torch=False,
            )
            assert len(ci_np) == rows[0]["n_cohort"], (
                f"{cohort_name}: {len(ci_np)} != {rows[0]['n_cohort']}"
            )
            assert d_fix == rows[0]["subsample"]
            assert test_t0 == rows[0]["test_t0"] and test_t1 == rows[0]["test_t1"]

            cohort_items = np.unique(te_i[ci_np])
            imask = np.zeros(n_items, dtype=bool)
            imask[cohort_items] = True

            cohort_users = None
            umask = None
            if not item_only:
                cohort_users = np.unique(te_u[ci_np])
                umask = np.zeros(n_users, dtype=bool)
                umask[cohort_users] = True

            cells = defaultdict(list)
            for r in rows:
                cells[round(r["age_days"], 2)].append(r["loss_cohort"])

            per_age = []
            for c in kept_a:
                a, b = np.searchsorted(ts, [c["train_t0"], c["train_t1"]])
                if d_fix < b - a:
                    rng = np.random.default_rng(0)
                    sel = a + np.sort(
                        rng.choice(b - a, size=int(d_fix), replace=False)
                    )
                else:
                    sel = np.arange(a, b)
                ev_i = int(imask[item_all[sel]].sum())
                mean_i = float(ev_i / len(cohort_items))
                ev_u = int(umask[user_all[sel]].sum()) if not item_only else None
                mean_u = (
                    float(ev_u / len(cohort_users)) if not item_only else None
                )
                ak = round(c["age_days"], 2)
                lc_vals = cells[ak]
                per_age.append(
                    dict(
                        age_days=float(c["age_days"]),
                        gap=int(c["gap"]),
                        train_ev_cohort_items=ev_i,
                        mean_ev_per_cohort_item=mean_i,
                        train_ev_cohort_users=ev_u,
                        mean_ev_per_cohort_user=mean_u,
                        loss_cohort_mean=float(np.mean(lc_vals)),
                        n_seeds=int(len(lc_vals)),
                    )
                )

            ages_c = np.array([x["age_days"] for x in per_age], dtype=float)
            ev_items = np.array(
                [x["train_ev_cohort_items"] for x in per_age], dtype=float
            )
            losses_c = np.array(
                [x["loss_cohort_mean"] for x in per_age], dtype=float
            )

            sp_raw = stats.spearmanr(ages_c, ev_items)
            dloss_raw = float(losses_c[-1] - losses_c[0])
            dlnev_raw = float(np.log(ev_items[-1] / ev_items[0]))

            qc = losses_c <= 1.0
            sp_qc = stats.spearmanr(ages_c[qc], ev_items[qc])
            dloss_qc = float(losses_c[qc][-1] - losses_c[qc][0])
            dlnev_qc = float(np.log(ev_items[qc][-1] / ev_items[qc][0]))

            sp_u = None
            dlnev_u_raw = None
            if not item_only:
                ev_users = np.array(
                    [x["train_ev_cohort_users"] for x in per_age], dtype=float
                )
                sp_u_res = stats.spearmanr(ages_c, ev_users)
                sp_u = _safe_float(sp_u_res.statistic)
                if ev_users[0] > 0 and ev_users[-1] > 0:
                    dlnev_u_raw = float(np.log(ev_users[-1] / ev_users[0]))

            cf_entry = collapse_domains[cohort_name]
            exp5_cohorts[cohort_name] = dict(
                domain=dom,
                item_only=item_only,
                n_cohort=int(len(ci)),
                n_cohort_items=int(len(cohort_items)),
                n_cohort_users=int(len(cohort_users)) if not item_only else None,
                d_fix=int(d_fix),
                collapse_descending=bool(cf_entry["descending"]),
                collapse_A=float(cf_entry["A"]),
                collapse_identified=bool(cf_entry["identified"]),
                delta_loss_cohort_raw=dloss_raw,
                delta_loss_cohort_qc=dloss_qc,
                delta_ln_train_ev_cohort_items_raw=dlnev_raw,
                delta_ln_train_ev_cohort_items_qc=dlnev_qc,
                spearman_tau_vs_ev_items_raw=_safe_float(sp_raw.statistic),
                spearman_tau_vs_ev_items_p_raw=_safe_float(sp_raw.pvalue),
                spearman_tau_vs_ev_items_qc=_safe_float(sp_qc.statistic),
                spearman_tau_vs_ev_users_raw=sp_u,
                delta_ln_train_ev_cohort_users_raw=dlnev_u_raw,
                n_dropped_qc=int((~qc).sum()),
                per_age=per_age,
            )

        # ---- Exp 4 Part 1: Model-free Game-UGC item-appeal autocorrelation -
        if dom in ("gameugc_finish", "gameugc_play"):
            if not verified_decorr_bins:
                _verify_decorr_bin_equivalence(
                    item_all, ts, y_all, 7 * C.DAY, 50
                )
                verified_decorr_bins = True
            dom_decorr = {}
            for bin_days, min_ev in ((7, 30), (3, 30), (7, 50)):
                curve, half = item_decorrelation_fast_exact(
                    item_all, ts, y_all, bin_days * C.DAY, min_ev
                )
                rs = [pt[1] for pt in curve]
                dom_decorr[f"bin{bin_days}d_minev{min_ev}"] = dict(
                    bin_days=bin_days,
                    min_ev=min_ev,
                    n_lags=len(curve),
                    half_life_days=half,
                    first_lag_days=curve[0][0] if curve else None,
                    first_corr=curve[0][1] if curve else None,
                    last_lag_days=curve[-1][0] if curve else None,
                    last_corr=curve[-1][1] if curve else None,
                    min_corr=float(min(rs)) if rs else None,
                    max_corr=float(max(rs)) if rs else None,
                    curve=[
                        {"lag_days": pt[0], "corr": pt[1], "n_pairs": pt[2]}
                        for pt in curve
                    ],
                )
            exp4_curves[dom] = dom_decorr

        del data_dict, ts, user_all, item_all, y_all
        print(
            f"  processed {dom:28s} ({time.time() - t_dom:.1f}s)", flush=True
        )

    assert len(exp5_cohorts) == 26
    assert exp1_checks["n_domains"] == 23

    # ---- Exp 5 summary across 26 cohorts ---------------------------------
    names = sorted(exp5_cohorts)
    dl_raw = np.array([exp5_cohorts[k]["delta_loss_cohort_raw"] for k in names])
    dl_qc = np.array([exp5_cohorts[k]["delta_loss_cohort_qc"] for k in names])
    de_raw = np.array(
        [exp5_cohorts[k]["delta_ln_train_ev_cohort_items_raw"] for k in names]
    )
    de_qc = np.array(
        [exp5_cohorts[k]["delta_ln_train_ev_cohort_items_qc"] for k in names]
    )
    sp_arr = np.array(
        [exp5_cohorts[k]["spearman_tau_vs_ev_items_raw"] for k in names]
    )
    is_desc = np.array([exp5_cohorts[k]["collapse_descending"] for k in names])
    no_drop = np.array([exp5_cohorts[k]["n_dropped_qc"] == 0 for k in names])

    def _group_stats(mask):
        return dict(
            n=int(mask.sum()),
            cohorts=[names[i] for i in np.flatnonzero(mask)],
            spearman_tau_ev_median=float(np.median(sp_arr[mask])),
            spearman_tau_ev_mean=float(np.mean(sp_arr[mask])),
            n_positive_spearman=int((sp_arr[mask] > 0).sum()),
            delta_ln_ev_median=float(np.median(de_qc[mask])),
            delta_ln_ev_mean=float(np.mean(de_qc[mask])),
            n_positive_delta_ln_ev=int((de_qc[mask] > 0).sum()),
            delta_loss_qc_median=float(np.median(dl_qc[mask])),
        )

    pr_raw = stats.pearsonr(dl_raw, de_raw)
    sr_raw = stats.spearmanr(dl_raw, de_raw)
    pr_qc = stats.pearsonr(dl_qc, de_qc)
    sr_qc = stats.spearmanr(dl_qc, de_qc)
    pr_nodrop = stats.pearsonr(dl_raw[no_drop], de_raw[no_drop])
    sr_nodrop = stats.spearmanr(dl_raw[no_drop], de_raw[no_drop])

    exp5_summary = dict(
        n_cohorts=len(names),
        all_26_raw=dict(
            pearson_r=float(pr_raw.statistic),
            pearson_p=float(pr_raw.pvalue),
            spearman_r=float(sr_raw.statistic),
            spearman_p=float(sr_raw.pvalue),
        ),
        all_26_qc_loss_le_1=dict(
            pearson_r=float(pr_qc.statistic),
            pearson_p=float(pr_qc.pvalue),
            spearman_r=float(sr_qc.statistic),
            spearman_p=float(sr_qc.pvalue),
        ),
        no_qc_drops_25=dict(
            n=int(no_drop.sum()),
            pearson_r=float(pr_nodrop.statistic),
            pearson_p=float(pr_nodrop.pvalue),
            spearman_r=float(sr_nodrop.statistic),
            spearman_p=float(sr_nodrop.pvalue),
        ),
        rising_by_collapse=_group_stats(~is_desc),
        descending_by_collapse=_group_stats(is_desc),
    )

    return (
        dict(checks=exp1_checks, domains=exp1_domains),
        dict(summary=exp5_summary, cohorts=exp5_cohorts),
        dict(bin_equivalence_verified=verified_decorr_bins, domains=exp4_curves),
    )


def main():
    ap = argparse.ArgumentParser(
        description="Stage 1 zero-new-training checks and summaries (reads results/ and data/*.npz)."
    )
    ap.parse_args()
    t0 = time.time()
    C.CR_RES.mkdir(parents=True, exist_ok=True)
    print("Running Exp 2A (RecFlow static-to-rolling AUC transfer & P*_AUC)...", flush=True)
    exp2a = run_exp2a()

    print("Running per-domain single-pass checks (Exp 1 Part 1, Exp 5, Exp 4 Part 1)...", flush=True)
    exp1_part1, exp5, exp4_part1 = run_domain_analyses()

    out = dict(
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        exp2a=exp2a,
        exp5=exp5,
        exp4_part1=exp4_part1,
        exp1_part1=exp1_part1,
    )
    out_path = C.CR_RES / "stage1_zero_run.json"
    out_path.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(f"\nSaved {out_path} in {time.time() - t0:.1f}s")

    # ---- Readable summary of key findings & verification checks ----------
    print("\n" + "=" * 78)
    print("EXP 2A SUMMARY")
    lt = exp2a["backtest_d300k"]["loss_transfer"]
    et = exp2a["backtest_d300k"]["one_minus_auc_transfer"]
    print(
        f"  Loss transfer : slope_ratio={lt['slope_ratio']:.3f}, "
        f"shape_corr={lt['shape_corr']:.4f} "
        f"(logP={lt['r_null_log_p']:.4f}, linP={lt['r_null_lin_p']:.4f})"
    )
    print(
        f"  1-AUC transfer: slope_ratio={et['slope_ratio']:.3f}, "
        f"shape_corr={et['shape_corr']:.4f} "
        f"(logP={et['r_null_log_p']:.4f}, linP={et['r_null_lin_p']:.4f})"
    )
    fp_auc = exp2a["fits_one_minus_auc"]["power"]
    fl_auc = exp2a["fits_one_minus_auc"]["log"]
    print(
        f"  1-AUC fits: power R0={fp_auc['R0']:.4f}, b={fp_auc['b_auc']:.4f}, "
        f"delta={fp_auc['delta_auc']:.4f}, R2={fp_auc['r2']:.4f} | "
        f"log c0={fl_auc['c0']:.4f}, beta={fl_auc['beta_auc']:.5f}, R2={fl_auc['r2']:.4f}"
    )
    for row in exp2a["p_star_auc"]:
        print(
            f"  lamC={row['lamC_auc']:.0e}: pow(a=0)={row['power_a0_closed']:.2f}, "
            f"log(a=0)={row['log_a0_closed']:.2f}, pow(a=1)={row['power_a1_exact']:.2f}, "
            f"log(a=1)={row['log_a1_exact']:.2f} | tested={row['tested_opt']}, "
            f"interp={row['interp_opt']:.2f}{' (edge)' if row['interp_at_edge'] else ''}"
        )
    vs = exp2a["backtest_valstop"]
    print(
        f"  Valstop AUC drop P=1->22: test-argmin={100*vs['auc_drop_test_argmin']:.2f} pts, "
        f"valstop={100*vs['auc_drop_valstop']:.2f} pts "
        f"(corr={vs['err_corr_valstop_vs_testargmin']:.4f}, "
        f"slope={vs['err_slope_valstop_vs_testargmin']:.3f})"
    )

    print("\n" + "=" * 78)
    print("EXP 5 SUMMARY (26 cohorts)")
    s5 = exp5["summary"]
    print(
        f"  All 26 cohorts matched len(cohort_idx) == n_cohort! "
        f"Rising n={s5['rising_by_collapse']['n']}, "
        f"Descending n={s5['descending_by_collapse']['n']}"
    )
    print(
        f"  Rising   : median spearman(tau, ev_items)={s5['rising_by_collapse']['spearman_tau_ev_median']:+.3f}, "
        f"median dlnev={s5['rising_by_collapse']['delta_ln_ev_median']:+.3f}, "
        f"pos_dlnev={s5['rising_by_collapse']['n_positive_delta_ln_ev']}/{s5['rising_by_collapse']['n']}"
    )
    print(
        f"  Descend. : median spearman(tau, ev_items)={s5['descending_by_collapse']['spearman_tau_ev_median']:+.3f}, "
        f"median dlnev={s5['descending_by_collapse']['delta_ln_ev_median']:+.3f}, "
        f"pos_dlnev={s5['descending_by_collapse']['n_positive_delta_ln_ev']}/{s5['descending_by_collapse']['n']}"
    )
    print(
        f"  Cross-cohort corr(delta_loss, delta_ln_ev): "
        f"raw r={s5['all_26_raw']['pearson_r']:+.3f} (p={s5['all_26_raw']['pearson_p']:.3f}), "
        f"rho={s5['all_26_raw']['spearman_r']:+.3f} (p={s5['all_26_raw']['spearman_p']:.3f}) | "
        f"QC r={s5['all_26_qc_loss_le_1']['pearson_r']:+.3f} (p={s5['all_26_qc_loss_le_1']['pearson_p']:.3f}), "
        f"rho={s5['all_26_qc_loss_le_1']['spearman_r']:+.3f} (p={s5['all_26_qc_loss_le_1']['spearman_p']:.3f})"
    )

    print("\n" + "=" * 78)
    print("EXP 4 PART 1 SUMMARY (Game-UGC item-appeal autocorrelation)")
    for dom, cfg_map in exp4_part1["domains"].items():
        for k, v in cfg_map.items():
            print(
                f"  {dom:15s} {k:16s}: n_lags={v['n_lags']:3d}, "
                f"first=({v['first_lag_days']:.0f}d, r={v['first_corr']:+.3f}), "
                f"last=({v['last_lag_days']:.0f}d, r={v['last_corr']:+.3f}), "
                f"min_r={v['min_corr']:+.3f}, half_life={v['half_life_days']}"
            )

    print("\n" + "=" * 78)
    print("EXP 1 PART 1 SUMMARY (4-way coverage shares across 23 domains)")
    chk = exp1_part1["checks"]
    print(
        f"  Checked {chk['n_domains']} domains, {chk['n_configs_checked']} configs, "
        f"{chk['n_seed_runs_checked']} seed runs: "
        f"max |sum(omega)-1|={chk['max_sum_omega_err']:.2e}, "
        f"max |omega_11-seen_frac|={chk['max_abs_diff_vs_seen_frac']:.2e}"
    )


if __name__ == "__main__":
    main()
