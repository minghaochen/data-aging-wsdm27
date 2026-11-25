"""Stage 2: Full four-way warm/cold decomposition across all 23 domains.

Runs Arm A across all 23 domains (SEEDS=[0, 1, 2], val_frac=None) and Arm A+C
on ("kuairand", "recflow", "ml25m", "amz_Video_Games"), logging full 4-way
coverage/loss/AUC metrics and fresh prior-shift recalibration (`loss_*_recal`).

Outputs exclusively to `results/`:
  - `fourway_<domain>.jsonl` (resumable, flushed per run)
  - `stage2_fourway_summary.json` (compatibility, partition identity checks,
    4-way midpoint attribution, internal cold decomposition raw & recal, and
    joint A+C Box-Cox fits).

Strictly read-only with respect to existing `results/` outputs.
PyTorch is imported lazily when training so `--help` and `--summary-only`
only require NumPy and SciPy. No matplotlib imports.

Usage: python src/stage2_fourway_cold.py [--summary-only] [--domains all|d1,d2]
"""
import os

for _k in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_k] = "2"

import argparse
import gc
import json
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import cr_lib as C

ALL_DOMAINS = C.PANEL + C.AMZ
AC_DOMAINS = ("kuairand", "recflow", "ml25m", "amz_Video_Games")
LARGE_DOMAINS = {
    "netflix",
    "gameugc_finish",
    "gameugc_play",
    "recflow",
    "ml25m",
    "steam",
    "epinions",
    "amz_CDs_and_Vinyl",
    "amz_Toys_and_Games",
    "amz_Automotive",
    "amz_Kindle_Store",
    "amz_Office_Products",
    "amz_Pet_Supplies",
}
GROUPS = ("11", "10", "01", "00")
COLD_GROUPS = ("10", "01", "00")
BOXCOX_METRICS = (
    "loss",
    "loss_recal",
    "loss_11",
    "loss_10",
    "loss_01",
    "loss_00",
    "loss_00_recal",
    "loss_11_recal",
    "loss_10_recal",
    "loss_01_recal",
)
N_BOOT = 300

_WORKER_STATE = {}


def _run_key(r):
    return (
        str(r["exp"]),
        int(r["train_t0"]),
        int(r["train_t1"]),
        int(r["seed"]),
        int(r["subsample"]),
    )


def _load_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _init_worker(dom):
    for _k in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[_k] = "2"
    import torch

    torch.set_num_threads(2)
    data_dict, n_users, n_items = C.load_domain_npz(dom)
    _WORKER_STATE.clear()
    _WORKER_STATE["dom"] = dom
    _WORKER_STATE["data_dict"] = data_dict
    _WORKER_STATE["n_users"] = n_users
    _WORKER_STATE["n_items"] = n_items


def _run_worker_task(task):
    dom = task["domain"]
    assert _WORKER_STATE.get("dom") == dom, f"Worker loaded {_WORKER_STATE.get('dom')} != {dom}"
    res = C.run_preloaded(
        _WORKER_STATE["data_dict"],
        _WORKER_STATE["n_users"],
        _WORKER_STATE["n_items"],
        task["train_t0"],
        task["train_t1"],
        task["test_t0"],
        task["test_t1"],
        steps=8000,
        batch=8192,
        dim=32,
        lr=3e-3,
        seed=task["seed"],
        subsample=task["subsample"],
        eval_every=task["eval_every"],
        probe="mf",
        device="cpu",
        calib_t0=task["calib_t0"],
        calib_t1=task["calib_t1"],
        calib_cap=50_000,
        val_frac=None,
        log_fourway=True,
    )
    res["exp"] = task["exp"]
    res["age_days"] = float(task["age_days"])
    res["vol"] = int(task["vol"])
    res["domain"] = dom
    res["calib_t0"] = int(task["calib_t0"])
    res["calib_t1"] = int(task["calib_t1"])
    if "gap" in task:
        res["gap"] = int(task["gap"])
    if "span" in task:
        res["span"] = int(task["span"])
    gc.collect()
    return res


def _build_domain_tasks(dom):
    spec = C.DOMAINS[dom]
    npz_path = C.DATA_DIR / spec["npz"]
    with np.load(npz_path) as z:
        ts = z["ts"]
    n_events = int(len(ts))
    kept, d_fix, test_t0, test_t1, span_avail, _ = C.build_window_schedule(ts, spec)
    calib_t0 = int(test_t0 - spec["unit"])
    calib_t1 = int(test_t0)
    ca, cb = np.searchsorted(ts, [calib_t0, calib_t1])
    n_calib_avail = int(cb - ca)
    assert n_calib_avail > 0, f"{dom}: empty calibration slice [{calib_t0}, {calib_t1})"
    assert calib_t1 <= test_t0, f"{dom}: calib_t1 {calib_t1} > test_t0 {test_t0}"

    arms = ("A", "C") if dom in AC_DOMAINS else ("A",)
    eval_every = int(spec.get("eval_every", 200))

    tasks = []
    for cfg in kept:
        if cfg["exp"] not in arms:
            continue
        for seed in C.SEEDS:
            t = {
                "domain": dom,
                "exp": cfg["exp"],
                "train_t0": int(cfg["train_t0"]),
                "train_t1": int(cfg["train_t1"]),
                "test_t0": int(test_t0),
                "test_t1": int(test_t1),
                "calib_t0": calib_t0,
                "calib_t1": calib_t1,
                "seed": int(seed),
                "subsample": int(d_fix),
                "eval_every": eval_every,
                "age_days": float(cfg["age_days"]),
                "vol": int(cfg["vol"]),
            }
            if "gap" in cfg:
                t["gap"] = int(cfg["gap"])
            if "span" in cfg:
                t["span"] = int(cfg["span"])
            tasks.append(t)

    del ts
    gc.collect()
    return tasks, n_events, n_calib_avail, d_fix, span_avail


def run_domain(dom):
    out_path = C.CR_RES / f"fourway_{dom}.jsonl"
    tasks, n_events, n_calib_avail, d_fix, span_avail = _build_domain_tasks(dom)
    existing = _load_jsonl(out_path)
    done_keys = {_run_key(r) for r in existing}
    todo = [t for t in tasks if _run_key(t) not in done_keys]

    arms_str = "A+C" if dom in AC_DOMAINS else "A"
    print(
        f"\n[{dom}] arms={arms_str} events={n_events:,} d_fix={d_fix:,} "
        f"calib_avail={n_calib_avail:,} total={len(tasks)} done={len(done_keys)} todo={len(todo)}",
        flush=True,
    )
    if not todo:
        return len(tasks), 0.0

    max_w = 2 if (dom in LARGE_DOMAINS or n_events > 10_000_000) else 3
    n_workers = min(max_w, len(todo))
    t0 = time.time()

    ctx = mp.get_context("spawn")
    with out_path.open("a") as fh:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(dom,),
        ) as pool:
            futures = {pool.submit(_run_worker_task, t): t for t in todo}
            completed = 0
            for fut in as_completed(futures):
                res = fut.result()
                fh.write(json.dumps(res) + "\n")
                fh.flush()
                completed += 1
                tag = (
                    f"gap={res['gap']}"
                    if "gap" in res
                    else f"span={res.get('span', '?')}"
                )
                print(
                    f"  [{dom}] ({completed}/{len(todo)}) {res['exp']} {tag:8s} "
                    f"s={res['seed']} loss={res['loss']:.5f} recal={res['loss_recal']:.5f} "
                    f"w11={res['omega_11']:.4f} w00={res['omega_00']:.4f} "
                    f"({res['train_time_s']}s)",
                    flush=True,
                )
    elapsed = time.time() - t0
    gc.collect()
    print(f"  [{dom}] finished {len(todo)} runs in {elapsed:.1f}s (workers={n_workers})", flush=True)
    return len(tasks), elapsed


def _weighted_loss_term(r, g, suffix=""):
    om = r[f"omega_{g}"]
    lg = r[f"loss_{g}{suffix}"]
    if om == 0 or lg is None:
        return 0.0
    return float(om) * float(lg)


def _check_domain_compatibility_and_partition(dom, rows, shipped_rows):
    shipped_by_key = {
        (r["exp"], round(float(r["age_days"]), 4), int(r["seed"])): r
        for r in shipped_rows
    }
    cpu_cells = defaultdict(list)
    shipped_cells = defaultdict(list)

    max_seed_loss_diff = 0.0
    max_shipped_sf_diff = 0.0
    max_cpu_sf_diff = 0.0
    max_sum_om_err = 0.0
    max_raw_part_err = 0.0
    max_recal_part_err = 0.0
    max_loss11_err = 0.0
    min_counts = {g: None for g in GROUPS}

    for r in rows:
        # 4-way partition identity checks
        om_sum = sum(float(r[f"omega_{g}"]) for g in GROUPS)
        max_sum_om_err = max(max_sum_om_err, abs(om_sum - 1.0))

        raw_part = sum(_weighted_loss_term(r, g, "") for g in GROUPS)
        raw_err = abs(raw_part - float(r["loss"]))
        max_raw_part_err = max(max_raw_part_err, raw_err)
        assert raw_err < 1e-6, f"{dom}: raw partition error {raw_err}"

        recal_part = sum(_weighted_loss_term(r, g, "_recal") for g in GROUPS)
        recal_err = abs(recal_part - float(r["loss_recal"]))
        max_recal_part_err = max(max_recal_part_err, recal_err)
        assert recal_err < 1e-6, f"{dom}: recal partition error {recal_err}"

        cpu_sf_diff = abs(float(r["seen_frac"]) - float(r["omega_11"]))
        max_cpu_sf_diff = max(max_cpu_sf_diff, cpu_sf_diff)
        assert cpu_sf_diff < 1e-12, f"{dom}: seen_frac vs omega_11 diff {cpu_sf_diff}"

        if r["n_11"] > 0:
            l11_err = abs(float(r["loss_11"]) - float(r["loss_seen"]))
            max_loss11_err = max(max_loss11_err, l11_err)
            assert l11_err < 1e-6, f"{dom}: loss_11 vs loss_seen diff {l11_err}"

        for g in GROUPS:
            cnt = int(r[f"n_{g}"])
            min_counts[g] = cnt if min_counts[g] is None else min(min_counts[g], cnt)

        k = (r["exp"], round(float(r["age_days"]), 4), int(r["seed"]))
        sh = shipped_by_key[k]
        max_seed_loss_diff = max(max_seed_loss_diff, abs(float(r["loss"]) - float(sh["loss"])))
        shipped_sf_diff = abs(float(sh["seen_frac"]) - float(r["omega_11"]))
        max_shipped_sf_diff = max(max_shipped_sf_diff, shipped_sf_diff)
        assert shipped_sf_diff < 1e-12, (
            f"{dom} {k}: shipped seen_frac {sh['seen_frac']} != omega_11 {r['omega_11']}"
        )

        ck = (r["exp"], round(float(r["age_days"]), 4))
        cpu_cells[ck].append(r)
        shipped_cells[ck].append(sh)

    max_mean_loss_diff = 0.0
    max_mean_loss_seen_diff = 0.0
    max_mean_auc_diff = 0.0
    for ck, cr in sorted(cpu_cells.items()):
        sr = shipped_cells[ck]
        assert len(cr) == len(sr) == 3, f"{dom} {ck}: seed count mismatch {len(cr)} vs {len(sr)}"
        max_mean_loss_diff = max(
            max_mean_loss_diff,
            abs(float(np.mean([x["loss"] for x in cr])) - float(np.mean([x["loss"] for x in sr]))),
        )
        max_mean_loss_seen_diff = max(
            max_mean_loss_seen_diff,
            abs(
                float(np.mean([x["loss_seen"] for x in cr]))
                - float(np.mean([x["loss_seen"] for x in sr]))
            ),
        )
        max_mean_auc_diff = max(
            max_mean_auc_diff,
            abs(float(np.mean([x["auc"] for x in cr])) - float(np.mean([x["auc"] for x in sr]))),
        )

    compat = {
        "n_rows": len(rows),
        "n_cells": len(cpu_cells),
        "max_abs_diff_mean_loss": max_mean_loss_diff,
        "max_abs_diff_seed_loss": max_seed_loss_diff,
        "max_abs_diff_mean_loss_seen": max_mean_loss_seen_diff,
        "max_abs_diff_mean_auc": max_mean_auc_diff,
        "max_abs_diff_shipped_seen_frac_vs_omega_11": max_shipped_sf_diff,
        "max_abs_diff_cpu_seen_frac_vs_omega_11": max_cpu_sf_diff,
    }
    part = {
        "n_rows": len(rows),
        "max_sum_omega_err": max_sum_om_err,
        "max_raw_partition_err": max_raw_part_err,
        "max_recal_partition_err": max_recal_part_err,
        "max_omega11_seen_frac_err": max_cpu_sf_diff,
        "max_loss11_loss_seen_err": max_loss11_err,
        "min_group_counts": min_counts,
    }
    return compat, part


def _midpoint_attribution_for_mode(cells_sorted, suffix=""):
    """Compute 4-way midpoint attribution, 2-way comparison, and internal cold decomposition."""
    n_ages = len(cells_sorted)
    w = {g: np.zeros(n_ages, dtype=float) for g in GROUPS}
    L = {g: np.zeros(n_ages, dtype=float) for g in GROUPS}
    mass = {g: np.zeros(n_ages, dtype=float) for g in GROUPS}
    L_tot = np.zeros(n_ages, dtype=float)
    L_tot_field = np.zeros(n_ages, dtype=float)
    loss_key = f"loss{suffix}"

    for t, (_, cell) in enumerate(cells_sorted):
        L_tot_field[t] = float(np.mean([float(r[loss_key]) for r in cell]))
        for g in GROUPS:
            wg = float(np.mean([float(r[f"omega_{g}"]) for r in cell]))
            mg = float(np.mean([_weighted_loss_term(r, g, suffix) for r in cell]))
            assert wg > 0.0, f"Empty group {g} in cell {t}"
            w[g][t] = wg
            mass[g][t] = mg
            L[g][t] = mg / wg
        L_tot[t] = sum(mass[g][t] for g in GROUPS)

    assert np.max(np.abs(L_tot - L_tot_field)) < 1e-6
    delta_L = float(L_tot[-1] - L_tot[0])
    mid_L = 0.5 * (L_tot[:-1] + L_tot[1:])

    within_4 = {}
    cov_4 = {}
    for g in GROUPS:
        mid_wg = 0.5 * (w[g][:-1] + w[g][1:])
        mid_Lg = 0.5 * (L[g][:-1] + L[g][1:])
        within_4[g] = float(np.sum(mid_wg * np.diff(L[g])))
        cov_4[g] = float(np.sum((mid_Lg - mid_L) * np.diff(w[g])))

    total_within_4 = float(sum(within_4.values()))
    total_cov_4 = float(sum(cov_4.values()))
    part_err = abs((total_within_4 + total_cov_4) - delta_L)
    assert part_err < 1e-10, f"4-way midpoint identity err {part_err}"

    # 2-way comparison (matching age_diagnostics.py)
    w_warm = w["11"]
    w_cold = 1.0 - w_warm
    L_warm = L["11"]
    L_cold = sum(mass[g] for g in COLD_GROUPS) / w_cold
    mid_w_warm = 0.5 * (w_warm[:-1] + w_warm[1:])
    mid_w_cold = 0.5 * (w_cold[:-1] + w_cold[1:])
    gap_2g = L_warm - L_cold
    within_warm_2g = float(np.sum(mid_w_warm * np.diff(L_warm)))
    within_cold_2g = float(np.sum(mid_w_cold * np.diff(L_cold)))
    cov_2g = float(np.sum(0.5 * (gap_2g[:-1] + gap_2g[1:]) * np.diff(w_warm)))
    err_2g = abs((within_warm_2g + within_cold_2g + cov_2g) - delta_L)
    assert err_2g < 1e-10, f"2-way midpoint identity err {err_2g}"

    # Internal decomposition of Delta L_cold = L_cold[-1] - L_cold[0]
    tilde_w = {g: w[g] / w_cold for g in COLD_GROUPS}
    assert np.max(np.abs(sum(tilde_w.values()) - 1.0)) < 1e-12
    assert np.max(np.abs(sum(tilde_w[g] * L[g] for g in COLD_GROUPS) - L_cold)) < 1e-12

    delta_L_cold = float(L_cold[-1] - L_cold[0])
    within_cold_internal = {}
    w_within_cold_internal = {}
    mix_within_cold = 0.0
    w_mix_within_cold = 0.0
    for g in COLD_GROUPS:
        mid_tw = 0.5 * (tilde_w[g][:-1] + tilde_w[g][1:])
        mid_Lg = 0.5 * (L[g][:-1] + L[g][1:])
        dLg = np.diff(L[g])
        dtw = np.diff(tilde_w[g])
        within_cold_internal[f"within_{g}"] = float(np.sum(mid_tw * dLg))
        w_within_cold_internal[f"within_{g}"] = float(np.sum(mid_w_cold * mid_tw * dLg))
        mix_within_cold += float(np.sum(mid_Lg * dtw))
        w_mix_within_cold += float(np.sum(mid_w_cold * mid_Lg * dtw))

    cold_int_err = abs(
        (sum(within_cold_internal.values()) + mix_within_cold) - delta_L_cold
    )
    assert cold_int_err < 1e-10, f"cold internal identity err {cold_int_err}"
    w_cold_int_err = abs(
        (sum(w_within_cold_internal.values()) + w_mix_within_cold) - within_cold_2g
    )
    assert w_cold_int_err < 1e-10, f"weighted cold internal identity err {w_cold_int_err}"

    return {
        "delta_L": delta_L,
        "L_freshest": float(L_tot[0]),
        "L_oldest": float(L_tot[-1]),
        "fourway": {
            "within": within_4,
            "coverage": cov_4,
            "total_within": total_within_4,
            "total_coverage": total_cov_4,
            "identity_err": float(part_err),
        },
        "twoway": {
            "warm": within_warm_2g,
            "cold": within_cold_2g,
            "coverage": cov_2g,
            "change": delta_L,
            "cold_share": float(within_cold_2g / delta_L) if abs(delta_L) > 1e-15 else None,
            "identity_err": float(err_2g),
        },
        "cold_internal": {
            "delta_L_cold": delta_L_cold,
            "L_cold_freshest": float(L_cold[0]),
            "L_cold_oldest": float(L_cold[-1]),
            **within_cold_internal,
            "mix_within_cold": float(mix_within_cold),
            "identity_err": float(cold_int_err),
            "weighted_contribution_to_delta_L": {
                "cold_2g": within_cold_2g,
                **w_within_cold_internal,
                "mix_within_cold": float(w_mix_within_cold),
                "identity_err": float(w_cold_int_err),
            },
        },
        "group_endpoints": {
            g: {
                "omega_freshest": float(w[g][0]),
                "omega_oldest": float(w[g][-1]),
                "delta_omega": float(w[g][-1] - w[g][0]),
                "L_freshest": float(L[g][0]),
                "L_oldest": float(L[g][-1]),
                "delta_L": float(L[g][-1] - L[g][0]),
                **({
                    "tilde_w_freshest": float(tilde_w[g][0]),
                    "tilde_w_oldest": float(tilde_w[g][-1]),
                    "delta_tilde_w": float(tilde_w[g][-1] - tilde_w[g][0]),
                } if g in COLD_GROUPS else {}),
            }
            for g in GROUPS
        },
    }


def _summarize_group_attributions(per_dom, dom_list):
    def _stats(vals):
        arr = np.asarray(vals, dtype=float)
        if len(arr) == 0:
            return {
                "n": 0,
                "median": None,
                "mean": None,
                "q25": None,
                "q75": None,
                "min": None,
                "max": None,
            }
        return {
            "n": int(len(arr)),
            "median": float(np.median(arr)),
            "mean": float(np.mean(arr)),
            "q25": float(np.percentile(arr, 25)),
            "q75": float(np.percentile(arr, 75)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    out = {"n_domains": len(dom_list), "domains": list(dom_list)}
    for mode in ("raw", "recal"):
        tw_shares = [per_dom[d][mode]["twoway"]["cold_share"] for d in dom_list]
        dL = [per_dom[d][mode]["delta_L"] for d in dom_list]
        fw_within_shares = {
            g: [
                per_dom[d][mode]["fourway"]["within"][g] / per_dom[d][mode]["delta_L"]
                for d in dom_list
                if abs(per_dom[d][mode]["delta_L"]) > 1e-12
            ]
            for g in GROUPS
        }
        fw_cov_shares = {
            g: [
                per_dom[d][mode]["fourway"]["coverage"][g] / per_dom[d][mode]["delta_L"]
                for d in dom_list
                if abs(per_dom[d][mode]["delta_L"]) > 1e-12
            ]
            for g in GROUPS
        }
        tot_within_shares = [
            per_dom[d][mode]["fourway"]["total_within"] / per_dom[d][mode]["delta_L"]
            for d in dom_list
            if abs(per_dom[d][mode]["delta_L"]) > 1e-12
        ]
        tot_cov_shares = [
            per_dom[d][mode]["fourway"]["total_coverage"] / per_dom[d][mode]["delta_L"]
            for d in dom_list
            if abs(per_dom[d][mode]["delta_L"]) > 1e-12
        ]

        ci_terms = ("within_10", "within_01", "within_00", "mix_within_cold")
        ci_raw_vals = {
            k: [per_dom[d][mode]["cold_internal"][k] for d in dom_list]
            for k in ci_terms
        }
        ci_share_of_cold_L = {
            k: [
                per_dom[d][mode]["cold_internal"][k]
                / per_dom[d][mode]["cold_internal"]["delta_L_cold"]
                for d in dom_list
                if abs(per_dom[d][mode]["cold_internal"]["delta_L_cold"]) > 1e-12
            ]
            for k in ci_terms
        }
        ci_w_share_of_cold2g = {
            k: [
                per_dom[d][mode]["cold_internal"]["weighted_contribution_to_delta_L"][k]
                / per_dom[d][mode]["twoway"]["cold"]
                for d in dom_list
                if abs(per_dom[d][mode]["twoway"]["cold"]) > 1e-12
            ]
            for k in ci_terms
        }
        ci_w_share_of_dL = {
            k: [
                per_dom[d][mode]["cold_internal"]["weighted_contribution_to_delta_L"][k]
                / per_dom[d][mode]["delta_L"]
                for d in dom_list
                if abs(per_dom[d][mode]["delta_L"]) > 1e-12
            ]
            for k in ci_terms
        }

        dL00 = [per_dom[d][mode]["group_endpoints"]["00"]["delta_L"] for d in dom_list]
        dL10 = [per_dom[d][mode]["group_endpoints"]["10"]["delta_L"] for d in dom_list]
        dL01 = [per_dom[d][mode]["group_endpoints"]["01"]["delta_L"] for d in dom_list]
        dL11 = [per_dom[d][mode]["group_endpoints"]["11"]["delta_L"] for d in dom_list]

        out[mode] = {
            "delta_L": _stats(dL),
            "twoway_cold_share": {
                **_stats([s for s in tw_shares if s is not None]),
                "n_at_least_half": int(sum(1 for s in tw_shares if s is not None and s >= 0.5)),
            },
            "fourway_share_of_delta_L": {
                "within": {g: _stats(fw_within_shares[g]) for g in GROUPS},
                "coverage": {g: _stats(fw_cov_shares[g]) for g in GROUPS},
                "total_within": _stats(tot_within_shares),
                "total_coverage": _stats(tot_cov_shares),
            },
            "cold_internal_raw_values": {k: _stats(v) for k, v in ci_raw_vals.items()},
            "cold_internal_share_of_delta_L_cold": {
                k: _stats(v) for k, v in ci_share_of_cold_L.items()
            },
            "cold_internal_weighted_share_of_cold_2g": {
                k: _stats(v) for k, v in ci_w_share_of_cold2g.items()
            },
            "cold_internal_weighted_share_of_delta_L": {
                k: _stats(v) for k, v in ci_w_share_of_dL.items()
            },
            "group_delta_L": {
                "11": _stats(dL11),
                "10": _stats(dL10),
                "01": _stats(dL01),
                "00": _stats(dL00),
            },
        }
    return out


def _fit_boxcox_with_boot(rows, metric, dbox):
    sub_rows = [r for r in rows if r.get(metric) is not None]
    base = C.fit_joint(sub_rows, seed=None, dbox=dbox, metric=metric)
    deltas, blogs = [], []
    for b in range(N_BOOT):
        try:
            fb = C.fit_joint(sub_rows, seed=b, dbox=dbox, metric=metric)
            deltas.append(fb["delta"])
            blogs.append(fb["b_log"])
        except Exception:
            pass
    lo_d, hi_d = np.percentile(deltas, [5, 95])
    lo_b, hi_b = np.percentile(blogs, [5, 95])
    no_drift = bool(base["beta"] < 1e-6 or base["r2"] < 0.2)
    return {
        "metric": metric,
        "delta_box": [float(dbox[0]), float(dbox[1])],
        "delta": float(base["delta"]),
        "delta_lo": float(lo_d),
        "delta_hi": float(hi_d),
        "b_log": float(base["b_log"]),
        "b_log_lo": float(lo_b),
        "b_log_hi": float(hi_b),
        "beta": float(base["beta"]),
        "r2": float(base["r2"]),
        "t0": float(base["t0"]),
        "arms": list(base["arms"]),
        "intercepts": [float(x) for x in base["intercepts"]],
        "n_cells": int(base["n_cells"]),
        "at_bound": bool(base["at_bound"]),
        "descending": bool(base["descending"]),
        "no_drift": no_drift,
        "n_boot": int(len(deltas)),
    }


def build_summary():
    age_diag = json.loads((C.REV_RES / "age_diagnostics.json").read_text())
    included_18 = sorted(d for d, v in age_diag["attribution"].items() if v["included"])
    assert len(included_18) == 18, f"Expected 18 included domains, got {len(included_18)}"

    compat_per_dom = {}
    part_per_dom = {}
    attr_per_dom = {}
    boxcox_ac = {}

    all_rows_count = 0
    all_cells_count = 0
    g_max_mean_loss_diff = 0.0
    g_max_seed_loss_diff = 0.0
    g_max_mean_loss_seen_diff = 0.0
    g_max_shipped_sf_diff = 0.0
    g_max_cpu_sf_diff = 0.0
    g_max_sum_om_err = 0.0
    g_max_raw_part_err = 0.0
    g_max_recal_part_err = 0.0
    g_max_loss11_err = 0.0
    g_min_counts = {g: None for g in GROUPS}

    for dom in ALL_DOMAINS:
        fw_path = C.CR_RES / f"fourway_{dom}.jsonl"
        rows = _load_jsonl(fw_path)
        shipped_rows = _load_jsonl(C.REV_RES / f"domain_{dom}.jsonl")
        compat, part = _check_domain_compatibility_and_partition(dom, rows, shipped_rows)
        compat_per_dom[dom] = compat
        part_per_dom[dom] = part

        all_rows_count += compat["n_rows"]
        all_cells_count += compat["n_cells"]
        g_max_mean_loss_diff = max(g_max_mean_loss_diff, compat["max_abs_diff_mean_loss"])
        g_max_seed_loss_diff = max(g_max_seed_loss_diff, compat["max_abs_diff_seed_loss"])
        g_max_mean_loss_seen_diff = max(
            g_max_mean_loss_seen_diff, compat["max_abs_diff_mean_loss_seen"]
        )
        g_max_shipped_sf_diff = max(
            g_max_shipped_sf_diff, compat["max_abs_diff_shipped_seen_frac_vs_omega_11"]
        )
        g_max_cpu_sf_diff = max(
            g_max_cpu_sf_diff, compat["max_abs_diff_cpu_seen_frac_vs_omega_11"]
        )
        g_max_sum_om_err = max(g_max_sum_om_err, part["max_sum_omega_err"])
        g_max_raw_part_err = max(g_max_raw_part_err, part["max_raw_partition_err"])
        g_max_recal_part_err = max(g_max_recal_part_err, part["max_recal_partition_err"])
        g_max_loss11_err = max(g_max_loss11_err, part["max_loss11_loss_seen_err"])
        for g in GROUPS:
            c_g = part["min_group_counts"][g]
            g_min_counts[g] = c_g if g_min_counts[g] is None else min(g_min_counts[g], c_g)

        # Arm A 4-way midpoint attribution & internal cold decomposition
        arm_a = [r for r in rows if r["exp"] == "A"]
        by_age = defaultdict(list)
        for r in arm_a:
            by_age[float(r["age_days"])].append(r)
        cells_sorted = sorted(by_age.items(), key=lambda kv: kv[0])
        raw_attr = _midpoint_attribution_for_mode(cells_sorted, "")
        recal_attr = _midpoint_attribution_for_mode(cells_sorted, "_recal")
        shipped_attr = age_diag["attribution"][dom]
        attr_per_dom[dom] = {
            "included_in_18": bool(shipped_attr["included"]),
            "ages_days": [a for a, _ in cells_sorted],
            "raw": raw_attr,
            "recal": recal_attr,
            "shipped_twoway": shipped_attr,
            "abs_diff_cold_share_vs_shipped": (
                abs(raw_attr["twoway"]["cold_share"] - shipped_attr["cold_share"])
                if raw_attr["twoway"]["cold_share"] is not None
                else None
            ),
        }

        # Joint A+C Box-Cox fits on the 4 representative domains
        if dom in AC_DOMAINS:
            dom_fits = {}
            for dbox in ((-1.0, 4.0), (-6.0, 4.0)):
                box_tag = f"box_{dbox[0]:g}_{dbox[1]:g}"
                dom_fits[box_tag] = {
                    m: _fit_boxcox_with_boot(rows, m, dbox) for m in BOXCOX_METRICS
                }
            boxcox_ac[dom] = dom_fits

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n_domains": len(ALL_DOMAINS),
        "ac_domains": list(AC_DOMAINS),
        "included_18_domains": included_18,
        "compatibility": {
            "n_rows_compared": all_rows_count,
            "n_cells_compared": all_cells_count,
            "max_abs_diff_mean_loss": g_max_mean_loss_diff,
            "max_abs_diff_seed_loss": g_max_seed_loss_diff,
            "max_abs_diff_mean_loss_seen": g_max_mean_loss_seen_diff,
            "max_abs_diff_shipped_seen_frac_vs_omega_11": g_max_shipped_sf_diff,
            "max_abs_diff_cpu_seen_frac_vs_omega_11": g_max_cpu_sf_diff,
            "per_domain": compat_per_dom,
        },
        "partition_checks": {
            "n_rows_checked": all_rows_count,
            "max_sum_omega_err": g_max_sum_om_err,
            "max_raw_partition_err": g_max_raw_part_err,
            "max_recal_partition_err": g_max_recal_part_err,
            "max_omega11_seen_frac_err": g_max_cpu_sf_diff,
            "max_loss11_loss_seen_err": g_max_loss11_err,
            "min_group_counts": g_min_counts,
            "per_domain": part_per_dom,
        },
        "fourway_attribution": {
            "aggregate_included_18": _summarize_group_attributions(attr_per_dom, included_18),
            "aggregate_all_23": _summarize_group_attributions(attr_per_dom, ALL_DOMAINS),
            "per_domain": attr_per_dom,
        },
        "boxcox_ac_fits": boxcox_ac,
    }
    out_path = C.CR_RES / "stage2_fourway_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary, out_path


def _print_summary(summary, out_path):
    comp = summary["compatibility"]
    part = summary["partition_checks"]
    agg18 = summary["fourway_attribution"]["aggregate_included_18"]

    print("\n" + "=" * 78)
    print(f"SAVED SUMMARY: {out_path}")
    print("COMPATIBILITY VS BASELINE:")
    print(
        f"  rows={comp['n_rows_compared']} cells={comp['n_cells_compared']} | "
        f"max |mean_loss_cpu - mean_loss_base| = {comp['max_abs_diff_mean_loss']:.6f} | "
        f"max |seed_loss_cpu - seed_loss_base| = {comp['max_abs_diff_seed_loss']:.6f}"
    )
    print(
        f"  max |base_seen_frac - omega_11| = {comp['max_abs_diff_shipped_seen_frac_vs_omega_11']:.2e} | "
        f"max |cpu_seen_frac - omega_11| = {comp['max_abs_diff_cpu_seen_frac_vs_omega_11']:.2e}"
    )
    print("PARTITION IDENTITY CHECKS:")
    print(
        f"  max |sum(omega)-1|={part['max_sum_omega_err']:.2e} | "
        f"max |sum(omega*loss)-loss|={part['max_raw_partition_err']:.2e} | "
        f"max |sum(omega*loss_recal)-loss_recal|={part['max_recal_partition_err']:.2e} | "
        f"min_counts={part['min_group_counts']}"
    )
    print("18-DOMAIN AGGREGATE (median across 18 included domains):")
    for mode in ("raw", "recal"):
        m = agg18[mode]
        fw = m["fourway_share_of_delta_L"]
        ci_s = m["cold_internal_share_of_delta_L_cold"]
        ci_w = m["cold_internal_weighted_share_of_delta_L"]
        print(
            f"  [{mode:5s}] cold_share(2g)={m['twoway_cold_share']['median']:.3f} "
            f"({m['twoway_cold_share']['n_at_least_half']}/18 >= 0.5) | "
            f"within(11,10,01,00)/dL = "
            f"({fw['within']['11']['median']:+.3f}, {fw['within']['10']['median']:+.3f}, "
            f"{fw['within']['01']['median']:+.3f}, {fw['within']['00']['median']:+.3f}) | "
            f"cov_tot/dL={fw['total_coverage']['median']:+.3f}"
        )
        print(
            f"          dL_cold shares (10, 01, 00, mix) = "
            f"({ci_s['within_10']['median']:+.3f}, {ci_s['within_01']['median']:+.3f}, "
            f"{ci_s['within_00']['median']:+.3f}, {ci_s['mix_within_cold']['median']:+.3f}) | "
            f"w_contrib/dL = ({ci_w['within_10']['median']:+.3f}, {ci_w['within_01']['median']:+.3f}, "
            f"{ci_w['within_00']['median']:+.3f}, {ci_w['mix_within_cold']['median']:+.3f})"
        )
    print("A+C BOX-COX FITS (dbox=(-1, 4)):")
    for dom in AC_DOMAINS:
        fits = summary["boxcox_ac_fits"][dom]["box_-1_4"]
        print(f"  {dom}:")
        for m in ("loss", "loss_recal", "loss_11", "loss_10", "loss_01", "loss_00", "loss_00_recal"):
            f = fits[m]
            flags = (" BOUND" if f["at_bound"] else "") + (" DESC" if f["descending"] else "") + (" NODRIFT" if f["no_drift"] else "")
            print(
                f"    {m:14s}: delta={f['delta']:+.3f} [{f['delta_lo']:+.3f},{f['delta_hi']:+.3f}] "
                f"b_log={f['b_log']:+.5f} R2={f['r2']:+.3f}{flags}"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--domains",
        default="all",
        help="Comma-separated domain names or 'all' (default: all 23 domains)",
    )
    ap.add_argument(
        "--summary-only",
        action="store_true",
        help="Skip training and build stage2_fourway_summary.json from existing JSONLs",
    )
    args = ap.parse_args()

    C.CR_RES.mkdir(parents=True, exist_ok=True)
    t_all = time.time()

    if not args.summary_only:
        doms = ALL_DOMAINS if args.domains == "all" else [d.strip() for d in args.domains.split(",") if d.strip()]
        total_runs = 0
        for dom in doms:
            assert dom in C.DOMAINS, f"Unknown domain: {dom}"
            n_dom, _ = run_domain(dom)
            total_runs += n_dom
        print(f"\nCompleted Stage 2 training sweep across {len(doms)} domains ({total_runs} runs) in {time.time() - t_all:.1f}s.", flush=True)

    # Only build full 23-domain summary when all 23 JSONLs are complete (or --summary-only)
    if args.domains == "all" or args.summary_only:
        summary, out_path = build_summary()
        _print_summary(summary, out_path)


if __name__ == "__main__":
    main()
