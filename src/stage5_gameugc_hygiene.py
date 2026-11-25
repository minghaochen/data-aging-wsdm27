"""Stage 5: GameUGC-Finish Cohort & Stopping-Rule Hygiene.

Executes all 453 new training runs on `gameugc_finish` (42.8M rows, strictly
CPU SparseAdam, max_workers=2, pre-loaded .npz once per worker, resumable JSONL
flushed per row, eval_every=50, calib_t0=test_t0-DAY, calib_t1=test_t0,
calib_cap=50_000, log_fourway=True):

  1. `gameugc_finish__icohort_draws.jsonl` (231 runs, val_frac=None):
       - subseed=1, seeds=0..8 (99 runs -> 9-seed parity with reference draw0)
       - subseed=2,3,4, seeds=0..2 (99 runs)
       - subseed=0, seeds=0..2 (33 runs -> adds calib + 4-way on draw0)
  2. `gameugc_finish__valstop.jsonl` (60 runs, apparent A+C, val_frac=0.1,
       fixed_step=1000, seeds=0..2)
  3. `gameugc_finish__icohort_valstop.jsonl` (66 runs, 11-gap standard A cohort,
       subseed=0,1, seeds=0..2, val_frac=0.1, fixed_step=1000)
  4. `gameugc_finish__icohort_finegrid.jsonl` (96 runs, 16-gap A schedule with
       1-day spacing in [2.5d..10.5d], subseed=0,1, seeds=0..2, val_frac=0.1,
       fixed_step=1000)

Produces `results/stage5_gameugc_summary.json`.

Strictly read-only with respect to existing `results/` baselines.
PyTorch is imported lazily when training so `--help` and `--summary-only`
only require NumPy and SciPy. No matplotlib imports.

Usage: python src/stage5_gameugc_hygiene.py [--summary-only] [--dry-run] [--queues ...]
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
import math
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import cr_lib as C

DOM = "gameugc_finish"
MAX_WORKERS = 2
EVAL_EVERY = 50
CALIB_CAP = 50_000
VAL_FRAC = 0.1
FIXED_STEP = 1000
N_BOOT_COHORT = 200
N_BOOT_BOXCOX = 300
GROUPS = ("11", "10", "01", "00")

FINE_GAPS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 32, 64, 128, 256, 512]
EXPECTED_STD_COHORT_SIZES = {
    0: 20446,
    1: 27091,
    2: 18985,
    3: 20878,
    4: 22967,
}
EXPECTED_COUNTS = {
    "draws": 231,
    "valstop": 60,
    "icohort_valstop": 66,
    "icohort_finegrid": 96,
}

OUT_FILES = {
    "draws": C.CR_RES / "gameugc_finish__icohort_draws.jsonl",
    "valstop": C.CR_RES / "gameugc_finish__valstop.jsonl",
    "icohort_valstop": C.CR_RES / "gameugc_finish__icohort_valstop.jsonl",
    "icohort_finegrid": C.CR_RES / "gameugc_finish__icohort_finegrid.jsonl",
}
SUMMARY_PATH = C.CR_RES / "stage5_gameugc_summary.json"

_WORKER_STATE = {}


def _safe_float(v):
    if v is None:
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def _load_jsonl(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if s:
            rows.append(json.loads(s))
    return rows


def _run_key(r, has_subseed):
    base = (
        str(r["exp"]),
        int(r["train_t0"]),
        int(r["train_t1"]),
        int(r["seed"]),
        int(r["subsample"]),
    )
    if has_subseed:
        return base + (int(r["subseed"]),)
    return base


def _validate_row(res, task):
    """Verify finiteness and exact four-way partition identity before persisting."""
    for k, v in res.items():
        if isinstance(v, float) and not math.isfinite(v):
            raise AssertionError(f"Non-finite float in {task['queue']}: {k}={v}")

    om_sum = sum(float(res[f"omega_{g}"]) for g in GROUPS)
    assert abs(om_sum - 1.0) < 1e-10, f"sum(omega)={om_sum}"

    part_raw = sum(
        (
            float(res[f"omega_{g}"]) * float(res[f"loss_{g}"])
            if res[f"omega_{g}"] > 0 and res[f"loss_{g}"] is not None
            else 0.0
        )
        for g in GROUPS
    )
    assert abs(part_raw - float(res["loss"])) < 1e-6, (
        f"raw partition err={abs(part_raw - float(res['loss']))}"
    )

    part_recal = sum(
        (
            float(res[f"omega_{g}"]) * float(res[f"loss_{g}_recal"])
            if res[f"omega_{g}"] > 0 and res[f"loss_{g}_recal"] is not None
            else 0.0
        )
        for g in GROUPS
    )
    assert abs(part_recal - float(res["loss_recal"])) < 1e-6, (
        f"recal partition err={abs(part_recal - float(res['loss_recal']))}"
    )

    sf_err = abs(float(res["seen_frac"]) - float(res["omega_11"]))
    assert sf_err < 1e-12, f"seen_frac vs omega_11 err={sf_err}"

    if task["expected_n_cohort"] is not None:
        assert int(res["n_cohort"]) == int(task["expected_n_cohort"]), (
            f"n_cohort mismatch: {res['n_cohort']} != {task['expected_n_cohort']}"
        )


def _init_worker(cohorts_np):
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

    data_dict, n_users, n_items = C.load_domain_npz(DOM)
    _WORKER_STATE.clear()
    _WORKER_STATE["data_dict"] = data_dict
    _WORKER_STATE["n_users"] = n_users
    _WORKER_STATE["n_items"] = n_items
    _WORKER_STATE["cohorts"] = {
        k: torch.from_numpy(np.asarray(v, dtype=np.int64))
        for k, v in cohorts_np.items()
    }


def _run_worker_task(task):
    cohort_idx = (
        _WORKER_STATE["cohorts"][task["cohort_key"]]
        if task["cohort_key"] is not None
        else None
    )
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
        eval_every=EVAL_EVERY,
        probe="mf",
        cohort_idx=cohort_idx,
        subsample_seed=task["subsample_seed"],
        device="cpu",
        calib_t0=task["calib_t0"],
        calib_t1=task["calib_t1"],
        calib_cap=CALIB_CAP,
        val_frac=task["val_frac"],
        fixed_step=task["fixed_step"],
        log_fourway=True,
    )
    res["exp"] = task["exp"]
    if "gap" in task:
        res["gap"] = int(task["gap"])
    if "span" in task:
        res["span"] = int(task["span"])
    res["age_days"] = float(task["age_days"])
    res["vol"] = int(task["vol"])
    res["domain"] = DOM
    res["calib_t0"] = int(task["calib_t0"])
    res["calib_t1"] = int(task["calib_t1"])
    if task["subseed"] is not None:
        res["subseed"] = int(task["subseed"])

    _validate_row(res, task)
    gc.collect()
    return res


def build_all_queues():
    """Load gameugc_finish once to build schedules, cohorts, and task queues."""
    data_dict, n_users, n_items = C.load_domain_npz(DOM)
    ts = data_dict["ts"].numpy()
    spec_std = C.DOMAINS[DOM]
    kept_std, d_fix, test_t0, test_t1, _, dropped_std = C.build_window_schedule(
        ts, spec_std
    )
    assert not dropped_std, f"Unexpected dropped std configs: {dropped_std}"
    assert d_fix == 27330, f"Expected d_fix=27330, got {d_fix}"
    kept_std_a = [c for c in kept_std if c["exp"] == "A"]
    assert len(kept_std) == 20 and len(kept_std_a) == 11

    calib_t0 = int(test_t0 - C.DAY)
    calib_t1 = int(test_t0)
    ca, cb = np.searchsorted(ts, [calib_t0, calib_t1])
    assert cb > ca, "Empty calibration slice"

    spec_fine = dict(spec_std)
    spec_fine["gaps"] = list(FINE_GAPS)
    spec_fine["spans"] = []
    kept_fine, d_fix_fine, test_t0_f, test_t1_f, _, dropped_fine = (
        C.build_window_schedule(ts, spec_fine)
    )
    assert not dropped_fine, f"Unexpected dropped fine configs: {dropped_fine}"
    assert d_fix_fine == 27330, f"Expected fine d_fix=27330, got {d_fix_fine}"
    assert test_t0_f == test_t0 and test_t1_f == test_t1
    assert len(kept_fine) == 16

    cohorts_np = {}
    for ss in range(5):
        ci = C.build_cohort_idx(
            data_dict,
            kept_std_a,
            d_fix,
            test_t0,
            test_t1,
            item_only=True,
            sub_seed=ss,
        ).numpy()
        assert len(ci) == EXPECTED_STD_COHORT_SIZES[ss], (
            f"11-gap subseed={ss}: {len(ci)} != {EXPECTED_STD_COHORT_SIZES[ss]}"
        )
        cohorts_np[f"std_ss{ss}"] = ci

    fine_cohort_sizes = {}
    for ss in (0, 1):
        ci_f = C.build_cohort_idx(
            data_dict,
            kept_fine,
            d_fix_fine,
            test_t0,
            test_t1,
            item_only=True,
            sub_seed=ss,
        ).numpy()
        assert len(ci_f) >= 300, f"16-gap subseed={ss}: cohort too small {len(ci_f)}"
        cohorts_np[f"fine_ss{ss}"] = ci_f
        fine_cohort_sizes[ss] = int(len(ci_f))

    def _make_task(
        queue_name,
        cfg,
        seed,
        subseed,
        subsample_seed,
        val_frac,
        fixed_step,
        cohort_key,
        exp_ncohort,
    ):
        t = {
            "queue": queue_name,
            "exp": cfg["exp"],
            "train_t0": int(cfg["train_t0"]),
            "train_t1": int(cfg["train_t1"]),
            "test_t0": int(test_t0),
            "test_t1": int(test_t1),
            "calib_t0": calib_t0,
            "calib_t1": calib_t1,
            "seed": int(seed),
            "subsample": int(d_fix),
            "age_days": float(cfg["age_days"]),
            "vol": int(cfg["vol"]),
            "subseed": subseed,
            "subsample_seed": subsample_seed,
            "val_frac": val_frac,
            "fixed_step": fixed_step,
            "cohort_key": cohort_key,
            "expected_n_cohort": exp_ncohort,
        }
        if "gap" in cfg:
            t["gap"] = int(cfg["gap"])
        if "span" in cfg:
            t["span"] = int(cfg["span"])
        return t

    queues = {
        "draws": [],
        "valstop": [],
        "icohort_valstop": [],
        "icohort_finegrid": [],
    }

    # 1. Multi-draw & 9-seed cohort parity (231 runs)
    draw_seed_plan = [
        (1, list(range(9))),
        (2, list(range(3))),
        (3, list(range(3))),
        (4, list(range(3))),
        (0, list(range(3))),
    ]
    for ss, seeds in draw_seed_plan:
        for cfg in kept_std_a:
            for s in seeds:
                queues["draws"].append(
                    _make_task(
                        "draws",
                        cfg,
                        s,
                        subseed=ss,
                        subsample_seed=ss,
                        val_frac=None,
                        fixed_step=None,
                        cohort_key=f"std_ss{ss}",
                        exp_ncohort=EXPECTED_STD_COHORT_SIZES[ss],
                    )
                )

    # 2. Apparent valstop (A + C, 60 runs)
    for cfg in kept_std:
        for s in C.SEEDS:
            queues["valstop"].append(
                _make_task(
                    "valstop",
                    cfg,
                    s,
                    subseed=None,
                    subsample_seed=None,
                    val_frac=VAL_FRAC,
                    fixed_step=FIXED_STEP,
                    cohort_key=None,
                    exp_ncohort=None,
                )
            )

    # 3. Item-cohort valstop (11-gap standard A cohort, subseed=0,1, 66 runs)
    for ss in (0, 1):
        for cfg in kept_std_a:
            for s in C.SEEDS:
                queues["icohort_valstop"].append(
                    _make_task(
                        "icohort_valstop",
                        cfg,
                        s,
                        subseed=ss,
                        subsample_seed=ss,
                        val_frac=VAL_FRAC,
                        fixed_step=FIXED_STEP,
                        cohort_key=f"std_ss{ss}",
                        exp_ncohort=EXPECTED_STD_COHORT_SIZES[ss],
                    )
                )

    # 4. 1-day fine age grid (16-gap A cohort, subseed=0,1, 96 runs)
    for ss in (0, 1):
        for cfg in kept_fine:
            for s in C.SEEDS:
                queues["icohort_finegrid"].append(
                    _make_task(
                        "icohort_finegrid",
                        cfg,
                        s,
                        subseed=ss,
                        subsample_seed=ss,
                        val_frac=VAL_FRAC,
                        fixed_step=FIXED_STEP,
                        cohort_key=f"fine_ss{ss}",
                        exp_ncohort=fine_cohort_sizes[ss],
                    )
                )

    for qname, exp_n in EXPECTED_COUNTS.items():
        assert len(queues[qname]) == exp_n, (
            f"{qname}: {len(queues[qname])} != {exp_n}"
        )

    meta = {
        "n_events": int(len(ts)),
        "n_users": int(n_users),
        "n_items": int(n_items),
        "d_fix": int(d_fix),
        "test_t0": int(test_t0),
        "test_t1": int(test_t1),
        "calib_t0": int(calib_t0),
        "calib_t1": int(calib_t1),
        "n_calib_avail": int(cb - ca),
        "std_cohort_sizes": dict(EXPECTED_STD_COHORT_SIZES),
        "fine_cohort_sizes": fine_cohort_sizes,
    }
    del data_dict, ts
    gc.collect()
    return queues, cohorts_np, meta


def execute_queue(queue_name, tasks, cohorts_np):
    """Run one queue with max_workers=2, preloading once per worker, resumable."""
    out_path = OUT_FILES[queue_name]
    has_subseed = queue_name != "valstop"
    existing = _load_jsonl(out_path)
    done_keys = {_run_key(r, has_subseed) for r in existing}
    todo = [t for t in tasks if _run_key(t, has_subseed) not in done_keys]

    print(
        f"\n[Stage5:{queue_name}] target={out_path.name} "
        f"total={len(tasks)} done={len(done_keys)} todo={len(todo)}",
        flush=True,
    )
    if not todo:
        assert len(existing) == len(tasks), (
            f"{out_path.name}: len={len(existing)} != {len(tasks)}"
        )
        return 0.0

    sub_cohorts = {
        k: v
        for k, v in cohorts_np.items()
        if k in {t["cohort_key"] for t in todo if t["cohort_key"] is not None}
    }
    n_workers = min(MAX_WORKERS, len(todo))
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with out_path.open("a") as fh:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(sub_cohorts,),
        ) as pool:
            futures = {pool.submit(_run_worker_task, t): t for t in todo}
            completed = 0
            for fut in as_completed(futures):
                res = fut.result()
                fh.write(json.dumps(res, allow_nan=False) + "\n")
                fh.flush()
                completed += 1
                ss_str = f" ss={res['subseed']}" if "subseed" in res else ""
                tag = (
                    f"gap={res['gap']}"
                    if "gap" in res
                    else f"span={res.get('span', '?')}"
                )
                extra = ""
                if "loss_cohort" in res:
                    extra += f" lc={res['loss_cohort']:.5f}"
                if "loss_cohort_recal" in res:
                    extra += f" lcr={res['loss_cohort_recal']:.5f}"
                if "loss_valstop" in res:
                    extra += f" lvs={res['loss_valstop']:.5f}"
                if "loss_cohort_valstop" in res:
                    extra += f" lcvs={res['loss_cohort_valstop']:.5f}"
                print(
                    f"  [{queue_name}] ({completed}/{len(todo)}) "
                    f"{res['exp']} {tag:8s}{ss_str} s={res['seed']} "
                    f"loss={res['loss']:.5f}{extra} ({res['train_time_s']}s)",
                    flush=True,
                )

    elapsed = time.time() - t0
    gc.collect()

    final_rows = _load_jsonl(out_path)
    assert len(final_rows) == len(tasks), (
        f"{out_path.name}: expected {len(tasks)} rows, got {len(final_rows)}"
    )
    assert len({_run_key(r, has_subseed) for r in final_rows}) == len(tasks), (
        f"{out_path.name}: duplicate keys detected"
    )
    print(
        f"  [{queue_name}] finished {len(todo)} runs in {elapsed:.1f}s "
        f"(total verified={len(final_rows)})",
        flush=True,
    )
    return elapsed


# ---------------------------------------------------------------------------
# Summary & Analysis Builders
# ---------------------------------------------------------------------------
def _summarize_cohort_curve(rows, metric="loss_cohort", n_boot=N_BOOT_COHORT):
    valid = [r for r in rows if r.get(metric) is not None and math.isfinite(float(r[metric]))]
    assert valid, f"No valid rows for metric={metric}"

    cells_all = defaultdict(list)
    for r in valid:
        cells_all[round(float(r["age_days"]), 1)].append(float(r[metric]))

    ages_all = np.array(sorted(cells_all), dtype=float)
    y_all = np.array([float(np.mean(cells_all[a])) for a in ages_all], dtype=float)
    sd_all = np.array([float(np.std(cells_all[a])) for a in ages_all], dtype=float)

    keep = y_all <= 1.0
    n_dropped_qc = int((~keep).sum())
    ages = ages_all[keep]
    y = y_all[keep]
    cells_qc = {float(a): cells_all[float(a)] for a in ages}

    def _stats_for_sub(sub_ages, sub_y):
        tau = _safe_float(C.tau_half_obs(sub_ages, sub_y))
        lo, hi = C.boot(cells_qc, sub_ages, n=n_boot, seed=0)
        rate = _safe_float(1000.0 * np.polyfit(np.log(sub_ages), sub_y, 1)[0])
        trunc = {}
        for cap in sub_ages[3:]:
            m = sub_ages <= cap
            v = C.tau_half_obs(sub_ages[m], sub_y[m])
            trunc[f"{cap:g}"] = _safe_float(v)
        fin = [v for v in trunc.values() if v is not None]
        spread = float(max(fin) / min(fin)) if len(fin) >= 2 else None
        pass_1_5x = (
            bool(spread < 1.5) if (len(fin) > 2 and spread is not None) else None
        )
        return {
            "tau_half": tau,
            "ci90": [_safe_float(lo), _safe_float(hi)],
            "rate_mnat_per_efold": rate,
            "amp_mnat": float(1000.0 * (sub_y.max() - sub_y.min())),
            "endpoint_delta_mnat": float(1000.0 * (sub_y[-1] - sub_y[0])),
            "truncations": trunc,
            "n_finite_truncations": int(len(fin)),
            "n_total_truncations": int(len(trunc)),
            "spread": spread,
            "pass_1_5x": pass_1_5x,
        }

    base_stats = _stats_for_sub(ages, y)
    omit_stats = _stats_for_sub(ages[1:], y[1:]) if len(ages) >= 5 else None

    curve_pts = [
        {
            "age_days": float(a),
            "mean_loss": float(m),
            "sd_loss": float(s),
            "delta_mnat": float(1000.0 * (m - y_all[0])),
            "n_seeds": int(len(cells_all[float(a)])),
        }
        for a, m, s in zip(ages_all, y_all, sd_all)
    ]

    return {
        "metric": metric,
        "n_rows": int(len(valid)),
        "n_ages": int(len(ages_all)),
        "seeds_per_age_min": int(min(len(cells_all[float(a)]) for a in ages_all)),
        "seeds_per_age_max": int(max(len(cells_all[float(a)]) for a in ages_all)),
        "n_cohort": int(valid[0]["n_cohort"]) if valid[0].get("n_cohort") is not None else None,
        "n_dropped_qc": n_dropped_qc,
        **base_stats,
        "omit_freshest": omit_stats,
        "curve": curve_pts,
    }


def _fit_boxcox_with_boot(rows, metric, dbox, n_boot=N_BOOT_BOXCOX):
    sub_rows = [
        r for r in rows if r.get(metric) is not None and math.isfinite(float(r[metric]))
    ]
    base = C.fit_joint(sub_rows, seed=None, dbox=dbox, metric=metric)
    deltas, blogs = [], []
    for b in range(n_boot):
        try:
            fb = C.fit_joint(sub_rows, seed=b, dbox=dbox, metric=metric)
            deltas.append(fb["delta"])
            blogs.append(fb["b_log"])
        except Exception:
            pass
    lo_d, hi_d = np.percentile(deltas, [5, 95]) if deltas else (np.nan, np.nan)
    lo_b, hi_b = np.percentile(blogs, [5, 95]) if blogs else (np.nan, np.nan)
    no_drift = bool(base["beta"] < 1e-6 or base["r2"] < 0.2)
    return {
        "metric": metric,
        "delta_box": [float(dbox[0]), float(dbox[1])],
        "delta": _safe_float(base["delta"]),
        "delta_lo": _safe_float(lo_d),
        "delta_hi": _safe_float(hi_d),
        "b_log": _safe_float(base["b_log"]),
        "b_log_lo": _safe_float(lo_b),
        "b_log_hi": _safe_float(hi_b),
        "beta": _safe_float(base["beta"]),
        "r2": _safe_float(base["r2"]),
        "t0": _safe_float(base["t0"]),
        "arms": list(base["arms"]),
        "intercepts": [_safe_float(x) for x in base["intercepts"]],
        "n_cells": int(base["n_cells"]),
        "at_bound": bool(base["at_bound"]),
        "descending": bool(base["descending"]),
        "no_drift": no_drift,
        "n_boot": int(len(deltas)),
    }


def _build_part1(draws_rows, shipped_d0, shipped_d1):
    draws_3seed = {}
    for ss in range(5):
        sub_3 = [
            r for r in draws_rows if int(r["subseed"]) == ss and int(r["seed"]) in (0, 1, 2)
        ]
        draws_3seed[f"subseed_{ss}"] = {
            "subseed": ss,
            "n_cohort": int(sub_3[0]["n_cohort"]),
            "loss_cohort": _summarize_cohort_curve(sub_3, "loss_cohort"),
            "loss_cohort_recal": _summarize_cohort_curve(sub_3, "loss_cohort_recal"),
        }

    shipped_d0_3 = [r for r in shipped_d0 if int(r["seed"]) in (0, 1, 2)]
    shipped_3seed_ref = {
        "draw0_shipped_3seed": _summarize_cohort_curve(shipped_d0_3, "loss_cohort"),
        "draw1_shipped_3seed": _summarize_cohort_curve(shipped_d1, "loss_cohort"),
    }

    def _agg_5draws(metric_key):
        taus = [draws_3seed[f"subseed_{ss}"][metric_key]["tau_half"] for ss in range(5)]
        spreads = [draws_3seed[f"subseed_{ss}"][metric_key]["spread"] for ss in range(5)]
        rates = [
            draws_3seed[f"subseed_{ss}"][metric_key]["rate_mnat_per_efold"]
            for ss in range(5)
        ]
        passes = [draws_3seed[f"subseed_{ss}"][metric_key]["pass_1_5x"] for ss in range(5)]
        taus_f = [x for x in taus if x is not None]
        spreads_f = [x for x in spreads if x is not None]
        return {
            "tau_half_by_draw": taus,
            "tau_half_min": float(min(taus_f)) if taus_f else None,
            "tau_half_median": float(np.median(taus_f)) if taus_f else None,
            "tau_half_max": float(max(taus_f)) if taus_f else None,
            "tau_half_max_over_min": (
                float(max(taus_f) / min(taus_f)) if len(taus_f) >= 2 else None
            ),
            "spread_by_draw": spreads,
            "spread_min": float(min(spreads_f)) if spreads_f else None,
            "spread_max": float(max(spreads_f)) if spreads_f else None,
            "rate_by_draw": rates,
            "rate_min": float(min(rates)),
            "rate_median": float(np.median(rates)),
            "rate_max": float(max(rates)),
            "pass_1_5x_by_draw": passes,
            "n_pass_1_5x": int(sum(1 for p in passes if p is True)),
        }

    draw1_9 = [r for r in draws_rows if int(r["subseed"]) == 1]
    assert len(draw1_9) == 99, f"Expected 99 rows for draw1 9-seed, got {len(draw1_9)}"
    d0_9_sum = _summarize_cohort_curve(shipped_d0, "loss_cohort")
    d1_9_raw_sum = _summarize_cohort_curve(draw1_9, "loss_cohort")
    d1_9_rec_sum = _summarize_cohort_curve(draw1_9, "loss_cohort_recal")

    def _max_cell_diff_mnat(rows_a, rows_b):
        ca = defaultdict(list)
        cb = defaultdict(list)
        for r in rows_a:
            ca[round(float(r["age_days"]), 1)].append(float(r["loss_cohort"]))
        for r in rows_b:
            cb[round(float(r["age_days"]), 1)].append(float(r["loss_cohort"]))
        assert set(ca) == set(cb)
        return float(max(1000.0 * abs(np.mean(ca[k]) - np.mean(cb[k])) for k in ca))

    d0_new_3 = [
        r for r in draws_rows if int(r["subseed"]) == 0 and int(r["seed"]) in (0, 1, 2)
    ]
    d1_new_3 = [
        r for r in draws_rows if int(r["subseed"]) == 1 and int(r["seed"]) in (0, 1, 2)
    ]

    t0_9, t1_9 = d0_9_sum["tau_half"], d1_9_raw_sum["tau_half"]
    parity_9seed = {
        "draw0_shipped_9seed": d0_9_sum,
        "draw1_new_9seed_raw": d1_9_raw_sum,
        "draw1_new_9seed_recal": d1_9_rec_sum,
        "draw0_vs_draw1_9seed_comparison": {
            "n_cohort_draw0": int(d0_9_sum["n_cohort"]),
            "n_cohort_draw1": int(d1_9_raw_sum["n_cohort"]),
            "tau_half_ratio_max_over_min": (
                float(max(t0_9, t1_9) / min(t0_9, t1_9))
                if (t0_9 is not None and t1_9 is not None)
                else None
            ),
            "draw0_pass_1_5x": d0_9_sum["pass_1_5x"],
            "draw1_pass_1_5x": d1_9_raw_sum["pass_1_5x"],
            "draw0_spread": d0_9_sum["spread"],
            "draw1_spread": d1_9_raw_sum["spread"],
            "draw0_rate_mnat_per_efold": d0_9_sum["rate_mnat_per_efold"],
            "draw1_rate_mnat_per_efold": d1_9_raw_sum["rate_mnat_per_efold"],
        },
        "cpu_vs_shipped_3seed_compat": {
            "draw0_3seed_max_abs_cell_diff_mnat": _max_cell_diff_mnat(
                shipped_d0_3, d0_new_3
            ),
            "draw1_3seed_max_abs_cell_diff_mnat": _max_cell_diff_mnat(
                shipped_d1, d1_new_3
            ),
        },
    }

    return {
        "draws_3seed": draws_3seed,
        "shipped_3seed_ref": shipped_3seed_ref,
        "five_draw_summary": {
            "loss_cohort": _agg_5draws("loss_cohort"),
            "loss_cohort_recal": _agg_5draws("loss_cohort_recal"),
        },
        "parity_9seed": parity_9seed,
    }


def _build_part2(draws_rows, icohort_vs_rows, shipped_d0, shipped_d1):
    out = {}
    for ss, label in ((0, "draw0"), (1, "draw1")):
        sh_rows = shipped_d0 if ss == 0 else shipped_d1
        dr_rows_all = [r for r in draws_rows if int(r["subseed"]) == ss]
        dr_rows_3 = [r for r in dr_rows_all if int(r["seed"]) in (0, 1, 2)]
        vs_rows = [r for r in icohort_vs_rows if int(r["subseed"]) == ss]

        rule_summaries = {
            "shipped_valfrac_none_raw": _summarize_cohort_curve(sh_rows, "loss_cohort"),
            "draws_valfrac_none_raw_3seed": _summarize_cohort_curve(
                dr_rows_3, "loss_cohort"
            ),
            "draws_valfrac_none_recal_3seed": _summarize_cohort_curve(
                dr_rows_3, "loss_cohort_recal"
            ),
            "valfrac01_test_argmin": _summarize_cohort_curve(vs_rows, "loss_cohort"),
            "valfrac01_valstop": _summarize_cohort_curve(
                vs_rows, "loss_cohort_valstop"
            ),
            "valfrac01_fixed1000": _summarize_cohort_curve(
                vs_rows, "loss_cohort_fixed"
            ),
            "valfrac01_recal": _summarize_cohort_curve(
                vs_rows, "loss_cohort_recal"
            ),
        }
        if ss == 1:
            rule_summaries["draws_valfrac_none_raw_9seed"] = _summarize_cohort_curve(
                dr_rows_all, "loss_cohort"
            )
            rule_summaries["draws_valfrac_none_recal_9seed"] = _summarize_cohort_curve(
                dr_rows_all, "loss_cohort_recal"
            )

        def _by_age(rows):
            d = defaultdict(list)
            for r in rows:
                d[round(float(r["age_days"]), 1)].append(r)
            return d

        sh_by_a = _by_age(sh_rows)
        dr_by_a = _by_age(dr_rows_3)
        dr9_by_a = _by_age(dr_rows_all) if ss == 1 else None
        vs_by_a = _by_age(vs_rows)
        ages = sorted(sh_by_a)

        def _base0(by_a, key):
            return float(np.mean([r[key] for r in by_a[ages[0]]]))

        b_sh = _base0(sh_by_a, "loss_cohort")
        b_dr = _base0(dr_by_a, "loss_cohort")
        b_drr = _base0(dr_by_a, "loss_cohort_recal")
        b_dr9 = _base0(dr9_by_a, "loss_cohort") if dr9_by_a is not None else None
        b_dr9r = _base0(dr9_by_a, "loss_cohort_recal") if dr9_by_a is not None else None
        b_vst = _base0(vs_by_a, "loss_cohort")
        b_vsv = _base0(vs_by_a, "loss_cohort_valstop")
        b_vsf = _base0(vs_by_a, "loss_cohort_fixed")
        b_vsr = _base0(vs_by_a, "loss_cohort_recal")

        age_table = []
        for a in ages:
            rs_sh = sh_by_a[a]
            rs_dr = dr_by_a[a]
            rs_vs = vs_by_a[a]

            def _m(rs, k):
                return float(np.mean([r[k] for r in rs]))

            row = {
                "age_days": float(a),
                "gap": int(rs_dr[0]["gap"]),
                "shipped_loss_cohort": _m(rs_sh, "loss_cohort"),
                "shipped_delta_mnat": float(1000.0 * (_m(rs_sh, "loss_cohort") - b_sh)),
                "shipped_best_step_mean": _m(rs_sh, "best_step"),
                "draws_loss_cohort": _m(rs_dr, "loss_cohort"),
                "draws_delta_mnat": float(1000.0 * (_m(rs_dr, "loss_cohort") - b_dr)),
                "draws_loss_cohort_recal": _m(rs_dr, "loss_cohort_recal"),
                "draws_recal_delta_mnat": float(
                    1000.0 * (_m(rs_dr, "loss_cohort_recal") - b_drr)
                ),
                "draws_apparent_loss": _m(rs_dr, "loss"),
                "draws_best_step_mean": _m(rs_dr, "best_step"),
                "draws_best_steps": [int(r["best_step"]) for r in rs_dr],
                "draws_train_pos_rate": _m(rs_dr, "train_pos_rate"),
                "draws_test_pos_rate": _m(rs_dr, "test_pos_rate"),
                "draws_calib_pos_rate": _m(rs_dr, "calib_pos_rate"),
                "draws_delta_mu": _m(rs_dr, "delta_mu"),
                "valstop_traj_loss_cohort": _m(rs_vs, "loss_cohort"),
                "valstop_traj_delta_mnat": float(
                    1000.0 * (_m(rs_vs, "loss_cohort") - b_vst)
                ),
                "valstop_loss_cohort": _m(rs_vs, "loss_cohort_valstop"),
                "valstop_delta_mnat": float(
                    1000.0 * (_m(rs_vs, "loss_cohort_valstop") - b_vsv)
                ),
                "fixed_loss_cohort": _m(rs_vs, "loss_cohort_fixed"),
                "fixed_delta_mnat": float(
                    1000.0 * (_m(rs_vs, "loss_cohort_fixed") - b_vsf)
                ),
                "valstop_traj_loss_cohort_recal": _m(rs_vs, "loss_cohort_recal"),
                "valstop_traj_recal_delta_mnat": float(
                    1000.0 * (_m(rs_vs, "loss_cohort_recal") - b_vsr)
                ),
                "valstop_apparent_loss": _m(rs_vs, "loss"),
                "valstop_apparent_loss_valstop": _m(rs_vs, "loss_valstop"),
                "valstop_best_step_test_argmin_mean": _m(rs_vs, "best_step"),
                "valstop_best_step_valstop_mean": _m(rs_vs, "best_step_valstop"),
                "valstop_best_steps_test_argmin": [int(r["best_step"]) for r in rs_vs],
                "valstop_best_steps_valstop": [
                    int(r["best_step_valstop"]) for r in rs_vs
                ],
                "valstop_train_pos_rate": _m(rs_vs, "train_pos_rate"),
                "valstop_delta_mu": _m(rs_vs, "delta_mu"),
            }
            if dr9_by_a is not None:
                rs_dr9 = dr9_by_a[a]
                row["draws_9seed_loss_cohort"] = _m(rs_dr9, "loss_cohort")
                row["draws_9seed_delta_mnat"] = float(
                    1000.0 * (_m(rs_dr9, "loss_cohort") - b_dr9)
                )
                row["draws_9seed_loss_cohort_recal"] = _m(rs_dr9, "loss_cohort_recal")
                row["draws_9seed_recal_delta_mnat"] = float(
                    1000.0 * (_m(rs_dr9, "loss_cohort_recal") - b_dr9r)
                )
                row["draws_9seed_best_step_mean"] = _m(rs_dr9, "best_step")
            age_table.append(row)

        by_age_map = {round(r["age_days"], 1): r for r in age_table}
        out[label] = {
            "subseed": ss,
            "n_cohort": int(sh_rows[0]["n_cohort"]),
            "rule_summaries": rule_summaries,
            "age_table": age_table,
            "spotlight_6_5d_and_34_5d": {
                "age_6_5d": by_age_map[6.5],
                "age_34_5d": by_age_map[34.5],
            },
        }
    return out


def _build_part3(valstop_rows, shipped_apparent):
    fits = {}
    for dbox in ((-1.0, 4.0), (-6.0, 4.0)):
        btag = f"box_{dbox[0]:g}_{dbox[1]:g}"
        fits[btag] = {
            "shipped_valfrac_none_loss": _fit_boxcox_with_boot(
                shipped_apparent, "loss", dbox
            ),
            "valfrac01_loss_test_argmin": _fit_boxcox_with_boot(
                valstop_rows, "loss", dbox
            ),
            "valfrac01_loss_valstop": _fit_boxcox_with_boot(
                valstop_rows, "loss_valstop", dbox
            ),
            "valfrac01_loss_fixed1000": _fit_boxcox_with_boot(
                valstop_rows, "loss_fixed", dbox
            ),
            "valfrac01_loss_recal": _fit_boxcox_with_boot(
                valstop_rows, "loss_recal", dbox
            ),
            "valfrac01_loss_seen_test_argmin": _fit_boxcox_with_boot(
                valstop_rows, "loss_seen", dbox
            ),
            "valfrac01_loss_seen_valstop": _fit_boxcox_with_boot(
                valstop_rows, "loss_seen_valstop", dbox
            ),
        }

    cells = defaultdict(list)
    for r in valstop_rows:
        cells[(r["exp"], round(float(r["age_days"]), 2))].append(r)

    per_cell = []
    for (exp, age), rs in sorted(cells.items()):
        l_ta = float(np.mean([r["loss"] for r in rs]))
        l_vs = float(np.mean([r["loss_valstop"] for r in rs]))
        l_fx = float(np.mean([r["loss_fixed"] for r in rs]))
        l_rc = float(np.mean([r["loss_recal"] for r in rs]))
        bs_ta = float(np.mean([r["best_step"] for r in rs]))
        bs_vs = float(np.mean([r["best_step_valstop"] for r in rs]))
        per_cell.append(
            {
                "exp": exp,
                "age_days": float(age),
                "gap_or_span": int(rs[0].get("gap", rs[0].get("span"))),
                "mean_loss_test_argmin": l_ta,
                "mean_loss_valstop": l_vs,
                "mean_loss_fixed": l_fx,
                "mean_loss_recal": l_rc,
                "loss_minus_valstop_mnat": float(1000.0 * (l_ta - l_vs)),
                "optimism_valstop_minus_loss_mnat": float(1000.0 * (l_vs - l_ta)),
                "mean_best_step_test_argmin": bs_ta,
                "mean_best_step_valstop": bs_vs,
                "mean_step_diff_test_minus_val": float(bs_ta - bs_vs),
                "mean_train_pos_rate": float(
                    np.mean([r["train_pos_rate"] for r in rs])
                ),
                "mean_delta_mu": float(np.mean([r["delta_mu"] for r in rs])),
            }
        )

    def _opt_trend(sub_cells):
        ages = np.array([c["age_days"] for c in sub_cells], dtype=float)
        opt = np.array(
            [c["optimism_valstop_minus_loss_mnat"] for c in sub_cells], dtype=float
        )
        diff = np.array([c["loss_minus_valstop_mnat"] for c in sub_cells], dtype=float)
        st_diff = np.array(
            [c["mean_step_diff_test_minus_val"] for c in sub_cells], dtype=float
        )
        sl_opt = float(np.polyfit(np.log(ages), opt, 1)[0])
        r_opt = (
            float(np.corrcoef(np.log(ages), opt)[0, 1]) if opt.std() > 0 else 0.0
        )
        return {
            "n_cells": int(len(sub_cells)),
            "mean_optimism_mnat": float(np.mean(opt)),
            "min_optimism_mnat": float(np.min(opt)),
            "max_optimism_mnat": float(np.max(opt)),
            "mean_loss_minus_valstop_mnat": float(np.mean(diff)),
            "optimism_slope_mnat_per_efold": sl_opt,
            "optimism_pearson_r_vs_ln_age": r_opt,
            "mean_best_step_diff": float(np.mean(st_diff)),
        }

    return {
        "n_val": int(valstop_rows[0]["n_val"]),
        "n_train_valstop": int(valstop_rows[0]["n_train"]),
        "n_train_shipped": int(shipped_apparent[0]["n_train"]),
        "boxcox_joint_ac": fits,
        "optimism_and_step_shift": {
            "joint_AC": _opt_trend(per_cell),
            "arm_A": _opt_trend([c for c in per_cell if c["exp"] == "A"]),
            "arm_C": _opt_trend([c for c in per_cell if c["exp"] == "C"]),
            "per_cell": per_cell,
        },
    }


def _build_part4(fine_rows):
    rules = (
        ("test_argmin", "loss_cohort"),
        ("valstop", "loss_cohort_valstop"),
        ("fixed", "loss_cohort_fixed"),
        ("recal", "loss_cohort_recal"),
    )
    out = {"draws": {}}
    for ss, label in ((0, "draw0"), (1, "draw1")):
        sub = [r for r in fine_rows if int(r["subseed"]) == ss]
        assert len(sub) == 48, f"Expected 48 rows for finegrid {label}, got {len(sub)}"
        by_age = defaultdict(list)
        for r in sub:
            by_age[round(float(r["age_days"]), 1)].append(r)

        rule_summaries = {}
        fine_region_diag = {}
        for rname, metric in rules:
            s = _summarize_cohort_curve(sub, metric)
            rule_summaries[rname] = s
            pts_1d = [p for p in s["curve"] if p["age_days"] <= 10.5 + 1e-6]
            jumps = []
            for p0, p1 in zip(pts_1d[:-1], pts_1d[1:]):
                jumps.append(
                    {
                        "from_age": p0["age_days"],
                        "to_age": p1["age_days"],
                        "jump_mnat": float(p1["delta_mnat"] - p0["delta_mnat"]),
                    }
                )
            max_jump = max(jumps, key=lambda x: abs(x["jump_mnat"]))
            val_6_5 = next(p["delta_mnat"] for p in pts_1d if abs(p["age_days"] - 6.5) < 1e-6)
            val_34_5 = next(
                p["delta_mnat"] for p in s["curve"] if abs(p["age_days"] - 34.5) < 1e-6
            )
            fine_region_diag[rname] = {
                "daily_deltas_2_5_to_10_5": [
                    {"age_days": p["age_days"], "delta_mnat": p["delta_mnat"]}
                    for p in pts_1d
                ],
                "daily_jumps_2_5_to_10_5": jumps,
                "max_abs_daily_jump": max_jump,
                "delta_mnat_at_6_5d": val_6_5,
                "delta_mnat_at_34_5d": val_34_5,
            }

        aux_by_age = []
        for a in sorted(by_age):
            rs = by_age[a]
            aux_by_age.append(
                {
                    "age_days": float(a),
                    "gap": int(rs[0]["gap"]),
                    "train_pos_rate": float(np.mean([r["train_pos_rate"] for r in rs])),
                    "delta_mu": float(np.mean([r["delta_mu"] for r in rs])),
                    "best_step_test_argmin_mean": float(
                        np.mean([r["best_step"] for r in rs])
                    ),
                    "best_step_valstop_mean": float(
                        np.mean([r["best_step_valstop"] for r in rs])
                    ),
                }
            )

        out["draws"][label] = {
            "subseed": ss,
            "n_cohort": int(sub[0]["n_cohort"]),
            "rules": rule_summaries,
            "fine_region_2_5_to_10_5": fine_region_diag,
            "aux_by_age": aux_by_age,
        }

    comp = {}
    for rname, _ in rules:
        s0 = out["draws"]["draw0"]["rules"][rname]
        s1 = out["draws"]["draw1"]["rules"][rname]
        t0, t1 = s0["tau_half"], s1["tau_half"]
        comp[rname] = {
            "draw0_tau_half": t0,
            "draw1_tau_half": t1,
            "tau_half_ratio_max_over_min": (
                float(max(t0, t1) / min(t0, t1))
                if (t0 is not None and t1 is not None)
                else None
            ),
            "draw0_spread": s0["spread"],
            "draw1_spread": s1["spread"],
            "draw0_pass_1_5x": s0["pass_1_5x"],
            "draw1_pass_1_5x": s1["pass_1_5x"],
            "draw0_6_5d_mnat": out["draws"]["draw0"]["fine_region_2_5_to_10_5"][rname][
                "delta_mnat_at_6_5d"
            ],
            "draw1_6_5d_mnat": out["draws"]["draw1"]["fine_region_2_5_to_10_5"][rname][
                "delta_mnat_at_6_5d"
            ],
        }
    out["draw0_vs_draw1_comparison"] = comp
    return out


def build_summary(meta=None):
    draws_rows = _load_jsonl(OUT_FILES["draws"])
    valstop_rows = _load_jsonl(OUT_FILES["valstop"])
    icohort_vs_rows = _load_jsonl(OUT_FILES["icohort_valstop"])
    fine_rows = _load_jsonl(OUT_FILES["icohort_finegrid"])

    assert len(draws_rows) == EXPECTED_COUNTS["draws"], len(draws_rows)
    assert len(valstop_rows) == EXPECTED_COUNTS["valstop"], len(valstop_rows)
    assert len(icohort_vs_rows) == EXPECTED_COUNTS["icohort_valstop"], len(
        icohort_vs_rows
    )
    assert len(fine_rows) == EXPECTED_COUNTS["icohort_finegrid"], len(fine_rows)

    shipped_d0 = _load_jsonl(C.REV_RES / "domain_gameugc_finish__icohort.jsonl")
    shipped_d1 = _load_jsonl(C.REV_RES / "domain_gameugc_finish__icohort__ss1.jsonl")
    shipped_app = _load_jsonl(C.REV_RES / "domain_gameugc_finish.jsonl")

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "domain": DOM,
        "meta": meta,
        "run_counts": {
            "draws": len(draws_rows),
            "valstop": len(valstop_rows),
            "icohort_valstop": len(icohort_vs_rows),
            "icohort_finegrid": len(fine_rows),
            "total": len(draws_rows)
            + len(valstop_rows)
            + len(icohort_vs_rows)
            + len(fine_rows),
        },
        "part1_cohort_draws_and_parity": _build_part1(
            draws_rows, shipped_d0, shipped_d1
        ),
        "part2_draw0_spike_dip_diagnostic": _build_part2(
            draws_rows, icohort_vs_rows, shipped_d0, shipped_d1
        ),
        "part3_apparent_valstop_vs_test_argmin": _build_part3(
            valstop_rows, shipped_app
        ),
        "part4_finegrid_1d_age": _build_part4(fine_rows),
    }

    payload = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    SUMMARY_PATH.write_text(payload)
    print(f"\n[Stage5] Wrote summary -> {SUMMARY_PATH}", flush=True)
    _print_highlights(summary)
    return summary


def _fmt(v, spec=".2f"):
    return "None" if v is None else format(v, spec)


def _print_highlights(summary):
    p1 = summary["part1_cohort_draws_and_parity"]
    print("\n=== PART 1: 5-Draw Cohort Summary (3 seeds, raw loss_cohort) ===")
    for ss in range(5):
        e = p1["draws_3seed"][f"subseed_{ss}"]
        lc = e["loss_cohort"]
        lcr = e["loss_cohort_recal"]
        print(
            f"  ss={ss} n={e['n_cohort']:5d} | "
            f"raw tau={_fmt(lc['tau_half'])} [{_fmt(lc['ci90'][0])},{_fmt(lc['ci90'][1])}] "
            f"rate={_fmt(lc['rate_mnat_per_efold'])} "
            f"spread={_fmt(lc['spread'], '.3f')} pass={lc['pass_1_5x']} | "
            f"recal tau={_fmt(lcr['tau_half'])} spread={_fmt(lcr['spread'], '.3f')} "
            f"pass={lcr['pass_1_5x']}"
        )
    par = p1["parity_9seed"]
    d0 = par["draw0_shipped_9seed"]
    d1 = par["draw1_new_9seed_raw"]
    print(
        f"  9-seed parity: draw0(baseline, n={d0['n_cohort']}) tau={_fmt(d0['tau_half'])} "
        f"spread={_fmt(d0['spread'], '.3f')} pass={d0['pass_1_5x']} vs "
        f"draw1(new, n={d1['n_cohort']}) tau={_fmt(d1['tau_half'])} "
        f"spread={_fmt(d1['spread'], '.3f')} pass={d1['pass_1_5x']}"
    )

    p3 = summary["part3_apparent_valstop_vs_test_argmin"]
    print("\n=== PART 3: Apparent A+C Box-Cox Fits ===")
    for btag in ("box_-1_4", "box_-6_4"):
        print(f"  [{btag}]")
        for k, v in p3["boxcox_joint_ac"][btag].items():
            print(
                f"    {k:32s}: delta={_fmt(v['delta'], '+.3f')} "
                f"[{_fmt(v['delta_lo'], '+.3f')},{_fmt(v['delta_hi'], '+.3f')}] "
                f"b_log={_fmt(v['b_log'], '+.5f')} r2={_fmt(v['r2'], '.3f')} "
                f"bound={v['at_bound']}"
            )

    p4 = summary["part4_finegrid_1d_age"]
    print("\n=== PART 4: 1-Day Fine Grid (16 gaps) ===")
    for rname, c in p4["draw0_vs_draw1_comparison"].items():
        print(
            f"  {rname:12s}: draw0 tau={_fmt(c['draw0_tau_half'])} "
            f"spread={_fmt(c['draw0_spread'], '.3f')} "
            f"6.5d={_fmt(c['draw0_6_5d_mnat'], '+.2f')}m | "
            f"draw1 tau={_fmt(c['draw1_tau_half'])} "
            f"spread={_fmt(c['draw1_spread'], '.3f')} "
            f"6.5d={_fmt(c['draw1_6_5d_mnat'], '+.2f')}m"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate schedules, cohort sizes, and run counts without training.",
    )
    ap.add_argument(
        "--summary-only",
        action="store_true",
        help="Recompute stage5_gameugc_summary.json from completed JSONLs.",
    )
    ap.add_argument(
        "--queues",
        nargs="+",
        choices=["draws", "valstop", "icohort_valstop", "icohort_finegrid"],
        default=["draws", "valstop", "icohort_valstop", "icohort_finegrid"],
        help="Subset of queues to execute.",
    )
    args = ap.parse_args()

    C.CR_RES.mkdir(parents=True, exist_ok=True)
    npz_path = C.DATA_DIR / C.DOMAINS[DOM]["npz"]

    if args.summary_only:
        meta = None
        if npz_path.exists():
            _, _, meta = build_all_queues()
            print("[Stage5] Meta:", json.dumps(meta, indent=2), flush=True)
        elif SUMMARY_PATH.exists():
            meta = json.loads(SUMMARY_PATH.read_text()).get("meta")
        if all(
            OUT_FILES[q].exists() and len(_load_jsonl(OUT_FILES[q])) == EXPECTED_COUNTS[q]
            for q in EXPECTED_COUNTS
        ):
            build_summary(meta=meta)
        else:
            print("[Stage5] Skipping summary: not all 4 JSONLs are complete yet.", flush=True)
        return

    if not npz_path.exists():
        raise SystemExit(
            f"Missing {npz_path}; build preprocessed .npz first via src/prep_gameugc.py."
        )
    queues, cohorts_np, meta = build_all_queues()
    print("[Stage5] Meta:", json.dumps(meta, indent=2), flush=True)

    if args.dry_run:
        for qname in args.queues:
            has_ss = qname != "valstop"
            existing = _load_jsonl(OUT_FILES[qname])
            done = {_run_key(r, has_ss) for r in existing}
            todo = [t for t in queues[qname] if _run_key(t, has_ss) not in done]
            print(
                f"  dry-run {qname:18s}: total={len(queues[qname])} "
                f"done={len(done)} todo={len(todo)}"
            )
        return

    t_all = time.time()
    for qname in args.queues:
        execute_queue(qname, queues[qname], cohorts_np)
    print(
        f"\n[Stage5] All requested queues finished in {time.time() - t_all:.1f}s",
        flush=True,
    )

    if all(
        OUT_FILES[q].exists() and len(_load_jsonl(OUT_FILES[q])) == EXPECTED_COUNTS[q]
        for q in EXPECTED_COUNTS
    ):
        build_summary(meta=meta)
    else:
        print("[Stage5] Skipping summary: not all 4 JSONLs are complete yet.", flush=True)


if __name__ == "__main__":
    main()
