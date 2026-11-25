"""Stage 3: Non-overlapping rolling-retrain backtests & cadence-pricing analysis.

Runs three strictly non-overlapping rolling-retrain backtests (exact static
D_fix volume matching, val_frac=None, eval_every=50, SEEDS=[0, 1], CPU
SparseAdam, log_fourway=True):
  1. kuairand_strict       -> backtest_kuairand_strict.jsonl
  2. recflow_nooverlap19d  -> backtest_recflow_nooverlap19d.jsonl
  3. gameugc_strict        -> backtest_gameugc_strict.jsonl

Then evaluates all four backtest series (including existing
`recflow_22d_existing` from results/) on BOTH `loss` and `1 - auc`,
writing `stage3_backtest_summary.json`.

Strictly read-only with respect to existing `results/` inputs.
PyTorch is imported lazily during training so `--help` and `--summary-only`
only require NumPy and SciPy. No matplotlib imports.

Usage: python src/stage3_backtests.py [--summary-only] [--check-schedules-only] [--only ...]
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

_SRC = str(Path(__file__).resolve().parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import cr_lib as C

BT_SEEDS = [0, 1]
EVAL_EVERY = 50
W_DAYS = 2
MAX_WORKERS = 2
GROUPS = ("11", "10", "01", "00")

# Comprehensive log-spaced price grid (quarter-decades from 1e-5 to 1.0);
# includes np.geomspace(1e-5, 1e-2, 7) and paper Table 6 prices (1e-3, 10^-2.5, 1e-2).
PRICE_GRID = [float(10.0 ** x) for x in np.linspace(-5.0, 0.0, 21)]
STD_PRICES = [1e-5, 10.0 ** (-4.5), 1e-4, 10.0 ** (-3.5), 1e-3, 10.0 ** (-2.5), 1e-2]

BACKTEST_CONFIGS = {
    "kuairand_strict": {
        "domain": "kuairand",
        "out_name": "backtest_kuairand_strict.jsonl",
        "W": 2,
    },
    "recflow_nooverlap19d": {
        "domain": "recflow",
        "out_name": "backtest_recflow_nooverlap19d.jsonl",
        "W": 2,
    },
    "gameugc_strict": {
        "domain": "gameugc_finish",
        "out_name": "backtest_gameugc_strict.jsonl",
        "W": 2,
    },
}

_WORKER_STATE = {}
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(64)


def _load_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _run_key(r):
    return (int(r["P"]), int(r["t_retrain"]), int(r["seed"]))


def _rel_posix(p):
    pp = Path(p)
    return pp.relative_to(C.CR_ROOT).as_posix() if pp.is_absolute() else pp.as_posix()


def _resolve_root(p):
    pp = Path(p)
    return pp if pp.is_absolute() else C.CR_ROOT / pp


def _load_static_domain_info(dom):
    src_path = C.REV_RES / f"domain_{dom}.jsonl"
    rows = _load_jsonl(src_path)
    assert rows, f"Empty static domain file: {src_path}"
    test_t0s = sorted({int(r["test_t0"]) for r in rows})
    test_t1s = sorted({int(r["test_t1"]) for r in rows})
    subsamples = sorted({int(r["subsample"]) for r in rows})
    assert len(test_t0s) == 1, f"{dom}: expected single test_t0, got {test_t0s}"
    assert len(test_t1s) == 1, f"{dom}: expected single test_t1, got {test_t1s}"
    assert len(subsamples) == 1, f"{dom}: expected single d_fix, got {subsamples}"
    arm_a = [r for r in rows if r["exp"] == "A"]
    assert arm_a, f"{dom}: no Arm A rows in {src_path}"
    return {
        "src_path": _rel_posix(src_path),
        "static_test_t0": test_t0s[0],
        "static_test_t1": test_t1s[0],
        "d_fix": subsamples[0],
        "arm_a_rows": arm_a,
    }


def build_backtest_schedule(name):
    """Build slice schedule and run strict zero-overlap pre-checks."""
    cfg = BACKTEST_CONFIGS[name]
    dom = cfg["domain"]
    W = int(cfg["W"])
    info = _load_static_domain_info(dom)
    static_test_t0 = info["static_test_t0"]
    d_fix = info["d_fix"]

    npz_path = C.DATA_DIR / C.DOMAINS[dom]["npz"]
    if not npz_path.exists():
        raise SystemExit(
            f"Missing {npz_path}; build preprocessed .npz first via src/prep_*.py."
        )
    with np.load(npz_path) as z:
        ts = z["ts"]
    ts0 = int(ts[0])
    ts_last = int(ts[-1])
    n_events = int(len(ts))

    if name == "kuairand_strict":
        serve_end_ts = static_test_t0
        max_H = int((serve_end_ts - ts0) // C.DAY) - W
        if max_H >= 22:
            H = 22
            serve_start_ts = serve_end_ts - 22 * C.DAY
            periods = [1, 2, 4, 8, 11, 22]
        else:
            H = max_H
            serve_start_ts = serve_end_ts - H * C.DAY
            periods = sorted({p for p in [1, 2, 4, 8, 11] if p < H} | {H})
        schedule_note = f"max_H={max_H}, selected H={H}"
    elif name == "recflow_nooverlap19d":
        serve_start_ts = ts0 + 7 * C.DAY
        serve_end_ts = static_test_t0
        H_days = (serve_end_ts - serve_start_ts) / C.DAY
        H = int(math.floor(H_days + 1e-9))
        if H == 19:
            periods = [1, 2, 4, 8, 11, 19]
        else:
            periods = sorted({p for p in [1, 2, 4, 8, 11] if p < H} | {H})
        schedule_note = f"H_days={H_days:.6f}, floored H={H}"
    elif name == "gameugc_strict":
        serve_end_ts = static_test_t0
        H = 32
        serve_start_ts = serve_end_ts - 32 * C.DAY
        periods = [1, 2, 4, 8, 16, 32]
        schedule_note = "H=32"
    else:
        raise ValueError(f"Unknown backtest name: {name}")

    assert serve_end_ts <= static_test_t0, (
        f"{name}: serve_end_ts={serve_end_ts} > static_test_t0={static_test_t0}"
    )
    assert serve_start_ts - W * C.DAY >= ts0, (
        f"{name}: earliest train_t0={serve_start_ts - W * C.DAY} < ts[0]={ts0}"
    )

    h_span_s = serve_end_ts - serve_start_ts
    total_horizon_days = h_span_s / C.DAY
    expected_serve_total = int(
        np.searchsorted(ts, serve_end_ts) - np.searchsorted(ts, serve_start_ts)
    )

    tasks = []
    windows_below_dfix = []
    serve_totals_by_P = {}
    min_train_vol = None
    min_serve_vol = None

    for P in periods:
        n_models = int(math.ceil(total_horizon_days / P - 1e-12))
        p_serve_sum = 0
        for k in range(n_models):
            t_re = int(serve_start_ts + k * P * C.DAY)
            t_serve_start = t_re
            t_serve_end = int(min(t_re + P * C.DAY, serve_end_ts))
            train_t0 = int(t_re - W * C.DAY)
            train_t1 = int(t_re)

            # Strict non-overlap and non-empty assertions
            assert t_serve_end <= static_test_t0, (
                f"{name} P={P} k={k}: t_serve_end={t_serve_end} > static_test_t0={static_test_t0}"
            )
            assert t_serve_start < t_serve_end, (
                f"{name} P={P} k={k}: empty serve range {t_serve_start}..{t_serve_end}"
            )
            assert train_t0 >= ts0, (
                f"{name} P={P} k={k}: train_t0={train_t0} < ts[0]={ts0}"
            )

            a, b = np.searchsorted(ts, [train_t0, train_t1])
            c, e = np.searchsorted(ts, [t_serve_start, t_serve_end])
            train_vol = int(b - a)
            serve_vol = int(e - c)
            assert train_vol > 0 and serve_vol > 0, (
                f"{name} P={P} k={k}: hollow slice train={train_vol} serve={serve_vol}"
            )
            assert train_vol >= 25_000 and serve_vol >= 1000, (
                f"{name} P={P} k={k}: slice below minimum threshold train={train_vol} serve={serve_vol}"
            )

            min_train_vol = train_vol if min_train_vol is None else min(min_train_vol, train_vol)
            min_serve_vol = serve_vol if min_serve_vol is None else min(min_serve_vol, serve_vol)
            p_serve_sum += serve_vol
            below_dfix = bool(train_vol < d_fix)
            if below_dfix:
                windows_below_dfix.append(
                    {
                        "P": int(P),
                        "k": int(k),
                        "t_retrain": t_re,
                        "train_t0": train_t0,
                        "train_t1": train_t1,
                        "train_vol_avail": train_vol,
                        "d_fix": int(d_fix),
                        "deficit": int(d_fix - train_vol),
                        "serve_vol": serve_vol,
                    }
                )

            for seed in BT_SEEDS:
                tasks.append(
                    {
                        "backtest_name": name,
                        "domain": dom,
                        "P": int(P),
                        "k": int(k),
                        "W": int(W),
                        "train_t0": train_t0,
                        "train_t1": train_t1,
                        "t_retrain": t_re,
                        "t_serve_start": t_serve_start,
                        "t_serve_end": t_serve_end,
                        "static_test_t0": int(static_test_t0),
                        "d_fix": int(d_fix),
                        "train_vol_avail": train_vol,
                        "below_dfix": below_dfix,
                        "expected_serve_vol": serve_vol,
                        "seed": int(seed),
                    }
                )
        serve_totals_by_P[int(P)] = p_serve_sum
        assert p_serve_sum == expected_serve_total, (
            f"{name} P={P}: sum(serve_vol)={p_serve_sum} != expected {expected_serve_total}"
        )

    del ts
    gc.collect()

    meta = {
        "backtest_name": name,
        "domain": dom,
        "npz_path": _rel_posix(npz_path),
        "out_path": _rel_posix(C.CR_RES / cfg["out_name"]),
        "n_events": n_events,
        "ts_first": ts0,
        "ts_last": ts_last,
        "static_domain_src": _rel_posix(info["src_path"]),
        "static_test_t0": int(static_test_t0),
        "static_test_t1": int(info["static_test_t1"]),
        "d_fix": int(d_fix),
        "W": int(W),
        "serve_start_ts": int(serve_start_ts),
        "serve_end_ts": int(serve_end_ts),
        "horizon_days": float(total_horizon_days),
        "periods": [int(p) for p in periods],
        "expected_serve_events": expected_serve_total,
        "serve_totals_by_P": serve_totals_by_P,
        "min_train_vol_avail": int(min_train_vol),
        "min_serve_vol_avail": int(min_serve_vol),
        "windows_below_dfix": windows_below_dfix,
        "n_models_total": len(tasks) // len(BT_SEEDS),
        "n_runs_total": len(tasks),
        "schedule_note": schedule_note,
    }
    return tasks, meta


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
    assert _WORKER_STATE.get("dom") == dom, (
        f"Worker loaded {_WORKER_STATE.get('dom')} != {dom}"
    )
    assert task["t_serve_end"] <= task["static_test_t0"], (
        f"Worker zero-overlap violation: {task['t_serve_end']} > {task['static_test_t0']}"
    )
    res = C.run_preloaded(
        _WORKER_STATE["data_dict"],
        _WORKER_STATE["n_users"],
        _WORKER_STATE["n_items"],
        task["train_t0"],
        task["train_t1"],
        task["t_serve_start"],
        task["t_serve_end"],
        steps=8000,
        batch=8192,
        dim=32,
        lr=3e-3,
        seed=task["seed"],
        subsample=task["d_fix"],
        eval_every=EVAL_EVERY,
        probe="mf",
        device="cpu",
        calib_t0=None,
        calib_t1=None,
        val_frac=None,
        fixed_step=None,
        log_fourway=True,
    )
    assert int(res["n_test"]) == int(task["expected_serve_vol"]), (
        f"n_test mismatch: {res['n_test']} != {task['expected_serve_vol']}"
    )
    res.update(
        {
            "domain": dom,
            "backtest_name": task["backtest_name"],
            "P": int(task["P"]),
            "k": int(task["k"]),
            "W": int(task["W"]),
            "t_retrain": int(task["t_retrain"]),
            "t_serve_start": int(task["t_serve_start"]),
            "t_serve_end": int(task["t_serve_end"]),
            "static_test_t0": int(task["static_test_t0"]),
            "train_vol_avail": int(task["train_vol_avail"]),
            "below_dfix": bool(task["below_dfix"]),
        }
    )
    gc.collect()
    return res


def _verify_backtest_rows(name, tasks, meta):
    out_path = _resolve_root(meta["out_path"])
    rows = _load_jsonl(out_path)
    expected_keys = {_run_key(t) for t in tasks}
    by_key = {}
    for r in rows:
        k = _run_key(r)
        if k in expected_keys:
            by_key[k] = r
    assert len(by_key) == len(expected_keys), (
        f"{name}: missing runs! got {len(by_key)}/{len(expected_keys)}"
    )

    static_t0 = meta["static_test_t0"]
    max_serve_end = None
    ntest_by_ps = defaultdict(int)
    for k, r in by_key.items():
        assert int(r["t_serve_end"]) <= static_t0, (
            f"{name} {k}: t_serve_end={r['t_serve_end']} > static_test_t0={static_t0}"
        )
        assert int(r["test_t1"]) <= static_t0, (
            f"{name} {k}: test_t1={r['test_t1']} > static_test_t0={static_t0}"
        )
        assert int(r["subsample"]) == meta["d_fix"], (
            f"{name} {k}: subsample={r['subsample']} != {meta['d_fix']}"
        )
        for fld in ("loss", "auc", "loss_seen", "seen_frac"):
            v = float(r[fld])
            assert math.isfinite(v), f"{name} {k}: non-finite {fld}={v}"
        om_sum = sum(float(r[f"omega_{g}"]) for g in GROUPS)
        assert abs(om_sum - 1.0) < 1e-9, f"{name} {k}: sum(omega)={om_sum}"
        max_serve_end = (
            int(r["t_serve_end"])
            if max_serve_end is None
            else max(max_serve_end, int(r["t_serve_end"]))
        )
        ntest_by_ps[(int(r["P"]), int(r["seed"]))] += int(r["n_test"])

    for P in meta["periods"]:
        for seed in BT_SEEDS:
            tot = ntest_by_ps[(P, seed)]
            assert tot == meta["expected_serve_events"], (
                f"{name} P={P} seed={seed}: sum(n_test)={tot} != {meta['expected_serve_events']}"
            )
    assert max_serve_end is not None and max_serve_end <= static_t0
    return len(rows)


def run_backtest(name):
    tasks, meta = build_backtest_schedule(name)
    out_path = _resolve_root(meta["out_path"])
    existing = _load_jsonl(out_path)
    done_keys = {_run_key(r) for r in existing}
    todo = [t for t in tasks if _run_key(t) not in done_keys]

    print(
        f"\n[{name}] dom={meta['domain']} d_fix={meta['d_fix']:,} "
        f"H={meta['horizon_days']:.1f}d periods={meta['periods']} "
        f"total={len(tasks)} done={len(done_keys)} todo={len(todo)} "
        f"(serve_end={meta['serve_end_ts']} <= static_test_t0={meta['static_test_t0']})",
        flush=True,
    )
    if meta["windows_below_dfix"]:
        print(
            f"  [{name}] NOTE: {len(meta['windows_below_dfix'])} window(s) have "
            f"train_vol_avail < d_fix: {meta['windows_below_dfix']}",
            flush=True,
        )

    elapsed = 0.0
    if todo:
        n_workers = min(MAX_WORKERS, len(todo))
        t0 = time.time()
        ctx = mp.get_context("spawn")
        with out_path.open("a") as fh:
            with ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=ctx,
                initializer=_init_worker,
                initargs=(meta["domain"],),
            ) as pool:
                futures = {pool.submit(_run_worker_task, t): t for t in todo}
                completed = 0
                for fut in as_completed(futures):
                    res = fut.result()
                    fh.write(json.dumps(res) + "\n")
                    fh.flush()
                    completed += 1
                    print(
                        f"  [{name}] ({completed}/{len(todo)}) P={res['P']:2d} "
                        f"k={res['k']:2d} s={res['seed']} loss={res['loss']:.5f} "
                        f"auc={res['auc']:.4f} n_test={res['n_test']:,} "
                        f"n_train={res['n_train']:,} ({res['train_time_s']}s)",
                        flush=True,
                    )
        elapsed = time.time() - t0
        gc.collect()
        print(
            f"  [{name}] completed {len(todo)} runs in {elapsed:.1f}s (workers={n_workers})",
            flush=True,
        )

    n_verified = _verify_backtest_rows(name, tasks, meta)
    print(f"  [{name}] verified {n_verified} rows, zero overlap & equal traffic OK.", flush=True)
    return meta, elapsed


# ---------------------------------------------------------------------------
# Summary analysis helpers
# ---------------------------------------------------------------------------
def _load_static_a_curve(src_path, metric):
    rows = _load_jsonl(_resolve_root(src_path))
    cells = defaultdict(list)
    for r in rows:
        if r["exp"] == "A":
            val = (1.0 - float(r["auc"])) if metric == "one_minus_auc" else float(r[metric])
            cells[round(float(r["age_days"]), 2)].append(val)
    ages = np.array(sorted(cells), dtype=float)
    ys = np.array([float(np.mean(cells[a])) for a in ages], dtype=float)
    return ages, ys, [r for r in rows if r["exp"] == "A"]


def _load_backtest_agg(bt_path, metric):
    rows = _load_jsonl(Path(bt_path))
    assert rows, f"No backtest rows in {bt_path}"
    agg = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    by_seed = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    fw_agg = defaultdict(lambda: {g: [0.0, 0.0] for g in GROUPS})
    has_fourway = all(f"omega_{g}" in rows[0] for g in GROUPS)

    for r in rows:
        P = int(r["P"])
        n = float(r["n_test"])
        val = (1.0 - float(r["auc"])) if metric == "one_minus_auc" else float(r["loss"])
        a = agg[P]
        a[0] += val * n
        a[1] += n
        a[2] += float(r["auc"]) * n
        a[3] += 1

        s = by_seed[P][int(r["seed"])]
        s[0] += val * n
        s[1] += n

        if has_fourway:
            for g in GROUPS:
                w_g = float(r[f"omega_{g}"]) * n
                fw_agg[P][g][0] += w_g
                l_g = r.get(f"loss_{g}")
                if l_g is not None:
                    fw_agg[P][g][1] += w_g * float(l_g)

    Ps = np.array(sorted(agg), dtype=int)
    vals = np.array([agg[int(p)][0] / agg[int(p)][1] for p in Ps], dtype=float)
    aucs = np.array([agg[int(p)][2] / agg[int(p)][1] for p in Ps], dtype=float)
    nev = np.array([int(agg[int(p)][3]) for p in Ps], dtype=int)

    seed_spreads = {}
    for p in Ps:
        pi = int(p)
        s0 = by_seed[pi][0][0] / by_seed[pi][0][1]
        s1 = by_seed[pi][1][0] / by_seed[pi][1][1]
        seed_spreads[str(pi)] = abs(float(s0 - s1))

    fourway_by_p = None
    if has_fourway:
        fourway_by_p = {}
        for p in Ps:
            pi = int(p)
            tot_n = agg[pi][1]
            entry = {}
            for g in GROUPS:
                wg_sum, lg_mass = fw_agg[pi][g]
                entry[f"omega_{g}"] = float(wg_sum / tot_n)
                entry[f"loss_{g}"] = float(lg_mass / wg_sum) if wg_sum > 0 else None
            fourway_by_p[str(pi)] = entry

    return rows, Ps, vals, aucs, nev, seed_spreads, fourway_by_p


def _boxcox_eval(tau, c, beta, delta, t0):
    if abs(delta) < 1e-9:
        return c + beta * np.log(tau / t0)
    return c + beta * ((tau / t0) ** delta - 1.0) / delta


def _boxcox_serving_avg(P, a, c, beta, delta, t0):
    P_arr = np.atleast_1d(P).astype(float)
    half = 0.5 * P_arr[:, None]
    mid = a + half
    tau = mid + half * _GL_NODES[None, :]
    L = _boxcox_eval(tau, c, beta, delta, t0)
    return 0.5 * np.sum(L * _GL_WEIGHTS[None, :], axis=1)


def _fit_power_serving_avg(P, a, L0, b, delta):
    P_arr = np.atleast_1d(P).astype(float)
    if a == 0.0:
        return L0 + (b / (delta + 1.0)) * (P_arr ** delta)
    half = 0.5 * P_arr[:, None]
    mid = a + half
    tau = mid + half * _GL_NODES[None, :]
    L = L0 + b * (tau ** delta)
    return 0.5 * np.sum(L * _GL_WEIGHTS[None, :], axis=1)


def _minimize_1d_loggrid(obj_fn, p_lo=1e-3, p_hi=1e4, n_grid=4000):
    """Find continuous 1D minimum on [p_lo, p_hi] via log-grid scan + golden section."""
    grid = np.geomspace(p_lo, p_hi, n_grid)
    vals = obj_fn(grid)
    k = int(np.argmin(vals))
    if k == 0 or k == n_grid - 1:
        return float(grid[k]), float(vals[k]), True
    lo = float(grid[k - 1])
    hi = float(grid[k + 1])
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c = hi - gr * (hi - lo)
    d = lo + gr * (hi - lo)
    fc = float(obj_fn(np.array([c]))[0])
    fd = float(obj_fn(np.array([d]))[0])
    for _ in range(60):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - gr * (hi - lo)
            fc = float(obj_fn(np.array([c]))[0])
        else:
            lo, c, fc = c, d, fd
            d = lo + gr * (hi - lo)
            fd = float(obj_fn(np.array([d]))[0])
        if (hi - lo) / max(lo, 1e-12) < 1e-9:
            break
    mid = 0.5 * (lo + hi)
    fm = float(obj_fn(np.array([mid]))[0])
    return float(mid), fm, False


def _compute_cadence_optima(Ps, measured, ages_static, y_static, arm_a_rows, metric, W):
    # 1. fit_power (matching backtest_analysis.py)
    L0_s, b_s, d_s = C.fit_power(ages_static, y_static)

    # 2. Box-Cox fit_joint on Arm A (handles -1 < delta < 0 and delta <= -1!)
    arm_a_prep = []
    for r in arm_a_rows:
        rc = dict(r)
        if metric == "one_minus_auc":
            rc["one_minus_auc"] = 1.0 - float(r["auc"])
        arm_a_prep.append(rc)
    fj_m1 = C.fit_joint(arm_a_prep, dbox=(-1.0, 4.0), metric=metric)
    fj = C.fit_joint(arm_a_prep, dbox=(-6.0, 4.0), metric=metric) if fj_m1["at_bound"] else fj_m1
    c_bc = float(fj["intercepts"][0])
    beta_bc = float(fj["beta"])
    delta_bc = float(fj["delta"])
    t0_bc = float(fj["t0"])

    # In-sample power diagnostic (matching backtest_analysis.py)
    pen = measured - measured[0]
    m_pos = pen > 0
    in_sample = None
    if int(m_pos.sum()) >= 2:
        d_i, lb = np.polyfit(np.log(Ps[m_pos]), np.log(pen[m_pos]), 1)
        b_serve_i = float(math.exp(lb))
        b_i = float(b_serve_i * (d_i + 1.0))
        in_sample = {
            "delta": float(d_i),
            "b_serve": b_serve_i,
            "b": b_i,
        }

    Pg = np.geomspace(float(Ps.min()), float(Ps.max()), 400)
    Lg = np.interp(np.log(Pg), np.log(Ps), measured)
    a_off = 0.5 * W  # 1.0 day for W=2

    price_rows = []
    for lamC in PRICE_GRID:
        # Tested & Interp.
        j_test = measured + lamC / Ps
        k_test = int(np.argmin(j_test))
        p_test = int(Ps[k_test])
        test_interior = bool(0 < k_test < len(Ps) - 1)

        j_emp = Lg + lamC / Pg
        k_emp = int(np.argmin(j_emp))
        p_emp = float(Pg[k_emp])
        emp_interior = bool(0 < k_emp < len(Pg) - 1)

        # Static fit_power: a=0 closed form & a=W/2 numerical integral
        p_fp_a0 = float((lamC * (d_s + 1.0) / (b_s * d_s)) ** (1.0 / (1.0 + d_s)))
        p_fp_a1, _, fp_a1_at_bound = _minimize_1d_loggrid(
            lambda P_vec: _fit_power_serving_avg(P_vec, a_off, L0_s, b_s, d_s)
            + lamC / P_vec
        )

        # Static Box-Cox: a=0 closed form (when delta > -1) & a=W/2 numerical integral
        if delta_bc <= -1.0 or beta_bc <= 0.0:
            p_bc_a0 = None
            bc_a0_note = (
                "undefined: delta <= -1 (a=0 serving integral diverges at tau->0+)"
                if delta_bc <= -1.0
                else "undefined: beta <= 0"
            )
        elif abs(delta_bc) < 1e-9:
            p_bc_a0 = float(lamC / beta_bc)
            bc_a0_note = "log-limit closed form"
        else:
            p_bc_a0 = float(
                (lamC * (delta_bc + 1.0) / (beta_bc * (t0_bc ** (-delta_bc))))
                ** (1.0 / (1.0 + delta_bc))
            )
            bc_a0_note = "Box-Cox closed form"

        if beta_bc <= 0.0:
            p_bc_a1, bc_a1_at_bound = None, True
        else:
            p_bc_a1, _, bc_a1_at_bound = _minimize_1d_loggrid(
                lambda P_vec: _boxcox_serving_avg(
                    P_vec, a_off, c_bc, beta_bc, delta_bc, t0_bc
                )
                + lamC / P_vec
            )

        p_ins = None
        if in_sample is not None and in_sample["b"] * in_sample["delta"] > 0 and in_sample["delta"] > -1.0:
            p_ins = float(
                (lamC * (in_sample["delta"] + 1.0) / (in_sample["b"] * in_sample["delta"]))
                ** (1.0 / (1.0 + in_sample["delta"]))
            )

        price_rows.append(
            {
                "lamC": float(lamC),
                "is_standard_table_price": bool(
                    any(abs(lamC - sp) / sp < 1e-9 for sp in (1e-3, 10.0 ** (-2.5), 1e-2))
                ),
                "static_fit_power_a0": p_fp_a0,
                "static_fit_power_a_W_half": p_fp_a1,
                "static_fit_power_a_W_half_at_bound": bool(fp_a1_at_bound),
                "static_boxcox_a0": p_bc_a0,
                "static_boxcox_a0_note": bc_a0_note,
                "static_boxcox_a_W_half": p_bc_a1,
                "static_boxcox_a_W_half_at_bound": bool(bc_a1_at_bound),
                "in_sample_p_star": p_ins,
                "tested": p_test,
                "tested_interior": test_interior,
                "interp": p_emp,
                "interp_interior": emp_interior,
                "ratio_fit_power_a0_to_interp": float(p_fp_a0 / p_emp),
                "ratio_fit_power_a1_to_interp": float(p_fp_a1 / p_emp),
                "ratio_boxcox_a0_to_interp": (
                    float(p_bc_a0 / p_emp) if p_bc_a0 is not None else None
                ),
                "ratio_boxcox_a1_to_interp": (
                    float(p_bc_a1 / p_emp) if p_bc_a1 is not None else None
                ),
            }
        )

    interior_rows = [r for r in price_rows if r["interp_interior"]]
    standard_rows = [r for r in price_rows if any(abs(r["lamC"] - sp) / sp < 1e-9 for sp in STD_PRICES)]
    return {
        "static_fit_power": {
            "L0": float(L0_s),
            "b": float(b_s),
            "delta": float(d_s),
        },
        "static_boxcox": {
            "c": c_bc,
            "beta": beta_bc,
            "delta": delta_bc,
            "t0": t0_bc,
            "r2": float(fj["r2"]),
            "b_log": float(fj["b_log"]),
            "dbox_1_4_delta": float(fj_m1["delta"]),
            "dbox_1_4_at_bound": bool(fj_m1["at_bound"]),
            "final_at_bound": bool(fj["at_bound"]),
        },
        "in_sample_backtest_fit": in_sample,
        "a_offset_days": float(a_off),
        "all_prices": price_rows,
        "interior_interp_prices": interior_rows,
        "standard_prices": standard_rows,
    }


def _analyze_metric(Ps, measured, aucs, nev, seed_spreads, ages_static, y_static, arm_a_rows, metric, W):
    def L_A(age):
        return np.interp(np.log(age), np.log(ages_static), y_static)

    eval_ages_per_P = {str(int(P)): (np.arange(int(P)) + 0.5 + 0.5 * W).tolist() for P in Ps}
    pred = np.array([float(np.mean(L_A(np.arange(int(P)) + 0.5 + 0.5 * W))) for P in Ps], dtype=float)

    dm = measured - measured[0]
    dp = pred - pred[0]
    null_log = np.log(Ps) - np.log(Ps[0])
    null_lin = Ps - Ps[0]

    slope_poly = np.polyfit(dp[1:], dm[1:], 1)
    slope_ratio = float(slope_poly[0])
    slope_intercept = float(slope_poly[1])

    corr_age_law = float(np.corrcoef(dm[1:], dp[1:])[0, 1])
    corr_log_p = float(np.corrcoef(dm[1:], null_log[1:])[0, 1])
    corr_lin_p = float(np.corrcoef(dm[1:], null_lin[1:])[0, 1])

    offsets = measured - pred
    max_seed_spread = float(max(seed_spreads.values()))

    min_eval_age = min(min(v) for v in eval_ages_per_P.values())
    max_eval_age = max(max(v) for v in eval_ages_per_P.values())
    static_min_age = float(ages_static.min())
    static_max_age = float(ages_static.max())

    per_period = []
    for P, m, p, a, n, off, dmv, dpv in zip(
        Ps, measured, pred, aucs, nev, offsets, dm, dp
    ):
        per_period.append(
            {
                "P": int(P),
                "n_evals": int(n),
                "measured": float(m),
                "predicted": float(p),
                "offset_m_minus_p": float(off),
                "offset_milli": float(1000.0 * off),
                "delta_measured": float(dmv),
                "delta_predicted": float(dpv),
                "delta_measured_milli": float(1000.0 * dmv),
                "delta_predicted_milli": float(1000.0 * dpv),
                "auc": float(a),
                "two_seed_spread": float(seed_spreads[str(int(P))]),
                "two_seed_spread_milli": float(1000.0 * seed_spreads[str(int(P))]),
            }
        )

    optima = _compute_cadence_optima(
        Ps, measured, ages_static, y_static, arm_a_rows, metric, W
    )

    return {
        "metric": metric,
        "static_ages": [float(x) for x in ages_static],
        "static_y": [float(x) for x in y_static],
        "age_coverage_check": {
            "eval_age_min_days": float(min_eval_age),
            "eval_age_max_days": float(max_eval_age),
            "static_age_min_days": static_min_age,
            "static_age_max_days": static_max_age,
            "u0_clamped_below_min_age": bool(min_eval_age < static_min_age),
            "all_ages_gt_2p5_inside_static_grid": bool(max_eval_age <= static_max_age),
            "extrapolates_above_static_max": bool(max_eval_age > static_max_age),
        },
        "per_period": per_period,
        "total_staleness_penalty": float(measured[-1] - measured[0]),
        "total_staleness_penalty_milli": float(1000.0 * (measured[-1] - measured[0])),
        "total_predicted_penalty": float(pred[-1] - pred[0]),
        "total_predicted_penalty_milli": float(1000.0 * (pred[-1] - pred[0])),
        "auc_start": float(aucs[0]),
        "auc_end": float(aucs[-1]),
        "auc_change_points": float(100.0 * (aucs[-1] - aucs[0])),
        "polyfit_slope_ratio": slope_ratio,
        "polyfit_intercept": slope_intercept,
        "shape_corr_age_law": corr_age_law,
        "shape_corr_log_P_null": corr_log_p,
        "shape_corr_linear_P_null": corr_lin_p,
        "n_corr_points": int(len(Ps) - 1),
        "two_seed_max_spread": max_seed_spread,
        "two_seed_max_spread_milli": float(1000.0 * max_seed_spread),
        "two_seed_spread_by_P": seed_spreads,
        "offset_min": float(offsets.min()),
        "offset_max": float(offsets.max()),
        "offset_min_milli": float(1000.0 * offsets.min()),
        "offset_max_milli": float(1000.0 * offsets.max()),
        "offset_span_milli": float(1000.0 * (offsets.max() - offsets.min())),
        "cadence_optima": optima,
    }


def analyze_series(series_name, bt_path, domain, W=2, schedule_meta=None):
    info = _load_static_domain_info(domain)
    static_src = info["src_path"]
    static_test_t0 = info["static_test_t0"]

    rows, Ps, vals_loss, aucs, nev, spreads_loss, fourway_by_p = _load_backtest_agg(
        bt_path, "loss"
    )
    _, _, vals_1m_auc, _, _, spreads_1m_auc, _ = _load_backtest_agg(
        bt_path, "one_minus_auc"
    )

    ages_l, y_l, arm_a_rows = _load_static_a_curve(static_src, "loss")
    ages_a, y_a, _ = _load_static_a_curve(static_src, "one_minus_auc")

    max_t_retrain = max(int(r["t_retrain"]) for r in rows)
    max_t_serve_end = (
        max(int(r["t_serve_end"]) for r in rows)
        if "t_serve_end" in rows[0]
        else None
    )
    zero_overlap_verified = (
        bool(max_t_serve_end <= static_test_t0)
        if max_t_serve_end is not None
        else False
    )

    ntest_by_ps = defaultdict(int)
    for r in rows:
        ntest_by_ps[(int(r["P"]), int(r["seed"]))] += int(r["n_test"])
    unique_totals = sorted({v for v in ntest_by_ps.values()})

    loss_res = _analyze_metric(
        Ps, vals_loss, aucs, nev, spreads_loss, ages_l, y_l, arm_a_rows, "loss", W
    )
    auc_res = _analyze_metric(
        Ps, vals_1m_auc, aucs, nev, spreads_1m_auc, ages_a, y_a, arm_a_rows, "one_minus_auc", W
    )

    return {
        "series": series_name,
        "domain": domain,
        "backtest_jsonl": _rel_posix(bt_path),
        "static_domain_jsonl": _rel_posix(static_src),
        "d_fix_static": info["d_fix"],
        "subsamples_in_backtest": sorted({int(r["subsample"]) for r in rows}),
        "static_test_t0": int(static_test_t0),
        "W_days": int(W),
        "n_rows": len(rows),
        "periods": [int(p) for p in Ps],
        "eval_counts_per_period": [int(x) for x in nev],
        "n_test_totals_per_period_seed": unique_totals,
        "max_t_retrain": int(max_t_retrain),
        "max_t_serve_end": max_t_serve_end,
        "zero_overlap_strict_ok": zero_overlap_verified,
        "schedule_meta": schedule_meta,
        "fourway_volume_weighted_by_P": fourway_by_p,
        "loss": loss_res,
        "one_minus_auc": auc_res,
    }


def build_summary(metas=None):
    metas = metas or {}
    series_specs = [
        (
            "recflow_22d_existing",
            C.REV_RES / "backtest_recflow__d300k.jsonl",
            "recflow",
            2,
        ),
        (
            "recflow_19d_nooverlap",
            C.CR_RES / "backtest_recflow_nooverlap19d.jsonl",
            "recflow",
            2,
        ),
        (
            "kuairand_strict",
            C.CR_RES / "backtest_kuairand_strict.jsonl",
            "kuairand",
            2,
        ),
        (
            "gameugc_strict",
            C.CR_RES / "backtest_gameugc_strict.jsonl",
            "gameugc_finish",
            2,
        ),
    ]

    results = {}
    for sname, bt_path, dom, w in series_specs:
        assert bt_path.exists(), f"Missing backtest output: {bt_path}"
        results[sname] = analyze_series(
            sname, bt_path, dom, W=w, schedule_meta=metas.get(sname)
        )

    # Strict global assertions across the 3 new non-overlapping series
    for sname in ("recflow_19d_nooverlap", "kuairand_strict", "gameugc_strict"):
        sr = results[sname]
        assert sr["zero_overlap_strict_ok"], f"{sname}: failed zero_overlap_strict_ok"
        assert len(sr["n_test_totals_per_period_seed"]) == 1, (
            f"{sname}: unequal n_test totals across P/seed: {sr['n_test_totals_per_period_seed']}"
        )

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "series_order": [s[0] for s in series_specs],
        "series": results,
    }
    out_path = C.CR_RES / "stage3_backtest_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"\nSAVED SUMMARY: {out_path}", flush=True)
    _print_summary(summary)
    return summary, out_path


def _print_summary(summary):
    print("\n" + "=" * 86)
    print("STAGE 3 BACKTEST SUMMARY")
    print("=" * 86)
    for sname in summary["series_order"]:
        sr = summary["series"][sname]
        print(
            f"\n[{sname}] domain={sr['domain']} periods={sr['periods']} "
            f"rows={sr['n_rows']} zero_overlap={sr['zero_overlap_strict_ok']} "
            f"max_serve_end={sr['max_t_serve_end']} static_t0={sr['static_test_t0']}"
        )
        for m in ("loss", "one_minus_auc"):
            mr = sr[m]
            fp = mr["cadence_optima"]["static_fit_power"]
            bc = mr["cadence_optima"]["static_boxcox"]
            cov = mr["age_coverage_check"]
            print(
                f"  <{m}> total_pen={mr['total_staleness_penalty_milli']:+.2f}m "
                f"(pred={mr['total_predicted_penalty_milli']:+.2f}m) | "
                f"slope_ratio={mr['polyfit_slope_ratio']:.3f} | "
                f"r_age={mr['shape_corr_age_law']:.4f} "
                f"r_logP={mr['shape_corr_log_P_null']:.4f} "
                f"r_linP={mr['shape_corr_linear_P_null']:.4f} | "
                f"2seed_max={mr['two_seed_max_spread_milli']:.3f}m | "
                f"offset=[{mr['offset_min_milli']:+.2f}, {mr['offset_max_milli']:+.2f}]m"
            )
            print(
                f"    ages=[{cov['eval_age_min_days']:.1f}..{cov['eval_age_max_days']:.1f}]d "
                f"inside static [{cov['static_age_min_days']:.1f}..{cov['static_age_max_days']:.1f}]d "
                f"(>2.5d inside={cov['all_ages_gt_2p5_inside_static_grid']})"
            )
            print(
                f"    fit_power(delta={fp['delta']:.3f}, b={fp['b']:.3e}) | "
                f"boxcox(delta={bc['delta']:.3f}, beta={bc['beta']:.3e}, r2={bc['r2']:.3f})"
            )
            int_rows = mr["cadence_optima"]["interior_interp_prices"]
            show_rows = int_rows if int_rows else mr["cadence_optima"]["standard_prices"]
            tag = "interior Interp. prices" if int_rows else "standard prices (no interior Interp.)"
            print(f"    P* ({tag}):")
            for pr in show_rows[:6]:
                bc0 = "None" if pr["static_boxcox_a0"] is None else f"{pr['static_boxcox_a0']:.2f}"
                bc1 = "None" if pr["static_boxcox_a_W_half"] is None else f"{pr['static_boxcox_a_W_half']:.2f}"
                print(
                    f"      lamC={pr['lamC']:.2e}: fp(a=0)={pr['static_fit_power_a0']:.2f} "
                    f"fp(a=1)={pr['static_fit_power_a_W_half']:.2f} | "
                    f"bc(a=0)={bc0} bc(a=1)={bc1} | "
                    f"Tested={pr['tested']} (int={pr['tested_interior']}) "
                    f"Interp={pr['interp']:.2f} (int={pr['interp_interior']})"
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        nargs="+",
        choices=list(BACKTEST_CONFIGS.keys()),
        default=list(BACKTEST_CONFIGS.keys()),
        help="Which backtests to run.",
    )
    parser.add_argument(
        "--check-schedules-only",
        action="store_true",
        help="Only verify schedules and zero-overlap assertions without training.",
    )
    parser.add_argument(
        "--analyze-only",
        "--summary-only",
        dest="analyze_only",
        action="store_true",
        help="Only compute stage3_backtest_summary.json from existing JSONLs.",
    )
    args = parser.parse_args()

    if args.check_schedules_only:
        for name in args.only:
            tasks, meta = build_backtest_schedule(name)
            print(json.dumps(meta, indent=2))
        return

    existing_summary_path = C.CR_RES / "stage3_backtest_summary.json"
    existing_series = (
        json.loads(existing_summary_path.read_text()).get("series", {})
        if existing_summary_path.exists()
        else {}
    )

    def _schedule_meta_or_fallback(name):
        npz_path = C.DATA_DIR / C.DOMAINS[BACKTEST_CONFIGS[name]["domain"]]["npz"]
        if npz_path.exists():
            _, meta = build_backtest_schedule(name)
            return meta
        return existing_series.get(name, {}).get("schedule_meta")

    metas = {}
    if not args.analyze_only:
        for name in args.only:
            meta, _ = run_backtest(name)
            metas[name] = meta
    else:
        for name in BACKTEST_CONFIGS:
            metas[name] = _schedule_meta_or_fallback(name)

    if set(args.only) == set(BACKTEST_CONFIGS.keys()) or args.analyze_only:
        # Ensure we attach schedule_meta for all 3 new series when available
        for name in BACKTEST_CONFIGS:
            if name not in metas:
                metas[name] = _schedule_meta_or_fallback(name)
        build_summary(metas)


if __name__ == "__main__":
    main()
