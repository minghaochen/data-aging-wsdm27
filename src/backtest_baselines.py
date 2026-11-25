"""Comparators for the rolling-retrain backtest.

The age law tells you how much staleness costs, but it does not by itself
show that *retraining more often* is the right lever. Two standard
engineering answers compete with it, and the main study compared against
neither:

  recency  -- train on a longer window but weight events by recency
              (exponential half-life), so a stale cadence is partly
              compensated by emphasising the newest data in the window;
  warm     -- warm-start each retrain from the previous model instead of
              training from scratch.

Both run on the identical serving schedule as the from-scratch backtest
(same periods, same windows, same subsample size). The optimisation budget
is NOT identical: recency runs at the from-scratch budget, but warm diverges
there (see WARM_STEP_FRAC below) and runs at steps/8 and lr 1e-4, so the
loss-side comparison against warm is between each policy's best feasible
configuration rather than at matched compute.

Usage: python src/backtest_baselines.py recflow [--mode recency|warm|both]
                                                [--halflife DAYS] [--device cuda|cpu]
                                                [--valstop F] [--fixedstep K]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SRC = str(Path(__file__).resolve().parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from train import run
from backtest import SPECS, SEEDS

ROOT = Path(__file__).resolve().parent.parent
DAY = 86400

# recency baseline: 4x the from-scratch window, half-life = the original
# window length, so the effective mass sits where the from-scratch model
# trains but old events still contribute.
WINDOW_MULT = 4

# warm start: an incremental update is NOT a full retrain from a warm
# initialisation. Running the full 8k-step budget on top of an already
# converged model, every day, on a two-day sliding window drives the probe
# far past the overfitting cliff -- our first attempt returned 0.91 nats at
# P=1, above the ln 2 chance baseline, which is exactly the divergence
# signature the protocol says to re-tune. Incremental updates therefore use
# a reduced budget and a lower learning rate, which is also what production
# incremental training does.
WARM_STEP_FRAC = 8          # steps // 8
WARM_LR = 1e-4
CHANCE = 0.6931471805599453


def schedule(spec, ts, t0_log):
    horizon = spec["serve_end"] - spec["serve_start"]
    for P in spec["periods"]:
        for k in range(int(np.ceil(horizon / P))):
            t_re = t0_log + (spec["serve_start"] + k * P) * DAY
            t_end = min(t_re + P * DAY, t0_log + spec["serve_end"] * DAY)
            yield P, k, t_re, t_end


def main(domain, mode, device, halflife=None, val_frac=None,
         fixed_step=None):
    if domain not in SPECS:
        raise SystemExit(
            f"Unknown domain {domain!r}. Known domains: {sorted(SPECS)}"
        )
    spec = SPECS[domain]
    npz = ROOT / "data" / spec["npz"]
    if not npz.exists():
        raise SystemExit(
            f"Missing {npz}; build preprocessed .npz first via src/prep_*.py."
        )
    with np.load(npz) as z:
        ts = z["ts"]
    t0_log = int(ts[0])
    modes = ["recency", "warm"] if mode == "both" else [mode]

    for m in modes:
        tag = "" if (halflife is None or m != "recency") else f"_hl{halflife:g}"
        tag += "" if not val_frac else "_valstop"
        out = ROOT / "results" / f"backtest_{domain}__{m}{tag}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        done = set()
        if out.exists():
            done = {(r["P"], r["t_retrain"], r["seed"])
                    for r in map(json.loads, out.read_text().splitlines())
                    if r}
        # warm start needs the models in chronological order per (P, seed)
        # Resume guard. `done` survives a restart but `prev` does not, so
        # resuming a partially-finished warm chain would silently rerun its
        # first link at the FULL from-scratch budget and file the result as
        # warm-started (whereas full uninterrupted chains carry warm_started on
        # all 78/90 continuation links). Refuse rather than corrupt.
        if m == "warm" and done:
            raise SystemExit(
                f"{out.name} already holds {len(done)} rows: warm chains "
                f"cannot be resumed, because the previous model's weights "
                f"are not persisted. Delete the file and rerun.")
        prev = {}
        with out.open("a") as fh:
            for P, k, t_re, t_end in schedule(spec, ts, t0_log):
                W = spec["W"] * (WINDOW_MULT if m == "recency" else 1)
                a, b = np.searchsorted(ts, [t_re - W * DAY, t_re])
                c, e = np.searchsorted(ts, [t_re, t_end])
                if b - a < 5000 or e - c < 1000:
                    print(f"skip P={P} k={k}: train={b-a} serve={e-c}")
                    continue
                for seed in SEEDS:
                    key = (P, seed)
                    if (P, t_re, seed) in done:
                        continue
                    kw = {}
                    if m == "recency":
                        kw["recency_halflife_days"] = (
                            spec["W"] if halflife is None else halflife)
                    if m == "warm" and key in prev:
                        kw["init_state"] = prev[key]
                        kw["steps"] = 8000 // WARM_STEP_FRAC
                        kw["lr"] = WARM_LR
                    res, state = run(
                        t_re - W * DAY, t_re, t_re, t_end, seed=seed,
                        subsample=spec["d_fix"], data_path=npz,
                        eval_every=spec["eval_every"], device=device,
                        return_state=True, val_frac=val_frac,
                        fixed_step=fixed_step, **kw)
                    if res["loss"] > CHANCE:
                        print(f"  WARNING diverged (loss {res['loss']:.4f} "
                              f"> ln 2) at P={P} k={k} seed={seed}")
                    prev[key] = state
                    res.update(domain=domain, P=P, t_retrain=int(t_re),
                               mode=m, window_days=W)
                    fh.write(json.dumps(res) + "\n")
                    fh.flush()
                    print(f"  [{m}] P={P:3d} k={k:2d} s={seed} "
                          f"loss={res['loss']:.5f} auc={res['auc']:.5f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Run recency-weighted and warm-start rolling-backtest baselines."
    )
    ap.add_argument("domain")
    ap.add_argument("--mode", default="both",
                    choices=["recency", "warm", "both"])
    ap.add_argument("--device", default="cuda")
    # Optional exponential recency half-life in days (defaults to window W).
    ap.add_argument("--halflife", type=float, default=None)
    # Selecting the checkpoint on held-out data removes the selection bias
    # from BOTH policies, which is what the from-scratch-vs-warm loss
    # comparison needs: at the full budget a model has 160 checkpoints to be
    # picked from and a warm-started one has 20.
    ap.add_argument("--valstop", type=float, default=None)
    ap.add_argument("--fixedstep", type=int, default=None)
    a = ap.parse_args()
    main(a.domain, a.mode, a.device, halflife=a.halflife,
         val_frac=a.valstop, fixed_step=a.fixedstep)
