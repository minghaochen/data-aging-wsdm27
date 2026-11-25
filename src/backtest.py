"""Rolling-retrain backtest: does the statically-measured age law predict
the serving loss of a periodically-retrained model, and does the derived
optimal period match the empirical optimum?

For retrain period P (days): models are trained at t_0, t_0+P, ... on the
window [t_i - W, t_i] (subsampled to D_fix, fixed budget) and serve events
in [t_i, t_i + P). Reported: volume-weighted mean serving loss Lbar(P).

Usage: python src/backtest.py kuairec|recflow [P=1,2,...] [tag=...] [valstop=0.1] [fixedstep=300] [dfix=300000] [--device cuda|cpu]
"""
import json
import sys
from pathlib import Path

import numpy as np

_SRC = str(Path(__file__).resolve().parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from train import run

ROOT = Path(__file__).resolve().parent.parent
DAY = 86400

SPECS = {
    # serve_start/serve_end in days from log start; W = training window days
    "kuairec": dict(npz="kuairec.npz", W=3, serve_start=30, serve_end=78,
                    periods=[1, 2, 4, 8, 16, 24, 48], d_fix=300_000,
                    eval_every=50),
    "recflow": dict(npz="recflow.npz", W=2, serve_start=7, serve_end=29,
                    periods=[1, 2, 4, 8, 11, 22], d_fix=217_575,
                    eval_every=50),
}
SEEDS = [0, 1]


def main(domain, only_periods=None, tag="", val_frac=None, fixed_step=None,
         d_fix=None, device="cuda"):
    if domain not in SPECS:
        raise SystemExit(
            f"Unknown domain {domain!r}. Known domains: {sorted(SPECS)}"
        )
    spec = SPECS[domain]
    if only_periods:
        spec = dict(spec, periods=[p for p in spec["periods"]
                                   if p in only_periods])
    if d_fix:
        # Compare D=300,000 against default backtest volume (217,575 on RecFlow)
        # to separate training-volume effects from period-difficulty offsets.
        spec = dict(spec, d_fix=int(d_fix))
    npz = ROOT / "data" / spec["npz"]
    if not npz.exists():
        raise SystemExit(
            f"Missing {npz}; build preprocessed .npz first via src/prep_*.py."
        )
    with np.load(npz) as z:
        ts = z["ts"]
    t0_log = int(ts[0])
    out = ROOT / "results" / f"backtest_{domain}{tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out.exists():
        done = {(r["P"], r["t_retrain"], r["seed"])
                for r in map(json.loads, out.read_text().splitlines()) if r}

    horizon = spec["serve_end"] - spec["serve_start"]
    for P in spec["periods"]:
        n_models = int(np.ceil(horizon / P))
        for k in range(n_models):
            t_re = t0_log + (spec["serve_start"] + k * P) * DAY
            t_serve_end = min(t_re + P * DAY,
                              t0_log + spec["serve_end"] * DAY)
            # skip hollow intervals (kuairec has inactive days)
            a, b = np.searchsorted(ts, [t_re - spec["W"] * DAY, t_re])
            c, e = np.searchsorted(ts, [t_re, t_serve_end])
            if b - a < 5000 or e - c < 1000:
                print(f"skip P={P} @day{spec['serve_start']+k*P}: "
                      f"train={b-a} serve={e-c}")
                continue
            for seed in SEEDS:
                if (P, t_re, seed) in done:
                    continue
                res = run(t_re - spec["W"] * DAY, t_re, t_re, t_serve_end,
                          seed=seed, subsample=spec["d_fix"], data_path=npz,
                          eval_every=spec["eval_every"], device=device,
                          val_frac=val_frac, fixed_step=fixed_step)
                rec = dict(domain=domain, P=P, t_retrain=int(t_re),
                           seed=seed, loss=res["loss"], n_test=res["n_test"],
                           auc=res["auc"], best_step=res["best_step"],
                           subsample=spec["d_fix"])
                # alternate stopping rules, when requested
                rec.update({k: v for k, v in res.items()
                            if k.endswith(("_valstop", "_fixed"))
                            or k in ("val_frac", "n_val")})
                with out.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"P={P} model@day{spec['serve_start']+k*P} seed={seed}: "
                      f"loss={res['loss']:.4f} n={res['n_test']:,}",
                      flush=True)

    # aggregate
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    print("\nP  Lbar(volume-weighted)")
    for P in spec["periods"]:
        rs = [r for r in rows if r["P"] == P]
        w = np.array([r["n_test"] for r in rs], float)
        l = np.array([r["loss"] for r in rs])
        print(f"{P:3d}  {np.sum(w * l) / np.sum(w):.5f}   "
              f"({len(rs)} model-evals)")


if __name__ == "__main__":
    raw_args = sys.argv[1:]
    if any(a in ("-h", "--help") for a in raw_args):
        print(__doc__.strip())
        sys.exit(0)
    if not raw_args:
        raise SystemExit(
            "Usage: python src/backtest.py kuairec|recflow [P=...] [tag=...] "
            "[valstop=...] [fixedstep=...] [dfix=...] [--device DEV]"
        )
    only = None
    tag = ""
    device = "cuda"
    val_frac = fixed_step = d_fix = None
    pos = []
    opts = []
    i = 0
    while i < len(raw_args):
        a = raw_args[i]
        if a == "--device":
            if i + 1 >= len(raw_args):
                raise SystemExit("--device requires an argument")
            device = raw_args[i + 1]
            i += 2
        elif a.startswith("--device="):
            device = a.split("=", 1)[1]
            i += 1
        elif a.startswith("device="):
            device = a.split("=", 1)[1]
            i += 1
        elif "=" in a:
            opts.append(a)
            i += 1
        else:
            pos.append(a)
            i += 1
    if not pos:
        raise SystemExit("Usage: python src/backtest.py kuairec|recflow ...")
    for a in opts:
        if a.startswith("P="):
            only = [int(x) for x in a[2:].split(",")]
        if a.startswith("tag="):
            tag = a[4:]
        if a.startswith("valstop="):
            val_frac = float(a[8:])
        if a.startswith("fixedstep="):
            fixed_step = int(a[10:])
        if a.startswith("dfix="):
            d_fix = int(a[5:])
    main(pos[0], only_periods=only, tag=tag, val_frac=val_frac,
         fixed_step=fixed_step, d_fix=d_fix, device=device)
