"""LEGACY EXPLORATORY -- not used by the paper; superseded by src/run_domain.py.

First-generation KuaiRand-Pure day-scale grid (Exp A/B/C). TEST covers the
last 3 days of the standard log. Results append to results/results_kuairand.jsonl.

Usage: python src/run_experiments_kuairand.py
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
OUT = ROOT / "results" / "results_kuairand.jsonl"
DATA = ROOT / "data" / "kuairand.npz"

DAY = 86400


def key(cfg):
    return json.dumps({"exp": cfg["exp"], "train_t0": cfg["train_t0"],
                       "train_t1": cfg["train_t1"], "seed": cfg["seed"],
                       "subsample": cfg.get("subsample")}, sort_keys=True)


def done_keys():
    if not OUT.exists():
        return set()
    return {key(json.loads(line)) for line in OUT.read_text().splitlines() if line}


def main():
    if not DATA.exists():
        raise SystemExit(
            f"Missing {DATA}; run src/prep_kuairand.py first."
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with np.load(DATA) as z:
        max_ts = int(z["ts"][-1])
        min_ts = int(z["ts"][0])
    test_t1 = max_ts + 1
    test_t0 = test_t1 - 3 * DAY
    span_days = (test_t0 - min_ts) / DAY
    print(f"history span before test: {span_days:.1f} days")
    configs = []

    # Daily volume is very uneven (80-270k/day in the first 5 days, ~15-25k
    # after), so Exp A windows are ALSO subsampled to fixed D to keep data
    # volume from confounding age.
    D_FIX = 25_000

    # Exp A: fixed window W=2 days at fixed D, slide back by gap (days)
    W = 2 * DAY
    max_gap = int(span_days - 2)
    for gap_d in [0, 1, 2, 4, 8, 16, max_gap]:
        t1 = test_t0 - gap_d * DAY
        configs.append({"exp": "A", "gap_days": gap_d, "subsample": D_FIX,
                        "train_t0": int(t1 - W), "train_t1": int(t1)})

    # Exp B: window ends at test start, vary length (days), natural volume
    for w_d in [1, 2, 4, 8, 16, int(span_days)]:
        configs.append({"exp": "B", "w_days": w_d,
                        "train_t0": int(test_t0 - w_d * DAY),
                        "train_t1": int(test_t0)})

    # Exp C: fixed sample count, vary span (spans small enough windows excluded)
    for w_d in [2, 4, 8, 16, int(span_days)]:
        configs.append({"exp": "C", "w_days": w_d, "subsample": D_FIX,
                        "train_t0": int(test_t0 - w_d * DAY),
                        "train_t1": int(test_t0)})

    seeds = [0, 1, 2]
    done = done_keys()
    todo = [dict(cfg, seed=s) for cfg in configs for s in seeds
            if key(dict(cfg, seed=s)) not in done]
    print(f"{len(todo)} runs to do ({len(done)} already done)")

    for n, cfg in enumerate(todo):
        res = run(cfg["train_t0"], cfg["train_t1"], test_t0, test_t1,
                  seed=cfg["seed"], subsample=cfg.get("subsample"),
                  data_path=DATA, eval_every=50)
        res.update({k: v for k, v in cfg.items() if k not in res})
        with OUT.open("a") as f:
            f.write(json.dumps(res) + "\n")
        tag = (f"gap={cfg.get('gap_days')}d" if cfg["exp"] == "A"
               else f"W={cfg.get('w_days')}d")
        print(f"[{n+1}/{len(todo)}] {cfg['exp']} {tag} seed={cfg['seed']}: "
              f"loss={res['loss']:.4f} seen={res['loss_seen']:.4f} "
              f"auc={res['auc']:.4f} ({res['train_time_s']}s)", flush=True)


if __name__ == "__main__":
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__.strip())
        sys.exit(0)
    main()
