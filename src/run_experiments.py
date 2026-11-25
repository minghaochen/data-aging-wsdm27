"""LEGACY EXPLORATORY -- not used by the paper; superseded by src/run_domain.py.

First-generation MovieLens-25M experiment runner: Exp A (window age),
Exp B (window length), and Exp C (fixed sample count). Results append to
results/results.jsonl; already-done configs are skipped.

Usage: python src/run_experiments.py
"""
import json
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from train import run, DAY

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "ml25m.npz"
OUT = ROOT / "results" / "results.jsonl"
OUT.parent.mkdir(exist_ok=True)

YEAR = 365 * DAY
MONTH = 30 * DAY

# ml-25m ends 2019-11-21; test = last ~6 months
TEST_T1 = 1574327703 + 1          # dataset max ts (verified in prep)
TEST_T0 = TEST_T1 - 6 * MONTH


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
            f"Missing {DATA}; run src/prep.py first."
        )
    configs = []

    # Exp A: fixed window length W=2y, slide back by gap
    W = 2 * YEAR
    for gap_m in [0, 3, 6, 12, 24, 48, 96, 144]:
        t1 = TEST_T0 - gap_m * MONTH
        configs.append({"exp": "A", "gap_months": gap_m,
                        "train_t0": int(t1 - W), "train_t1": int(t1)})

    # Exp B: window ends at test start, vary length (log-spaced months)
    for w_m in [2, 4, 8, 16, 32, 64, 128, 250]:
        configs.append({"exp": "B", "w_months": w_m,
                        "train_t0": int(TEST_T0 - w_m * MONTH),
                        "train_t1": int(TEST_T0)})

    # Exp C: fixed sample count D, vary time span (orthogonal to Exp B).
    # Subsample uniformly from windows of increasing span; D fits the
    # smallest span (8 months has ~850k events in 2019).
    D_FIX = 500_000
    for w_m in [8, 16, 32, 64, 128, 250]:
        configs.append({"exp": "C", "w_months": w_m, "subsample": D_FIX,
                        "train_t0": int(TEST_T0 - w_m * MONTH),
                        "train_t1": int(TEST_T0)})

    seeds = [0, 1]
    done = done_keys()
    todo = []
    for cfg in configs:
        for s in seeds:
            c = dict(cfg, seed=s)
            if key(c) not in done:
                todo.append(c)
    print(f"{len(todo)} runs to do ({len(done)} already done)")

    for n, cfg in enumerate(todo):
        res = run(cfg["train_t0"], cfg["train_t1"], TEST_T0, TEST_T1,
                  seed=cfg["seed"], subsample=cfg.get("subsample"))
        res.update({k: v for k, v in cfg.items() if k not in res})
        with OUT.open("a") as f:
            f.write(json.dumps(res) + "\n")
        tag = f"gap={cfg.get('gap_months')}m" if cfg["exp"] == "A" \
            else f"W={cfg.get('w_months')}m"
        print(f"[{n+1}/{len(todo)}] {cfg['exp']} {tag} seed={cfg['seed']}: "
              f"loss={res['loss']:.4f} seen={res['loss_seen']:.4f} "
              f"auc={res['auc']:.4f} ({res['train_time_s']}s)", flush=True)


if __name__ == "__main__":
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__.strip())
        sys.exit(0)
    main()
