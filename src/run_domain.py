"""Generic fixed-D age-effect grid (Exp A' translated + C' widening) for the
cross-domain delta dictionary.

Protocol per domain:
- test window  = last TEST units of the log
- Exp A': window of W_A units translated back by log-spaced gaps
- Exp C': windows [TEST_T0 - span, TEST_T0] for log-spaced spans
- every training window subsampled to the same D_fix; configs whose window
  holds fewer than D_MIN events are dropped (logged), and D_fix is the min
  volume among surviving configs, capped at D_CAP.

Usage: python src/run_domain.py <domain> [mf|mlp] [cohort|icohort] [--device cuda|cpu]
                              [lr=3e-3] [dim=32] [seeds=3] [subseed=N]
                              [valstop=F] [fixedstep=K]
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
MONTH = 30 * DAY

# unit is the time quantum for gaps/spans; all recorded ages are in DAYS.
DOMAINS = {
    "ml25m": dict(npz="ml25m.npz", unit=MONTH, test=6, wA=24,
                  gaps=[0, 3, 6, 12, 24, 48, 96, 144],
                  spans=[8, 16, 32, 64, 128, 250]),
    "kuairand": dict(npz="kuairand.npz", unit=DAY, test=3, wA=2,
                     gaps=[0, 1, 2, 4, 8, 16, 24],
                     spans=[2, 4, 8, 16, 26], eval_every=50),
    "amz_Video_Games": dict(npz="amz_Video_Games.npz", unit=MONTH, test=6,
                            wA=12, gaps=[0, 6, 12, 24, 48, 96, 192],
                            spans=[6, 12, 24, 48, 96, 192, 320]),
    "amz_CDs_and_Vinyl": dict(npz="amz_CDs_and_Vinyl.npz", unit=MONTH, test=6,
                              wA=12, gaps=[0, 6, 12, 24, 48, 96, 192],
                              spans=[6, 12, 24, 48, 96, 192, 320]),
    "amz_Software": dict(npz="amz_Software.npz", unit=MONTH, test=6,
                         wA=12, gaps=[0, 6, 12, 24, 48, 96, 192],
                         spans=[6, 12, 24, 48, 96, 192, 320]),
    "amz_Appliances": dict(npz="amz_Appliances.npz", unit=MONTH, test=6,
                           wA=12, gaps=[0, 6, 12, 24, 48, 96, 192],
                           spans=[6, 12, 24, 48, 96, 192, 320]),
    "amz_Baby_Products": dict(npz="amz_Baby_Products.npz", unit=MONTH, test=6,
                              wA=12, gaps=[0, 6, 12, 24, 48, 96, 192],
                              spans=[6, 12, 24, 48, 96, 192, 320]),
    # held-out domains for the blind prediction test
    "amz_Toys_and_Games": dict(npz="amz_Toys_and_Games.npz", unit=MONTH,
                               test=6, wA=12, gaps=[0, 6, 12, 24, 48, 96, 192],
                               spans=[6, 12, 24, 48, 96, 192, 320]),
    "amz_Office_Products": dict(npz="amz_Office_Products.npz", unit=MONTH,
                                test=6, wA=12,
                                gaps=[0, 6, 12, 24, 48, 96, 192],
                                spans=[6, 12, 24, 48, 96, 192, 320]),
    "amz_Pet_Supplies": dict(npz="amz_Pet_Supplies.npz", unit=MONTH, test=6,
                             wA=12, gaps=[0, 6, 12, 24, 48, 96, 192],
                             spans=[6, 12, 24, 48, 96, 192, 320]),
    # round-2 hold-outs, stratified by catalog size
    "amz_Musical_Instruments": dict(
        npz="amz_Musical_Instruments.npz", unit=MONTH, test=6, wA=12,
        gaps=[0, 6, 12, 24, 48, 96, 192], spans=[6, 12, 24, 48, 96, 192, 320]),
    "amz_Arts_Crafts_and_Sewing": dict(
        npz="amz_Arts_Crafts_and_Sewing.npz", unit=MONTH, test=6, wA=12,
        gaps=[0, 6, 12, 24, 48, 96, 192], spans=[6, 12, 24, 48, 96, 192, 320]),
    "amz_Automotive": dict(
        npz="amz_Automotive.npz", unit=MONTH, test=6, wA=12,
        gaps=[0, 6, 12, 24, 48, 96, 192], spans=[6, 12, 24, 48, 96, 192, 320]),
    "amz_Kindle_Store": dict(
        npz="amz_Kindle_Store.npz", unit=MONTH, test=6, wA=12,
        gaps=[0, 6, 12, 24, 48, 96, 192], spans=[6, 12, 24, 48, 96, 192, 320]),
    # production feed logs
    "recflow": dict(npz="recflow.npz", unit=DAY, test=3, wA=2,
                    gaps=[0, 1, 2, 4, 8, 16, 24],
                    spans=[2, 4, 8, 16, 26], eval_every=50),
    "kuairec": dict(npz="kuairec.npz", unit=DAY, test=5, wA=3,
                    gaps=[0, 2, 4, 8, 16, 32, 64],
                    spans=[3, 6, 12, 24, 48, 74], eval_every=50),
    "gameugc_play": dict(npz="gameugc_play.npz", unit=DAY, test=3, wA=2,
                         gaps=[0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                         spans=[2, 4, 8, 16, 32, 64, 128, 256, 512],
                         eval_every=50),
    "gameugc_finish": dict(npz="gameugc_finish.npz", unit=DAY, test=3, wA=2,
                           gaps=[0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                           spans=[2, 4, 8, 16, 32, 64, 128, 256, 512],
                           eval_every=50),
    # cross-platform domains
    "epinions": dict(npz="epinions.npz", unit=MONTH, test=2, wA=2,
                     gaps=[0, 1, 2, 4, 8, 12], spans=[2, 4, 8, 14]),
    "beer": dict(npz="beer.npz", unit=MONTH, test=6, wA=12,
                 gaps=[0, 6, 12, 24, 48, 84],
                 spans=[6, 12, 24, 48, 96, 180]),
    "yelp": dict(npz="yelp.npz", unit=MONTH, test=6, wA=12,
                 gaps=[0, 6, 12, 24, 48, 96, 168],
                 spans=[6, 12, 24, 48, 96, 192]),
    "netflix": dict(npz="netflix.npz", unit=MONTH, test=6, wA=12,
                    gaps=[0, 3, 6, 12, 24, 48],
                    spans=[6, 12, 24, 48, 72]),
    "steam": dict(npz="steam.npz", unit=MONTH, test=6, wA=12,
                  gaps=[0, 6, 12, 24, 48, 96],
                  spans=[6, 12, 24, 48, 96, 120]),
}

D_MIN = 25_000
D_CAP = 300_000
SEEDS = [0, 1, 2]


def key(cfg):
    return json.dumps({"exp": cfg["exp"], "train_t0": cfg["train_t0"],
                       "train_t1": cfg["train_t1"], "seed": cfg["seed"],
                       "subsample": cfg["subsample"]}, sort_keys=True)


def main(domain, probe="mf", cohort=False, item_only=False, lr=3e-3, dim=32,
         val_frac=None, fixed_step=None, sub_seed_override=None, device="cuda"):
    if domain not in DOMAINS:
        raise SystemExit(
            f"Unknown domain {domain!r}. Known domains: {sorted(DOMAINS)}"
        )
    spec = DOMAINS[domain]
    npz = ROOT / "data" / spec["npz"]
    if not npz.exists():
        raise SystemExit(
            f"Missing {npz}; build preprocessed .npz first via src/prep_*.py."
        )
    suffix = ("__icohort" if (cohort and item_only) else
              "__cohort" if cohort else "") + \
        ("" if probe == "mf" else f"__{probe}") + \
        ("" if lr == 3e-3 else f"__lr{lr:g}") + \
        ("" if dim == 32 else f"__dim{dim}") + \
        ("" if not val_frac else "__valstop") + \
        ("" if sub_seed_override in (None, 0) else f"__ss{sub_seed_override}")
    out = ROOT / "results" / f"domain_{domain}{suffix}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with np.load(npz) as z:
        ts = z["ts"]
        user_all = z["user"] if cohort else None
        item_all = z["item"] if cohort else None
    u = spec["unit"]
    test_t1 = int(ts[-1]) + 1
    test_t0 = test_t1 - spec["test"] * u
    span_avail = (test_t0 - int(ts[0])) / u

    def volume(t0, t1):
        a, b = np.searchsorted(ts, [t0, t1])
        return int(b - a)

    cands = []
    for g in spec["gaps"]:
        t1 = test_t0 - g * u
        t0 = t1 - spec["wA"] * u
        if t0 < ts[0]:
            print(f"drop A gap={g}: window exceeds history")
            continue
        cands.append({"exp": "A", "gap": g, "train_t0": int(t0),
                      "train_t1": int(t1), "vol": volume(t0, t1),
                      "age_days": (g + spec["wA"] / 2 + spec["test"] / 2)
                      * u / DAY})
    for s in spec["spans"]:
        t0 = test_t0 - s * u
        if t0 < ts[0] - 2 * u:   # allow slight overhang for "full history"
            print(f"drop C span={s}: exceeds history")
            continue
        t0 = max(t0, int(ts[0]))
        cands.append({"exp": "C", "span": s, "train_t0": int(t0),
                      "train_t1": int(test_t0), "vol": volume(t0, test_t0),
                      "age_days": (s / 2 + spec["test"] / 2) * u / DAY})

    kept = [c for c in cands if c["vol"] >= D_MIN]
    for c in cands:
        if c["vol"] < D_MIN:
            print(f"drop {c['exp']} {c.get('gap', c.get('span'))}: "
                  f"vol={c['vol']} < {D_MIN}")
    if not kept:
        raise SystemExit(
            f"{domain}: no candidate windows satisfy D_MIN={D_MIN}"
        )
    d_fix = min(D_CAP, min(c["vol"] for c in kept))
    print(f"{domain}: span={span_avail:.0f} units, {len(kept)} configs, "
          f"D_fix={d_fix}, test events={volume(test_t0, test_t1)}")

    cohort_idx = None
    sub_seed = None
    if cohort:
        # fixed cohort under a FIXED subsample: every run uses the seed-0
        # subsample of its window (model seeds only vary init/order), and the
        # cohort is the set of test events whose user AND item appear in the
        # seed-0 subsample of EVERY A window -> composition is exactly
        # constant across ages.
        # The cohort is defined by ONE subsample realization. Fixing it makes
        # the estimator well defined but leaves cohort membership a function
        # of that draw; sub_seed_override reruns the whole construction on a
        # different draw to measure how much tau_half depends on it.
        import torch

        sub_seed = 0 if sub_seed_override is None else sub_seed_override
        a_wins = [c for c in kept if c["exp"] == "A"]
        kept = a_wins  # cohort mode runs the A arm only
        c0, e0 = np.searchsorted(ts, [test_t0, test_t1])
        te_u, te_i = user_all[c0:e0], item_all[c0:e0]
        mask = np.ones(e0 - c0, dtype=bool)
        for c in a_wins:
            a, b = np.searchsorted(ts, [c["train_t0"], c["train_t1"]])
            if d_fix < b - a:
                rng = np.random.default_rng(sub_seed)
                sel = a + np.sort(rng.choice(b - a, size=int(d_fix),
                                             replace=False))
            else:
                sel = np.arange(a, b)
            su = np.zeros(user_all.max() + 1, dtype=bool)
            si = np.zeros(item_all.max() + 1, dtype=bool)
            su[user_all[sel]] = True
            si[item_all[sel]] = True
            # item_only: review platforms have no returning users, so the
            # user-side intersection is empty by construction; the item-side
            # cohort (with uniformly-cold users) is the measurable pure-drift
            # object there.
            mask &= si[te_i] if item_only else (su[te_u] & si[te_i])
        cohort_idx = np.flatnonzero(mask)
        print(f"cohort: {len(cohort_idx)} of {e0 - c0} test events")
        if len(cohort_idx) < 300:
            print("cohort too small; aborting domain")
            return
        cohort_idx = torch.from_numpy(cohort_idx)

    done = set()
    if out.exists():
        done = {key(json.loads(l)) for l in out.read_text().splitlines()
                if l.strip()}
    todo = [dict(c, seed=s, subsample=int(d_fix)) for c in kept for s in SEEDS
            if key(dict(c, seed=s, subsample=int(d_fix))) not in done]
    print(f"{len(todo)} runs to do ({len(done)} done)")

    for n, cfg in enumerate(todo):
        res = run(cfg["train_t0"], cfg["train_t1"], test_t0, test_t1,
                  seed=cfg["seed"], subsample=cfg["subsample"],
                  data_path=npz, eval_every=spec.get("eval_every", 200),
                  probe=probe, cohort_idx=cohort_idx,
                  subsample_seed=sub_seed, lr=lr, dim=dim, device=device,
                  val_frac=val_frac, fixed_step=fixed_step)
        res.update({k: v for k, v in cfg.items() if k not in res})
        res["domain"] = domain
        with out.open("a") as f:
            f.write(json.dumps(res) + "\n")
        tag = (f"gap={cfg.get('gap')}" if cfg["exp"] == "A"
               else f"span={cfg.get('span')}")
        print(f"[{n+1}/{len(todo)}] {cfg['exp']} {tag} seed={cfg['seed']}: "
              f"loss={res['loss']:.4f} auc={res['auc']:.4f} "
              f"({res['train_time_s']}s)", flush=True)


if __name__ == "__main__":
    raw_args = sys.argv[1:]
    if any(a in ("-h", "--help") for a in raw_args):
        print(__doc__.strip())
        sys.exit(0)
    if not raw_args:
        raise SystemExit(
            "Usage: python src/run_domain.py <domain> [mf|mlp] [cohort|icohort] [--device cuda|cpu]"
        )
    lr = 3e-3
    dim = 32
    device = "cuda"
    val_frac = fixed_step = sub_seed_override = None
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
        elif "=" in a or a in ("cohort", "icohort"):
            opts.append(a)
            i += 1
        else:
            pos.append(a)
            i += 1
    if not pos:
        raise SystemExit("Usage: python src/run_domain.py <domain> [mf|mlp] ...")
    domain_arg = pos[0]
    probe_arg = pos[1] if len(pos) > 1 else "mf"
    for a in opts:
        if a.startswith("subseed="):
            sub_seed_override = int(a[8:])
        if a.startswith("lr="):
            lr = float(a[3:])
        if a.startswith("seeds="):
            SEEDS = list(range(int(a[6:])))
        if a.startswith("dim="):
            dim = int(a[4:])
        # valstop=0.1 : hold out 10% of the training subsample, report the
        # test loss at the step chosen ON THAT SLICE alongside the test-argmin
        # convention (see train.py).
        if a.startswith("valstop="):
            val_frac = float(a[8:])
        if a.startswith("fixedstep="):
            fixed_step = int(a[10:])
    main(domain_arg, probe_arg,
         cohort=any(a in ("cohort", "icohort") for a in opts),
         item_only="icohort" in opts, lr=lr, dim=dim,
         val_frac=val_frac, fixed_step=fixed_step,
         sub_seed_override=sub_seed_override, device=device)
