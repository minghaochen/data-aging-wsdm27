"""Control: is the feed-log drift timescale an artefact of weekday composition?

The main protocol probes the feed logs with 2-day training windows translated
by log-spaced gaps.  Over a 30-day log those windows land on different
weekday pairs (KuaiRand: Wed/Thu at the freshest age, Sat/Sun at 6.5 d,
Sun/Mon at 26.5 d), so weekday composition varies systematically with age
while the test period is held fixed.  Any weekly periodicity in engagement
therefore enters the measured age curve, and the reported timescale of
~2 weeks is uncomfortably close to two weekly cycles.

This control removes the confound by construction: every training window is
7 days long, so every window covers all seven weekdays in equal proportion.
Windows are still subsampled to the same event count as the main run, so the
only thing that changes relative to the main measurement is weekday balance
(and the window length that buys it).

If the drift timescale is a weekday artefact, tau_1/2 should move.  If it is
real, it should not.

Usage: python src/control_weekday.py <kuairand|recflow> [--device cpu]
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

ROOT = Path(__file__).resolve().parent.parent
DAY = 86400

SPEC = {
    # wA = 7 days: each window spans a full week, so weekday composition is
    # identical across ages.  gaps chosen to keep the age range comparable to
    # the main run (2.5-26.5 d) inside a 30-day log.
    "kuairand": dict(npz="kuairand.npz", test=3, wA=7,
                     gaps=[0, 1, 2, 4, 8, 13], dfix=28849, eval_every=50),
    "recflow": dict(npz="recflow.npz", test=3, wA=7,
                    gaps=[0, 1, 2, 4, 8, 13], dfix=300000, eval_every=50),
}
SEEDS = [0, 1, 2]


def main(domain, device="cuda"):
    if domain not in SPEC:
        raise SystemExit(
            f"Unknown domain {domain!r}. Known domains: {sorted(SPEC)}"
        )
    spec = SPEC[domain]
    npz = ROOT / "data" / spec["npz"]
    if not npz.exists():
        raise SystemExit(
            f"Missing {npz}; build preprocessed .npz first via src/prep_*.py."
        )
    out = ROOT / "results" / f"domain_{domain}__w7cohort.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with np.load(npz) as z:
        ts, user_all, item_all = z["ts"], z["user"], z["item"]
    test_t1 = int(ts[-1]) + 1
    test_t0 = test_t1 - spec["test"] * DAY

    wins = []
    for g in spec["gaps"]:
        t1 = test_t0 - g * DAY
        t0 = t1 - spec["wA"] * DAY
        if t0 < ts[0]:
            print(f"drop gap={g}: window exceeds history")
            continue
        a, b = np.searchsorted(ts, [t0, t1])
        wins.append(dict(gap=g, train_t0=int(t0), train_t1=int(t1),
                         vol=int(b - a),
                         age_days=g + spec["wA"] / 2 + spec["test"] / 2))
    d_fix = min(spec["dfix"], min(w["vol"] for w in wins))
    print(f"{domain} w7 control: {len(wins)} windows, D_fix={d_fix}")

    # same composition-fixed cohort construction as the main protocol
    c0, e0 = np.searchsorted(ts, [test_t0, test_t1])
    te_u, te_i = user_all[c0:e0], item_all[c0:e0]
    mask = np.ones(e0 - c0, dtype=bool)
    for w in wins:
        a, b = np.searchsorted(ts, [w["train_t0"], w["train_t1"]])
        if d_fix < b - a:
            rng = np.random.default_rng(0)
            sel = a + np.sort(rng.choice(b - a, size=int(d_fix),
                                         replace=False))
        else:
            sel = np.arange(a, b)
        su = np.zeros(user_all.max() + 1, dtype=bool)
        si = np.zeros(item_all.max() + 1, dtype=bool)
        su[user_all[sel]] = True
        si[item_all[sel]] = True
        mask &= su[te_u] & si[te_i]
    cohort_idx = np.flatnonzero(mask)
    print(f"cohort size: {len(cohort_idx)} of {e0 - c0} test events")

    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["train_t0"], r["seed"]))

    with out.open("a") as fh:
        for w in wins:
            for seed in SEEDS:
                if (w["train_t0"], seed) in done:
                    continue
                res = run(w["train_t0"], w["train_t1"], test_t0, test_t1,
                          seed=seed, subsample=int(d_fix), subsample_seed=0,
                          data_path=str(npz), eval_every=spec["eval_every"],
                          cohort_idx=cohort_idx, device=device)
                res.update(exp="A", gap=w["gap"], vol=w["vol"],
                           age_days=w["age_days"], domain=f"{domain}__w7")
                fh.write(json.dumps(res) + "\n")
                fh.flush()
                print(f"  age={w['age_days']:5.1f} seed={seed} "
                      f"loss={res['loss']:.5f} "
                      f"cohort={res['loss_cohort']:.5f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Run 7-day equal-weekday-balance cohort control on KuaiRand or RecFlow."
    )
    ap.add_argument("domain")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    main(a.domain, a.device)
