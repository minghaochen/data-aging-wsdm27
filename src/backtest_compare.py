"""Retrain-from-scratch versus two standard engineering alternatives.

The age law prices staleness, but it does not by itself show that retraining
more often is the right lever. This compares three policies on the identical
serving schedule (same periods, same schedule, and same subsample size D_fix):

  scratch  -- retrain from random init on the last W days (steps=8000, lr=3e-3)
  recency  -- retrain from random init on the last 4W days, with sample
              weights halving every W days (default) at full budget
  warm     -- retrain on the last W days warm-started from the previous
              model (reduced budget: steps//8, lr=1e-4 to avoid divergence)

Reported per policy: volume-weighted serving loss and AUC against the
retraining period, plus the loss and AUC gap to daily retraining, which is
the quantity a cadence decision actually trades against compute.

Usage: python src/backtest_compare.py <domain>
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

MODES = [("scratch", ""), ("recency", "__recency"), ("warm", "__warm")]


def curve(domain, suffix):
    fp = RES / f"backtest_{domain}{suffix}.jsonl"
    if not fp.exists():
        return None
    rows = [json.loads(l) for l in fp.read_text().splitlines() if l]
    agg = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    for r in rows:
        a = agg[r["P"]]
        a[0] += r["loss"] * r["n_test"]
        a[1] += r["n_test"]
        a[2] += r["auc"] * r["n_test"]
        a[3] += 1
    Ps = np.array(sorted(agg))
    return dict(P=Ps,
                loss=np.array([agg[p][0] / agg[p][1] for p in Ps]),
                auc=np.array([agg[p][2] / agg[p][1] for p in Ps]),
                n=np.array([agg[p][3] for p in Ps]))


def main(domain):
    got = {name: curve(domain, sfx) for name, sfx in MODES}
    got = {k: v for k, v in got.items() if v is not None}
    if "scratch" not in got:
        print("no from-scratch backtest found")
        return

    print(f"{domain}: serving loss / AUC by retraining period\n")
    hdr = "  P  " + "".join(f"{k:>22s}" for k in got)
    print(hdr)
    print("     " + "".join(f"{'loss':>11s}{'AUC':>11s}" for _ in got))
    for i, P in enumerate(got["scratch"]["P"]):
        line = f"{int(P):3d}  "
        for k, v in got.items():
            j = np.where(v["P"] == P)[0]
            if len(j) == 0:
                line += f"{'-':>11s}{'-':>11s}"
            else:
                line += f"{v['loss'][j[0]]:11.5f}{v['auc'][j[0]]:11.5f}"
        print(line)

    print("\nstaleness cost relative to daily retraining "
          "(P=1 of the same policy):")
    print(f"{'policy':10s} {'loss @Pmax - @P1':>18s} {'AUC @P1 - @Pmax':>18s} "
          f"{'loss vs scratch@P1':>20s}")
    base = got["scratch"]
    for k, v in got.items():
        dl = 1000 * (v["loss"][-1] - v["loss"][0])
        da = 100 * (v["auc"][0] - v["auc"][-1])
        rel = 1000 * (v["loss"][0] - base["loss"][0])
        print(f"{k:10s} {dl:15.1f} mnat {da:15.2f} pts {rel:+17.1f} mnat")

    print("\nRead: a policy helps only if it lowers the loss at a GIVEN "
          "period,\nnot merely if its own P=1 baseline is different.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for k, v in got.items():
        axes[0].plot(v["P"], v["loss"], "o-", label=k)
        axes[1].plot(v["P"], v["auc"], "o-", label=k)
    for ax, lab in ((axes[0], "volume-weighted serving loss"),
                    (axes[1], "volume-weighted serving AUC")):
        ax.set_xscale("log")
        ax.set_xlabel("retraining period P (days)")
        ax.set_ylabel(lab)
        ax.legend(fontsize=8)
    axes[0].set_title(f"{domain}: cadence vs training policy")
    axes[1].set_title("ranking quality")
    fig.tight_layout()
    fig.savefig(RES / f"backtest_{domain}_policies.png", dpi=150)
    print(f"\nsaved backtest_{domain}_policies.png")


if __name__ == "__main__":
    args = sys.argv[1:]
    if any(a in ("-h", "--help") for a in args):
        print(__doc__.strip())
        sys.exit(0)
    if not args:
        raise SystemExit("Usage: python src/backtest_compare.py <domain>")
    main(args[0])
