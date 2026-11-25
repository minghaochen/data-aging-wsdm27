"""Recency half-life sensitivity sweep vs from-scratch retraining.

The default recency baseline fixed the exponential half-life at the
training-window length (2 d). Here the half-life is swept over {1, 2, 4, 8} d
on the same serving schedule and scored against the from-scratch curve.

Usage: python src/recency_halflife_analysis.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

# half-life (days) -> results file; 2 d matches the default W=2 d window
ARMS = {2.0: "backtest_recflow__recency.jsonl",
        1.0: "backtest_recflow__recency_hl1.jsonl",
        4.0: "backtest_recflow__recency_hl4.jsonl",
        8.0: "backtest_recflow__recency_hl8.jsonl"}


def agg(fp, metric):
    if not fp.exists():
        return None
    rows = [json.loads(line) for line in fp.read_text().splitlines() if line]
    per = defaultdict(list)
    for r in rows:
        if metric in r:
            per[r["P"]].append(r)
    out = {}
    for P, rs in per.items():
        w = np.array([r["n_test"] for r in rs], float)
        out[P] = float(np.sum(w * np.array([r[metric] for r in rs]))
                       / np.sum(w))
    return out or None


def main():
    base_l = agg(RES / "backtest_recflow.jsonl", "loss")
    base_a = agg(RES / "backtest_recflow.jsonl", "auc")
    Ps = sorted(base_l)
    print("=" * 78)
    print("RECENCY HALF-LIFE SWEEP (RecFlow, D=217,575, vs from-scratch)")
    print(f"{'half-life':>10s} " + "".join(f"{'P=' + str(P):>10s}"
                                           for P in Ps)
          + f"{'AUC@P1':>9s} {'AUC slope':>10s}")
    print(f"{'from-scratch':>10s} "
          + "".join(f"{1000*base_l[P]:10.2f}" for P in Ps)
          + f"{100*base_a[1]:9.2f} "
          + f"{100*(base_a[1]-base_a[Ps[-1]]):10.2f}")
    print("  (loss in mnat, absolute; AUC slope = points lost P=1 -> P=22)")
    print()
    rows_out = {}
    for hl in sorted(ARMS):
        L = agg(RES / ARMS[hl], "loss")
        A = agg(RES / ARMS[hl], "auc")
        if L is None:
            print(f"{hl:10.0f} d  not run yet")
            continue
        d = {P: 1000 * (L[P] - base_l[P]) for P in Ps if P in L}
        slope = 100 * (A[1] - A[Ps[-1]]) if 1 in A and Ps[-1] in A else np.nan
        tag = " (default 2 d)" if hl == 2.0 else ""
        print(f"{hl:8.0f} d  " + "".join(f"{d.get(P, np.nan):+10.2f}"
                                         for P in Ps)
              + f"{100*(A[1]-base_a[1]):+9.2f} {slope:10.2f}{tag}")
        rows_out[hl] = dict(loss_delta_mnat=d,
                            auc_delta_p1=100 * (A[1] - base_a[1]),
                            auc_slope=float(slope),
                            beats_scratch_on_loss=[P for P in d if d[P] < 0],
                            beats_scratch_on_auc=[P for P in Ps
                                                  if P in A and A[P] > base_a[P]])
    print("  (loss numbers are recency MINUS from-scratch: positive = worse)")

    print()
    any_win = {hl: v["beats_scratch_on_loss"] for hl, v in rows_out.items()
               if v["beats_scratch_on_loss"]}
    any_auc = {hl: v["beats_scratch_on_auc"] for hl, v in rows_out.items()
               if v["beats_scratch_on_auc"]}
    print(f"  half-lives beating from-scratch on loss at any cadence: "
          f"{any_win if any_win else 'none'}")
    print(f"  half-lives beating from-scratch on AUC at any cadence:  "
          f"{any_auc if any_auc else 'none'}")
    if rows_out:
        best = min(rows_out, key=lambda h: min(rows_out[h]
                                               ["loss_delta_mnat"].values()))
        print(f"  best half-life by smallest loss gap: {best:.0f} d "
              f"(default was 2 d)")
    (RES / "recency_halflife.json").write_text(
        json.dumps(rows_out, indent=2, default=str))
    print("\nsaved results/recency_halflife.json")


if __name__ == "__main__":
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__.strip())
        sys.exit(0)
    main()
