"""Does the from-scratch-vs-warm loss ranking survive a held-out
stopping rule?

The evaluation-set stopping rule is asymmetric between the two policies:
at the full budget a from-scratch model offers 160 candidate checkpoints and
a warm-started one offers 20, and a minimum over more draws is lower.

Choosing the checkpoint on a held-out slice of the training subsample
removes the exploitable asymmetry for both policies at once. This script
reports all three policies under both stopping rules.

Usage: python src/policy_valstop_analysis.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

# the from-scratch backtest tags its variant with "__valstop" (backtest.py
# takes a free-form tag) while the policy runs tag with "_valstop"
# (backtest_baselines.py appends to an existing "__mode" separator)
POLICIES = [("scratch", "backtest_recflow.jsonl",
             "backtest_recflow__valstop.jsonl"),
            ("recency", "backtest_recflow__recency.jsonl",
             "backtest_recflow__recency_valstop.jsonl"),
            ("warm", "backtest_recflow__warm.jsonl",
             "backtest_recflow__warm_valstop.jsonl")]


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


def table(which, loss_key, auc_key, label):
    print(f"\n{label}")
    curves = {}
    for name, base_f, val_f in POLICIES:
        fp = RES / (base_f if which == "baseline" else val_f)
        L = agg(fp, loss_key)
        A = agg(fp, auc_key)
        if L:
            curves[name] = (L, A)
    if "scratch" not in curves:
        print("  scratch run missing under this rule")
        return {}
    Ps = sorted(curves["scratch"][0])
    print(f"  {'P':>3s} " + "".join(f"{n:>22s}" for n in curves))
    for P in Ps:
        line = f"  {P:3d} "
        for n, (L, A) in curves.items():
            line += (f"{L[P]:12.5f}/{100*A[P]:6.2f}" if P in L
                     else f"{'--':>22s}")
        line += "   (loss / AUC%)"
        print(line)
    base = curves["scratch"][0]
    print("  vs from-scratch, mnat of loss (positive = worse than scratch):")
    for n, (L, A) in curves.items():
        if n == "scratch":
            continue
        d = {P: 1000 * (L[P] - base[P]) for P in Ps if P in L}
        warm = [v for P, v in d.items() if P != 22]
        print(f"    {n:10s} " + " ".join(f"P{P}={v:+.1f}" for P, v in d.items())
              + (f"   [range over cadences with a warm start: "
                 f"{min(warm):+.1f} to {max(warm):+.1f}]" if n == "warm"
                 and warm else ""))
    print("  vs from-scratch, AUC points at daily cadence:")
    for n, (L, A) in curves.items():
        if n == "scratch" or A is None:
            continue
        print(f"    {n:10s} {100*(A[1]-curves['scratch'][1][1]):+.2f}")
    return curves


def main():
    print("=" * 78)
    print("POLICY COMPARISON UNDER TWO STOPPING RULES (RecFlow, D=217,575)")
    a = table("baseline", "loss", "auc",
              "[A] test-argmin rule: checkpoint chosen on the serving events")
    b = table("valstop", "loss_valstop", "auc_valstop",
              "[B] held-out rule: checkpoint chosen on 10% of the training "
              "subsample")
    # does the ranking flip?
    print("\n" + "-" * 78)
    for label, c in (("test-argmin rule", a), ("held-out rule", b)):
        if not c or "scratch" not in c:
            continue
        Ps = sorted(c["scratch"][0])
        wins = {n: sum(1 for P in Ps
                       if n in c and P in c[n][0]
                       and c[n][0][P] < c["scratch"][0][P])
                for n in c if n != "scratch"}
        print(f"  {label}: cadences where a policy beats from-scratch on "
              f"loss -> {wins} (of {len(Ps)})")
    (RES / "policy_valstop.json").write_text(json.dumps(
        {"baseline": {k: v[0] for k, v in a.items()},
         "valstop": {k: v[0] for k, v in b.items()}}, indent=2))
    print("\nsaved results/policy_valstop.json")


if __name__ == "__main__":
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__.strip())
        sys.exit(0)
    main()
