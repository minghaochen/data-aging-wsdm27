"""Paired early-stopping rule robustness check.

The baseline runs report test loss at the minimum over checkpoints evaluated on
the evaluation set, with no held-out validation split. The validation-stop runs
record, from the same optimization trajectory, the test loss under three
checkpoint rules:

    loss           test-argmin        (baseline convention)
    loss_valstop   argmin on a held-out 10% of the training subsample
    loss_fixed     a fixed step, no selection at all

Because the three come from the same trajectory they are paired, isolating
stopping-rule effects. We refit the age law and recompute the backtest
quantities under each rule and compare.

Usage: python src/valstop_analysis.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_SRC = str(Path(__file__).resolve().parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import fit_domains as FD

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
RULES = [("loss", "test-argmin (baseline)"),
         ("loss_valstop", "validation-argmin"),
         ("loss_fixed", "fixed step")]


def fit_under(rows, metric, nboot=200):
    rows = [r for r in rows if metric in r]
    if len(rows) < 8:
        return None
    base = FD.fit_joint(rows, metric=metric)
    ds = []
    for b in range(nboot):
        try:
            ds.append(FD.fit_joint(rows, seed=b, metric=metric)["delta"])
        except Exception:
            pass
    lo, hi = np.percentile(ds, [5, 95]) if ds else (np.nan, np.nan)
    return base, float(lo), float(hi)


def static_curves():
    print("=" * 78)
    print("STATIC AGE LAW UNDER THREE STOPPING RULES")
    out = {}
    for dom in ("recflow", "kuairand"):
        fp = RES / f"domain_{dom}__valstop.jsonl"
        if not fp.exists():
            print(f"  {dom}: not run yet")
            continue
        rows = [json.loads(l) for l in fp.read_text().splitlines() if l]
        rel = json.loads((RES / "delta_dictionary.json").read_text())[dom]
        print(f"\n{dom}  ({len(rows)} runs, val_frac="
              f"{rows[0].get('val_frac')}, n_val={rows[0].get('n_val'):,})")
        print(f"  {'rule':26s} {'delta':>8s} {'[90% CI]':>18s} {'R2':>6s} "
              f"{'mean loss':>10s}")
        print(f"  {'baseline dictionary':26s} {rel['delta']:+8.3f} "
              f"[{rel['delta_lo']:+7.3f},{rel['delta_hi']:+7.3f}] "
              f"{rel['r2']:6.3f}")
        rec = {}
        for metric, label in RULES:
            got = fit_under(rows, metric)
            if got is None:
                continue
            b, lo, hi = got
            ml = float(np.mean([r[metric] for r in rows if metric in r]))
            rec[metric] = dict(delta=b["delta"], lo=lo, hi=hi, r2=b["r2"],
                               mean_loss=ml)
            print(f"  {label:26s} {b['delta']:+8.3f} [{lo:+7.3f},{hi:+7.3f}] "
                  f"{b['r2']:6.3f} {ml:10.5f}")
        # The decisive question is not how big the optimism is but whether it
        # varies with AGE: a constant offset cannot change a slope, an
        # age-dependent one can. Also separate the volume effect: these runs
        # train on (1-val_frac)*D, so their test-argmin delta differs from
        # the full-volume one by volume alone.
        cells = defaultdict(list)
        for r in rows:
            if "loss_valstop" in r:
                cells[round(r["age_days"], 2)].append(
                    1000 * (r["loss_valstop"] - r["loss"]))
        ages = np.array(sorted(cells))
        opt = np.array([np.mean(cells[a]) for a in ages])
        sl = np.polyfit(np.log(ages), opt, 1)[0]
        rr = np.corrcoef(np.log(ages), opt)[0, 1]
        print(f"  optimism of test-argmin rule, per age cell (mnat): "
              f"{', '.join(f'{a:.1f}d:{o:+.2f}' for a, o in zip(ages, opt))}")
        print(f"  -> optimism vs ln(age): slope {sl:+.3f} mnat/e-fold, "
              f"r={rr:+.2f}  (a constant offset cannot bend a curve; this is "
              f"the part that can)")
        rec["optimism_slope_mnat_per_efold"] = float(sl)
        rec["optimism_r"] = float(rr)
        if "loss" in rec:
            dv = rec["loss"]["delta"] - rel["delta"]
            print(f"  -> volume: these runs train on {100*(1-rows[0]['val_frac']):.0f}% "
                  f"of D, which alone moves the test-argmin delta by "
                  f"{dv:+.3f} vs full-D {rel['delta']:+.3f}")
        if "loss" in rec and "loss_valstop" in rec:
            d0, d1 = rec["loss"]["delta"], rec["loss_valstop"]["delta"]
            w = rec["loss"]["hi"] - rec["loss"]["lo"]
            print(f"  -> validation-stop moves delta by {d1-d0:+.3f} "
                  f"({abs(d1-d0)/w:.2f} x its own seed CI); optimism of "
                  f"test-argmin = "
                  f"{1000*(rec['loss_valstop']['mean_loss']-rec['loss']['mean_loss']):.2f}"
                  f" mnat mean")
        out[dom] = rec
    return out


def backtest_curves():
    print("\n" + "=" * 78)
    print("BACKTEST UNDER THREE STOPPING RULES (volume-weighted serving loss)")
    fp = RES / "backtest_recflow__valstop.jsonl"
    if not fp.exists():
        print("  not run yet")
        return {}
    rows = [json.loads(l) for l in fp.read_text().splitlines() if l]
    per = defaultdict(list)
    for r in rows:
        per[r["P"]].append(r)
    out = {}
    hdr = "  P   " + "".join(f"{lab:>22s}" for _, lab in RULES)
    print(hdr)
    for P in sorted(per):
        rs = per[P]
        w = np.array([r["n_test"] for r in rs], float)
        line = f"  {P:3d} "
        rec = {}
        for metric, _ in RULES:
            if metric not in rs[0]:
                line += f"{'--':>22s}"
                continue
            v = np.array([r[metric] for r in rs])
            lb = float(np.sum(w * v) / np.sum(w))
            rec[metric] = lb
            line += f"{lb:22.5f}"
        out[P] = rec
        print(line)
    # the two quantities the paper reads off this curve
    print("\n  cadence effect (P=22 minus P=1), in mnat:")
    for metric, label in RULES:
        if all(metric in v for v in out.values()):
            ps = sorted(out)
            print(f"    {label:26s} "
                  f"{1000*(out[ps[-1]][metric]-out[ps[0]][metric]):+7.2f}")
    return out


def volume_control():
    print("\n" + "=" * 78)
    print("BACKTEST AT THE STATIC SWEEP'S TRAINING VOLUME")
    fp = RES / "backtest_recflow__d300k.jsonl"
    base = RES / "backtest_recflow.jsonl"
    if not fp.exists():
        print("  not run yet")
        return {}
    out = {}
    for tag, f in (("D=217,575 (default)", base), ("D=300,000", fp)):
        rows = [json.loads(l) for l in f.read_text().splitlines() if l]
        per = defaultdict(list)
        for r in rows:
            per[r["P"]].append(r)
        agg = {}
        for P, rs in per.items():
            w = np.array([r["n_test"] for r in rs], float)
            agg[P] = float(np.sum(w * np.array([r["loss"] for r in rs]))
                           / np.sum(w))
        out[tag] = agg
        print(f"  {tag:22s} " +
              " ".join(f"P{P}={agg[P]:.5f}" for P in sorted(agg)))
    a, b = out["D=217,575 (default)"], out["D=300,000"]
    common = sorted(set(a) & set(b))
    if common:
        d = [1000 * (a[P] - b[P]) for P in common]
        print(f"  volume effect (217.5k minus 300k), mnat: "
              + " ".join(f"P{P}={x:+.2f}" for P, x in zip(common, d)))
        print(f"  mean {np.mean(d):+.2f} mnat, spread across P "
              f"{max(d)-min(d):.2f} mnat")
        print("  (the paper attributes a +3.2 to +4.3 mnat offset between the "
              "static curve\n   and the backtest to period difficulty; this "
              "is how much of it is volume)")
    return out


def main():
    out = dict(static=static_curves(), backtest=backtest_curves(),
               volume=volume_control())
    (RES / "valstop_analysis.json").write_text(
        json.dumps(out, indent=2, default=str))
    print("\nsaved results/valstop_analysis.json")


if __name__ == "__main__":
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__.strip())
        sys.exit(0)
    main()
