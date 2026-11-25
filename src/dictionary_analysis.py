"""Coverage-elasticity dictionary across domains.

Finding: the shape exponent delta of the data-age law is NOT predicted by
naive catalog stats (popularity churn, new-item share) because the age
penalty is a mixture of two mechanisms:

  (1) coverage decay   - test entities absent from older train windows
                         (saturating; pushes delta down/negative)
  (2) preference drift - popularity/taste change among covered entities
                         (accumulating; pushes delta up)

The mixture weight is captured by a training-free statistic: the elasticity
of warm coverage seen_frac w.r.t. window age (slope of ln seen_frac vs
ln age). Domains where coverage barely decays expose pure drift (delta>0,
accelerating); domains where coverage collapses expose the saturating
coverage term (delta<=0).

Markers in LABELS: * round-1 Amazon hold-out; ** round-2 Amazon hold-out;
dagger cross-platform generalization domain.

Usage: python src/dictionary_analysis.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

LABELS = {
    "ml25m": "MovieLens",
    "kuairand": "KuaiRand",
    "amz_Video_Games": "AMZ Games",
    "amz_CDs_and_Vinyl": "AMZ CDs",
    "amz_Software": "AMZ Software",
    "amz_Appliances": "AMZ Appliances",
    "amz_Baby_Products": "AMZ Baby",
    "amz_Toys_and_Games": "AMZ Toys*",
    "amz_Office_Products": "AMZ Office*",
    "amz_Pet_Supplies": "AMZ Pet*",
    "epinions": "Epinions†",
    "beer": "BeerAdvocate†",
    "recflow": "RecFlow†",
    "kuairec": "KuaiRec†",
    "yelp": "Yelp†",
    "netflix": "Netflix†",
    "steam": "Steam†",
    "amz_Musical_Instruments": "AMZ Instruments**",
    "amz_Arts_Crafts_and_Sewing": "AMZ Arts**",
    "amz_Automotive": "AMZ Auto**",
    "amz_Kindle_Store": "AMZ Kindle**",
    "gameugc_play": "Game-UGC play",
    "gameugc_finish": "Game-UGC finish",
}
HOLDOUT = {"amz_Toys_and_Games", "amz_Office_Products", "amz_Pet_Supplies",
           "amz_Musical_Instruments", "amz_Arts_Crafts_and_Sewing",
           "amz_Automotive", "amz_Kindle_Store"}


def coverage_elasticity(dom):
    fp = RES / f"domain_{dom}.jsonl"
    rows = [json.loads(l) for l in fp.read_text().splitlines() if l]
    cells = defaultdict(list)
    for r in rows:
        if r["exp"] == "A":   # translated windows: clean coverage-vs-age
            cells[round(r["age_days"], 2)].append(r["seen_frac"])
    ages = np.array(sorted(cells))
    sf = np.array([np.mean(cells[a]) for a in ages])
    m = sf > 1e-4
    if m.sum() < 3:
        return np.nan, ages, sf
    slope = np.polyfit(np.log(ages[m]), np.log(sf[m]), 1)[0]
    return slope, ages, sf


def main():
    dd = json.loads((RES / "delta_dictionary.json").read_text())
    ds = json.loads((RES / "domain_stats.json").read_text())
    doms = [d for d in dd if d in ds]

    rows = []
    curves = {}
    for d in doms:
        e_cov, ages, sf = coverage_elasticity(d)
        curves[d] = (ages, sf)
        rows.append(dict(
            dom=d, delta=dd[d]["delta"], dlo=dd[d]["delta_lo"],
            dhi=dd[d]["delta_hi"], r2=dd[d]["r2"], beta=dd[d]["beta"],
            b_log=dd[d]["b_log"], e_cov=e_cov,
            slope=ds[d]["rho_decay_slope"],
            new_share=ds[d]["new_item_share"],
            no_drift=dd[d]["beta"] < 1e-6 or dd[d]["r2"] < 0.2))
    rows.sort(key=lambda r: -r["delta"])

    print(f"{'domain':26s} {'delta [90% CI]':>22s} {'R2':>6s} "
          f"{'e_cov':>7s} {'rho_slp':>8s} {'new%':>6s}")
    for r in rows:
        flag = " (no drift: beta~0)" if r["no_drift"] else ""
        print(f"{r['dom']:26s} {r['delta']:6.2f} "
              f"[{r['dlo']:5.2f},{r['dhi']:5.2f}]  {r['r2']:6.2f} "
              f"{r['e_cov']:7.2f} {r['slope']:8.3f} "
              f"{r['new_share']*100:5.1f}%{flag}")

    use = [r for r in rows if not r["no_drift"]]
    ins = [r for r in use if r["dom"] not in HOLDOUT]
    x = np.array([r["e_cov"] for r in ins])
    y = np.array([r["delta"] for r in ins])
    sp = stats.spearmanr(x, y)
    pe = stats.pearsonr(x, y)
    print(f"\ndelta vs e_cov IN-SAMPLE (n={len(ins)}): "
          f"spearman={sp.statistic:+.3f} (p={sp.pvalue:.4f})  "
          f"pearson={pe.statistic:+.3f} (p={pe.pvalue:.4f})")
    xa = np.array([r["e_cov"] for r in use])
    ya = np.array([r["delta"] for r in use])
    spa = stats.spearmanr(xa, ya)
    pea = stats.pearsonr(xa, ya)
    print(f"delta vs e_cov ALL DRIFT DOMAINS (n={len(use)}, incl. hold-out): "
          f"spearman={spa.statistic:+.3f} (p={spa.pvalue:.4f})  "
          f"pearson={pea.statistic:+.3f} (p={pea.pvalue:.4f})")
    for alt in ("slope", "new_share"):
        xalt = np.array([r[alt] for r in use])
        s2 = stats.spearmanr(xalt, ya)
        print(f"delta vs {alt:>10s}: spearman={s2.statistic:+.3f} "
              f"(p={s2.pvalue:.4f})   [naive predictor, all domains]")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax = axes[0]
    for r in rows:
        hold = r["dom"] in HOLDOUT
        color = "C3" if r["no_drift"] else ("C1" if hold else "C0")
        ax.errorbar([r["e_cov"]], [r["delta"]],
                    yerr=[[max(0, r["delta"] - r["dlo"])],
                          [max(0, r["dhi"] - r["delta"])]],
                    fmt="^" if hold else "o", ms=8 if hold else 7,
                    capsize=3, color=color)
        ax.annotate(LABELS.get(r["dom"], r["dom"]),
                    (r["e_cov"], r["delta"]), fontsize=8,
                    textcoords="offset points", xytext=(7, 4))
    b, a = np.polyfit(x, y, 1)
    xg = np.linspace(xa.min() - 0.1, xa.max() + 0.1, 50)
    ax.plot(xg, a + b * xg, "--", color="gray", lw=1,
            label=(f"in-sample line: δ = {a:.2f} + {b:.2f}·e_cov "
                   f"(ρ={sp.statistic:.2f})\n"
                   f"all domains incl. hold-out (▲): "
                   f"ρ={spa.statistic:.2f} (p={spa.pvalue:.3f})"))
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("coverage elasticity  d ln(seen_frac) / d ln(age)")
    ax.set_ylabel("drift shape exponent δ (Box-Cox)")
    ax.set_title("Blind test: hold-outs (▲) fall below the in-sample line")
    ax.legend(fontsize=8)

    ax = axes[1]
    for r in rows:
        ages, sf = curves[r["dom"]]
        m = sf > 1e-4
        ax.loglog(ages[m], sf[m], "o-", ms=4, lw=1,
                  label=LABELS.get(r["dom"], r["dom"]))
    ax.set_xlabel("train-window mean age (days)")
    ax.set_ylabel("warm coverage (seen_frac)")
    ax.set_title("Coverage decay curves (training-free)")
    ax.legend(fontsize=7)
    fig.suptitle("The drift dictionary: two mechanisms behind the data-age "
                 "scaling law")
    fig.tight_layout()
    fig.savefig(RES / "delta_dictionary.png", dpi=150)
    print(f"saved {RES / 'delta_dictionary.png'}")


if __name__ == "__main__":
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__.strip())
        sys.exit(0)
    main()
