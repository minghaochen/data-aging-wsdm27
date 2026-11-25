"""Reproduce the finite-change attribution, game-UGC redraw and cadence checks.
Run from any directory: python src/age_diagnostics.py
"""
import json
import sys
from pathlib import Path

import numpy as np

_SRC = str(Path(__file__).resolve().parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from backtest_analysis import a_curve, backtest_curve, fit_power
from cohort_seed_analysis import boot, tau_half_obs
from paper_numbers import AMZ, PANEL, RES, rows_of


def attribution():
    fits = json.loads((RES / "delta_dictionary.json").read_text())
    output = {}
    for domain in PANEL + AMZ:
        rows = [r for r in rows_of(domain) if r["exp"] == "A"]
        cells = []
        for age in sorted({r["age_days"] for r in rows}):
            cell = [r for r in rows if r["age_days"] == age]
            w, loss, warm_mass = np.mean(
                [(r["seen_frac"], r["loss"], r["seen_frac"] * r["loss_seen"])
                 for r in cell], axis=0)
            cells.append((w, loss, warm_mass / w, (loss - warm_mass) / (1 - w)))
        w, loss, warm, cold = np.asarray(cells).T
        mid_w = (w[:-1] + w[1:]) / 2
        gap = warm - cold
        terms = [np.sum(mid_w * np.diff(warm)),
                 np.sum((1 - mid_w) * np.diff(cold)),
                 np.sum((gap[:-1] + gap[1:]) / 2 * np.diff(w))]
        change = loss[-1] - loss[0]
        np.testing.assert_allclose(sum(terms), change, atol=1e-12)
        output[domain] = dict(warm=terms[0], cold=terms[1], coverage=terms[2],
                              change=change, cold_share=terms[1] / change,
                              included=bool(change > 0 and not fits[domain]["no_drift"]))
    return output


def game_curve(tag, n_seeds):
    rows = [json.loads(line) for line in (RES / tag).read_text().splitlines()]
    rows = [r for r in rows if r["seed"] < n_seeds]
    ages = np.array(sorted({r["age_days"] for r in rows}))
    cells = {a: [r["loss_cohort"] for r in rows if r["age_days"] == a] for a in ages}
    loss = np.array([np.mean(cells[a]) for a in ages])
    assert all(len(cells[a]) == n_seeds for a in ages) and np.all(loss <= 1)
    times = {}
    for n in range(4, len(ages) + 1):
        t = tau_half_obs(ages[:n], loss[:n])
        times[str(ages[n-1])] = float(t) if np.isfinite(t) else None
    finite = [v for v in times.values() if v is not None]
    return dict(n_cohort=rows[0]["n_cohort"], seeds=n_seeds,
                delta_mnat=dict(zip(map(str, ages), 1000*(loss-loss[0]))),
                tau=tau_half_obs(ages, loss), ci90=boot(cells, ages).tolist(),
                truncations=times, spread=max(finite)/min(finite),
                omit_freshest=tau_half_obs(ages[1:], loss[1:]))


def cadence():
    periods, loss, auc, _ = backtest_curve("recflow", "__d300k")
    _, b, delta = fit_power(*a_curve("recflow"))
    grid = np.geomspace(periods.min(), periods.max(), 400)
    interp = np.interp(np.log(grid), np.log(periods), loss)
    rows = []
    for price in (1e-3, 10**(-2.5), 1e-2):
        rows.append(dict(price=price, static=(price*(1+delta)/(b*delta))**(1/(1+delta)),
                         observed=int(periods[np.argmin(loss+price/periods)]),
                         interpolated=grid[np.argmin(interp+price/grid)]))
    return dict(optima=rows, periods=periods.tolist(), auc=auc.tolist())


if __name__ == "__main__":
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__.strip())
        sys.exit(0)
    cold = attribution()
    shares = [v["cold_share"] for v in cold.values() if v["included"]]
    base = "domain_gameugc_finish__icohort"
    results = dict(attribution=cold, cold_summary=dict(n=len(shares),
                   at_least_half=sum(bool(s >= 0.5) for s in shares),
                   median=float(np.median(shares))),
                   game_draw0=game_curve(base+".jsonl", 9),
                   game_draw0_matched=game_curve(base+".jsonl", 3),
                   game_draw1=game_curve(base+"__ss1.jsonl", 3),
                   cadence=cadence())
    (RES / "age_diagnostics.json").write_text(json.dumps(results, indent=2, allow_nan=False)+"\n")
