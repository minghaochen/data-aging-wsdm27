"""Offline verifier & summary reporter for the extended diagnostic records.

Runs strictly from `results/` alone (no prepared `.npz` required) to audit all
33 extended-study JSONLs (1,385 runs) and the five `stage*_summary.json` files:
  - 4-way partition identities (11/10/01/00) across all runs
  - strict zero serving/test overlap on the 3 non-overlapping rolling backtests
  - exact omega parity between SideFeatMF and ID-only MF
  - consistency with the recorded Stage 1-5 summary JSONs

Usage: python src/camera_ready_reproduce.py
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
GROUPS = ("11", "10", "01", "00")


def _check_finite(obj, path=""):
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise AssertionError(f"Non-finite float at {path}: {obj}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _check_finite(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _check_finite(v, f"{path}[{i}]")


def _audit_jsonl(path, expected_n=None):
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if expected_n is not None:
        assert len(rows) == expected_n, f"{path.name}: expected {expected_n}, got {len(rows)}"
    max_om_err = 0.0
    max_part_err = 0.0
    for idx, r in enumerate(rows):
        _check_finite(r, f"{path.name}[{idx}]")
        if "omega_11" in r:
            om_sum = sum(float(r[f"omega_{g}"]) for g in GROUPS)
            max_om_err = max(max_om_err, abs(om_sum - 1.0))
            assert abs(om_sum - 1.0) < 1e-10
            part = sum(
                (
                    float(r[f"omega_{g}"]) * float(r[f"loss_{g}"])
                    if r[f"omega_{g}"] > 0 and r[f"loss_{g}"] is not None
                    else 0.0
                )
                for g in GROUPS
            )
            pe = abs(part - float(r["loss"]))
            max_part_err = max(max_part_err, pe)
            assert pe < 1e-6
            assert abs(float(r["seen_frac"]) - float(r["omega_11"])) < 1e-12
    return len(rows), max_om_err, max_part_err


def main():
    s1 = json.loads((RES / "stage1_zero_run.json").read_text())
    s2 = json.loads((RES / "stage2_fourway_summary.json").read_text())
    s3 = json.loads((RES / "stage3_backtest_summary.json").read_text())
    s4 = json.loads((RES / "stage4_side_features_summary.json").read_text())
    s5 = json.loads((RES / "stage5_gameugc_summary.json").read_text())
    for tag, obj in (("stage1", s1), ("stage2", s2), ("stage3", s3), ("stage4", s4), ("stage5", s5)):
        _check_finite(obj, tag)

    fw_files = sorted(RES.glob("fourway_*.jsonl"))
    assert len(fw_files) == 23
    n_fw = sum(_audit_jsonl(p)[0] for p in fw_files)
    assert n_fw == 522

    bt_exp = {
        "backtest_kuairand_strict.jsonl": 90,
        "backtest_recflow_nooverlap19d.jsonl": 80,
        "backtest_gameugc_strict.jsonl": 126,
    }
    n_bt = 0
    for fn, exp_n in bt_exp.items():
        fp = RES / fn
        n_bt += _audit_jsonl(fp, expected_n=exp_n)[0]
        rows = [json.loads(l) for l in fp.read_text().splitlines() if l.strip()]
        assert max(r["t_serve_end"] - r["static_test_t0"] for r in rows) <= 0
    assert n_bt == 296

    sf_exp = {
        "sidefeat_kuairand.jsonl": 36,
        "sidefeat_ml25m.jsonl": 42,
        "sidefeat_recflow.jsonl": 36,
    }
    n_sf = 0
    max_om_diff = 0.0
    for fn, exp_n in sf_exp.items():
        fp = RES / fn
        n_sf += _audit_jsonl(fp, expected_n=exp_n)[0]
        dom = fn.replace("sidefeat_", "").replace(".jsonl", "")
        sf_rows = [json.loads(l) for l in fp.read_text().splitlines() if l.strip()]
        fw_rows = [json.loads(l) for l in (RES / f"fourway_{dom}.jsonl").read_text().splitlines() if l.strip()]
        fw_map = {(r["exp"], int(r["train_t0"]), int(r["train_t1"]), int(r["seed"])): r for r in fw_rows}
        for r in sf_rows:
            m = fw_map[(r["exp"], int(r["train_t0"]), int(r["train_t1"]), int(r["seed"]))]
            for g in GROUPS:
                max_om_diff = max(max_om_diff, abs(r[f"omega_{g}"] - m[f"omega_{g}"]))
    assert n_sf == 114 and max_om_diff < 1e-12

    g5_exp = {
        "gameugc_finish__icohort_draws.jsonl": 231,
        "gameugc_finish__valstop.jsonl": 60,
        "gameugc_finish__icohort_valstop.jsonl": 66,
        "gameugc_finish__icohort_finegrid.jsonl": 96,
    }
    n_g5 = sum(_audit_jsonl(RES / fn, expected_n=exp_n)[0] for fn, exp_n in g5_exp.items())
    assert n_g5 == 453

    total_cr = n_fw + n_bt + n_sf + n_g5
    assert total_cr == 1385

    agg18 = s2["fourway_attribution"]["aggregate_included_18"]["raw"]
    fw_sh = agg18["fourway_share_of_delta_L"]
    ci_sh = agg18["cold_internal_share_of_delta_L_cold"]
    ci_dl = agg18["cold_internal_weighted_share_of_delta_L"]
    print(f"Verified 33 extended-study JSONLs: {total_cr} runs (4-way={n_fw}, backtests={n_bt}, sidefeat={n_sf}, gameugc={n_g5}).")
    print(
        "4-way medians (18 domains): "
        f"within11={fw_sh['within']['11']['median']:.3f}, "
        f"within01={fw_sh['within']['01']['median']:.3f}, "
        f"within10={fw_sh['within']['10']['median']:.3f}, "
        f"within00={fw_sh['within']['00']['median']:.3f}, "
        f"cov={fw_sh['total_coverage']['median']:.3f}; "
        f"cold_within01={ci_sh['within_01']['median']:.3f}, "
        f"cold_mix={ci_sh['mix_within_cold']['median']:.3f} ({ci_dl['mix_within_cold']['median']:.3f} of dL)"
    )
    print(f"Side-feature omega parity max diff: {max_om_diff:.2e}")
    for name in s3["series_order"]:
        sr = s3["series"][name]
        print(
            f"Backtest [{name}]: zero_overlap={sr['zero_overlap_strict_ok']} "
            f"loss_sr={sr['loss']['polyfit_slope_ratio']:.3f} "
            f"auc_sr={sr['one_minus_auc']['polyfit_slope_ratio']:.3f}"
        )


if __name__ == "__main__":
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__.strip())
        sys.exit(0)
    main()
