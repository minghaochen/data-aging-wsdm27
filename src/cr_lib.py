"""Shared utilities, constants, model probes, and preloaded runner.

Strictly read-only with respect to existing `results/` outputs.
PyTorch is imported lazily via _require_torch() so offline summary/help modes
only require NumPy. No matplotlib imports.
"""
import os

for _k in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_k] = "2"

import time
from collections import defaultdict
from pathlib import Path

import numpy as np

torch = None
nn = None
_MF_CLS = None
_MLP_CLS = None


def _require_torch():
    """Import PyTorch lazily and build probe model classes once."""
    global torch, nn, _MF_CLS, _MLP_CLS
    if torch is not None:
        return torch, nn
    import torch as _torch
    import torch.nn as _nn

    _torch.set_num_threads(2)

    class MF(_nn.Module):
        """Biased matrix factorization probe with sparse embeddings."""

        def __init__(self, n_users, n_items, dim):
            super().__init__()
            self.pu = _nn.Embedding(n_users, dim, sparse=True)
            self.qi = _nn.Embedding(n_items, dim, sparse=True)
            self.bu = _nn.Embedding(n_users, 1, sparse=True)
            self.bi = _nn.Embedding(n_items, 1, sparse=True)
            self.mu = _nn.Parameter(_torch.zeros(1))
            _nn.init.normal_(self.pu.weight, std=0.01)
            _nn.init.normal_(self.qi.weight, std=0.01)
            _nn.init.zeros_(self.bu.weight)
            _nn.init.zeros_(self.bi.weight)

        def forward(self, u, i):
            return (
                (self.pu(u) * self.qi(i)).sum(-1)
                + self.bu(u).squeeze(-1)
                + self.bi(i).squeeze(-1)
                + self.mu
            )

    class MLP(_nn.Module):
        """NeuMF-lite two-tower probe with an MLP interaction head."""

        def __init__(self, n_users, n_items, dim):
            super().__init__()
            self.pu = _nn.Embedding(n_users, dim, sparse=True)
            self.qi = _nn.Embedding(n_items, dim, sparse=True)
            self.bu = _nn.Embedding(n_users, 1, sparse=True)
            self.bi = _nn.Embedding(n_items, 1, sparse=True)
            self.mu = _nn.Parameter(_torch.zeros(1))
            self.head = _nn.Sequential(
                _nn.Linear(3 * dim, 64), _nn.ReLU(), _nn.Linear(64, 1)
            )
            _nn.init.normal_(self.pu.weight, std=0.01)
            _nn.init.normal_(self.qi.weight, std=0.01)
            _nn.init.zeros_(self.bu.weight)
            _nn.init.zeros_(self.bi.weight)

        def forward(self, u, i):
            pu, qi = self.pu(u), self.qi(i)
            h = _torch.cat([pu, qi, pu * qi], -1)
            return (
                self.head(h).squeeze(-1)
                + self.bu(u).squeeze(-1)
                + self.bi(i).squeeze(-1)
                + self.mu
            )

    torch = _torch
    nn = _nn
    _MF_CLS = MF
    _MLP_CLS = MLP
    return torch, nn


def __getattr__(name):
    if name in ("MF", "MLP"):
        _require_torch()
        return _MF_CLS if name == "MF" else _MLP_CLS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

CR_ROOT = Path(__file__).resolve().parent.parent
CR_SRC = CR_ROOT / "src"
CR_RES = CR_ROOT / "results"
SCALING_ROOT = CR_ROOT
DATA_DIR = CR_ROOT / "data"
REV_RES = CR_ROOT / "results"

DAY = 86400
MONTH = 30 * DAY
D_MIN = 25_000
D_CAP = 300_000
SEEDS = [0, 1, 2]

DOMAINS = {
    "ml25m": dict(
        npz="ml25m.npz",
        unit=MONTH,
        test=6,
        wA=24,
        gaps=[0, 3, 6, 12, 24, 48, 96, 144],
        spans=[8, 16, 32, 64, 128, 250],
    ),
    "kuairand": dict(
        npz="kuairand.npz",
        unit=DAY,
        test=3,
        wA=2,
        gaps=[0, 1, 2, 4, 8, 16, 24],
        spans=[2, 4, 8, 16, 26],
        eval_every=50,
    ),
    "amz_Video_Games": dict(
        npz="amz_Video_Games.npz",
        unit=MONTH,
        test=6,
        wA=12,
        gaps=[0, 6, 12, 24, 48, 96, 192],
        spans=[6, 12, 24, 48, 96, 192, 320],
    ),
    "amz_CDs_and_Vinyl": dict(
        npz="amz_CDs_and_Vinyl.npz",
        unit=MONTH,
        test=6,
        wA=12,
        gaps=[0, 6, 12, 24, 48, 96, 192],
        spans=[6, 12, 24, 48, 96, 192, 320],
    ),
    "amz_Software": dict(
        npz="amz_Software.npz",
        unit=MONTH,
        test=6,
        wA=12,
        gaps=[0, 6, 12, 24, 48, 96, 192],
        spans=[6, 12, 24, 48, 96, 192, 320],
    ),
    "amz_Appliances": dict(
        npz="amz_Appliances.npz",
        unit=MONTH,
        test=6,
        wA=12,
        gaps=[0, 6, 12, 24, 48, 96, 192],
        spans=[6, 12, 24, 48, 96, 192, 320],
    ),
    "amz_Baby_Products": dict(
        npz="amz_Baby_Products.npz",
        unit=MONTH,
        test=6,
        wA=12,
        gaps=[0, 6, 12, 24, 48, 96, 192],
        spans=[6, 12, 24, 48, 96, 192, 320],
    ),
    "amz_Toys_and_Games": dict(
        npz="amz_Toys_and_Games.npz",
        unit=MONTH,
        test=6,
        wA=12,
        gaps=[0, 6, 12, 24, 48, 96, 192],
        spans=[6, 12, 24, 48, 96, 192, 320],
    ),
    "amz_Office_Products": dict(
        npz="amz_Office_Products.npz",
        unit=MONTH,
        test=6,
        wA=12,
        gaps=[0, 6, 12, 24, 48, 96, 192],
        spans=[6, 12, 24, 48, 96, 192, 320],
    ),
    "amz_Pet_Supplies": dict(
        npz="amz_Pet_Supplies.npz",
        unit=MONTH,
        test=6,
        wA=12,
        gaps=[0, 6, 12, 24, 48, 96, 192],
        spans=[6, 12, 24, 48, 96, 192, 320],
    ),
    "amz_Musical_Instruments": dict(
        npz="amz_Musical_Instruments.npz",
        unit=MONTH,
        test=6,
        wA=12,
        gaps=[0, 6, 12, 24, 48, 96, 192],
        spans=[6, 12, 24, 48, 96, 192, 320],
    ),
    "amz_Arts_Crafts_and_Sewing": dict(
        npz="amz_Arts_Crafts_and_Sewing.npz",
        unit=MONTH,
        test=6,
        wA=12,
        gaps=[0, 6, 12, 24, 48, 96, 192],
        spans=[6, 12, 24, 48, 96, 192, 320],
    ),
    "amz_Automotive": dict(
        npz="amz_Automotive.npz",
        unit=MONTH,
        test=6,
        wA=12,
        gaps=[0, 6, 12, 24, 48, 96, 192],
        spans=[6, 12, 24, 48, 96, 192, 320],
    ),
    "amz_Kindle_Store": dict(
        npz="amz_Kindle_Store.npz",
        unit=MONTH,
        test=6,
        wA=12,
        gaps=[0, 6, 12, 24, 48, 96, 192],
        spans=[6, 12, 24, 48, 96, 192, 320],
    ),
    "recflow": dict(
        npz="recflow.npz",
        unit=DAY,
        test=3,
        wA=2,
        gaps=[0, 1, 2, 4, 8, 16, 24],
        spans=[2, 4, 8, 16, 26],
        eval_every=50,
    ),
    "kuairec": dict(
        npz="kuairec.npz",
        unit=DAY,
        test=5,
        wA=3,
        gaps=[0, 2, 4, 8, 16, 32, 64],
        spans=[3, 6, 12, 24, 48, 74],
        eval_every=50,
    ),
    "gameugc_play": dict(
        npz="gameugc_play.npz",
        unit=DAY,
        test=3,
        wA=2,
        gaps=[0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
        spans=[2, 4, 8, 16, 32, 64, 128, 256, 512],
        eval_every=50,
    ),
    "gameugc_finish": dict(
        npz="gameugc_finish.npz",
        unit=DAY,
        test=3,
        wA=2,
        gaps=[0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
        spans=[2, 4, 8, 16, 32, 64, 128, 256, 512],
        eval_every=50,
    ),
    "epinions": dict(
        npz="epinions.npz",
        unit=MONTH,
        test=2,
        wA=2,
        gaps=[0, 1, 2, 4, 8, 12],
        spans=[2, 4, 8, 14],
    ),
    "beer": dict(
        npz="beer.npz",
        unit=MONTH,
        test=6,
        wA=12,
        gaps=[0, 6, 12, 24, 48, 84],
        spans=[6, 12, 24, 48, 96, 180],
    ),
    "yelp": dict(
        npz="yelp.npz",
        unit=MONTH,
        test=6,
        wA=12,
        gaps=[0, 6, 12, 24, 48, 96, 168],
        spans=[6, 12, 24, 48, 96, 192],
    ),
    "netflix": dict(
        npz="netflix.npz",
        unit=MONTH,
        test=6,
        wA=12,
        gaps=[0, 3, 6, 12, 24, 48],
        spans=[6, 12, 24, 48, 72],
    ),
    "steam": dict(
        npz="steam.npz",
        unit=MONTH,
        test=6,
        wA=12,
        gaps=[0, 6, 12, 24, 48, 96],
        spans=[6, 12, 24, 48, 96, 120],
    ),
}

PANEL = [
    "kuairand",
    "recflow",
    "kuairec",
    "ml25m",
    "netflix",
    "yelp",
    "steam",
    "beer",
    "epinions",
    "gameugc_play",
    "gameugc_finish",
]

AMZ = [
    "amz_Appliances",
    "amz_Arts_Crafts_and_Sewing",
    "amz_Automotive",
    "amz_Baby_Products",
    "amz_CDs_and_Vinyl",
    "amz_Kindle_Store",
    "amz_Musical_Instruments",
    "amz_Office_Products",
    "amz_Pet_Supplies",
    "amz_Software",
    "amz_Toys_and_Games",
    "amz_Video_Games",
]

N_GRID = 241
N_REFINE = 60
BOUND_TOL = 1e-3


def auc_score(y, p):
    """Exact rank-sum ROC-AUC mirror of src/train.py (NaN if empty or single-class)."""
    order = np.argsort(p)
    y = y[order]
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = np.where(y == 1)[0].sum() + n_pos
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fit_logit_offset(logits, y, iters=50):
    """Scalar shift d minimizing BCE(logit + d, y) via Newton steps."""
    _require_torch()
    d = 0.0
    for _ in range(iters):
        p = torch.sigmoid(logits + d)
        grad = float((p - y).sum())
        hess = float((p * (1 - p)).sum())
        if hess < 1e-12:
            break
        step = grad / hess
        d -= step
        if abs(step) < 1e-9:
            break
    return d


def _fit_cells(rows, metric="loss"):
    cells = defaultdict(list)
    for r in rows:
        if metric in r:
            cells[(r["exp"], round(r["age_days"], 3))].append(r[metric])
    return cells


def _boxcox(t, t0, delta):
    r = t / t0
    if abs(delta) < 1e-9:
        return np.log(r)
    return (r ** delta - 1) / delta


def fit_joint(rows, seed=None, dbox=(-1.0, 4.0), metric="loss"):
    """Pure-NumPy joint Box-Cox age-law fitter matching src/fit_domains.py."""
    cells = _fit_cells(rows, metric)
    if seed is not None:
        rng = np.random.default_rng(seed)
        cells = {
            k: list(rng.choice(v, size=len(v), replace=True))
            for k, v in cells.items()
        }

    arms = sorted({k[0] for k in cells})
    xs, ys, gs = [], [], []
    for (exp, age), vals in sorted(cells.items()):
        xs.append(age)
        ys.append(np.mean(vals))
        gs.append(arms.index(exp))
    xs, ys, gs = np.asarray(xs), np.asarray(ys), np.asarray(gs)
    t0 = xs.min()
    D = np.column_stack([(gs == a).astype(float) for a in range(len(arms))])
    sstot = float(np.sum((ys - ys.mean()) ** 2))

    c0 = np.linalg.solve(D.T @ D, D.T @ ys)
    ss0 = float(np.sum((D @ c0 - ys) ** 2))

    def solve(delta):
        X = np.column_stack([D, _boxcox(xs, t0, delta)])
        G = X.T @ X
        try:
            coef = np.linalg.solve(G + 1e-12 * np.eye(len(G)), X.T @ ys)
        except np.linalg.LinAlgError:
            return np.append(c0, 0.0), ss0
        if coef[-1] < 0:
            return np.append(c0, 0.0), ss0
        return coef, float(np.sum((X @ coef - ys) ** 2))

    grid = np.linspace(dbox[0], dbox[1], N_GRID)
    ssq = np.array([solve(d)[1] for d in grid])
    k = int(np.argmin(ssq))
    d_best, ss_best = grid[k], ssq[k]
    a = grid[max(k - 1, 0)]
    b = grid[min(k + 1, N_GRID - 1)]
    gr = (np.sqrt(5) - 1) / 2
    c_, d_ = b - gr * (b - a), a + gr * (b - a)
    fc, fd = solve(c_)[1], solve(d_)[1]
    for _ in range(N_REFINE):
        if fc < fd:
            b, d_, fd = d_, c_, fc
            c_ = b - gr * (b - a)
            fc = solve(c_)[1]
        else:
            a, c_, fc = c_, d_, fd
            d_ = a + gr * (b - a)
            fd = solve(d_)[1]
        if b - a < 1e-7:
            break
    for cand in (c_, d_, grid[k]):
        coef, s = solve(cand)
        if s <= ss_best:
            d_best, ss_best = cand, s
    coef_best = solve(d_best)[0]

    beta = float(coef_best[-1])
    r2 = 1 - ss_best / sstot if sstot > 0 else float("nan")

    A = np.column_stack([D, np.log(xs)])
    b_log = float(np.linalg.lstsq(A, ys, rcond=None)[0][-1])

    at_bound = abs(d_best - dbox[0]) < BOUND_TOL or abs(d_best - dbox[1]) < BOUND_TOL
    return dict(
        delta=float(d_best),
        beta=beta,
        r2=float(r2),
        b_log=b_log,
        t0=float(t0),
        arms=arms,
        at_bound=bool(at_bound),
        intercepts=[float(c) for c in coef_best[:-1]],
        descending=bool(beta <= 0 and b_log < 0),
        xs=xs,
        ys=ys,
        gs=gs,
        n_cells=len(xs),
    )


def tau_half_obs(ages, y):
    """Model-free half-rise age matching src/collapse.py."""
    d = y - y[0]
    target = 0.5 * d[-1]
    if not np.isfinite(target) or target <= 0:
        return float("nan")
    for i in range(1, len(ages)):
        if d[i] >= target:
            if d[i] == d[i - 1]:
                return float(ages[i])
            f = (target - d[i - 1]) / (d[i] - d[i - 1])
            return float(
                np.exp(
                    np.log(ages[i - 1])
                    + f * (np.log(ages[i]) - np.log(ages[i - 1]))
                )
            )
    return float("nan")


def boot(cells, ages, n=200, seed=0):
    """Bootstrap 90% CI for tau_half_obs matching cohort_seed_analysis.py."""
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        yb = np.array(
            [
                np.mean(rng.choice(cells[a], size=len(cells[a]), replace=True))
                for a in ages
            ]
        )
        v = tau_half_obs(ages, yb)
        if np.isfinite(v):
            bs.append(v)
    return np.percentile(bs, [5, 95]) if len(bs) > 20 else (np.nan, np.nan)


def fit_power(ages, y):
    """Fit L = L0 + b * age^delta by grid over delta (matching backtest_analysis.py)."""
    ages = np.asarray(ages, dtype=float)
    y = np.asarray(y, dtype=float)
    best = None
    for d in np.linspace(-0.9, 4.0, 981):
        X = np.column_stack([np.ones_like(ages), ages ** d])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        if coef[1] <= 0:
            continue
        ss = float(np.sum((X @ coef - y) ** 2))
        if best is None or ss < best[0]:
            best = (ss, float(coef[0]), float(coef[1]), float(d))
    _, L0, b, d = best
    return L0, b, d


def load_domain_npz(domain_or_path):
    """Load domain .npz once into CPU torch tensors (user, item, label, ts)."""
    p = Path(domain_or_path)
    if not p.exists():
        key = str(domain_or_path)
        if key in DOMAINS:
            p = DATA_DIR / DOMAINS[key]["npz"]
        else:
            cand = DATA_DIR / f"{key}.npz"
            if cand.exists():
                p = cand
    if not p.exists():
        raise SystemExit(
            f"Missing {p}; build preprocessed .npz first via src/prep_*.py."
        )
    _require_torch()
    with np.load(p) as z:
        d = {k: torch.from_numpy(z[k]) for k in ("user", "item", "label", "ts")}
    n_users = int(d["user"].max()) + 1
    n_items = int(d["item"].max()) + 1
    return d, n_users, n_items


def build_window_schedule(ts, spec):
    """Mirror src/run_domain.py lines 131-168."""
    if hasattr(ts, "detach") and callable(getattr(ts, "numpy", None)):
        ts_np = ts.detach().cpu().numpy()
    else:
        ts_np = np.asarray(ts)
    if isinstance(spec, str):
        spec = DOMAINS[spec]

    u = spec["unit"]
    test_t1 = int(ts_np[-1]) + 1
    test_t0 = test_t1 - spec["test"] * u
    span_avail = (test_t0 - int(ts_np[0])) / u

    def volume(t0, t1):
        a, b = np.searchsorted(ts_np, [t0, t1])
        return int(b - a)

    cands = []
    dropped = []
    for g in spec["gaps"]:
        t1 = test_t0 - g * u
        t0 = t1 - spec["wA"] * u
        if t0 < ts_np[0]:
            dropped.append(
                {"exp": "A", "gap": g, "reason": "window_exceeds_history"}
            )
            continue
        cands.append(
            {
                "exp": "A",
                "gap": g,
                "train_t0": int(t0),
                "train_t1": int(t1),
                "vol": volume(t0, t1),
                "age_days": (g + spec["wA"] / 2 + spec["test"] / 2) * u / DAY,
            }
        )
    for s in spec["spans"]:
        t0 = test_t0 - s * u
        if t0 < ts_np[0] - 2 * u:
            dropped.append(
                {"exp": "C", "span": s, "reason": "exceeds_history"}
            )
            continue
        t0 = max(t0, int(ts_np[0]))
        cands.append(
            {
                "exp": "C",
                "span": s,
                "train_t0": int(t0),
                "train_t1": int(test_t0),
                "vol": volume(t0, test_t0),
                "age_days": (s / 2 + spec["test"] / 2) * u / DAY,
            }
        )

    kept = [c for c in cands if c["vol"] >= D_MIN]
    for c in cands:
        if c["vol"] < D_MIN:
            dropped.append(
                {
                    "exp": c["exp"],
                    **({"gap": c["gap"]} if "gap" in c else {"span": c["span"]}),
                    "vol": c["vol"],
                    "reason": "vol_below_d_min",
                }
            )
    d_fix = int(min(D_CAP, min(c["vol"] for c in kept))) if kept else 0
    return kept, d_fix, int(test_t0), int(test_t1), float(span_avail), dropped


def build_cohort_idx(
    data_dict, kept_a, d_fix, test_t0, test_t1, item_only=False, sub_seed=0,
    as_torch=True,
):
    """Mirror src/run_domain.py lines 182-212 for fixed cohort index construction."""
    def _np(arr):
        return (
            arr.detach().cpu().numpy()
            if hasattr(arr, "detach") and callable(getattr(arr, "numpy", None))
            else np.asarray(arr)
        )

    ts = _np(data_dict["ts"])
    user_all = _np(data_dict["user"])
    item_all = _np(data_dict["item"])

    c0, e0 = np.searchsorted(ts, [test_t0, test_t1])
    te_u, te_i = user_all[c0:e0], item_all[c0:e0]
    mask = np.ones(e0 - c0, dtype=bool)
    n_users_arr = int(user_all.max()) + 1
    n_items_arr = int(item_all.max()) + 1

    for c in kept_a:
        a, b = np.searchsorted(ts, [c["train_t0"], c["train_t1"]])
        if d_fix < b - a:
            rng = np.random.default_rng(sub_seed)
            sel = a + np.sort(
                rng.choice(b - a, size=int(d_fix), replace=False)
            )
        else:
            sel = np.arange(a, b)
        su = np.zeros(n_users_arr, dtype=bool)
        si = np.zeros(n_items_arr, dtype=bool)
        su[user_all[sel]] = True
        si[item_all[sel]] = True
        mask &= si[te_i] if item_only else (su[te_u] & si[te_i])

    cohort_idx = np.flatnonzero(mask)
    if not as_torch:
        return cohort_idx
    _require_torch()
    return torch.from_numpy(cohort_idx)


def _compute_fourway_fields(
    logits, te_y_cpu, y_np, masks_list, suffix=""
):
    """Compute four-way coverage/loss/AUC metrics on a given logit tensor."""
    _require_torch()
    bce = nn.BCEWithLogitsLoss(reduction="none")(logits, te_y_cpu)
    p_np = torch.sigmoid(logits).numpy()
    out = {}
    for tag, mask_t, mask_np, cnt in masks_list:
        if cnt == 0:
            out[f"loss_{tag}{suffix}"] = None
            out[f"auc_{tag}{suffix}"] = None
        else:
            out[f"loss_{tag}{suffix}"] = float(bce[mask_t].mean())
            a = auc_score(y_np[mask_np], p_np[mask_np])
            out[f"auc_{tag}{suffix}"] = None if np.isnan(a) else float(a)
    return out


def run_preloaded(
    data_dict,
    n_users,
    n_items,
    train_t0,
    train_t1,
    test_t0,
    test_t1,
    steps=8000,
    batch=8192,
    dim=32,
    lr=3e-3,
    seed=0,
    subsample=None,
    eval_every=200,
    probe="mf",
    cohort_idx=None,
    subsample_seed=None,
    device="cpu",
    calib_t0=None,
    calib_t1=None,
    calib_cap=50_000,
    recency_halflife_days=None,
    val_frac=None,
    fixed_step=None,
    log_fourway=True,
    init_state=None,
    return_state=False,
):
    """Mirror src/train.py::run on preloaded tensors, with optional 4-way logging."""
    _require_torch()
    torch.manual_seed(seed)
    d = data_dict
    ts = d["ts"].numpy()

    a, b = np.searchsorted(ts, [train_t0, train_t1])
    c, e = np.searchsorted(ts, [test_t0, test_t1])
    assert b > a and e > c, f"empty window: train[{a}:{b}] test[{c}:{e}]"

    sel = slice(a, b)
    if subsample is not None and subsample < b - a:
        rng = np.random.default_rng(
            seed if subsample_seed is None else subsample_seed
        )
        sel = a + np.sort(rng.choice(b - a, size=subsample, replace=False))
        sel = torch.from_numpy(sel)
    tr_u = d["user"][sel].long().to(device)
    tr_i = d["item"][sel].long().to(device)
    tr_y = d["label"][sel].float().to(device)

    tr_w = None
    if recency_halflife_days is not None:
        age_d = (train_t1 - d["ts"][sel].numpy()) / DAY
        w = 0.5 ** (age_d / recency_halflife_days)
        w = w / w.mean()
        tr_w = torch.from_numpy(w.astype(np.float32)).to(device)

    va_u = va_i = va_y = None
    if val_frac:
        vrng = np.random.default_rng(
            10_000 + (seed if subsample_seed is None else subsample_seed)
        )
        n_all = len(tr_u)
        perm = torch.from_numpy(vrng.permutation(n_all)).to(device)
        n_val = max(1, int(round(val_frac * n_all)))
        vi, ti = perm[:n_val], perm[n_val:]
        va_u, va_i, va_y = tr_u[vi], tr_i[vi], tr_y[vi]
        tr_u, tr_i, tr_y = tr_u[ti], tr_i[ti], tr_y[ti]
        if tr_w is not None:
            tr_w = tr_w[ti]

    cls = {"mf": _MF_CLS, "mlp": _MLP_CLS}[probe]
    model = cls(n_users, n_items, dim).to(device)
    if init_state is not None:
        model.load_state_dict(init_state)
    emb_params = [
        model.pu.weight,
        model.qi.weight,
        model.bu.weight,
        model.bi.weight,
    ]
    dense_params = [
        p for _, p in model.named_parameters() if not any(p is e_ for e_ in emb_params)
    ]
    opt_emb = torch.optim.SparseAdam(emb_params, lr=lr)
    opt_mu = torch.optim.Adam(dense_params, lr=lr)
    lossf = nn.BCEWithLogitsLoss()

    te_u = d["user"][c:e].long().to(device)
    te_i = d["item"][c:e].long().to(device)
    te_y = d["label"][c:e].float().to(device)

    def eval_logits():
        outs = []
        with torch.no_grad():
            for k in range(0, len(te_u), 262144):
                outs.append(model(te_u[k : k + 262144], te_i[k : k + 262144]))
        return torch.cat(outs)

    n_train = len(tr_u)
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    best = {"loss": float("inf"), "step": 0, "logits": None}
    best_val = {"val": float("inf"), "step": 0, "logits": None}
    at_fixed = {"step": None, "logits": None}
    t_start = time.time()
    for step in range(steps):
        idx = torch.randint(0, n_train, (batch,), device=device, generator=g)
        logit = model(tr_u[idx], tr_i[idx])
        if tr_w is None:
            loss = lossf(logit, tr_y[idx])
        else:
            per = nn.BCEWithLogitsLoss(reduction="none")(logit, tr_y[idx])
            loss = (per * tr_w[idx]).mean()
        opt_emb.zero_grad(set_to_none=True)
        opt_mu.zero_grad(set_to_none=True)
        loss.backward()
        opt_emb.step()
        opt_mu.step()
        if (step + 1) % eval_every == 0 or step + 1 == steps:
            lg = eval_logits()
            l = float(nn.BCEWithLogitsLoss()(lg, te_y))
            if l < best["loss"]:
                best = {"loss": l, "step": step + 1, "logits": lg.cpu()}
            if va_u is not None:
                with torch.no_grad():
                    vl = float(nn.BCEWithLogitsLoss()(model(va_u, va_i), va_y))
                if vl < best_val["val"]:
                    best_val = {
                        "val": vl,
                        "step": step + 1,
                        "logits": lg.cpu(),
                    }
            if fixed_step is not None and step + 1 == fixed_step:
                at_fixed = {"step": step + 1, "logits": lg.cpu()}
    train_time = time.time() - t_start

    seen_u = torch.zeros(n_users, dtype=torch.bool)
    seen_i = torch.zeros(n_items, dtype=torch.bool)
    seen_u[tr_u.unique().cpu()] = True
    seen_i[tr_i.unique().cpu()] = True
    te_u_cpu, te_i_cpu, te_y_cpu = te_u.cpu(), te_i.cpu(), te_y.cpu()
    su_te = seen_u[te_u_cpu]
    si_te = seen_i[te_i_cpu]
    seen_mask = su_te & si_te

    ci_t = None
    if cohort_idx is not None:
        ci_t = (
            cohort_idx.long()
            if isinstance(cohort_idx, torch.Tensor)
            else torch.as_tensor(cohort_idx, dtype=torch.long)
        )

    logits = best["logits"]
    bce = nn.BCEWithLogitsLoss(reduction="none")(logits, te_y_cpu)

    p = torch.sigmoid(logits).numpy()
    y = te_y_cpu.numpy()
    m = seen_mask.numpy()

    masks_list = None
    if log_fourway:
        n_te = len(te_y_cpu)
        m11 = seen_mask
        m10 = su_te & (~si_te)
        m01 = (~su_te) & si_te
        m00 = (~su_te) & (~si_te)
        masks_list = []
        for fw_tag, mt in (("11", m11), ("10", m10), ("01", m01), ("00", m00)):
            mnp = mt.numpy()
            cnt = int(mnp.sum())
            masks_list.append((fw_tag, mt, mnp, cnt))

    recal = {}
    bce_r = None
    if calib_t0 is not None and calib_t1 is not None:
        ca, cb = np.searchsorted(ts, [calib_t0, calib_t1])
        if cb > ca:
            idx = np.arange(ca, cb)
            if len(idx) > calib_cap:
                rng = np.random.default_rng(0)
                idx = np.sort(rng.choice(idx, size=calib_cap, replace=False))
            cu = d["user"][idx].long().to(device)
            ci = d["item"][idx].long().to(device)
            cy = d["label"][idx].float()
            with torch.no_grad():
                clog = model(cu, ci).cpu()
            dmu = fit_logit_offset(clog, cy)
            bce_r = nn.BCEWithLogitsLoss(reduction="none")(
                logits + dmu, te_y_cpu
            )
            recal = {
                "delta_mu": float(dmu),
                "n_calib": int(len(idx)),
                "calib_pos_rate": float(cy.mean()),
                "loss_recal": float(bce_r.mean()),
                "loss_seen_recal": float(bce_r[seen_mask].mean()),
                **(
                    {"loss_cohort_recal": float(bce_r[ci_t].mean())}
                    if ci_t is not None
                    else {}
                ),
            }
            if log_fourway and masks_list is not None:
                for fw_tag, mt, _, cnt in masks_list:
                    recal[f"loss_{fw_tag}_recal"] = (
                        None if cnt == 0 else float(bce_r[mt].mean())
                    )

    res = {
        "train_pos_rate": float(tr_y.mean()),
        "test_pos_rate": float(te_y_cpu.mean()),
        **recal,
        "train_t0": int(train_t0),
        "train_t1": int(train_t1),
        "test_t0": int(test_t0),
        "test_t1": int(test_t1),
        "n_train": int(n_train),
        "n_test": int(e - c),
        "subsample": subsample,
        "probe": probe,
        "steps": steps,
        "batch": batch,
        "dim": dim,
        "lr": lr,
        "seed": seed,
        "loss": float(bce.mean()),
        "loss_seen": float(bce[seen_mask].mean()),
        "best_step": best["step"],
        **(
            {
                "loss_cohort": float(bce[ci_t].mean()),
                "n_cohort": int(len(ci_t)),
            }
            if ci_t is not None
            else {}
        ),
        "seen_frac": float(m.mean()),
        "auc": auc_score(y, p),
        "auc_seen": auc_score(y[m], p[m]),
        "train_time_s": round(train_time, 1),
    }

    if log_fourway and masks_list is not None:
        n_te = len(te_y_cpu)
        for fw_tag, _, _, cnt in masks_list:
            res[f"n_{fw_tag}"] = cnt
            res[f"omega_{fw_tag}"] = float(cnt / n_te)
        res.update(_compute_fourway_fields(logits, te_y_cpu, y, masks_list))

    for tag, sel_ in (("valstop", best_val), ("fixed", at_fixed)):
        lgt = sel_.get("logits")
        if lgt is None:
            continue
        b_ = nn.BCEWithLogitsLoss(reduction="none")(lgt, te_y_cpu)
        p_ = torch.sigmoid(lgt).numpy()
        res[f"loss_{tag}"] = float(b_.mean())
        res[f"loss_seen_{tag}"] = float(b_[seen_mask].mean())
        res[f"auc_{tag}"] = auc_score(y, p_)
        res[f"best_step_{tag}"] = sel_["step"]
        if ci_t is not None:
            res[f"loss_cohort_{tag}"] = float(b_[ci_t].mean())
        if log_fourway and tag == "valstop" and val_frac and masks_list is not None:
            res.update(
                _compute_fourway_fields(
                    lgt, te_y_cpu, y, masks_list, suffix="_valstop"
                )
            )

    if val_frac:
        res["val_frac"] = val_frac
        res["n_val"] = int(len(va_u))
        res["val_loss_at_stop"] = best_val["val"]
    if recency_halflife_days is not None:
        res["recency_halflife_days"] = recency_halflife_days
    if init_state is not None:
        res["warm_started"] = True
    if return_state:
        return res, {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
    return res
