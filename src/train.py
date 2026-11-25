"""Biased MF with logistic loss, fixed optimization budget.

Design: optimization budget (steps K, batch B, lr schedule) is IDENTICAL across
runs so the optimization-error term is held constant; only the training window
(size / age) varies. Unseen users/items at eval keep their near-zero init, so
prediction degrades gracefully to the global bias (cold-start fallback).

Usage: python src/train.py --train-t0 T0 --train-t1 T1 --test-t0 U0 --test-t1 U1
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "ml25m.npz"

DAY = 86400

torch = None
nn = None
_MF_CLS = None
_MLP_CLS = None


def _require_torch():
    """Import PyTorch lazily so CLI --help and sibling imports work without torch."""
    global torch, nn, _MF_CLS, _MLP_CLS
    if torch is not None:
        return torch, nn
    import torch as _torch
    import torch.nn as _nn

    class MF(_nn.Module):
        # sparse gradients + SparseAdam: only touched embedding rows update each
        # step, so per-step cost is independent of vocabulary size (Amazon
        # categories have millions of users; dense Adam would sweep the full
        # table every step).
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
            return ((self.pu(u) * self.qi(i)).sum(-1)
                    + self.bu(u).squeeze(-1) + self.bi(i).squeeze(-1) + self.mu)

    class MLP(_nn.Module):
        """Second probe class for the delta model-invariance test: NeuMF-lite
        two-tower with an MLP head instead of a plain dot product."""
        def __init__(self, n_users, n_items, dim):
            super().__init__()
            self.pu = _nn.Embedding(n_users, dim, sparse=True)
            self.qi = _nn.Embedding(n_items, dim, sparse=True)
            self.bu = _nn.Embedding(n_users, 1, sparse=True)
            self.bi = _nn.Embedding(n_items, 1, sparse=True)
            self.mu = _nn.Parameter(_torch.zeros(1))
            self.head = _nn.Sequential(_nn.Linear(3 * dim, 64), _nn.ReLU(),
                                       _nn.Linear(64, 1))
            _nn.init.normal_(self.pu.weight, std=0.01)
            _nn.init.normal_(self.qi.weight, std=0.01)
            _nn.init.zeros_(self.bu.weight)
            _nn.init.zeros_(self.bi.weight)

        def forward(self, u, i):
            pu, qi = self.pu(u), self.qi(i)
            h = _torch.cat([pu, qi, pu * qi], -1)
            return (self.head(h).squeeze(-1)
                    + self.bu(u).squeeze(-1) + self.bi(i).squeeze(-1) + self.mu)

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


def load_data(device, data_path=None):
    p = Path(data_path) if data_path is not None else DATA
    if not p.exists():
        raise SystemExit(
            f"Missing {p}; build preprocessed .npz first via src/prep_*.py."
        )
    _require_torch()
    z = np.load(p)
    d = {k: torch.from_numpy(z[k]) for k in ("user", "item", "label", "ts")}
    n_users = int(d["user"].max()) + 1
    n_items = int(d["item"].max()) + 1
    # ts stays on CPU for searchsorted; index arrays go to GPU
    return d, n_users, n_items


def auc_score(y, p):
    # Plain rank sum: ties are broken by argsort order rather than given the
    # average rank. Exact ties are vanishingly rare at float32 logit
    # resolution, so this matches the textbook estimator on our data, but it
    # is not the textbook estimator and near-tied cold-cold pairs are the
    # place it could matter.
    order = np.argsort(p)
    y = y[order]
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = np.where(y == 1)[0].sum() + n_pos  # 1-based ranks of positives
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fit_logit_offset(logits, y, iters=50):
    """Scalar shift d minimising BCE(logit + d, y). Convex in d; Newton.

    This is the prior-shift control: a training window of a different age
    also has a different positive rate, so the probe learns a different
    global bias, and the test BCE moves even with zero preference drift.
    Fitting one scalar on a small FRESH slice removes exactly that channel
    and nothing else (it cannot change any ranking).
    """
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


def run(train_t0, train_t1, test_t0, test_t1, steps=8000, batch=8192,
        dim=32, lr=3e-3, seed=0, subsample=None, data_path=None,
        eval_every=200, probe="mf", cohort_idx=None, subsample_seed=None,
        device="cuda", calib_t0=None, calib_t1=None, calib_cap=50_000,
        recency_halflife_days=None, init_state=None, return_state=False,
        val_frac=None, fixed_step=None):
    """val_frac controls WHERE the early-stopping step is chosen.

    val_frac=None (default, and what every released number uses): the
        reported checkpoint is the one with the lowest loss on the evaluation
        set itself. There is no held-out validation split, so the reported
        loss is a minimum over checkpoints and carries that optimistic bias.

    val_frac=f (0<f<1): hold f of the training subsample out of training and
        use it to pick the step. The held-out slice is never trained on and
        is disjoint from the test period, so the selection uses only data a
        production system would have. The run then reports THREE checkpoint
        selections from the same trajectory:
          loss              -- test-argmin (the released convention)
          loss_valstop      -- test loss at the val-argmin step
          loss_fixed        -- test loss at `fixed_step` (if given)
        Because all three read the same optimization trajectory, the
        comparison is exactly paired and costs one training run, not three.
        Training volume is (1-f)*D rather than D, uniformly at every age, so
        curve shapes stay comparable while absolute levels shift slightly.
    """
    _require_torch()
    torch.manual_seed(seed)
    d, n_users, n_items = load_data(device, data_path)
    ts = d["ts"].numpy()

    a, b = np.searchsorted(ts, [train_t0, train_t1])
    c, e = np.searchsorted(ts, [test_t0, test_t1])
    assert b > a and e > c, f"empty window: train[{a}:{b}] test[{c}:{e}]"

    sel = slice(a, b)
    if subsample is not None and subsample < b - a:
        rng = np.random.default_rng(
            seed if subsample_seed is None else subsample_seed)
        sel = a + np.sort(rng.choice(b - a, size=subsample, replace=False))
        sel = torch.from_numpy(sel)
    tr_u = d["user"][sel].long().to(device)
    tr_i = d["item"][sel].long().to(device)
    tr_y = d["label"][sel].float().to(device)


    # Optional recency weighting: an alternative to retraining more often is
    # to train on a longer window but down-weight old events. Weights are
    # normalised to mean 1 so the effective loss scale (and hence the fixed
    # optimisation budget) is unchanged.
    tr_w = None
    if recency_halflife_days is not None:
        age_d = (train_t1 - d["ts"][sel].numpy()) / DAY
        w = 0.5 ** (age_d / recency_halflife_days)
        w = w / w.mean()
        tr_w = torch.from_numpy(w.astype(np.float32)).to(device)

    # Optional held-out validation slice, carved from the training subsample
    # so that the stopping step can be chosen without consulting the test
    # set. The subsample is already a uniform draw from the window, so a
    # random split of it is exchangeable in age with the part we keep.
    va_u = va_i = va_y = None
    if val_frac:
        vrng = np.random.default_rng(
            10_000 + (seed if subsample_seed is None else subsample_seed))
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
    emb_params = [model.pu.weight, model.qi.weight,
                  model.bu.weight, model.bi.weight]
    dense_params = [p for n, p in model.named_parameters()
                    if not any(p is e for e in emb_params)]
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
                outs.append(model(te_u[k:k + 262144], te_i[k:k + 262144]))
        return torch.cat(outs)

    # loss at the optimal early-stopping point within the budget: without this,
    # small windows are measured deep past the overfitting cliff and the BCE
    # blows up from overconfidence rather than tracing the data terms.
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
                    best_val = {"val": vl, "step": step + 1,
                                "logits": lg.cpu()}
            if fixed_step is not None and step + 1 == fixed_step:
                at_fixed = {"step": step + 1, "logits": lg.cpu()}
    train_time = time.time() - t_start

    seen_u = torch.zeros(n_users, dtype=torch.bool)
    seen_i = torch.zeros(n_items, dtype=torch.bool)
    seen_u[tr_u.unique().cpu()] = True
    seen_i[tr_i.unique().cpu()] = True
    te_u_cpu, te_i_cpu, te_y_cpu = te_u.cpu(), te_i.cpu(), te_y.cpu()
    seen_mask = seen_u[te_u_cpu] & seen_i[te_i_cpu]

    logits = best["logits"]
    bce = nn.BCEWithLogitsLoss(reduction="none")(logits, te_y_cpu)

    p = torch.sigmoid(logits).numpy()
    y = te_y_cpu.numpy()
    m = seen_mask.numpy()

    # ---- prior-shift control -------------------------------------------
    # Fit one scalar logit offset on a small slice of fresh data ending at
    # the test boundary (data any production system has), then report the
    # test loss under that offset. If the age curve survives this, the
    # measured ageing is not a base-rate artefact.
    recal = {}
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
                logits + dmu, te_y_cpu)
            recal = {
                "delta_mu": float(dmu),
                "n_calib": int(len(idx)),
                "calib_pos_rate": float(cy.mean()),
                "loss_recal": float(bce_r.mean()),
                "loss_seen_recal": float(bce_r[seen_mask].mean()),
                **({"loss_cohort_recal": float(bce_r[cohort_idx].mean())}
                   if cohort_idx is not None else {}),
            }

    res = {
        "train_pos_rate": float(tr_y.mean()),
        "test_pos_rate": float(te_y_cpu.mean()),
        **recal,
        "train_t0": int(train_t0), "train_t1": int(train_t1),
        "test_t0": int(test_t0), "test_t1": int(test_t1),
        "n_train": int(n_train), "n_test": int(e - c),
        "subsample": subsample, "probe": probe,
        "steps": steps, "batch": batch, "dim": dim, "lr": lr, "seed": seed,
        "loss": float(bce.mean()),
        "loss_seen": float(bce[seen_mask].mean()),
        "best_step": best["step"],
        **({"loss_cohort": float(bce[cohort_idx].mean()),
            "n_cohort": int(len(cohort_idx))}
           if cohort_idx is not None else {}),
        "seen_frac": float(m.mean()),
        "auc": auc_score(y, p),
        "auc_seen": auc_score(y[m], p[m]),
        "train_time_s": round(train_time, 1),
    }
    # Alternate stopping rules, read off the same trajectory so the
    # comparison with the released convention is exactly paired.
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
        if cohort_idx is not None:
            res[f"loss_cohort_{tag}"] = float(b_[cohort_idx].mean())
    if val_frac:
        res["val_frac"] = val_frac
        res["n_val"] = int(len(va_u))
        res["val_loss_at_stop"] = best_val["val"]
    if recency_halflife_days is not None:
        res["recency_halflife_days"] = recency_halflife_days
    if init_state is not None:
        res["warm_started"] = True
    if return_state:
        return res, {k: v.detach().cpu().clone()
                     for k, v in model.state_dict().items()}
    return res


def main():
    ap = argparse.ArgumentParser(
        description="Train a single fixed-budget MF/MLP probe window."
    )
    ap.add_argument("--train-t0", type=int, required=True)
    ap.add_argument("--train-t1", type=int, required=True)
    ap.add_argument("--test-t0", type=int, required=True)
    ap.add_argument("--test-t1", type=int, required=True)
    ap.add_argument("--data-path", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    res = run(args.train_t0, args.train_t1, args.test_t0, args.test_t1,
              steps=args.steps, batch=args.batch, dim=args.dim,
              lr=args.lr, seed=args.seed, data_path=args.data_path,
              device=args.device)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
