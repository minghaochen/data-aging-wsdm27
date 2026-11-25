"""Stage 4: Side-feature cold-start resolution test (`SideFeatMF` vs ID-only `MF`).

Part A — Strict ID-alignment verification (`np.array_equal` against `.npz`) and
static side-feature extraction/caching for `kuairand`, `ml25m`, and `recflow`.
Part B — `SideFeatMF` training across Exp A+C (`SEEDS=[0, 1, 2]`), reusing exact
Stage 2 windows, subsampling, recalibration, and ID-level 4-way masks.
Part C — Summary comparison vs Stage 2 CPU `MF` (`fourway_<dom>.jsonl`):
partition identity checks, endpoint losses/AUCs, joint A+C Box-Cox fits, and
Arm-A 2-way/4-way midpoint attribution.

Outputs exclusively to `results/`:
  - `feat_cache_<domain>.npz`
  - `sidefeat_<domain>.jsonl` (resumable, flushed per run)
  - `stage4_side_features_summary.json`

Strictly read-only with respect to existing `results/` outputs.
PyTorch and pandas are imported lazily so `--help` and `--summary-only`
only require NumPy and SciPy. No matplotlib imports.

Usage: python src/stage4_side_features.py [--summary-only] [--prep-only] [--domains ...]
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

import argparse
import csv
import gc
import glob
import json
import multiprocessing as mp
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

_SRC = str(Path(__file__).resolve().parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import cr_lib as C

pd = None
torch = None
nn = None
_SIDEFEAT_MF_CLS = None


def _require_pandas():
    global pd
    if pd is None:
        import pandas as _pd
        pd = _pd
    return pd


def _require_torch():
    """Import PyTorch lazily and build SideFeatMF once."""
    global torch, nn, _SIDEFEAT_MF_CLS
    if torch is not None:
        return torch, nn
    import torch as _torch
    import torch.nn as _nn

    _torch.set_num_threads(2)

    class SideFeatMF(_nn.Module):
        """MF backbone + sparse author embedding + averaged discrete embeddings + genre linear."""

        def __init__(self, n_users, n_items, dim, feat_spec):
            super().__init__()
            self.dim = dim
            self.pu = _nn.Embedding(n_users, dim, sparse=True)
            self.qi = _nn.Embedding(n_items, dim, sparse=True)
            self.bu = _nn.Embedding(n_users, 1, sparse=True)
            self.bi = _nn.Embedding(n_items, 1, sparse=True)
            self.mu = _nn.Parameter(_torch.zeros(1))
            _nn.init.normal_(self.pu.weight, std=0.01)
            _nn.init.normal_(self.qi.weight, std=0.01)
            _nn.init.zeros_(self.bu.weight)
            _nn.init.zeros_(self.bi.weight)

            self.user_cat_feats = feat_spec["user_cat_feats"]
            self.has_u_cat = self.user_cat_feats is not None
            if self.has_u_cat:
                self.u_eb = _nn.EmbeddingBag(feat_spec["u_vocab_total"], dim + 1, mode="mean")
                _nn.init.normal_(self.u_eb.weight[:, :dim], std=0.01)
                _nn.init.zeros_(self.u_eb.weight[:, dim])

            self.item_author = feat_spec["item_author"]
            self.has_author = self.item_author is not None
            if self.has_author:
                n_authors = feat_spec["n_authors"]
                self.pa = _nn.Embedding(n_authors, dim, sparse=True)
                self.ba = _nn.Embedding(n_authors, 1, sparse=True)
                _nn.init.normal_(self.pa.weight, std=0.01)
                _nn.init.zeros_(self.ba.weight)

            self.item_cat_feats = feat_spec["item_cat_feats"]
            self.has_i_cat = self.item_cat_feats is not None
            if self.has_i_cat:
                self.i_eb = _nn.EmbeddingBag(feat_spec["i_vocab_total"], dim + 1, mode="mean")
                _nn.init.normal_(self.i_eb.weight[:, :dim], std=0.01)
                _nn.init.zeros_(self.i_eb.weight[:, dim])

            self.item_genres = feat_spec["item_genres"]
            self.has_genre = self.item_genres is not None
            if self.has_genre:
                n_g = self.item_genres.shape[1]
                self.genre_proj = _nn.Linear(n_g, dim, bias=False)
                self.genre_bias = _nn.Linear(n_g, 1, bias=False)
                _nn.init.normal_(self.genre_proj.weight, std=0.01)
                _nn.init.zeros_(self.genre_bias.weight)

        def user_reprs(self, u):
            uv = self.pu(u)
            bu = self.bu(u).squeeze(-1)
            if self.has_u_cat:
                if len(u) > 2048:
                    uu, uinv = _torch.unique(u, return_inverse=True)
                    ue = self.u_eb(self.user_cat_feats[uu])[uinv]
                else:
                    ue = self.u_eb(self.user_cat_feats[u])
                uv = uv + ue[:, : self.dim]
                bu = bu + ue[:, self.dim]
            return uv, bu

        def all_user_reprs(self):
            uv = self.pu.weight
            bu = self.bu.weight.squeeze(-1)
            if self.has_u_cat:
                ue = self.u_eb(self.user_cat_feats)
                uv = uv + ue[:, : self.dim]
                bu = bu + ue[:, self.dim]
            return uv, bu

        def item_reprs(self, i):
            iv = self.qi(i)
            bi = self.bi(i).squeeze(-1)
            if self.has_author:
                a = self.item_author[i]
                iv = iv + self.pa(a)
                bi = bi + self.ba(a).squeeze(-1)
            if self.has_i_cat:
                ie = self.i_eb(self.item_cat_feats[i])
                iv = iv + ie[:, : self.dim]
                bi = bi + ie[:, self.dim]
            if self.has_genre:
                g = self.item_genres[i]
                iv = iv + self.genre_proj(g)
                bi = bi + self.genre_bias(g).squeeze(-1)
            return iv, bi

        def forward(self, u, i):
            uv, bu = self.user_reprs(u)
            iv, bi = self.item_reprs(i)
            return (uv * iv).sum(-1) + bu + bi + self.mu

        def sparse_parameters(self):
            ps = [self.pu.weight, self.qi.weight, self.bu.weight, self.bi.weight]
            if self.has_author:
                ps.extend([self.pa.weight, self.ba.weight])
            return ps

        def dense_parameters(self):
            sparse_ids = {id(p) for p in self.sparse_parameters()}
            return [p for p in self.parameters() if id(p) not in sparse_ids]

    torch = _torch
    nn = _nn
    _SIDEFEAT_MF_CLS = SideFeatMF
    return torch, nn


def __getattr__(name):
    if name == "SideFeatMF":
        _require_torch()
        return _SIDEFEAT_MF_CLS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

DOMAINS_STAGE4 = ("kuairand", "ml25m", "recflow")
GROUPS = ("11", "10", "01", "00")
COLD_GROUPS = ("10", "01", "00")
BOXCOX_METRICS = (
    "loss",
    "loss_recal",
    "loss_11",
    "loss_10",
    "loss_01",
    "loss_00",
    "loss_11_recal",
    "loss_10_recal",
    "loss_01_recal",
    "loss_00_recal",
    "loss_cold",
    "loss_cold_recal",
)
N_BOOT = 300

KUAIRAND_USER_COLS = (
    [
        "user_active_degree",
        "is_lowactive_period",
        "is_live_streamer",
        "is_video_author",
        "follow_user_num_range",
        "fans_user_num_range",
        "friend_user_num_range",
        "register_days_range",
    ]
    + [f"onehot_feat{k}" for k in range(18)]
)
KUAIRAND_VIDEO_CAT_COLS = ["video_type", "upload_type", "music_type", "tag"]
ML25M_GENRES = [
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "IMAX",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]
RECFLOW_FEATHER_COLS = [
    "user_id",
    "video_id",
    "request_timestamp",
    "age",
    "gender",
    "province",
    "author_id",
    "duration",
    "category_level_one",
    " category_level_two",  # raw RecFlow Feather schema column has a leading space
    "upload_type",
]

_WORKER_STATE = {}


def _run_key(r):
    return (
        str(r["exp"]),
        int(r["train_t0"]),
        int(r["train_t1"]),
        int(r["seed"]),
        int(r["subsample"]),
    )


def _load_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _encode_cat_series(series):
    """Map categorical values cleanly to int32 codes (0=missing/UNKNOWN, 1..K=sorted)."""
    _require_pandas()
    codes = np.zeros(len(series), dtype=np.int32)
    norm_vals = [None] * len(series)
    valid_set = set()
    for idx, v in enumerate(series.tolist()):
        if pd.isna(v):
            continue
        if isinstance(v, (float, np.floating)):
            s = str(int(v)) if float(v).is_integer() else str(v)
        else:
            s = str(v).strip()
        if not s or s.upper() == "UNKNOWN" or s.lower() == "nan":
            continue
        norm_vals[idx] = s
        valid_set.add(s)
    vocab = sorted(valid_set)
    mapping = {val: i + 1 for i, val in enumerate(vocab)}
    for idx, s in enumerate(norm_vals):
        if s is not None:
            codes[idx] = mapping[s]
    return codes, len(vocab) + 1


def _duration_buckets_ms(dur_arr):
    """Map duration (ms) -> 0 if missing/<=0 else 1 + clip(floor(log2(dur/1000+1)), 0, 12)."""
    out = np.zeros(len(dur_arr), dtype=np.int32)
    dur_f = dur_arr.astype(np.float64, copy=False)
    valid = np.isfinite(dur_f) & (dur_f > 0.0)
    if np.any(valid):
        b = np.floor(np.log2(dur_f[valid] / 1000.0 + 1.0)).astype(np.int32)
        out[valid] = 1 + np.clip(b, 0, 12)
    return out, 14


def _entity_mode_from_chronological(codes, n_entities, raw_arr):
    """Compute per-entity mode from chronologically sorted raw_arr, with constancy counts."""
    uniq_vals, val_codes = np.unique(raw_arr, return_inverse=True)
    k_vals = len(uniq_vals)
    first_v = np.empty(n_entities, dtype=np.int64)
    first_v[codes[::-1]] = val_codes[::-1]
    mismatch = val_codes != first_v[codes]
    n_bad_rows = int(mismatch.sum())
    if n_bad_rows == 0:
        return uniq_vals[first_v], val_codes, 0, 0, k_vals

    bad_ents = np.unique(codes[mismatch])
    n_bad_entities = int(len(bad_ents))
    mask = np.isin(codes, bad_ents)
    bc = codes[mask]
    bv = val_codes[mask]
    u_bad, local_e = np.unique(bc, return_inverse=True)
    key = np.sort(local_e.astype(np.int64) * np.int64(k_vals) + bv.astype(np.int64))
    idx_starts = np.concatenate(([0], np.flatnonzero(key[1:] != key[:-1]) + 1))
    counts = np.diff(np.append(idx_starts, len(key)))
    uniq_keys = key[idx_starts]
    e_idx = uniq_keys // np.int64(k_vals)
    v_idx = uniq_keys % np.int64(k_vals)
    ord2 = np.lexsort((-counts, e_idx))
    e_s = e_idx[ord2]
    v_s = v_idx[ord2]
    first_e = np.concatenate(([True], e_s[1:] != e_s[:-1]))
    out = first_v.copy()
    out[u_bad] = v_s[first_e]
    return uniq_vals[out], out, n_bad_rows, n_bad_entities, k_vals


def _prepare_kuairand():
    _require_pandas()
    raw_dir = C.DATA_DIR / "KuaiRand-Pure" / "data"
    files = sorted(glob.glob(str(raw_dir / "log_standard_*.csv")))
    assert files, f"No KuaiRand standard log files under {raw_dir}"

    users, items, tss = [], [], []
    for fp in files:
        with open(fp, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            iu = header.index("user_id")
            ii = header.index("video_id")
            it = header.index("time_ms")
            for row in reader:
                users.append(int(row[iu]))
                items.append(int(row[ii]))
                tss.append(int(row[it]) // 1000)
    user = np.array(users, dtype=np.int32)
    item = np.array(items, dtype=np.int32)
    ts = np.array(tss, dtype=np.int64)
    del users, items, tss

    order = np.argsort(ts, kind="stable")
    user, item, ts = user[order], item[order], ts[order]
    u_uniq, u_codes = np.unique(user, return_inverse=True)
    i_uniq, i_codes = np.unique(item, return_inverse=True)

    with np.load(C.DATA_DIR / "kuairand.npz") as z:
        assert len(ts) == len(z["ts"]), f"kuairand len {len(ts)} != {len(z['ts'])}"
        assert np.array_equal(u_codes, z["user"]), "kuairand user ID mismatch!"
        assert np.array_equal(i_codes, z["item"]), "kuairand item ID mismatch!"
        assert np.array_equal(ts, z["ts"]), "kuairand ts mismatch!"

    n_events = int(len(ts))
    del user, item, ts, order, u_codes, i_codes
    gc.collect()

    uf = pd.read_csv(raw_dir / "user_features_pure.csv")
    vf = pd.read_csv(raw_dir / "video_features_basic_pure.csv")
    assert uf["user_id"].duplicated().sum() == 0, "Duplicate user_id in user_features_pure"
    assert vf["video_id"].duplicated().sum() == 0, "Duplicate video_id in video_features_basic_pure"

    missing_u = int(len(set(u_uniq.tolist()) - set(uf["user_id"].tolist())))
    missing_i = int(len(set(i_uniq.tolist()) - set(vf["video_id"].tolist())))
    assert missing_u == 0 and missing_i == 0, f"kuairand missing metadata u={missing_u} i={missing_i}"

    uf_al = uf.set_index("user_id").reindex(u_uniq)
    vf_al = vf.set_index("video_id").reindex(i_uniq)

    u_cols_arrs, u_vocabs = [], []
    for c in KUAIRAND_USER_COLS:
        codes, vs = _encode_cat_series(uf_al[c])
        u_cols_arrs.append(codes)
        u_vocabs.append(int(vs))
    user_feats = np.column_stack(u_cols_arrs).astype(np.int32)
    user_vocab_sizes = np.array(u_vocabs, dtype=np.int32)

    author_codes, n_authors = _encode_cat_series(vf_al["author_id"])
    i_cols_arrs, i_vocabs = [author_codes], [int(n_authors)]
    for c in KUAIRAND_VIDEO_CAT_COLS:
        codes, vs = _encode_cat_series(vf_al[c])
        i_cols_arrs.append(codes)
        i_vocabs.append(int(vs))

    dur_ms = vf_al["video_duration"].to_numpy(dtype=np.float64)
    dur_b, dur_vs = _duration_buckets_ms(dur_ms)
    i_cols_arrs.append(dur_b)
    i_vocabs.append(int(dur_vs))

    item_feats = np.column_stack(i_cols_arrs).astype(np.int32)
    item_vocab_sizes = np.array(i_vocabs, dtype=np.int32)

    meta = {
        "domain": "kuairand",
        "n_events": n_events,
        "n_users": int(len(u_uniq)),
        "n_items": int(len(i_uniq)),
        "array_equal_user": True,
        "array_equal_item": True,
        "array_equal_ts": True,
        "missing_user_ids": missing_u,
        "missing_item_ids": missing_i,
        "user_cols": list(KUAIRAND_USER_COLS),
        "user_vocab_sizes": user_vocab_sizes.tolist(),
        "item_cols": ["author_id"] + list(KUAIRAND_VIDEO_CAT_COLS) + ["duration_bucket"],
        "item_vocab_sizes": item_vocab_sizes.tolist(),
        "has_author_col0": True,
        "duration_unit": "ms",
        "duration_missing_items": int(np.sum(dur_b == 0)),
    }

    cache_path = C.CR_RES / "feat_cache_kuairand.npz"
    np.savez_compressed(
        cache_path,
        user_feats=user_feats,
        user_vocab_sizes=user_vocab_sizes,
        item_feats=item_feats,
        item_vocab_sizes=item_vocab_sizes,
        meta_json=np.array(json.dumps(meta)),
    )
    del uf, vf, uf_al, vf_al, u_uniq, i_uniq, user_feats, item_feats
    gc.collect()
    return cache_path, meta


def _prepare_ml25m():
    _require_pandas()
    raw_dir = C.DATA_DIR / "ml-25m"
    n_rows_alloc = 25_000_095
    user = np.empty(n_rows_alloc, dtype=np.int32)
    item = np.empty(n_rows_alloc, dtype=np.int32)
    ts = np.empty(n_rows_alloc, dtype=np.int64)

    with (raw_dir / "ratings.csv").open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        n = 0
        for row in reader:
            user[n] = int(row[0])
            item[n] = int(row[1])
            ts[n] = int(row[3])
            n += 1
    user, item, ts = user[:n], item[:n], ts[:n]
    order = np.argsort(ts, kind="stable")
    user, item, ts = user[order], item[order], ts[order]
    u_uniq, u_codes = np.unique(user, return_inverse=True)
    i_uniq, i_codes = np.unique(item, return_inverse=True)

    with np.load(C.DATA_DIR / "ml25m.npz") as z:
        assert len(ts) == 25_000_095 and len(ts) == len(z["ts"]), f"ml25m len {len(ts)}"
        assert np.array_equal(u_codes, z["user"]), "ml25m user ID mismatch!"
        assert np.array_equal(i_codes, z["item"]), "ml25m item ID mismatch!"
        assert np.array_equal(ts, z["ts"]), "ml25m ts mismatch!"

    n_events = int(len(ts))
    del user, item, ts, order, u_codes, i_codes
    gc.collect()

    mv = pd.read_csv(raw_dir / "movies.csv")
    assert mv["movieId"].duplicated().sum() == 0, "Duplicate movieId in movies.csv"
    missing_i = int(len(set(i_uniq.tolist()) - set(mv["movieId"].tolist())))
    assert missing_i == 0, f"ml25m missing movieId count={missing_i}"

    mv_al = mv.set_index("movieId").reindex(i_uniq)
    genre_idx = {g: j for j, g in enumerate(ML25M_GENRES)}
    n_items = len(i_uniq)
    item_genres = np.zeros((n_items, len(ML25M_GENRES)), dtype=np.float32)

    no_genre_cnt = 0
    for idx, g_str in enumerate(mv_al["genres"].tolist()):
        if pd.isna(g_str) or g_str == "(no genres listed)":
            no_genre_cnt += 1
            continue
        gs = [genre_idx[p] for p in str(g_str).split("|") if p in genre_idx]
        if not gs:
            no_genre_cnt += 1
        else:
            item_genres[idx, gs] = 1.0 / len(gs)

    years = np.zeros(n_items, dtype=np.int32)
    missing_yr = 0
    for idx, title in enumerate(mv_al["title"].tolist()):
        ms = re.findall(r"\((\d{4})\)", str(title))
        if ms:
            y = int(ms[-1])
            if 1870 <= y <= 2025:
                years[idx] = y
                continue
        missing_yr += 1

    decades_raw = np.where(years > 0, (years // 10) * 10, 0)
    uniq_dec = sorted({int(d) for d in decades_raw if d > 0})
    dec_map = {d: i + 1 for i, d in enumerate(uniq_dec)}
    decade_codes = np.array([dec_map.get(int(d), 0) for d in decades_raw], dtype=np.int32)
    item_feats = decade_codes[:, None]
    item_vocab_sizes = np.array([len(uniq_dec) + 1], dtype=np.int32)

    meta = {
        "domain": "ml25m",
        "n_events": n_events,
        "n_users": int(len(u_uniq)),
        "n_items": int(n_items),
        "array_equal_user": True,
        "array_equal_item": True,
        "array_equal_ts": True,
        "missing_item_ids": missing_i,
        "n_genres": len(ML25M_GENRES),
        "genres": list(ML25M_GENRES),
        "no_genre_items": int(no_genre_cnt),
        "missing_year_items": int(missing_yr),
        "decades": uniq_dec,
        "item_cols": ["decade_bucket"],
        "item_vocab_sizes": item_vocab_sizes.tolist(),
        "has_author_col0": False,
    }

    cache_path = C.CR_RES / "feat_cache_ml25m.npz"
    np.savez_compressed(
        cache_path,
        item_genres=item_genres,
        item_feats=item_feats,
        item_vocab_sizes=item_vocab_sizes,
        meta_json=np.array(json.dumps(meta)),
    )
    del mv, mv_al, u_uniq, i_uniq, item_genres, item_feats
    gc.collect()
    return cache_path, meta


def _prepare_recflow():
    _require_pandas()
    raw_dir = C.DATA_DIR / "recflow"
    files = sorted(glob.glob(str(raw_dir / "*.feather")))
    if not files:
        raise SystemExit(f"No .feather files under {raw_dir}; place raw RecFlow files first.")

    parts = []
    skipped = []
    for fp in files:
        try:
            parts.append(pd.read_feather(fp, columns=RECFLOW_FEATHER_COLS))
        except Exception as ex:
            skipped.append({"file": Path(fp).name, "error": f"{type(ex).__name__}: {ex}"})

    if not parts:
        raise SystemExit(
            f"Failed to load any RecFlow .feather files under {raw_dir}: {skipped}"
        )
    if skipped and os.environ.get("ALLOW_PARTIAL") != "1":
        raise SystemExit(
            f"Skipped {len(skipped)} unreadable RecFlow .feather file(s): {skipped}. "
            "Set ALLOW_PARTIAL=1 to bypass."
        )

    df = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()

    ts = (df["request_timestamp"].to_numpy() // 1000).astype(np.int64)
    order = np.argsort(ts, kind="stable")
    ts = ts[order]
    user = df["user_id"].to_numpy()[order]
    item = df["video_id"].to_numpy()[order]
    u_uniq, u_codes = np.unique(user, return_inverse=True)
    i_uniq, i_codes = np.unique(item, return_inverse=True)
    del user, item
    gc.collect()

    recflow_npz = C.DATA_DIR / "recflow.npz"
    if not recflow_npz.exists():
        raise SystemExit(f"Missing {recflow_npz}; run src/prep_recflow.py first.")
    with np.load(recflow_npz) as z:
        assert len(ts) == len(z["ts"]), f"recflow len {len(ts)} != {len(z['ts'])}"
        assert np.array_equal(u_codes, z["user"]), "recflow user ID mismatch!"
        assert np.array_equal(i_codes, z["item"]), "recflow item ID mismatch!"
        assert np.array_equal(ts, z["ts"]), "recflow ts mismatch!"

    n_events = int(len(ts))
    n_users = int(len(u_uniq))
    n_items = int(len(i_uniq))
    del ts, u_uniq, i_uniq
    gc.collect()

    constancy = {}
    u_cols_arrs, u_vocabs = [], []
    for c in ("age", "gender", "province"):
        arr_ord = df[c].to_numpy()[order]
        _, mode_codes, br, be, k_vals = _entity_mode_from_chronological(
            u_codes, n_users, arr_ord
        )
        # Reserve 0 for unknown/missing; observed values occupy 1..k_vals
        u_cols_arrs.append((mode_codes + 1).astype(np.int32))
        u_vocabs.append(int(k_vals + 1))
        constancy[f"user:{c}"] = {
            "non_const_rows": br,
            "non_const_entities": be,
            "n_unique": int(k_vals),
        }
        del arr_ord

    user_feats = np.column_stack(u_cols_arrs).astype(np.int32)
    user_vocab_sizes = np.array(u_vocabs, dtype=np.int32)
    assert user_feats.shape == (n_users, 3), (
        f"Unexpected user_feats shape {user_feats.shape} vs ({n_users}, 3)"
    )

    i_cols_arrs, i_vocabs = [], []
    # Raw RecFlow Feather schema uses a leading space on " category_level_two"
    for c in ("author_id", "category_level_one", " category_level_two", "upload_type"):
        arr_ord = df[c].to_numpy()[order]
        _, mode_codes, br, be, k_vals = _entity_mode_from_chronological(
            i_codes, n_items, arr_ord
        )
        i_cols_arrs.append((mode_codes + 1).astype(np.int32))
        i_vocabs.append(int(k_vals + 1))
        constancy[f"item:{c.strip()}"] = {
            "non_const_rows": br,
            "non_const_entities": be,
            "n_unique": int(k_vals),
        }
        del arr_ord

    dur_ord = df["duration"].to_numpy()[order]
    dur_vals, _, br_d, be_d, k_d = _entity_mode_from_chronological(
        i_codes, n_items, dur_ord
    )
    constancy["item:duration"] = {
        "non_const_rows": br_d,
        "non_const_entities": be_d,
        "n_unique": int(k_d),
    }
    dur_b, dur_vs = _duration_buckets_ms(dur_vals)
    i_cols_arrs.append(dur_b)
    i_vocabs.append(int(dur_vs))

    item_feats = np.column_stack(i_cols_arrs).astype(np.int32)
    item_vocab_sizes = np.array(i_vocabs, dtype=np.int32)
    assert item_feats.shape == (n_items, 5), (
        f"Unexpected item_feats shape {item_feats.shape} vs ({n_items}, 5)"
    )

    meta = {
        "domain": "recflow",
        "n_events": n_events,
        "n_users": n_users,
        "n_items": n_items,
        "n_feather_total": len(files),
        "n_feather_loaded": len(files) - len(skipped),
        "skipped_files": skipped,
        "array_equal_user": True,
        "array_equal_item": True,
        "array_equal_ts": True,
        "user_cols": ["age", "gender", "province"],
        "user_vocab_sizes": user_vocab_sizes.tolist(),
        "item_cols": [
            "author_id",
            "category_level_one",
            "category_level_two",
            "upload_type",
            "duration_bucket",
        ],
        "item_vocab_sizes": item_vocab_sizes.tolist(),
        "has_author_col0": True,
        "duration_unit": "ms",
        "duration_zero_items": int(np.sum(dur_b == 0)),
        "metadata_constancy_and_mode_resolution": constancy,
    }

    del df, order, u_codes, i_codes, dur_ord, dur_vals
    gc.collect()

    cache_path = C.CR_RES / "feat_cache_recflow.npz"
    np.savez_compressed(
        cache_path,
        user_feats=user_feats,
        user_vocab_sizes=user_vocab_sizes,
        item_feats=item_feats,
        item_vocab_sizes=item_vocab_sizes,
        meta_json=np.array(json.dumps(meta)),
    )
    del user_feats, item_feats
    gc.collect()
    return cache_path, meta


def prepare_domain_features(dom):
    """Run Part A strict alignment checks and build/cache static feature arrays."""
    t0 = time.time()
    if dom == "kuairand":
        cache_path, meta = _prepare_kuairand()
    elif dom == "ml25m":
        cache_path, meta = _prepare_ml25m()
    elif dom == "recflow":
        cache_path, meta = _prepare_recflow()
    else:
        raise ValueError(f"Unsupported domain: {dom}")
    dt = round(time.time() - t0, 1)
    print(
        f"[{dom}] ALIGNMENT PASSED in {dt}s | events={meta['n_events']:,} "
        f"users={meta['n_users']:,} items={meta['n_items']:,} | cache={cache_path.name}",
        flush=True,
    )
    return cache_path, meta


def _load_feature_cache(dom):
    """Load feat_cache_<dom>.npz into CPU torch tensors for SideFeatMF."""
    _require_torch()
    cache_path = C.CR_RES / f"feat_cache_{dom}.npz"
    assert cache_path.exists(), f"Missing feature cache: {cache_path}"
    with np.load(cache_path) as z:
        meta = json.loads(str(z["meta_json"].item()))
        has_author = bool(meta["has_author_col0"])

        user_cat_feats = None
        u_vocab_total = 0
        if "user_feats" in z:
            uf_np = z["user_feats"].astype(np.int64)
            u_vs = z["user_vocab_sizes"].astype(np.int64)
            u_offsets = np.concatenate(([0], np.cumsum(u_vs[:-1])))
            uf_np = uf_np + u_offsets[None, :]
            user_cat_feats = torch.from_numpy(uf_np)
            u_vocab_total = int(u_vs.sum())

        item_author = None
        n_authors = 0
        item_cat_feats = None
        i_vocab_total = 0
        if "item_feats" in z:
            it_np = z["item_feats"].astype(np.int64)
            i_vs = z["item_vocab_sizes"].astype(np.int64)
            if has_author:
                item_author = torch.from_numpy(it_np[:, 0].copy())
                n_authors = int(i_vs[0])
                it_dense = it_np[:, 1:]
                i_vs_dense = i_vs[1:]
            else:
                it_dense = it_np
                i_vs_dense = i_vs
            if it_dense.shape[1] > 0:
                i_offsets = np.concatenate(([0], np.cumsum(i_vs_dense[:-1])))
                it_dense = it_dense + i_offsets[None, :]
                item_cat_feats = torch.from_numpy(it_dense.copy())
                i_vocab_total = int(i_vs_dense.sum())

        item_genres = None
        if "item_genres" in z:
            item_genres = torch.from_numpy(z["item_genres"].astype(np.float32))

    return {
        "meta": meta,
        "user_cat_feats": user_cat_feats,
        "u_vocab_total": u_vocab_total,
        "item_author": item_author,
        "n_authors": n_authors,
        "item_cat_feats": item_cat_feats,
        "i_vocab_total": i_vocab_total,
        "item_genres": item_genres,
    }


def run_sidefeat_preloaded(
    data_dict,
    feat_spec,
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
    calib_t0=None,
    calib_t1=None,
    calib_cap=50_000,
):
    """Train SideFeatMF matching Stage 2 protocol and log full 4-way + cold metrics."""
    _require_torch()
    torch.manual_seed(seed)
    d = data_dict
    ts = d["ts"].numpy()

    a, b = np.searchsorted(ts, [train_t0, train_t1])
    c, e = np.searchsorted(ts, [test_t0, test_t1])
    assert b > a and e > c, f"empty window: train[{a}:{b}] test[{c}:{e}]"

    sel = slice(a, b)
    if subsample is not None and subsample < b - a:
        rng = np.random.default_rng(seed)
        sel = a + np.sort(rng.choice(b - a, size=subsample, replace=False))
        sel = torch.from_numpy(sel)
    tr_u = d["user"][sel].long()
    tr_i = d["item"][sel].long()
    tr_y = d["label"][sel].float()

    model = _SIDEFEAT_MF_CLS(n_users, n_items, dim, feat_spec)
    opt_emb = torch.optim.SparseAdam(model.sparse_parameters(), lr=lr)
    opt_dense = torch.optim.Adam(model.dense_parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()

    te_u = d["user"][c:e].long()
    te_i = d["item"][c:e].long()
    te_y = d["label"][c:e].float()
    te_i_uniq, te_i_inv = torch.unique(te_i, return_inverse=True)

    def eval_logits():
        outs = []
        with torch.no_grad():
            u_vec_all, b_u_all = model.all_user_reprs()
            i_vec_u, b_i_u = model.item_reprs(te_i_uniq)
            mu = model.mu
            for k in range(0, len(te_u), 262144):
                uc = te_u[k : k + 262144]
                ic = te_i_inv[k : k + 262144]
                outs.append(
                    (u_vec_all[uc] * i_vec_u[ic]).sum(-1)
                    + b_u_all[uc]
                    + b_i_u[ic]
                    + mu
                )
        return torch.cat(outs)

    n_train = len(tr_u)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    best = {"loss": float("inf"), "step": 0, "logits": None}
    t_start = time.time()
    for step in range(steps):
        idx = torch.randint(0, n_train, (batch,), device="cpu", generator=g)
        logit = model(tr_u[idx], tr_i[idx])
        loss = lossf(logit, tr_y[idx])
        opt_emb.zero_grad(set_to_none=True)
        opt_dense.zero_grad(set_to_none=True)
        loss.backward()
        opt_emb.step()
        opt_dense.step()
        if (step + 1) % eval_every == 0 or step + 1 == steps:
            lg = eval_logits()
            l = float(lossf(lg, te_y))
            if l < best["loss"]:
                best = {"loss": l, "step": step + 1, "logits": lg.cpu()}
    train_time = time.time() - t_start

    # Exact ID-level seen masks matching Stage 2 (cr_lib.py lines 778-785)
    seen_u = torch.zeros(n_users, dtype=torch.bool)
    seen_i = torch.zeros(n_items, dtype=torch.bool)
    seen_u[tr_u.unique()] = True
    seen_i[tr_i.unique()] = True
    su_te = seen_u[te_u]
    si_te = seen_i[te_i]
    seen_mask = su_te & si_te
    cold_mask = ~seen_mask

    m11 = seen_mask
    m10 = su_te & (~si_te)
    m01 = (~su_te) & si_te
    m00 = (~su_te) & (~si_te)
    masks_list = []
    for fw_tag, mt in (("11", m11), ("10", m10), ("01", m01), ("00", m00)):
        mnp = mt.numpy()
        masks_list.append((fw_tag, mt, mnp, int(mnp.sum())))

    logits = best["logits"]
    bce = nn.BCEWithLogitsLoss(reduction="none")(logits, te_y)
    p = torch.sigmoid(logits).numpy()
    y = te_y.numpy()
    m = seen_mask.numpy()
    cm = cold_mask.numpy()

    recal = {}
    if calib_t0 is not None and calib_t1 is not None:
        ca, cb = np.searchsorted(ts, [calib_t0, calib_t1])
        if cb > ca:
            idx = np.arange(ca, cb)
            if len(idx) > calib_cap:
                rng = np.random.default_rng(0)
                idx = np.sort(rng.choice(idx, size=calib_cap, replace=False))
            cu = d["user"][idx].long()
            ci = d["item"][idx].long()
            cy = d["label"][idx].float()
            with torch.no_grad():
                clog = model(cu, ci).cpu()
            dmu = C.fit_logit_offset(clog, cy)
            bce_r = nn.BCEWithLogitsLoss(reduction="none")(logits + dmu, te_y)
            recal = {
                "delta_mu": float(dmu),
                "n_calib": int(len(idx)),
                "calib_pos_rate": float(cy.mean()),
                "loss_recal": float(bce_r.mean()),
                "loss_seen_recal": float(bce_r[seen_mask].mean()),
                "loss_cold_recal": None if int(cm.sum()) == 0 else float(bce_r[cold_mask].mean()),
            }
            for fw_tag, mt, _, cnt in masks_list:
                recal[f"loss_{fw_tag}_recal"] = None if cnt == 0 else float(bce_r[mt].mean())

    n_te = len(te_y)
    n_cold = int(cm.sum())
    auc_cold = None
    if n_cold > 0:
        ac = C.auc_score(y[cm], p[cm])
        auc_cold = None if np.isnan(ac) else float(ac)

    res = {
        "train_pos_rate": float(tr_y.mean()),
        "test_pos_rate": float(te_y.mean()),
        **recal,
        "train_t0": int(train_t0),
        "train_t1": int(train_t1),
        "test_t0": int(test_t0),
        "test_t1": int(test_t1),
        "n_train": int(n_train),
        "n_test": int(e - c),
        "subsample": subsample,
        "probe": "sidefeat_mf",
        "steps": steps,
        "batch": batch,
        "dim": dim,
        "lr": lr,
        "seed": seed,
        "loss": float(bce.mean()),
        "loss_seen": float(bce[seen_mask].mean()),
        "loss_cold": None if n_cold == 0 else float(bce[cold_mask].mean()),
        "best_step": best["step"],
        "seen_frac": float(m.mean()),
        "n_cold": n_cold,
        "omega_cold": float(n_cold / n_te),
        "auc": C.auc_score(y, p),
        "auc_seen": C.auc_score(y[m], p[m]),
        "auc_cold": auc_cold,
        "train_time_s": round(train_time, 1),
    }
    for fw_tag, _, _, cnt in masks_list:
        res[f"n_{fw_tag}"] = cnt
        res[f"omega_{fw_tag}"] = float(cnt / n_te)
    res.update(C._compute_fourway_fields(logits, te_y, y, masks_list))
    return res


def _init_worker(dom):
    for _k in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[_k] = "2"
    _require_torch()
    data_dict, n_users, n_items = C.load_domain_npz(dom)
    feat_spec = _load_feature_cache(dom)
    _WORKER_STATE.clear()
    _WORKER_STATE["dom"] = dom
    _WORKER_STATE["data_dict"] = data_dict
    _WORKER_STATE["n_users"] = n_users
    _WORKER_STATE["n_items"] = n_items
    _WORKER_STATE["feat_spec"] = feat_spec


def _run_worker_task(task):
    dom = task["domain"]
    assert _WORKER_STATE.get("dom") == dom, f"Worker loaded {_WORKER_STATE.get('dom')} != {dom}"
    res = run_sidefeat_preloaded(
        _WORKER_STATE["data_dict"],
        _WORKER_STATE["feat_spec"],
        _WORKER_STATE["n_users"],
        _WORKER_STATE["n_items"],
        task["train_t0"],
        task["train_t1"],
        task["test_t0"],
        task["test_t1"],
        steps=8000,
        batch=8192,
        dim=32,
        lr=3e-3,
        seed=task["seed"],
        subsample=task["subsample"],
        eval_every=task["eval_every"],
        calib_t0=task["calib_t0"],
        calib_t1=task["calib_t1"],
        calib_cap=50_000,
    )
    res["exp"] = task["exp"]
    res["age_days"] = float(task["age_days"])
    res["vol"] = int(task["vol"])
    res["domain"] = dom
    res["calib_t0"] = int(task["calib_t0"])
    res["calib_t1"] = int(task["calib_t1"])
    if "gap" in task:
        res["gap"] = int(task["gap"])
    if "span" in task:
        res["span"] = int(task["span"])
    gc.collect()
    return res


def _build_domain_tasks(dom):
    spec = C.DOMAINS[dom]
    with np.load(C.DATA_DIR / spec["npz"]) as z:
        ts = z["ts"]
    n_events = int(len(ts))
    kept, d_fix, test_t0, test_t1, span_avail, _ = C.build_window_schedule(ts, spec)
    calib_t0 = int(test_t0 - spec["unit"])
    calib_t1 = int(test_t0)
    ca, cb = np.searchsorted(ts, [calib_t0, calib_t1])
    n_calib_avail = int(cb - ca)
    assert n_calib_avail > 0, f"{dom}: empty calibration slice"
    assert calib_t1 <= test_t0, f"{dom}: calib_t1 > test_t0"

    eval_every = int(spec.get("eval_every", 200))
    tasks = []
    for cfg in kept:
        if cfg["exp"] not in ("A", "C"):
            continue
        for seed in C.SEEDS:
            t = {
                "domain": dom,
                "exp": cfg["exp"],
                "train_t0": int(cfg["train_t0"]),
                "train_t1": int(cfg["train_t1"]),
                "test_t0": int(test_t0),
                "test_t1": int(test_t1),
                "calib_t0": calib_t0,
                "calib_t1": calib_t1,
                "seed": int(seed),
                "subsample": int(d_fix),
                "eval_every": eval_every,
                "age_days": float(cfg["age_days"]),
                "vol": int(cfg["vol"]),
            }
            if "gap" in cfg:
                t["gap"] = int(cfg["gap"])
            if "span" in cfg:
                t["span"] = int(cfg["span"])
            tasks.append(t)

    del ts
    gc.collect()
    return tasks, n_events, n_calib_avail, d_fix, span_avail


def run_domain(dom, verify_alignment=True):
    align_meta = None
    if verify_alignment or not (C.CR_RES / f"feat_cache_{dom}.npz").exists():
        _, align_meta = prepare_domain_features(dom)
    else:
        with np.load(C.CR_RES / f"feat_cache_{dom}.npz") as z:
            align_meta = json.loads(str(z["meta_json"].item()))

    out_path = C.CR_RES / f"sidefeat_{dom}.jsonl"
    tasks, n_events, n_calib_avail, d_fix, _ = _build_domain_tasks(dom)
    existing = _load_jsonl(out_path)
    done_keys = {_run_key(r) for r in existing}
    todo = [t for t in tasks if _run_key(t) not in done_keys]

    print(
        f"\n[{dom}] SideFeatMF A+C | events={n_events:,} d_fix={d_fix:,} "
        f"calib_avail={n_calib_avail:,} total={len(tasks)} done={len(done_keys)} todo={len(todo)}",
        flush=True,
    )
    if not todo:
        return len(tasks), 0.0, align_meta

    n_workers = min(2, len(todo))
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with out_path.open("a") as fh:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(dom,),
        ) as pool:
            futures = {pool.submit(_run_worker_task, t): t for t in todo}
            completed = 0
            for fut in as_completed(futures):
                res = fut.result()
                fh.write(json.dumps(res) + "\n")
                fh.flush()
                completed += 1
                tag = (
                    f"gap={res['gap']}"
                    if "gap" in res
                    else f"span={res.get('span', '?')}"
                )
                print(
                    f"  [{dom}] ({completed}/{len(todo)}) {res['exp']} {tag:8s} "
                    f"s={res['seed']} loss={res['loss']:.5f} recal={res['loss_recal']:.5f} "
                    f"auc={res['auc']:.4f} auc10={res['auc_10']:.4f} "
                    f"auc01={res['auc_01']:.4f} auc00={res['auc_00']:.4f} "
                    f"({res['train_time_s']}s)",
                    flush=True,
                )
    elapsed = time.time() - t0
    gc.collect()
    print(f"  [{dom}] finished {len(todo)} runs in {elapsed:.1f}s (workers={n_workers})", flush=True)
    return len(tasks), elapsed, align_meta


def _weighted_loss_term(r, g, suffix=""):
    om = r[f"omega_{g}"]
    lg = r[f"loss_{g}{suffix}"]
    if om == 0 or lg is None:
        return 0.0
    return float(om) * float(lg)


def _augment_cold_losses(rows):
    """Ensure loss_cold / loss_cold_recal exist for both MF and SideFeatMF rows."""
    out = []
    for r in rows:
        rc = dict(r)
        w_cold = sum(float(r[f"omega_{g}"]) for g in COLD_GROUPS)
        if w_cold > 0:
            l_cold_calc = sum(_weighted_loss_term(r, g, "") for g in COLD_GROUPS) / w_cold
            lr_cold_calc = sum(_weighted_loss_term(r, g, "_recal") for g in COLD_GROUPS) / w_cold
        else:
            l_cold_calc = None
            lr_cold_calc = None
        if rc.get("loss_cold") is not None and l_cold_calc is not None:
            assert abs(float(rc["loss_cold"]) - l_cold_calc) < 1e-6
        else:
            rc["loss_cold"] = l_cold_calc
        if rc.get("loss_cold_recal") is not None and lr_cold_calc is not None:
            assert abs(float(rc["loss_cold_recal"]) - lr_cold_calc) < 1e-6
        else:
            rc["loss_cold_recal"] = lr_cold_calc
        out.append(rc)
    return out


def _check_omega_and_partition(dom, sf_rows, mf_rows):
    mf_by_key = {_run_key(r): r for r in mf_rows}
    sf_by_key = {_run_key(r): r for r in sf_rows}
    assert set(mf_by_key.keys()) == set(sf_by_key.keys()), f"{dom}: key mismatch MF vs SideFeatMF"

    max_om_diffs = {g: 0.0 for g in GROUPS}
    max_sum_om_err = 0.0
    max_raw_part_err = 0.0
    max_recal_part_err = 0.0
    max_sf_err = 0.0
    max_l11_err = 0.0
    min_counts = {g: None for g in GROUPS}

    for k, sr in sf_by_key.items():
        mr = mf_by_key[k]
        for g in GROUPS:
            d_om = abs(float(sr[f"omega_{g}"]) - float(mr[f"omega_{g}"]))
            max_om_diffs[g] = max(max_om_diffs[g], d_om)
            assert d_om < 1e-12, f"{dom} {k}: omega_{g} diff {d_om}"
            assert int(sr[f"n_{g}"]) == int(mr[f"n_{g}"]), f"{dom} {k}: n_{g} mismatch"
            cnt = int(sr[f"n_{g}"])
            min_counts[g] = cnt if min_counts[g] is None else min(min_counts[g], cnt)

        om_sum = sum(float(sr[f"omega_{g}"]) for g in GROUPS)
        max_sum_om_err = max(max_sum_om_err, abs(om_sum - 1.0))

        raw_part = sum(_weighted_loss_term(sr, g, "") for g in GROUPS)
        raw_err = abs(raw_part - float(sr["loss"]))
        max_raw_part_err = max(max_raw_part_err, raw_err)
        assert raw_err < 1e-6, f"{dom}: raw partition err {raw_err}"

        recal_part = sum(_weighted_loss_term(sr, g, "_recal") for g in GROUPS)
        recal_err = abs(recal_part - float(sr["loss_recal"]))
        max_recal_part_err = max(max_recal_part_err, recal_err)
        assert recal_err < 1e-6, f"{dom}: recal partition err {recal_err}"

        sf_err = abs(float(sr["seen_frac"]) - float(sr["omega_11"]))
        max_sf_err = max(max_sf_err, sf_err)
        assert sf_err < 1e-12

        if sr["n_11"] > 0:
            l11_err = abs(float(sr["loss_11"]) - float(sr["loss_seen"]))
            max_l11_err = max(max_l11_err, l11_err)
            assert l11_err < 1e-6

    return {
        "n_rows": len(sf_rows),
        "max_omega_diff_vs_stage2": max_om_diffs,
        "max_omega_diff_overall": float(max(max_om_diffs.values())),
        "max_sum_omega_err": max_sum_om_err,
        "max_raw_partition_err": max_raw_part_err,
        "max_recal_partition_err": max_recal_part_err,
        "max_omega11_seen_frac_err": max_sf_err,
        "max_loss11_loss_seen_err": max_l11_err,
        "min_group_counts": min_counts,
    }


def _cell_summary(cell_rows):
    fields = [
        "loss",
        "loss_recal",
        "loss_seen",
        "loss_cold",
        "loss_cold_recal",
        "auc",
        "auc_seen",
        "auc_cold",
        "omega_11",
        "omega_10",
        "omega_01",
        "omega_00",
        "loss_11",
        "loss_10",
        "loss_01",
        "loss_00",
        "loss_11_recal",
        "loss_10_recal",
        "loss_01_recal",
        "loss_00_recal",
        "auc_11",
        "auc_10",
        "auc_01",
        "auc_00",
    ]
    out = {
        "exp": cell_rows[0]["exp"],
        "age_days": float(cell_rows[0]["age_days"]),
        "n_seeds": len(cell_rows),
    }
    if "gap" in cell_rows[0]:
        out["gap"] = int(cell_rows[0]["gap"])
    if "span" in cell_rows[0]:
        out["span"] = int(cell_rows[0]["span"])
    for f in fields:
        vals = [float(r[f]) for r in cell_rows if r.get(f) is not None]
        out[f] = float(np.mean(vals)) if vals else None
    return out


def _summarize_endpoints(mf_rows, sf_rows):
    def group_cells(rows, exp):
        cells = defaultdict(list)
        for r in rows:
            if r["exp"] == exp:
                cells[float(r["age_days"])].append(r)
        return [_cell_summary(cells[a]) for a in sorted(cells.keys())]

    mf_a = group_cells(mf_rows, "A")
    sf_a = group_cells(sf_rows, "A")
    mf_c = group_cells(mf_rows, "C")
    sf_c = group_cells(sf_rows, "C")

    def pair_diff(sf_cell, mf_cell):
        keys = [
            "loss",
            "loss_recal",
            "loss_cold",
            "loss_cold_recal",
            "loss_11",
            "loss_10",
            "loss_01",
            "loss_00",
            "loss_11_recal",
            "loss_10_recal",
            "loss_01_recal",
            "loss_00_recal",
            "auc",
            "auc_11",
            "auc_10",
            "auc_01",
            "auc_00",
        ]
        return {
            f"delta_{k}": (
                float(sf_cell[k] - mf_cell[k])
                if (sf_cell.get(k) is not None and mf_cell.get(k) is not None)
                else None
            )
            for k in keys
        }

    def auc_stats(cells):
        res = {}
        for tag in ("auc_10", "auc_01", "auc_00", "auc_cold"):
            vals = [c[tag] for c in cells if c.get(tag) is not None]
            if vals:
                res[tag] = {
                    "min": float(min(vals)),
                    "mean": float(np.mean(vals)),
                    "max": float(max(vals)),
                    "n_gt_050": int(sum(1 for v in vals if v > 0.5)),
                    "n_cells": int(len(vals)),
                }
            else:
                res[tag] = None
        return res

    return {
        "arm_A_freshest": {
            "mf": mf_a[0],
            "sidefeat": sf_a[0],
            "diff_sidefeat_minus_mf": pair_diff(sf_a[0], mf_a[0]),
        },
        "arm_A_oldest": {
            "mf": mf_a[-1],
            "sidefeat": sf_a[-1],
            "diff_sidefeat_minus_mf": pair_diff(sf_a[-1], mf_a[-1]),
        },
        "arm_A_all_cells": {"mf": mf_a, "sidefeat": sf_a},
        "arm_C_all_cells": {"mf": mf_c, "sidefeat": sf_c},
        "cold_auc_range_all_AC_cells": {
            "mf": auc_stats(mf_a + mf_c),
            "sidefeat": auc_stats(sf_a + sf_c),
        },
    }


def _fit_boxcox_with_boot(rows, metric, dbox):
    sub_rows = [r for r in rows if r.get(metric) is not None]
    base = C.fit_joint(sub_rows, seed=None, dbox=dbox, metric=metric)
    deltas, blogs = [], []
    for b in range(N_BOOT):
        try:
            fb = C.fit_joint(sub_rows, seed=b, dbox=dbox, metric=metric)
            deltas.append(fb["delta"])
            blogs.append(fb["b_log"])
        except Exception:
            pass
    lo_d, hi_d = np.percentile(deltas, [5, 95])
    lo_b, hi_b = np.percentile(blogs, [5, 95])
    no_drift = bool(base["beta"] < 1e-6 or base["r2"] < 0.2)
    return {
        "metric": metric,
        "delta_box": [float(dbox[0]), float(dbox[1])],
        "delta": float(base["delta"]),
        "delta_lo": float(lo_d),
        "delta_hi": float(hi_d),
        "b_log": float(base["b_log"]),
        "b_log_lo": float(lo_b),
        "b_log_hi": float(hi_b),
        "beta": float(base["beta"]),
        "r2": float(base["r2"]),
        "t0": float(base["t0"]),
        "arms": list(base["arms"]),
        "intercepts": [float(x) for x in base["intercepts"]],
        "n_cells": int(base["n_cells"]),
        "at_bound": bool(base["at_bound"]),
        "descending": bool(base["descending"]),
        "no_drift": no_drift,
        "n_boot": int(len(deltas)),
    }


def _midpoint_attribution_for_mode(cells_sorted, suffix=""):
    n_ages = len(cells_sorted)
    w = {g: np.zeros(n_ages, dtype=float) for g in GROUPS}
    L = {g: np.zeros(n_ages, dtype=float) for g in GROUPS}
    mass = {g: np.zeros(n_ages, dtype=float) for g in GROUPS}
    L_tot = np.zeros(n_ages, dtype=float)
    L_tot_field = np.zeros(n_ages, dtype=float)
    loss_key = f"loss{suffix}"

    for t, (_, cell) in enumerate(cells_sorted):
        L_tot_field[t] = float(np.mean([float(r[loss_key]) for r in cell]))
        for g in GROUPS:
            wg = float(np.mean([float(r[f"omega_{g}"]) for r in cell]))
            mg = float(np.mean([_weighted_loss_term(r, g, suffix) for r in cell]))
            assert wg > 0.0, f"Empty group {g} in cell {t}"
            w[g][t] = wg
            mass[g][t] = mg
            L[g][t] = mg / wg
        L_tot[t] = sum(mass[g][t] for g in GROUPS)

    assert np.max(np.abs(L_tot - L_tot_field)) < 1e-6
    delta_L = float(L_tot[-1] - L_tot[0])
    mid_L = 0.5 * (L_tot[:-1] + L_tot[1:])

    within_4, cov_4 = {}, {}
    for g in GROUPS:
        mid_wg = 0.5 * (w[g][:-1] + w[g][1:])
        mid_Lg = 0.5 * (L[g][:-1] + L[g][1:])
        within_4[g] = float(np.sum(mid_wg * np.diff(L[g])))
        cov_4[g] = float(np.sum((mid_Lg - mid_L) * np.diff(w[g])))
    total_within_4 = float(sum(within_4.values()))
    total_cov_4 = float(sum(cov_4.values()))
    part_err = abs((total_within_4 + total_cov_4) - delta_L)
    assert part_err < 1e-10

    w_warm = w["11"]
    w_cold = 1.0 - w_warm
    L_warm = L["11"]
    L_cold = sum(mass[g] for g in COLD_GROUPS) / w_cold
    mid_w_warm = 0.5 * (w_warm[:-1] + w_warm[1:])
    mid_w_cold = 0.5 * (w_cold[:-1] + w_cold[1:])
    gap_2g = L_warm - L_cold
    within_warm_2g = float(np.sum(mid_w_warm * np.diff(L_warm)))
    within_cold_2g = float(np.sum(mid_w_cold * np.diff(L_cold)))
    cov_2g = float(np.sum(0.5 * (gap_2g[:-1] + gap_2g[1:]) * np.diff(w_warm)))
    err_2g = abs((within_warm_2g + within_cold_2g + cov_2g) - delta_L)
    assert err_2g < 1e-10

    tilde_w = {g: w[g] / w_cold for g in COLD_GROUPS}
    delta_L_cold = float(L_cold[-1] - L_cold[0])
    within_cold_int = {}
    w_within_cold_int = {}
    mix_within_cold = 0.0
    w_mix_within_cold = 0.0
    for g in COLD_GROUPS:
        mid_tw = 0.5 * (tilde_w[g][:-1] + tilde_w[g][1:])
        mid_Lg = 0.5 * (L[g][:-1] + L[g][1:])
        dLg = np.diff(L[g])
        dtw = np.diff(tilde_w[g])
        within_cold_int[f"within_{g}"] = float(np.sum(mid_tw * dLg))
        w_within_cold_int[f"within_{g}"] = float(np.sum(mid_w_cold * mid_tw * dLg))
        mix_within_cold += float(np.sum(mid_Lg * dtw))
        w_mix_within_cold += float(np.sum(mid_w_cold * mid_Lg * dtw))

    return {
        "delta_L": delta_L,
        "L_freshest": float(L_tot[0]),
        "L_oldest": float(L_tot[-1]),
        "fourway": {
            "within": within_4,
            "coverage": cov_4,
            "total_within": total_within_4,
            "total_coverage": total_cov_4,
            "within_share_of_dL": {
                g: float(within_4[g] / delta_L) if abs(delta_L) > 1e-12 else None
                for g in GROUPS
            },
            "coverage_share_of_dL": {
                g: float(cov_4[g] / delta_L) if abs(delta_L) > 1e-12 else None
                for g in GROUPS
            },
            "identity_err": float(part_err),
        },
        "twoway": {
            "warm": within_warm_2g,
            "cold": within_cold_2g,
            "coverage": cov_2g,
            "change": delta_L,
            "cold_share": float(within_cold_2g / delta_L) if abs(delta_L) > 1e-15 else None,
            "identity_err": float(err_2g),
        },
        "cold_internal": {
            "delta_L_cold": delta_L_cold,
            "L_cold_freshest": float(L_cold[0]),
            "L_cold_oldest": float(L_cold[-1]),
            **within_cold_int,
            "mix_within_cold": float(mix_within_cold),
            "weighted_contribution_to_delta_L": {
                "cold_2g": within_cold_2g,
                **w_within_cold_int,
                "mix_within_cold": float(w_mix_within_cold),
            },
        },
    }


def _attribution_for_rows(rows):
    arm_a = [r for r in rows if r["exp"] == "A"]
    by_age = defaultdict(list)
    for r in arm_a:
        by_age[float(r["age_days"])].append(r)
    cells_sorted = sorted(by_age.items(), key=lambda kv: kv[0])
    return {
        "ages_days": [a for a, _ in cells_sorted],
        "raw": _midpoint_attribution_for_mode(cells_sorted, ""),
        "recal": _midpoint_attribution_for_mode(cells_sorted, "_recal"),
    }


def build_summary(align_metas=None):
    align_metas = align_metas or {}
    per_dom = {}

    for dom in DOMAINS_STAGE4:
        sf_path = C.CR_RES / f"sidefeat_{dom}.jsonl"
        mf_path = C.CR_RES / f"fourway_{dom}.jsonl"
        assert sf_path.exists(), f"Missing {sf_path}"
        assert mf_path.exists(), f"Missing {mf_path}"

        if dom not in align_metas:
            cache_path = C.CR_RES / f"feat_cache_{dom}.npz"
            assert cache_path.exists(), f"Missing {cache_path}"
            with np.load(cache_path) as z:
                align_metas[dom] = json.loads(str(z["meta_json"].item()))

        sf_rows = _augment_cold_losses(_load_jsonl(sf_path))
        mf_rows = _augment_cold_losses(_load_jsonl(mf_path))
        checks = _check_omega_and_partition(dom, sf_rows, mf_rows)
        endpoints = _summarize_endpoints(mf_rows, sf_rows)

        boxcox = {}
        for dbox in ((-1.0, 4.0), (-6.0, 4.0)):
            btag = f"box_{dbox[0]:g}_{dbox[1]:g}"
            boxcox[btag] = {
                "mf": {m: _fit_boxcox_with_boot(mf_rows, m, dbox) for m in BOXCOX_METRICS},
                "sidefeat": {
                    m: _fit_boxcox_with_boot(sf_rows, m, dbox) for m in BOXCOX_METRICS
                },
            }

        attribution = {
            "mf": _attribution_for_rows(mf_rows),
            "sidefeat": _attribution_for_rows(sf_rows),
        }

        per_dom[dom] = {
            "alignment_and_features": align_metas[dom],
            "partition_and_omega_checks": checks,
            "endpoints": endpoints,
            "boxcox_ac": boxcox,
            "arm_A_attribution": attribution,
        }

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "domains": list(DOMAINS_STAGE4),
        "total_runs": sum(per_dom[d]["partition_and_omega_checks"]["n_rows"] for d in DOMAINS_STAGE4),
        "max_omega_diff_vs_stage2": float(
            max(per_dom[d]["partition_and_omega_checks"]["max_omega_diff_overall"] for d in DOMAINS_STAGE4)
        ),
        "per_domain": per_dom,
    }
    out_path = C.CR_RES / "stage4_side_features_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary, out_path


def _print_summary(summary, out_path):
    print("\n" + "=" * 82)
    print(f"SAVED SUMMARY: {out_path}")
    print(
        f"Total runs={summary['total_runs']} | "
        f"max |omega_SideFeatMF - omega_Stage2MF| = {summary['max_omega_diff_vs_stage2']:.2e}"
    )
    for dom in DOMAINS_STAGE4:
        d = summary["per_domain"][dom]
        chk = d["partition_and_omega_checks"]
        ep = d["endpoints"]
        print(f"\n--- {dom} (runs={chk['n_rows']}, max_om_diff={chk['max_omega_diff_overall']:.2e}) ---")
        for end_tag in ("arm_A_freshest", "arm_A_oldest"):
            mf_c = ep[end_tag]["mf"]
            sf_c = ep[end_tag]["sidefeat"]
            print(
                f"  {end_tag:14s} (age={sf_c['age_days']:6.1f}d): "
                f"loss MF={mf_c['loss']:.5f} SF={sf_c['loss']:.5f} | "
                f"AUC MF={mf_c['auc']:.4f} SF={sf_c['auc']:.4f}"
            )
            print(
                f"    AUC (11,10,01,00,cold): "
                f"MF=({mf_c['auc_11']:.4f}, {mf_c['auc_10']:.4f}, {mf_c['auc_01']:.4f}, {mf_c['auc_00']:.4f}, n/a) | "
                f"SF=({sf_c['auc_11']:.4f}, {sf_c['auc_10']:.4f}, {sf_c['auc_01']:.4f}, {sf_c['auc_00']:.4f}, {sf_c['auc_cold']:.4f})"
            )
            print(
                f"    Loss(11,10,01,00,cold): "
                f"MF=({mf_c['loss_11']:.5f}, {mf_c['loss_10']:.5f}, {mf_c['loss_01']:.5f}, {mf_c['loss_00']:.5f}, {mf_c['loss_cold']:.5f}) | "
                f"SF=({sf_c['loss_11']:.5f}, {sf_c['loss_10']:.5f}, {sf_c['loss_01']:.5f}, {sf_c['loss_00']:.5f}, {sf_c['loss_cold']:.5f})"
            )
        print("  Box-Cox A+C (box_-1_4):")
        bc = d["boxcox_ac"]["box_-1_4"]
        for m in ("loss", "loss_recal", "loss_11", "loss_10", "loss_01", "loss_00"):
            fm, fs = bc["mf"][m], bc["sidefeat"][m]
            print(
                f"    {m:11s}: MF d={fm['delta']:+.3f}[{fm['delta_lo']:+.3f},{fm['delta_hi']:+.3f}] b={fm['b_log']:+.5f} | "
                f"SF d={fs['delta']:+.3f}[{fs['delta_lo']:+.3f},{fs['delta_hi']:+.3f}] b={fs['b_log']:+.5f}"
            )
        at = d["arm_A_attribution"]
        for mode in ("raw", "recal"):
            am = at["mf"][mode]["twoway"]
            as_ = at["sidefeat"][mode]["twoway"]
            print(
                f"  Attribution [{mode:5s}]: MF cold_share={am['cold_share']:.3f} (dL={am['change']:+.5f}) | "
                f"SF cold_share={as_['cold_share']:.3f} (dL={as_['change']:+.5f})"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--domains",
        default=",".join(DOMAINS_STAGE4),
        help="Comma-separated subset of kuairand,ml25m,recflow",
    )
    ap.add_argument(
        "--prep-only",
        action="store_true",
        help="Run Part A alignment verification and feature caching only",
    )
    ap.add_argument(
        "--summary-only",
        action="store_true",
        help="Skip training and build stage4_side_features_summary.json",
    )
    args = ap.parse_args()

    C.CR_RES.mkdir(parents=True, exist_ok=True)
    doms = [d.strip() for d in args.domains.split(",") if d.strip()]
    for d in doms:
        assert d in DOMAINS_STAGE4, f"Unsupported domain: {d}"

    t_all = time.time()
    align_metas = {}

    if args.prep_only:
        for dom in doms:
            _, meta = prepare_domain_features(dom)
            align_metas[dom] = meta
        print(f"\nPart A completed in {time.time() - t_all:.1f}s.", flush=True)
        return

    if not args.summary_only:
        total_runs = 0
        for dom in doms:
            n_dom, _, meta = run_domain(dom, verify_alignment=True)
            align_metas[dom] = meta
            total_runs += n_dom
            gc.collect()
        print(
            f"\nCompleted Stage 4 SideFeatMF training across {len(doms)} domains "
            f"({total_runs} runs) in {time.time() - t_all:.1f}s.",
            flush=True,
        )

    if set(doms) == set(DOMAINS_STAGE4) or args.summary_only:
        summary, out_path = build_summary(align_metas)
        _print_summary(summary, out_path)


if __name__ == "__main__":
    main()
