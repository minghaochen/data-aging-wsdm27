"""click_play.csv.gz -> two npz (label=is_play, label=is_finish), standard
schema. Also writes gameugc_meta.json with population stats.

Note: Game-UGC interaction logs are proprietary and not redistributed.
Place `click_play.csv.gz` under `data/` or pass its path as an argument.

Usage: python src/prep_gameugc.py [PATH_TO_CLICK_PLAY_CSV_GZ]
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "click_play.csv.gz"


def main(raw_path=RAW):
    import pandas as pd

    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise SystemExit(
            f"Missing raw file {raw_path}; place proprietary click_play.csv.gz in data/ or pass its path."
        )
    (ROOT / "data").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_csv(raw_path, compression="gzip", encoding="utf-8-sig",
                     dtype={"user_id": "int64", "ugc_item_id": "int64",
                            "is_play": "int8", "is_finish": "int8"})
    print(f"read {len(df):,} rows in {time.time()-t0:.0f}s; cols={list(df.columns)}")

    # resolution-robust unix seconds (pandas>=2 may default to non-nano)
    ts = pd.to_datetime(df["timestamp"]).values.astype(
        "datetime64[s]").astype("int64")
    order = np.argsort(ts, kind="stable")
    ts = ts[order]
    _, user = np.unique(df["user_id"].to_numpy()[order], return_inverse=True)
    _, item = np.unique(df["ugc_item_id"].to_numpy()[order], return_inverse=True)
    y_play = df["is_play"].to_numpy()[order].astype(np.int8)
    y_fin = df["is_finish"].to_numpy()[order].astype(np.int8)
    user = user.astype(np.int32); item = item.astype(np.int32)

    for lab, y in [("play", y_play), ("finish", y_fin)]:
        np.savez_compressed(ROOT / "data" / f"gameugc_{lab}.npz",
                            user=user, item=item, label=y, ts=ts)

    def fmt(t):
        return datetime.fromtimestamp(int(t), tz=timezone.utc).strftime("%Y-%m-%d")

    nu, ni = int(user.max()) + 1, int(item.max()) + 1
    # per-user / per-item event counts
    uc = np.bincount(user)
    ic = np.bincount(item)
    # returning users: active in >=2 distinct 14-day bins
    binid = (ts - ts[0]) // (14 * 86400)
    ub = defaultdict_bins(user, binid, nu)
    ret = int((ub >= 2).sum())

    meta = dict(rows=int(len(ts)), users=nu, items=ni,
                span_days=float((ts[-1] - ts[0]) / 86400),
                span=f"{fmt(ts[0])}..{fmt(ts[-1])}",
                ev_per_user=float(len(ts) / nu),
                ev_per_item=float(len(ts) / ni),
                play_rate=float(y_play.mean()),
                finish_rate=float(y_fin.mean()),
                finish_rate_given_play=float(y_fin[y_play == 1].mean()),
                users_ge2_events=int((uc >= 2).sum()),
                users_ge5_events=int((uc >= 5).sum()),
                returning_users_2bins=ret,
                returning_frac=float(ret / nu))
    (ROOT / "data" / "gameugc_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


def defaultdict_bins(user, binid, nu):
    """number of distinct bins each user appears in."""
    key = user.astype(np.int64) * (binid.max() + 1) + binid
    uniq = np.unique(key)
    u_of = (uniq // (binid.max() + 1)).astype(np.int64)
    return np.bincount(u_of, minlength=nu)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Preprocess Game-UGC click_play.csv.gz into gameugc_{play,finish}.npz."
    )
    ap.add_argument("raw_csv", nargs="?", default=str(RAW),
                    help="Path to click_play.csv.gz (default: data/click_play.csv.gz)")
    args = ap.parse_args()
    main(args.raw_csv)
