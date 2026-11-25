"""Preprocess MovieLens-25M: sort by time, remap ids, save compact npz.

numpy-only (no pandas) so it runs in any torch venv.

Usage: python src/prep.py
"""
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "ml-25m" / "ratings.csv"
OUT = ROOT / "data" / "ml25m.npz"

N_ROWS = 25_000_095


def main():
    if not RAW.exists():
        raise SystemExit(
            f"Missing raw file {RAW}; place MovieLens-25M ratings.csv under data/ml-25m/."
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    user = np.empty(N_ROWS, dtype=np.int32)
    item = np.empty(N_ROWS, dtype=np.int32)
    rating = np.empty(N_ROWS, dtype=np.float32)
    ts = np.empty(N_ROWS, dtype=np.int64)

    t0 = time.time()
    with RAW.open(newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        n = 0
        for row in reader:
            user[n] = int(row[0])
            item[n] = int(row[1])
            rating[n] = float(row[2])
            ts[n] = int(row[3])
            n += 1
    user, item, rating, ts = user[:n], item[:n], rating[:n], ts[:n]
    print(f"read {n:,} rows in {time.time()-t0:.0f}s")

    order = np.argsort(ts, kind="stable")
    user, item, rating, ts = user[order], item[order], rating[order], ts[order]

    u_uniq, u_codes = np.unique(user, return_inverse=True)
    i_uniq, i_codes = np.unique(item, return_inverse=True)
    label = (rating >= 4.0).astype(np.int8)

    np.savez_compressed(OUT, user=u_codes.astype(np.int32),
                        item=i_codes.astype(np.int32), label=label, ts=ts)
    print(f"users={len(u_uniq):,}  items={len(i_uniq):,}  pos_rate={label.mean():.4f}")

    def fmt(t):
        return datetime.fromtimestamp(int(t), tz=timezone.utc).strftime("%Y-%m-%d")

    print(f"time span: {fmt(ts[0])} .. {fmt(ts[-1])}  (max_ts={ts[-1]})")

    # yearly volume for slicing decisions
    years = ts // (365 * 86400) + 1970
    yv, yc = np.unique(years, return_counts=True)
    for y, c in zip(yv, yc):
        print(f"  ~{y}: {c:,}")


if __name__ == "__main__":
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__.strip())
        sys.exit(0)
    main()
