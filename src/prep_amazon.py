"""Amazon Reviews 2023 rating-only CSV -> npz (same schema as ml25m.npz).

CSV columns: user_id,parent_asin,rating,timestamp (ms). Label = rating >= 4.
Usage: python src/prep_amazon.py Video_Games
"""
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main(cat):
    raw = ROOT / "data" / "amazon" / f"{cat}.csv"
    if not raw.exists():
        raise SystemExit(
            f"Missing raw file {raw}; place Amazon Reviews 2023 CSV under data/amazon/."
        )
    out = ROOT / "data" / f"amz_{cat}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)

    users, items, labels, tss = [], [], [], []
    t0 = time.time()
    with raw.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        iu = header.index("user_id")
        ii = header.index("parent_asin")
        ir = header.index("rating")
        it = header.index("timestamp")
        for row in reader:
            users.append(row[iu])
            items.append(row[ii])
            labels.append(1 if float(row[ir]) >= 4.0 else 0)
            tss.append(int(row[it]) // 1000)
    print(f"{cat}: read {len(users):,} rows in {time.time()-t0:.0f}s")

    u_uniq, u_codes = np.unique(np.array(users), return_inverse=True)
    i_uniq, i_codes = np.unique(np.array(items), return_inverse=True)
    label = np.array(labels, dtype=np.int8)
    ts = np.array(tss, dtype=np.int64)

    order = np.argsort(ts, kind="stable")
    np.savez_compressed(out, user=u_codes[order].astype(np.int32),
                        item=i_codes[order].astype(np.int32),
                        label=label[order], ts=ts[order])
    ts = ts[order]
    print(f"  users={len(u_uniq):,} items={len(i_uniq):,} "
          f"pos_rate={label.mean():.4f}")

    def fmt(t):
        return datetime.fromtimestamp(int(t), tz=timezone.utc).strftime("%Y-%m")

    print(f"  span: {fmt(ts[0])} .. {fmt(ts[-1])} (max_ts={ts[-1]})")
    years = 1970 + ts // (365 * 86400)
    yv, yc = np.unique(years, return_counts=True)
    print("  yearly:", {int(y): int(c) for y, c in zip(yv, yc) if c > 1000})


if __name__ == "__main__":
    args = sys.argv[1:]
    if any(a in ("-h", "--help") for a in args):
        print(__doc__.strip())
        sys.exit(0)
    if not args:
        raise SystemExit("Usage: python src/prep_amazon.py <Category>")
    main(args[0])
