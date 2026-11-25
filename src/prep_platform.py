"""Preprocess cross-platform datasets -> npz (same schema as ml25m.npz).

- epinions: KONECT out.* TSV (user item rating [mult] timestamp), rating 1-5
- beer:     BeerAdvocate CSV (review_profilename, beer_beerid,
            review_overall, review_time)
Label = rating >= 5 (epinions) or >= 4 (beer).
Usage: python src/prep_platform.py epinions|beer
"""
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def read_epinions():
    raw_dir = ROOT / "data" / "epinions-rating"
    fp = next(raw_dir.glob("out.*"), None)
    if fp is None:
        raise SystemExit(
            f"Missing raw file {raw_dir / 'out.*'}; place KONECT epinions-rating data first."
        )
    users, items, ratings, tss = [], [], [], []
    with fp.open() as f:
        for line in f:
            if line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            users.append(int(parts[0]))
            items.append(int(parts[1]))
            ratings.append(float(parts[2]))
            tss.append(int(parts[-1]))
    return users, items, ratings, tss


def read_beer():
    fp = ROOT / "data" / "beer_reviews.csv"
    if not fp.exists():
        raise SystemExit(
            f"Missing raw file {fp}; place beer_reviews.csv under data/."
        )
    users, items, ratings, tss = [], [], [], []
    with fp.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
        iu = header.index("review_profilename")
        ii = header.index("beer_beerid")
        ir = header.index("review_overall")
        it = header.index("review_time")
        for row in reader:
            if not row[iu]:
                continue
            users.append(row[iu])
            items.append(int(row[ii]))
            ratings.append(float(row[ir]))
            tss.append(int(row[it]))
    return users, items, ratings, tss


def main(name):
    if name not in ("epinions", "beer"):
        raise SystemExit(
            f"Unknown platform {name!r}; usage: python src/prep_platform.py epinions|beer"
        )
    users, items, ratings, tss = (read_epinions() if name == "epinions"
                                  else read_beer())
    user = np.array(users)
    item = np.array(items)
    # epinions ratings are extremely top-heavy (>=4 is 92%); use ==5 there
    thr = 5.0 if name == "epinions" else 4.0
    label = (np.array(ratings, dtype=np.float32) >= thr).astype(np.int8)
    ts = np.array(tss, dtype=np.int64)

    u_uniq, u_codes = np.unique(user, return_inverse=True)
    i_uniq, i_codes = np.unique(item, return_inverse=True)
    order = np.argsort(ts, kind="stable")
    out = ROOT / "data" / f"{name}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, user=u_codes[order].astype(np.int32),
                        item=i_codes[order].astype(np.int32),
                        label=label[order], ts=ts[order])
    ts = ts[order]

    def fmt(t):
        return datetime.fromtimestamp(int(t), tz=timezone.utc).strftime("%Y-%m")

    print(f"{name}: rows={len(ts):,} users={len(u_uniq):,} "
          f"items={len(i_uniq):,} pos_rate={label.mean():.4f}")
    print(f"  span: {fmt(ts[0])} .. {fmt(ts[-1])} (max_ts={int(ts[-1])})")
    years = 1970 + ts // (365 * 86400)
    yv, yc = np.unique(years, return_counts=True)
    print("  yearly:", {int(y): int(c) for y, c in zip(yv, yc) if c > 500})


if __name__ == "__main__":
    args = sys.argv[1:]
    if any(a in ("-h", "--help") for a in args):
        print(__doc__.strip())
        sys.exit(0)
    if not args:
        raise SystemExit("Usage: python src/prep_platform.py epinions|beer")
    main(args[0])
