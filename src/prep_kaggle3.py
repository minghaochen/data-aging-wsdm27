"""Stream-preprocess Yelp / Netflix Prize / Steam directly from their zips.

Outputs npz with the standard schema (user, item, label, ts).
Usage: python src/prep_kaggle3.py yelp|netflix|steam
"""
import csv
import io
import json
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def finish(name, users, items, labels, tss):
    DATA.mkdir(parents=True, exist_ok=True)
    user = np.array(users)
    item = np.array(items)
    label = np.array(labels, dtype=np.int8)
    ts = np.array(tss, dtype=np.int64)
    u_uniq, u_codes = np.unique(user, return_inverse=True)
    i_uniq, i_codes = np.unique(item, return_inverse=True)
    order = np.argsort(ts, kind="stable")
    np.savez_compressed(DATA / f"{name}.npz",
                        user=u_codes[order].astype(np.int32),
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
    print("  yearly:", {int(y): int(c) for y, c in zip(yv, yc) if c > 5000})


def d2ts(datestr):
    return int(datetime.strptime(datestr[:10], "%Y-%m-%d")
               .replace(tzinfo=timezone.utc).timestamp())


def prep_yelp():
    zp = DATA / "Yelp Dataset.zip"
    if not zp.exists():
        raise SystemExit(f"Missing raw archive {zp}; download Yelp Dataset.zip into data/.")
    users, items, labels, tss = [], [], [], []
    t0 = time.time()
    with zipfile.ZipFile(zp) as z:
        with z.open("yelp_academic_dataset_review.json") as f:
            for line in io.TextIOWrapper(f, encoding="utf-8"):
                r = json.loads(line)
                users.append(r["user_id"])
                items.append(r["business_id"])
                labels.append(1 if r["stars"] >= 4 else 0)
                tss.append(d2ts(r["date"]))
    print(f"read in {time.time()-t0:.0f}s")
    finish("yelp", users, items, labels, tss)


def prep_netflix():
    zp = DATA / "Netflix Prize data.zip"
    if not zp.exists():
        raise SystemExit(f"Missing raw archive {zp}; download Netflix Prize data.zip into data/.")
    users, items, labels, tss = [], [], [], []
    t0 = time.time()
    with zipfile.ZipFile(zp) as z:
        for member in ["combined_data_1.txt", "combined_data_2.txt",
                       "combined_data_3.txt", "combined_data_4.txt"]:
            with z.open(member) as f:
                movie = None
                for line in io.TextIOWrapper(f, encoding="utf-8"):
                    line = line.strip()
                    if line.endswith(":"):
                        movie = int(line[:-1])
                        continue
                    u, r, d = line.split(",")
                    users.append(int(u))
                    items.append(movie)
                    labels.append(1 if int(r) >= 4 else 0)
                    tss.append(d2ts(d))
            print(f"  {member} done ({time.time()-t0:.0f}s)")
    finish("netflix", users, items, labels, tss)


def prep_steam():
    zp = DATA / "Steam Reviews Dataset 2021.zip"
    if not zp.exists():
        raise SystemExit(f"Missing raw archive {zp}; download Steam Reviews Dataset 2021.zip into data/.")
    csv.field_size_limit(1 << 25)   # review texts exceed the default 128KB
    users, items, labels, tss = [], [], [], []
    t0 = time.time()
    with zipfile.ZipFile(zp) as z:
        with z.open("steam_reviews.csv") as f:
            reader = csv.reader(io.TextIOWrapper(f, encoding="utf-8",
                                                 errors="replace"))
            header = next(reader)
            iu = header.index("author.steamid")
            ii = header.index("app_id")
            ir = header.index("recommended")
            it = header.index("timestamp_created")
            n = 0
            for row in reader:
                try:
                    users.append(int(row[iu]))
                    items.append(int(row[ii]))
                    labels.append(1 if row[ir] in ("True", "true", "1")
                                  else 0)
                    tss.append(int(row[it]))
                except (ValueError, IndexError):
                    continue
                n += 1
                if n % 5000000 == 0:
                    print(f"  {n:,} rows ({time.time()-t0:.0f}s)")
    finish("steam", users, items, labels, tss)


if __name__ == "__main__":
    args = sys.argv[1:]
    if any(a in ("-h", "--help") for a in args):
        print(__doc__.strip())
        sys.exit(0)
    runners = {"yelp": prep_yelp, "netflix": prep_netflix, "steam": prep_steam}
    if not args or args[0] not in runners:
        raise SystemExit("Usage: python src/prep_kaggle3.py yelp|netflix|steam")
    runners[args[0]]()
