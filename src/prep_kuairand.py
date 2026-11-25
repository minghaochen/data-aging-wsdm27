"""Preprocess KuaiRand-Pure standard logs -> npz (same schema as ml25m.npz).

Uses only the standard recommendation logs (both periods); the
randomly-exposed logs come from a different serving distribution and are
excluded. Label = is_click.

Usage: python src/prep_kuairand.py
"""
import csv
import glob
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "KuaiRand-Pure" / "data"
OUT = ROOT / "data" / "kuairand.npz"


def main():
    files = sorted(glob.glob(str(RAW_DIR / "log_standard_*.csv")))
    if not files:
        raise SystemExit(
            f"No KuaiRand standard logs found under {RAW_DIR}; download KuaiRand-Pure first."
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print("files:", [Path(f).name for f in files])

    users, items, labels, tss = [], [], [], []
    t0 = time.time()
    for fp in files:
        with open(fp, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            iu = header.index("user_id")
            ii = header.index("video_id")
            ic = header.index("is_click")
            it = header.index("time_ms")
            for row in reader:
                users.append(int(row[iu]))
                items.append(int(row[ii]))
                labels.append(int(row[ic]))
                tss.append(int(row[it]) // 1000)
    user = np.array(users, dtype=np.int32)
    item = np.array(items, dtype=np.int32)
    label = np.array(labels, dtype=np.int8)
    ts = np.array(tss, dtype=np.int64)
    print(f"read {len(user):,} rows in {time.time()-t0:.0f}s")

    order = np.argsort(ts, kind="stable")
    user, item, label, ts = user[order], item[order], label[order], ts[order]

    u_uniq, u_codes = np.unique(user, return_inverse=True)
    i_uniq, i_codes = np.unique(item, return_inverse=True)

    np.savez_compressed(OUT, user=u_codes.astype(np.int32),
                        item=i_codes.astype(np.int32), label=label, ts=ts)
    print(f"users={len(u_uniq):,}  items={len(i_uniq):,}  "
          f"click_rate={label.mean():.4f}")

    def fmt(t):
        return datetime.fromtimestamp(int(t), tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M")

    print(f"time span: {fmt(ts[0])} .. {fmt(ts[-1])}  (max_ts={ts[-1]})")
    days = ts // 86400
    dv, dc = np.unique(days, return_counts=True)
    for d, c in zip(dv, dc):
        print(f"  {datetime.fromtimestamp(int(d*86400), tz=timezone.utc).strftime('%m-%d')}: {c:,}")


if __name__ == "__main__":
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__.strip())
        sys.exit(0)
    main()
