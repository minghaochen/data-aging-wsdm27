"""KuaiRec 2.0 big_matrix -> npz. label = watch_ratio >= 1.0 (the video was
watched through at least once); ts = unix timestamp.

The threshold is 1.0: at >= 2 the positive rate is too low for the
fixed-budget probe to train stably on the narrow windows this study
needs. The label semantics are
domain-internal (delta and tau_half are within-domain quantities), but the
threshold has to be stated correctly to reproduce the numbers.

Usage: python src/prep_kuairec.py
"""
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "KuaiRec 2.0" / "data" / "big_matrix.csv"


def main():
    if not RAW.exists():
        raise SystemExit(
            f"Missing raw file {RAW}; place KuaiRec 2.0 big_matrix.csv under data/KuaiRec 2.0/data/."
        )
    (ROOT / "data").mkdir(parents=True, exist_ok=True)
    users, items, labels, tss = [], [], [], []
    t0 = time.time()
    with RAW.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        iu, ii = header.index("user_id"), header.index("video_id")
        iw, it = header.index("watch_ratio"), header.index("timestamp")
        for row in reader:
            try:
                ts = float(row[it])
                w = float(row[iw])
            except ValueError:
                continue
            if ts < 1.5e9 or ts > 1.7e9:   # a few corrupt timestamps exist
                continue
            users.append(int(row[iu]))
            items.append(int(row[ii]))
            labels.append(1 if w >= 1.0 else 0)
            tss.append(int(ts))
    print(f"read {len(users):,} rows in {time.time()-t0:.0f}s")

    user, item = np.array(users), np.array(items)
    label = np.array(labels, dtype=np.int8)
    ts = np.array(tss, dtype=np.int64)
    u_uniq, u_codes = np.unique(user, return_inverse=True)
    i_uniq, i_codes = np.unique(item, return_inverse=True)
    order = np.argsort(ts, kind="stable")
    np.savez_compressed(ROOT / "data" / "kuairec.npz",
                        user=u_codes[order].astype(np.int32),
                        item=i_codes[order].astype(np.int32),
                        label=label[order], ts=ts[order])
    ts = ts[order]

    def fmt(t):
        return datetime.fromtimestamp(int(t), tz=timezone.utc).strftime(
            "%Y-%m-%d")

    print(f"users={len(u_uniq):,} items={len(i_uniq):,} "
          f"pos_rate={label.mean():.4f}")
    print(f"span: {fmt(ts[0])} .. {fmt(ts[-1])} (max_ts={int(ts[-1])})")
    days = ts // 86400
    dv, dc = np.unique(days, return_counts=True)
    print(f"days={len(dv)}  daily volume: min={dc.min()} "
          f"median={int(np.median(dc))} max={dc.max()}")


if __name__ == "__main__":
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__.strip())
        sys.exit(0)
    main()
