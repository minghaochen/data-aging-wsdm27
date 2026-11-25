"""RecFlow realshow daily feathers -> npz (same schema as ml25m.npz).

label = effective_view; ts = request_timestamp (ms -> s).
Reads every daily file under data/recflow/ and refuses to build a
partial npz: skipping unreadable days silently would make a short download
indistinguishable from a genuine gap in the log and silently shift every
downstream window volume.

Usage: python src/prep_recflow.py
"""
import glob
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "recflow"
OUT = ROOT / "data" / "recflow.npz"


def main():
    import pandas as pd

    files = sorted(glob.glob(str(RAW / "*.feather")))
    if not files:
        raise SystemExit(
            f"No .feather files found under {RAW}; download RecFlow raw data first."
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"{len(files)} daily files: {Path(files[0]).stem} .. "
          f"{Path(files[-1]).stem}")
    parts, skipped = [], []
    for fp in files:
        try:
            df = pd.read_feather(fp, columns=["user_id", "video_id",
                                              "request_timestamp",
                                              "effective_view"])
        except Exception as ex:
            print(f"skip {Path(fp).name}: {type(ex).__name__} (truncated?)")
            skipped.append(Path(fp).name)
            continue
        parts.append(df)
    # A silently short npz changes every window volume downstream, and a
    # partially-downloaded day looks exactly like a real gap in the log.
    # Fail loudly instead: the caller must acknowledge an incomplete build.
    if skipped and not os.environ.get("ALLOW_PARTIAL"):
        raise SystemExit(
            f"{len(skipped)} of {len(files)} daily files failed to read "
            f"({', '.join(skipped)}). Re-download them, or re-run with "
            f"ALLOW_PARTIAL=1 to build a knowingly incomplete npz.")
    if not parts:
        raise SystemExit(
            f"All {len(files)} daily files under {RAW} failed to read: {skipped}"
        )
    df = pd.concat(parts, ignore_index=True)
    print(f"rows={len(df):,} from {len(parts)} of {len(files)} daily files")

    ts = (df["request_timestamp"].to_numpy() // 1000).astype(np.int64)
    user = df["user_id"].to_numpy()
    item = df["video_id"].to_numpy()
    label = df["effective_view"].to_numpy().astype(np.int8)

    u_uniq, u_codes = np.unique(user, return_inverse=True)
    i_uniq, i_codes = np.unique(item, return_inverse=True)
    order = np.argsort(ts, kind="stable")
    np.savez_compressed(OUT, user=u_codes[order].astype(np.int32),
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
    print("daily volume: min=%d median=%d max=%d" %
          (dc.min(), np.median(dc), dc.max()))


if __name__ == "__main__":
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__.strip())
        sys.exit(0)
    main()
