from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd


def _load_csv_auto(path: Path) -> pd.DataFrame:
    last_err: Exception | None = None
    for enc in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception as e:
            last_err = e
    raise last_err if last_err else RuntimeError("CSV load failed")


def main() -> None:
    ap = argparse.ArgumentParser(description="Create anonymized portfolio sample data from raw.csv")
    ap.add_argument("--input", type=str, default="data/raw.csv")
    ap.add_argument("--output", type=str, default="data/raw_sample.csv")
    ap.add_argument("--rows", type=int, default=24 * 45, help="keep last N rows (default: 45 days hourly)")
    ap.add_argument("--time-col", type=str, default="날짜")
    args = ap.parse_args()

    p_in = Path(args.input)
    p_out = Path(args.output)

    df = _load_csv_auto(p_in)
    if args.time_col in df.columns:
        ts = pd.to_datetime(df[args.time_col], errors="coerce", format="%Y-%m-%d %H:%M:%S")
        if not ts.notna().any():
            ts = pd.to_datetime(df[args.time_col], errors="coerce")
        if ts.notna().any():
            # Shift timeline so portfolio data does not expose real operation dates.
            shift = pd.Timestamp("2031-01-01 00:00:00") - ts.dropna().iloc[0]
            df[args.time_col] = (ts + shift).dt.strftime("%Y-%m-%d %H:%M:%S")

    if args.rows > 0 and len(df) > args.rows:
        df = df.tail(args.rows).copy()

    preferred = [
        args.time_col,
        "FINAL_TOC",
        "FLOW",
        "TEMP",
        "AERB_DO",
        "AERB_RET",
        "AERB_WIT",
        "AERB_MLSS",
        "pred_t6",
        "pred_t12",
        "pred_t24",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]

    p_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p_out, index=False, encoding="utf-8-sig")
    print(f"saved: {p_out} rows={len(df)} cols={len(df.columns)}")


if __name__ == "__main__":
    main()
