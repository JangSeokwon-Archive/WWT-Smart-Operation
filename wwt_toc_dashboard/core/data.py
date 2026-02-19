import pandas as pd
import numpy as np

def load_raw_csv(path: str) -> pd.DataFrame:
    # Support UTF-8 and Korean-encoded csvs used by automl pipeline.
    last_err = None
    df = None
    for enc in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
        try:
            df = pd.read_csv(path, low_memory=False, encoding=enc)
            break
        except Exception as e:
            last_err = e
    if df is None:
        raise last_err if last_err else ValueError(f"CSV read failed: {path}")

    if "날짜" not in df.columns:
        raise ValueError("raw.csv에 '날짜' 컬럼이 없습니다.")
    df = df.dropna(subset=["날짜"]).copy()
    # Support both "26-1-30 23:00" and "2026-01-30 23:00" layouts.
    ts = pd.to_datetime(df["날짜"], format="%y-%m-%d %H:%M", errors="coerce")
    if ts.isna().mean() > 0.2:
        ts2 = pd.to_datetime(df["날짜"], format="%Y-%m-%d %H:%M", errors="coerce")
        ts = ts2.where(ts.isna(), ts)
    if ts.isna().any():
        ts3 = pd.to_datetime(df["날짜"], errors="coerce")
        ts = ts3.where(ts.isna(), ts)
    df["날짜"] = ts
    df = df.dropna(subset=["날짜"]).sort_values("날짜")
    df = df.set_index("날짜")

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def pick_latest(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return float("nan")
    s = df[col].dropna()
    return float(s.iloc[-1]) if len(s) else float("nan")

def slice_by_range(df: pd.DataFrame, range_key: str) -> pd.DataFrame:
    now = df.index.max()
    if pd.isna(now):
        return df
    mapping = {"24h": 1, "3d": 3, "7d": 7, "30d": 30, "90d": 90}
    days = mapping.get(range_key, 30)
    start = now - pd.Timedelta(days=days)
    return df.loc[df.index >= start].copy()

def ensure_pred_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["pred_final", "pred_t12", "pred_t24", "pred_t36"]:
        if c not in df.columns:
            df[c] = np.nan
    return df
