import re
import warnings
import pandas as pd

def _get_prefix(col: str) -> str | None:
    # 예: AERA_DO -> AERA
    m = re.match(r"^([A-Z]+[A-Z0-9]*)_", col)
    return m.group(1) if m else None

def _add_trend_features(df: pd.DataFrame, cols: list[str], lags: list[int], rolls: list[int]) -> None:
    for col in cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        for k in lags:
            df[f"{col}__trend_lag{k}h"] = s.shift(k)
        for w in rolls:
            df[f"{col}__trend_rollmean{w}h"] = s.rolling(w).mean()

        # slope / momentum features (explicit trend shape)
        df[f"{col}__slope3h"] = (s - s.shift(3)) / 3.0
        df[f"{col}__slope6h"] = (s - s.shift(6)) / 6.0
        df[f"{col}__slope12h"] = (s - s.shift(12)) / 12.0
        df[f"{col}__acc3to12h"] = df[f"{col}__slope3h"] - df[f"{col}__slope12h"]
        df[f"{col}__zscore24h"] = (s - s.rolling(24).mean()) / (s.rolling(24).std() + 1e-6)
        df[f"{col}__zscore72h"] = (s - s.rolling(72).mean()) / (s.rolling(72).std() + 1e-6)

def build_features_equipment_aware(
    df: pd.DataFrame,
    time_col: str,
    target_col: str,
    equipment_prefixes: list[str],
    global_cols: list[str],
    global_feature_params: dict | None,
    per_equipment: dict,
    cross_features: dict,
    leakage_guard: dict,
) -> pd.DataFrame:
    df = df.copy()

    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    trend_cfg = cross_features.get("trend_features", {}) if isinstance(cross_features, dict) else {}
    trend_cols_cfg = set(trend_cfg.get("cols", ["FINAL_TOC_raw", "EQ_TOC", "DAF_TOC", "AER_TOC__mean"]))

    # ---- 1) feature 후보 컬럼 분류
    feat_cols = [c for c in df.columns if c not in [time_col, target_col]]

    # 누출 방지: FINAL_TOC는 feature에서 제거
    if leakage_guard.get("drop_final_toc_as_feature", True):
        if "FINAL_TOC" in feat_cols:
            feat_cols.remove("FINAL_TOC")

    # 그룹 맵: {prefix: [cols...]}
    groups: dict[str, list[str]] = {p: [] for p in equipment_prefixes}
    others: list[str] = []

    for c in feat_cols:
        if c in global_cols:
            continue
        p = _get_prefix(c)
        if p in groups:
            groups[p].append(c)
        else:
            others.append(c)

        # ---- 0) prefix 단위 컬럼 제외 (운영 예측형에서 FINAL 제거 등)
    exclude_prefixes = leakage_guard.get("exclude_prefixes", [])
    if exclude_prefixes:
        drop_cols = []
        for c in feat_cols:
            if c in trend_cols_cfg:
                continue
            p = _get_prefix(c)
            if p in exclude_prefixes:
                drop_cols.append(c)
        if drop_cols:
            df = df.drop(columns=drop_cols, errors="ignore")
            feat_cols = [c for c in feat_cols if c not in drop_cols]

    # ---- 2) 그룹별 lag/rolling/diff 생성
    for prefix, cols in groups.items():
        if not cols:
            continue
        cfg = per_equipment.get(prefix, {})
        lags = cfg.get("lags", [])
        rolls = cfg.get("rolls", [])
        diffs = cfg.get("diffs", [])

        for col in cols:
            # lag
            for k in lags:
                df[f"{col}__lag{k}h"] = df[col].shift(k)

            # rolling mean
            for w in rolls:
                df[f"{col}__rollmean{w}h"] = df[col].rolling(w).mean()

            # diff
            for d in diffs:
                df[f"{col}__diff{d}h"] = df[col].diff(d)

    # ---- 3) global cols 처리(FLOW/TEMP)
    gcfg = global_feature_params or {}
    g_lags = gcfg.get("lags", [1, 2, 3, 6, 12, 24])
    g_rolls = gcfg.get("rolls", [3, 6, 12, 24])
    g_diffs = gcfg.get("diffs", [1, 3, 6])

    for col in global_cols:
        if col not in df.columns:
            continue
        for k in g_lags:
            df[f"{col}__lag{k}h"] = df[col].shift(k)
        for w in g_rolls:
            df[f"{col}__rollmean{w}h"] = df[col].rolling(w).mean()
        for d in g_diffs:
            df[f"{col}__diff{d}h"] = df[col].diff(d)

    # ---- 4) A/B 병렬 설비 교차피처 (평균/차이)
    def mean_diff(a: str, b: str, name: str):
        if a in df.columns and b in df.columns:
            df[f"{name}__mean"] = (df[a] + df[b]) / 2
            df[f"{name}__diff"] = (df[a] - df[b])

    if cross_features.get("aer_ab_mean_diff", True):
        for suffix in ["TOC", "pH", "DO", "MLSS", "WIT", "RET", "SV", "SRT", "FM", "SVI", "RET_CON"]:
            mean_diff(f"AERA_{suffix}", f"AERB_{suffix}", f"AER_{suffix}")

    if cross_features.get("cla_ab_mean_diff", True):
        for suffix in ["TOC", "pH", "SS"]:
            mean_diff(f"CLAA_{suffix}", f"CLAB_{suffix}", f"CLA_{suffix}")

    # ---- 5) explicit trend features (final toc inertia + EQ/DAF trend)
    if trend_cfg.get("enabled", True):
        trend_cols = list(trend_cols_cfg)
        trend_lags = trend_cfg.get("lags", [1, 3, 6, 12, 24, 48])
        trend_rolls = trend_cfg.get("rolls", [6, 12, 24, 48])
        _add_trend_features(df, trend_cols, trend_lags, trend_rolls)

    # ---- 6) NaN 제거(shift/rolling 때문에 앞쪽 NaN 발생)
    df = df.dropna().reset_index(drop=True)
    return df
