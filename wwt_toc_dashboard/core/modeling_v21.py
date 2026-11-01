"""Experimental change forecasts and persistence hybrids; no automatic promotion."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from core.forecast_training import lgbm_params

REGIME_COLUMNS = ("organic_load_proxy__shock", "EQ_TOC__shock", "TOC_AVAILABLE__diff12", "TOC_AVAILABLE__diff3", "TOC_AVAILABLE__missing")


def seasonal_baseline(observed: pd.Series, available: pd.Series, horizon: int, delay: int) -> pd.Series:
    """Use the most recent available observation at the target's daily clock time."""
    lag = 24 - horizon
    while lag < delay:
        lag += 24
    return observed.shift(lag).fillna(available)


def regime_matrix(X: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=X.index)
    for col in REGIME_COLUMNS:
        if col == "organic_load_proxy__shock" and col not in X:
            result[col] = X.get("EQ_TOC__shock", pd.Series(np.nan, index=X.index))
        else:
            result[col] = X.get(col, pd.Series(np.nan, index=X.index))
    return result.replace([np.inf, -np.inf], np.nan)


def choose_columns(X: pd.DataFrame, y: pd.Series, n: int, estimators: int = 80) -> list[str]:
    """Select with training rows only; rank gain and remove highly correlated duplicates."""
    model = LGBMRegressor(**lgbm_params(estimators)).fit(X, y)
    rank = np.argsort(-model.booster_.feature_importance(importance_type="gain"), kind="stable")
    candidates = [X.columns[i] for i in rank[: min(len(rank), n * 3)] if model.booster_.feature_importance(importance_type="gain")[i] > 0]
    # Correlations are estimated from a bounded sample of the same training fold.
    corr = X[candidates].tail(3000).corr().abs()
    selected = []
    for name in candidates:
        if all(not np.isfinite(corr.loc[name, other]) or corr.loc[name, other] < 0.98 for other in selected):
            selected.append(name)
        if len(selected) == n:
            break
    if "TOC_AVAILABLE" not in selected:
        selected.append("TOC_AVAILABLE")
    return selected


@dataclass
class DeltaModel:
    model: object
    columns: list[str]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        current = X["TOC_AVAILABLE"].to_numpy(dtype=float)
        return current + self.model.predict(X[self.columns])


def fit_delta(X: pd.DataFrame, y: pd.Series, *, selected: bool = False, estimators: int = 150) -> DeltaModel:
    if X["TOC_AVAILABLE"].isna().any():
        raise ValueError("Delta training requires an available target baseline")
    delta = y - X["TOC_AVAILABLE"]
    columns = choose_columns(X, delta, 60) if selected else list(X.columns)
    model = LGBMRegressor(**lgbm_params(estimators)).fit(X[columns], delta)
    return DeltaModel(model, columns)


def fit_fixed_weight(y, base, ml) -> float:
    y, base, ml = (np.asarray(x, dtype=float) for x in (y, base, ml))
    weights = np.linspace(0, 1, 21)
    return float(min(weights, key=lambda w: np.mean(np.abs(y - (base + w * (ml - base))))))


def fit_regime_weights(y, base, ml, load, threshold) -> tuple[float, float]:
    mask = np.isfinite(load) & (np.asarray(load) >= threshold)
    y, base, ml = (np.asarray(x, dtype=float) for x in (y, base, ml))
    overall = fit_fixed_weight(y, base, ml)
    return tuple(fit_fixed_weight(y[m], base[m], ml[m]) if m.sum() >= 50 else overall for m in (~mask, mask))


def fit_learned_gate(X, y, base, ml):
    difference = np.asarray(ml) - np.asarray(base)
    # A ridge gate predicts a bounded mixing weight from OOF forecasts only.
    weight = np.clip((np.asarray(y) - np.asarray(base)) / np.where(np.abs(difference) > 0.1, difference, np.nan), 0, 1)
    mask = np.isfinite(weight)
    if mask.sum() < 100:
        return None
    gate = make_pipeline(SimpleImputer(strategy="constant", fill_value=0), StandardScaler(), Ridge(alpha=100))
    return gate.fit(regime_matrix(X).loc[mask], weight[mask])


def blend(base, ml, weights):
    base, ml, weights = (np.asarray(v, dtype=float) for v in (base, ml, weights))
    # Missing or non-finite ML outputs must fall back to persistence.
    return np.where(np.isfinite(ml) & np.isfinite(weights), base + np.clip(weights, 0, 1) * (ml - base), base)


def block_bootstrap_difference(y, ref, candidate, index=None, block: int = 24, draws: int = 1000, seed: int = 42):
    diff = np.abs(np.asarray(y) - np.asarray(ref)) - np.abs(np.asarray(y) - np.asarray(candidate))
    finite = np.isfinite(diff)
    index = pd.DatetimeIndex(index)[finite] if index is not None else None
    diff = diff[finite]
    if len(diff) < block * 2:
        return None
    rng = np.random.default_rng(seed)
    starts = np.arange(len(diff) - block + 1)
    if index is not None:
        starts = np.array([s for s in starts if index[s + block - 1] - index[s] == pd.Timedelta(hours=block - 1)], dtype=int)
    if len(starts) == 0:
        return None
    values = []
    for _ in range(draws):
        chosen = rng.choice(starts, size=int(np.ceil(len(diff) / block)), replace=True)
        samples = np.concatenate([diff[s:s + block] for s in chosen])[:len(diff)]
        values.append(float(samples.mean()))
    return {"mean_mae_reduction": float(diff.mean()), "ci95": np.quantile(values, [0.025, 0.975]).tolist(),
            "block_hours": block, "draws": draws, "contiguous_hours_required": index is not None}
