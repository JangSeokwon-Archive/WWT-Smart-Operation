"""Deterministic model factories and purged out-of-fold residual fitting."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from core.forecast_evaluation import purged_folds

SEED = 42


def lgbm_params(estimators: int = 300) -> dict:
    return {"objective": "regression", "n_estimators": estimators, "learning_rate": 0.035,
            "num_leaves": 31, "min_child_samples": 30, "reg_lambda": 2.0,
            "colsample_bytree": 0.85, "random_state": SEED, "n_jobs": 1,
            "deterministic": True, "force_col_wise": True, "verbosity": -1}


@dataclass
class ResidualOOFModel:
    base: object
    correction: object

    @property
    def n_features_in_(self):
        return self.base.n_features_in_

    def predict(self, X):
        return self.base.predict(X) + self.correction.predict(X)


def fit_model(name: str, X: pd.DataFrame, y: pd.Series, horizon: int, delay: int,
              estimators: int = 300, legacy_params: dict | None = None):
    params = lgbm_params(estimators)
    if name == "legacy_feature_lgbm_refit":
        params.update(legacy_params or {})
        params.update(n_estimators=estimators, random_state=SEED, n_jobs=1, deterministic=True, force_col_wise=True, verbosity=-1)
    if name == "xgboost":
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=estimators, learning_rate=0.035, max_depth=4,
                             min_child_weight=10, reg_lambda=2.0, subsample=1.0,
                             colsample_bytree=0.85, random_state=SEED, n_jobs=1, tree_method="hist")
        return model.fit(X, y)
    if name == "lgbm_oof_residual":
        oof = pd.Series(np.nan, index=X.index)
        for train, valid in purged_folds(X.index, horizon, delay, n_splits=2):
            inner = LGBMRegressor(**params).fit(X.iloc[train], y.iloc[train])
            oof.iloc[valid] = inner.predict(X.iloc[valid])
        valid = oof.notna()
        residual_params = {**params, "n_estimators": min(100, estimators), "num_leaves": 7, "min_child_samples": 50}
        correction = LGBMRegressor(**residual_params).fit(X.loc[valid], y.loc[valid] - oof.loc[valid])
        base = LGBMRegressor(**params).fit(X, y)
        return ResidualOOFModel(base, correction)
    return LGBMRegressor(**params).fit(X, y)
