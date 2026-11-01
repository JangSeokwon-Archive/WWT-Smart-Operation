from dataclasses import dataclass
import lightgbm as lgb
import numpy as np

@dataclass
class LGBMRegressorWrapper:
    params: dict

    def fit(self, X, y):
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
