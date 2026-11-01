import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def history():
    index = pd.date_range("2025-01-01", periods=500, freq="h")
    x = np.arange(len(index), dtype=float)
    return pd.DataFrame({"FINAL_TOC": 10 + np.sin(x / 24), "FLOW": 100 + x / 100,
                         "EQ_TOC": 30 + np.sin(x / 12), "DAF_TOC": 20 + np.sin(x / 12),
                         "AERA_DO": 1.8 + x / 10000, "AERB_DO": 2 + x / 10000,
                         "AERA_MLSS": 2500 + x, "AERB_MLSS": 2400 + x,
                         "AERB_RET": 120 + np.sin(x / 3), "AERB_WIT": 5 + np.cos(x / 4)}, index=index)
