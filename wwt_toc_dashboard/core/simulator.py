import numpy as np
import pandas as pd

def smooth_step(n: int, ramp: int = 12):
    ramp = max(1, min(ramp, n))
    x = np.linspace(-3, 3, ramp)
    s = 1 / (1 + np.exp(-x))
    w = np.ones(n)
    w[:ramp] = s
    return w

def _poly_features(do_v: np.ndarray, ret_v: np.ndarray, wit_v: np.ndarray) -> np.ndarray:
    return np.column_stack([
        do_v,
        ret_v,
        wit_v,
        do_v ** 2,
        ret_v ** 2,
        wit_v ** 2,
        do_v * ret_v,
        do_v * wit_v,
        ret_v * wit_v,
    ])

def fit_quadratic_response_model(
    df: pd.DataFrame,
    target_col: str = "FINAL_TOC",
    do_col: str = "AERB_DO",
    ret_col: str = "AERB_RET",
    wit_col: str = "AERB_WIT",
    ridge: float = 1.0,
) -> dict:
    cols = [target_col, do_col, ret_col, wit_col]
    miss = [c for c in cols if c not in df.columns]
    if miss:
        return {"ok": False, "reason": f"missing columns: {miss}"}

    sub = df[cols].dropna().copy()
    if len(sub) < 300:
        return {"ok": False, "reason": "too few rows"}

    # Robust clipping to reduce extreme raw spikes impact.
    for c in [target_col, do_col, ret_col, wit_col]:
        lo, hi = sub[c].quantile(0.005), sub[c].quantile(0.995)
        sub = sub[(sub[c] >= lo) & (sub[c] <= hi)]
    if len(sub) < 300:
        return {"ok": False, "reason": "too few rows after clipping"}

    y = sub[target_col].to_numpy(dtype=float)
    do_v = sub[do_col].to_numpy(dtype=float)
    ret_v = sub[ret_col].to_numpy(dtype=float)
    wit_v = sub[wit_col].to_numpy(dtype=float)
    f = _poly_features(do_v, ret_v, wit_v)

    n = len(f)
    cut = max(200, int(n * 0.8))
    f_tr, f_va = f[:cut], f[cut:]
    y_tr, y_va = y[:cut], y[cut:]

    mu = f_tr.mean(axis=0)
    sigma = f_tr.std(axis=0)
    sigma[sigma < 1e-8] = 1.0

    x_tr = np.column_stack([np.ones(len(f_tr)), (f_tr - mu) / sigma])
    x_va = np.column_stack([np.ones(len(f_va)), (f_va - mu) / sigma]) if len(f_va) else np.empty((0, x_tr.shape[1]))

    reg = np.eye(x_tr.shape[1]) * float(ridge)
    reg[0, 0] = 0.0
    beta = np.linalg.solve((x_tr.T @ x_tr) + reg, (x_tr.T @ y_tr))

    pred_tr = x_tr @ beta
    tr_rmse = float(np.sqrt(np.mean((y_tr - pred_tr) ** 2)))
    if len(f_va):
        pred_va = x_va @ beta
        va_rmse = float(np.sqrt(np.mean((y_va - pred_va) ** 2)))
    else:
        va_rmse = float("nan")

    return {
        "ok": True,
        "beta": beta.tolist(),
        "mu": mu.tolist(),
        "sigma": sigma.tolist(),
        "n_rows": int(n),
        "n_train": int(len(f_tr)),
        "n_valid": int(len(f_va)),
        "rmse_train": tr_rmse,
        "rmse_valid": va_rmse,
        "target_col": target_col,
        "do_col": do_col,
        "ret_col": ret_col,
        "wit_col": wit_col,
    }

def predict_response(model: dict, do_v: float, ret_v: float, wit_v: float) -> float:
    if not model or not model.get("ok"):
        return float("nan")
    if not (np.isfinite(do_v) and np.isfinite(ret_v) and np.isfinite(wit_v)):
        return float("nan")
    beta = np.asarray(model["beta"], dtype=float)
    mu = np.asarray(model["mu"], dtype=float)
    sigma = np.asarray(model["sigma"], dtype=float)
    f = _poly_features(
        np.array([float(do_v)], dtype=float),
        np.array([float(ret_v)], dtype=float),
        np.array([float(wit_v)], dtype=float),
    )[0]
    x = np.concatenate(([1.0], (f - mu) / sigma))
    return float(x @ beta)

def operational_safety_penalty(do_v: float, ret_v: float, wit_v: float) -> float:
    # Process guard: over-aeration / oxygen shortage / too-high return can hurt treatment stability.
    over_do = max(0.0, float(do_v) - 2.8)
    low_do = max(0.0, 1.0 - float(do_v))
    over_ret = max(0.0, float(ret_v) - 170.0)
    low_ret = max(0.0, 90.0 - float(ret_v))
    over_wit = max(0.0, float(wit_v) - 9.5)
    return (0.75 * (over_do ** 2)) + (0.50 * (low_do ** 2)) + (0.012 * over_ret) + (0.010 * low_ret) + (0.10 * over_wit)

def do_optimal_target_by_mlss(current_mlss: float, mlss_q_low: float, mlss_q_high: float) -> float:
    # Operationally conservative DO setpoint window by MLSS regime.
    if not (np.isfinite(current_mlss) and np.isfinite(mlss_q_low) and np.isfinite(mlss_q_high) and mlss_q_high > mlss_q_low):
        return 1.9
    if current_mlss <= mlss_q_low:
        return 1.75
    if current_mlss >= mlss_q_high:
        return 2.15
    t = (current_mlss - mlss_q_low) / (mlss_q_high - mlss_q_low)
    return float(1.75 * (1.0 - t) + 2.15 * t)

def do_overaeration_penalty(do_v: float, do_opt: float) -> float:
    # Symmetric basin around do_opt plus strong extra penalty above high bound.
    x = float(do_v)
    opt = float(do_opt)
    basin = 0.85 * ((x - opt) ** 2)
    high = max(0.0, x - (opt + 0.55))
    low = max(0.0, (opt - 0.85) - x)
    return basin + (2.6 * (high ** 2)) + (0.9 * (low ** 2))

def _gaussian_kernel(horizon_hours: int, peak_h: float, width_h: float = 6.0) -> np.ndarray:
    h = np.arange(1, int(horizon_hours) + 1, dtype=float)
    w = np.exp(-0.5 * ((h - float(peak_h)) / max(float(width_h), 1.0)) ** 2)
    s = float(w.sum())
    if s <= 0:
        return np.ones_like(h) / len(h)
    return w / s

def _mlss_scale(cur_mlss: float, q_low: float, q_high: float, low_eff: float, mid_eff: float, high_eff: float) -> float:
    if not (np.isfinite(cur_mlss) and np.isfinite(q_low) and np.isfinite(q_high) and q_high > q_low):
        return 1.0
    if cur_mlss <= q_low:
        ref = low_eff
    elif cur_mlss >= q_high:
        ref = high_eff
    else:
        t = (cur_mlss - q_low) / (q_high - q_low)
        ref = low_eff * (1.0 - t) + high_eff * t
    den = mid_eff if np.isfinite(mid_eff) and abs(mid_eff) > 1e-8 else 1.0
    return float(ref / den)

def delay_profile_effect_curve(
    horizon_hours: int,
    do_delta: float,
    ret_delta_pct: float,
    wit_delta: float,
    delay_profile: dict | None,
    current_mlss: float = float("nan"),
    mlss_q_low: float = float("nan"),
    mlss_q_high: float = float("nan"),
    cap: float = 3.5,
) -> np.ndarray:
    """
    Build cumulative TOC delta curve from multivariate delay profile (automl report).
    Returns length horizon_hours+1 with 0 at index 0.
    """
    n = int(horizon_hours)
    if n <= 0 or not delay_profile:
        return np.zeros(max(n + 1, 1), dtype=float)
    summary = delay_profile.get("summary", {})
    if not isinstance(summary, dict) or not summary:
        return np.zeros(n + 1, dtype=float)

    ctrl_to_delta = {
        "AERB_DO": float(do_delta),
        "AERB_RET": float(ret_delta_pct),  # absolute %p shift
        "AERB_WIT": float(wit_delta),
    }
    dy_total = np.zeros(n, dtype=float)

    for ctrl, dval in ctrl_to_delta.items():
        if ctrl not in summary or not np.isfinite(dval) or abs(dval) < 1e-12:
            continue
        s = summary.get(ctrl, {})
        peak = float(s.get("peak_lag_main_h", 12))
        mid_eff = float(s.get("peak_coef_main", 0.0))
        low_eff = float(s.get("peak_eff_low_mlss", mid_eff))
        high_eff = float(s.get("peak_eff_high_mlss", mid_eff))
        cum_24_main = float(s.get("cum_coef_0_6h", 0.0)) + float(s.get("cum_coef_7_24h", 0.0))
        if not np.isfinite(cum_24_main):
            cum_24_main = 0.0

        scale = _mlss_scale(
            cur_mlss=float(current_mlss),
            q_low=float(mlss_q_low),
            q_high=float(mlss_q_high),
            low_eff=low_eff,
            mid_eff=mid_eff,
            high_eff=high_eff,
        )
        target_24 = cum_24_main * scale * dval
        ker = _gaussian_kernel(horizon_hours=n, peak_h=peak, width_h=6.0)
        dy_total += target_24 * ker

    level = np.concatenate(([0.0], np.cumsum(dy_total)))
    return np.clip(level, -float(cap), float(cap))

def apply_simulation(base_pred: pd.Series, do_delta: float, ret_delta_pct: float, wit_delta: float,
                     alpha_do: float = -0.55, alpha_ret: float = -1.4, alpha_wit: float = 0.22,
                     ramp_hours: int = 12, cap: float = 5.0,
                     current_do: float = float("nan"), current_ret: float = float("nan"), current_wit: float = float("nan"),
                     response_model: dict | None = None,
                     current_mlss: float = float("nan"), mlss_q_low: float = float("nan"), mlss_q_high: float = float("nan")) -> pd.Series:
    n = len(base_pred)
    if n == 0:
        return base_pred

    do_now = float(current_do) if np.isfinite(current_do) else 1.8
    ret_now = float(current_ret) if np.isfinite(current_ret) else 140.0
    wit_now = float(current_wit) if np.isfinite(current_wit) else 6.0
    do_sim = do_now + float(do_delta)
    # RET slider uses absolute percentage-point shift (e.g., +10 means +10%p).
    ret_sim = ret_now + float(ret_delta_pct)
    wit_sim = wit_now + float(wit_delta)

    if response_model and response_model.get("ok"):
        y_base = predict_response(response_model, do_now, ret_now, wit_now)
        y_sim = predict_response(response_model, do_sim, ret_sim, wit_sim)
        if np.isfinite(y_base) and np.isfinite(y_sim):
            delta = y_sim - y_base
        else:
            delta = alpha_do * do_delta + alpha_ret * np.tanh(float(ret_delta_pct) / 35.0) + alpha_wit * wit_delta
    else:
        delta = alpha_do * do_delta + alpha_ret * np.tanh(float(ret_delta_pct) / 35.0) + alpha_wit * wit_delta

    # Hybrid process guard (physics/operations prior).
    delta += operational_safety_penalty(do_sim, ret_sim, wit_sim) - operational_safety_penalty(do_now, ret_now, wit_now)
    do_opt = do_optimal_target_by_mlss(current_mlss=current_mlss, mlss_q_low=mlss_q_low, mlss_q_high=mlss_q_high)
    delta += do_overaeration_penalty(do_sim, do_opt) - do_overaeration_penalty(do_now, do_opt)

    delta = float(np.clip(delta, -cap, cap))
    w = smooth_step(n=n, ramp=ramp_hours)
    return base_pred + delta * w


def _base_curve_24h(current_toc: float, p6: float, p12: float, p24: float) -> np.ndarray:
    y0 = float(current_toc) if np.isfinite(current_toc) else float("nan")
    y6 = float(p6) if np.isfinite(p6) else y0
    y12 = float(p12) if np.isfinite(p12) else y6
    y24 = float(p24) if np.isfinite(p24) else y12
    x_anchor = np.array([0.0, 6.0, 12.0, 24.0], dtype=float)
    y_anchor = np.array([y0, y6, y12, y24], dtype=float)
    x = np.arange(0.0, 25.0, 1.0)
    return np.interp(x, x_anchor, y_anchor)


def estimate_sim_t24(
    current_toc: float,
    p6: float,
    p12: float,
    p24: float,
    current_do: float,
    current_ret: float,
    current_wit: float,
    do_delta: float,
    ret_delta_pct: float,
    wit_delta: float,
    response_model: dict | None = None,
    delay_profile: dict | None = None,
    current_mlss: float = float("nan"),
    mlss_q_low: float = float("nan"),
    mlss_q_high: float = float("nan"),
) -> dict:
    base_curve = _base_curve_24h(current_toc=current_toc, p6=p6, p12=p12, p24=p24)
    base_t12 = float(base_curve[12]) if len(base_curve) > 12 else float("nan")
    base_t24 = float(base_curve[-1]) if len(base_curve) else float("nan")

    do_now = float(current_do) if np.isfinite(current_do) else 1.8
    ret_now = float(current_ret) if np.isfinite(current_ret) else 140.0
    wit_now = float(current_wit) if np.isfinite(current_wit) else 6.0
    do_sim = do_now + float(do_delta)
    ret_sim = ret_now + float(ret_delta_pct)
    wit_sim = wit_now + float(wit_delta)

    static_delta = 0.0
    if response_model and response_model.get("ok"):
        y_base = predict_response(response_model, do_now, ret_now, wit_now)
        y_sim = predict_response(response_model, do_sim, ret_sim, wit_sim)
        if np.isfinite(y_base) and np.isfinite(y_sim):
            static_delta += float(y_sim - y_base)

    static_delta += operational_safety_penalty(do_sim, ret_sim, wit_sim) - operational_safety_penalty(do_now, ret_now, wit_now)
    do_opt = do_optimal_target_by_mlss(current_mlss=current_mlss, mlss_q_low=mlss_q_low, mlss_q_high=mlss_q_high)
    static_delta += do_overaeration_penalty(do_sim, do_opt) - do_overaeration_penalty(do_now, do_opt)
    static_delta = float(np.clip(static_delta, -3.5, 3.5))

    delay_curve = delay_profile_effect_curve(
        horizon_hours=24,
        do_delta=do_delta,
        ret_delta_pct=ret_delta_pct,
        wit_delta=wit_delta,
        delay_profile=delay_profile,
        current_mlss=current_mlss,
        mlss_q_low=mlss_q_low,
        mlss_q_high=mlss_q_high,
        cap=3.5,
    )
    delay_t24 = float(delay_curve[-1]) if len(delay_curve) else 0.0
    delay_t12 = float(delay_curve[12]) if len(delay_curve) > 12 else 0.0
    sim_t12 = float(base_t12 + static_delta + delay_t12)
    sim_t24 = float(base_t24 + static_delta + delay_t24)

    return {
        "base_t12": base_t12,
        "base_t24": base_t24,
        "sim_t12": sim_t12,
        "sim_t24": sim_t24,
        "delta_t12": float(sim_t12 - base_t12),
        "delta_t24": float(sim_t24 - base_t24),
        "do_delta": float(do_delta),
        "ret_delta_pct": float(ret_delta_pct),
        "wit_delta": float(wit_delta),
    }


def recommend_optimal_controls_24h(
    current_toc: float,
    p6: float,
    p12: float,
    p24: float,
    current_do: float,
    current_ret: float,
    current_wit: float,
    response_model: dict | None = None,
    delay_profile: dict | None = None,
    current_mlss: float = float("nan"),
    mlss_q_low: float = float("nan"),
    mlss_q_high: float = float("nan"),
) -> dict:
    do_grid = [-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6]
    ret_grid = [-10.0, -6.0, -3.0, 0.0, 3.0, 6.0, 10.0]
    wit_grid = [-20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0]

    base = estimate_sim_t24(
        current_toc=current_toc,
        p6=p6,
        p12=p12,
        p24=p24,
        current_do=current_do,
        current_ret=current_ret,
        current_wit=current_wit,
        do_delta=0.0,
        ret_delta_pct=0.0,
        wit_delta=0.0,
        response_model=response_model,
        delay_profile=delay_profile,
        current_mlss=current_mlss,
        mlss_q_low=mlss_q_low,
        mlss_q_high=mlss_q_high,
    )
    base_t24 = float(base.get("sim_t24", float("nan")))

    cand: list[dict] = []
    for d_do in do_grid:
        for d_ret in ret_grid:
            for d_wit in wit_grid:
                do_abs = float(current_do) + float(d_do) if np.isfinite(current_do) else (1.8 + float(d_do))
                ret_abs = float(current_ret) + float(d_ret) if np.isfinite(current_ret) else (140.0 + float(d_ret))
                wit_abs = float(current_wit) + float(d_wit) if np.isfinite(current_wit) else (6.0 + float(d_wit))
                if not (0.8 <= do_abs <= 3.2):
                    continue
                if not (80.0 <= ret_abs <= 180.0):
                    continue
                if not (0.0 <= wit_abs <= 20.0):
                    continue

                est = estimate_sim_t24(
                    current_toc=current_toc,
                    p6=p6,
                    p12=p12,
                    p24=p24,
                    current_do=current_do,
                    current_ret=current_ret,
                    current_wit=current_wit,
                    do_delta=d_do,
                    ret_delta_pct=d_ret,
                    wit_delta=d_wit,
                    response_model=response_model,
                    delay_profile=delay_profile,
                    current_mlss=current_mlss,
                    mlss_q_low=mlss_q_low,
                    mlss_q_high=mlss_q_high,
                )
                est["improve_vs_base"] = float(base_t24 - float(est.get("sim_t24", float("nan"))))
                cand.append(est)

    if not cand:
        return {"ok": False, "reason": "no valid candidate"}

    ranked = sorted(cand, key=lambda x: float(x.get("sim_t24", 1e9)))
    best = ranked[0]
    top3 = ranked[:3]
    return {
        "ok": True,
        "base_t24": float(base_t24),
        "best": best,
        "top3": top3,
    }


def recommend_optimal_controls_12h(
    current_toc: float,
    p6: float,
    p12: float,
    p24: float,
    current_do: float,
    current_ret: float,
    current_wit: float,
    response_model: dict | None = None,
    delay_profile: dict | None = None,
    current_mlss: float = float("nan"),
    mlss_q_low: float = float("nan"),
    mlss_q_high: float = float("nan"),
) -> dict:
    do_grid = [-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6]
    ret_grid = [-10.0, -6.0, -3.0, 0.0, 3.0, 6.0, 10.0]
    wit_grid = [-20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0]

    base = estimate_sim_t24(
        current_toc=current_toc,
        p6=p6,
        p12=p12,
        p24=p24,
        current_do=current_do,
        current_ret=current_ret,
        current_wit=current_wit,
        do_delta=0.0,
        ret_delta_pct=0.0,
        wit_delta=0.0,
        response_model=response_model,
        delay_profile=delay_profile,
        current_mlss=current_mlss,
        mlss_q_low=mlss_q_low,
        mlss_q_high=mlss_q_high,
    )
    base_t12 = float(base.get("sim_t12", float("nan")))

    cand: list[dict] = []
    for d_do in do_grid:
        for d_ret in ret_grid:
            for d_wit in wit_grid:
                do_abs = float(current_do) + float(d_do) if np.isfinite(current_do) else (1.8 + float(d_do))
                ret_abs = float(current_ret) + float(d_ret) if np.isfinite(current_ret) else (140.0 + float(d_ret))
                wit_abs = float(current_wit) + float(d_wit) if np.isfinite(current_wit) else (6.0 + float(d_wit))
                if not (0.8 <= do_abs <= 3.2):
                    continue
                if not (80.0 <= ret_abs <= 180.0):
                    continue
                if not (0.0 <= wit_abs <= 20.0):
                    continue

                est = estimate_sim_t24(
                    current_toc=current_toc,
                    p6=p6,
                    p12=p12,
                    p24=p24,
                    current_do=current_do,
                    current_ret=current_ret,
                    current_wit=current_wit,
                    do_delta=d_do,
                    ret_delta_pct=d_ret,
                    wit_delta=d_wit,
                    response_model=response_model,
                    delay_profile=delay_profile,
                    current_mlss=current_mlss,
                    mlss_q_low=mlss_q_low,
                    mlss_q_high=mlss_q_high,
                )
                est["improve_vs_base"] = float(base_t12 - float(est.get("sim_t12", float("nan"))))
                cand.append(est)

    if not cand:
        return {"ok": False, "reason": "no valid candidate"}

    ranked = sorted(cand, key=lambda x: float(x.get("sim_t12", 1e9)))
    best = ranked[0]
    top3 = ranked[:3]
    return {
        "ok": True,
        "base_t12": float(base_t12),
        "best": best,
        "top3": top3,
    }
