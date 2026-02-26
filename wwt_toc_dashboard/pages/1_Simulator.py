import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from core.data import load_raw_csv, slice_by_range, ensure_pred_columns
from core.simulator import (
    apply_simulation,
    fit_quadratic_response_model,
    predict_response,
    operational_safety_penalty,
    delay_profile_effect_curve,
    do_optimal_target_by_mlss,
    do_overaeration_penalty,
)
from core.signals import classify_signal, badge_pill_html
from core.automl_infer import predict_multi_leads_from_automl
from core.settings_store import load_app_settings

st.set_page_config(page_title="Simulator", layout="wide", initial_sidebar_state="collapsed")

APP_DIR = Path(__file__).resolve().parents[1]
css_path = APP_DIR / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

raw_path = str(APP_DIR / "data" / "raw.csv")
automl_raw_path = APP_DIR.parent / "wwt-toc-automl" / "data" / "raw" / "raw.csv"
if automl_raw_path.exists():
    raw_path = str(automl_raw_path)
AUTOML_ROOT = APP_DIR.parent / "wwt-toc-automl"
MODEL_BUNDLE = APP_DIR / "model_bundle"
if AUTOML_ROOT.exists():
    REPORT_ROOT = AUTOML_ROOT / "outputs" / "reports"
else:
    REPORT_ROOT = MODEL_BUNDLE / "reports"
WARN_TOC = 15.0
ALARM_TOC = 20.0
APP_SETTINGS = load_app_settings(APP_DIR)
sim_default_target = float(APP_SETTINGS.get("sim_default_target_toc", 20.0))
target_limit = float(st.session_state.get("sim_target_limit", sim_default_target))
SIM_PRED_CACHE_VERSION = 4
SIM_RESP_CACHE_VERSION = 2


def calc_efficiency(inflow: float, outflow: float) -> float:
    if inflow is None:
        return float("nan")
    try:
        if inflow == 0:
            return float("nan")
        return (float(inflow) - float(outflow)) / float(inflow)
    except Exception:
        return float("nan")


@st.cache_data
def load_df(path: str):
    df0 = load_raw_csv(path)
    df0 = ensure_pred_columns(df0)
    return df0


@st.cache_data(ttl=60)
def load_model_preds(raw_path: str, automl_root_path: str, raw_mtime: float):
    cache_path = APP_DIR / ".cache" / "sim_model_preds.json"
    try:
        if cache_path.exists():
            c = json.loads(cache_path.read_text(encoding="utf-8"))
            if int(c.get("version", -1)) == int(SIM_PRED_CACHE_VERSION) and float(c.get("raw_mtime", -1.0)) == float(raw_mtime):
                preds = c.get("preds", {})
                if isinstance(preds, dict):
                    return preds
    except Exception:
        pass
    # Cloud-safe fallback: use existing pred columns in raw data if present.
    try:
        dfr = load_raw_csv(raw_path)
        out = {}
        for c in ("pred_t6", "pred_t12", "pred_t24"):
            if c in dfr.columns:
                v = pd.to_numeric(dfr[c], errors="coerce").dropna()
                out[c] = float(v.iloc[-1]) if len(v) else float("nan")
            else:
                out[c] = float("nan")
        if any(pd.notna(out.get(k, float("nan"))) for k in ("pred_t6", "pred_t12", "pred_t24")):
            return out
    except Exception:
        pass
    try:
        preds = predict_multi_leads_from_automl(
            raw_path=raw_path,
            automl_root_path=automl_root_path,
            leads=(6, 12, 24),
        )
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps({"version": int(SIM_PRED_CACHE_VERSION), "raw_mtime": float(raw_mtime), "preds": preds}, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass
        return preds
    except Exception:
        return {"pred_t6": float("nan"), "pred_t12": float("nan"), "pred_t24": float("nan")}

@st.cache_data(ttl=600)
def load_response_model(raw_path: str, raw_mtime: float):
    cache_path = APP_DIR / ".cache" / "sim_response_model.json"
    try:
        if cache_path.exists():
            c = json.loads(cache_path.read_text(encoding="utf-8"))
            if int(c.get("version", -1)) == int(SIM_RESP_CACHE_VERSION) and float(c.get("raw_mtime", -1.0)) == float(raw_mtime):
                model = c.get("model", {})
                if isinstance(model, dict) and model:
                    return model
    except Exception:
        pass
    try:
        df0 = load_raw_csv(raw_path)
        model = fit_quadratic_response_model(df0)
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps({"version": int(SIM_RESP_CACHE_VERSION), "raw_mtime": float(raw_mtime), "model": model}, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass
        return model
    except Exception:
        return {"ok": False}

@st.cache_data(ttl=600)
def load_delay_profile(path: str, mtime: float):
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _future_curve(
    last_actual: float,
    p6: float,
    p12: float,
    p24: float,
    do_delta: float,
    ret_delta: float,
    wit_delta: float,
    cur_do: float,
    cur_ret: float,
    cur_wit: float,
    cur_mlss: float,
    mlss_q_low: float,
    mlss_q_high: float,
    response_model: dict | None,
    delay_profile: dict | None,
    horizon_hours: int = 24,
):
    """Build baseline/simulated future curves from +1h..+24h with smooth bending."""
    if not np.isfinite(last_actual):
        return None, None
    if not np.isfinite(p6):
        p6 = last_actual
    if not np.isfinite(p12):
        p12 = p6
    if not np.isfinite(p24):
        p24 = p12

    x_anchor = np.array([0.0, 6.0, 12.0, 24.0], dtype=float)
    y_anchor = np.array([last_actual, p6, p12, p24], dtype=float)

    x = np.arange(0.0, float(horizon_hours) + 1.0, 1.0)
    base = np.interp(x, x_anchor, y_anchor)
    if horizon_hours > 24:
        slope24 = (p24 - p12) / 12.0
        tail_mask = x > 24.0
        base[tail_mask] = p24 + (x[tail_mask] - 24.0) * slope24

    do_now = float(cur_do) if np.isfinite(cur_do) else 1.8
    ret_now = float(cur_ret) if np.isfinite(cur_ret) else 140.0
    wit_now = float(cur_wit) if np.isfinite(cur_wit) else 6.0

    do_sim = do_now + float(do_delta)
    # RET slider uses absolute percentage-point shift.
    ret_sim = ret_now + float(ret_delta)
    wit_sim = wit_now + float(wit_delta)
    if response_model and response_model.get("ok"):
        y_base = predict_response(response_model, do_now, ret_now, wit_now)
        y_sim = predict_response(response_model, do_sim, ret_sim, wit_sim)
        delta = (y_sim - y_base) if (np.isfinite(y_base) and np.isfinite(y_sim)) else 0.0
    else:
        delta = 0.0
    delta += operational_safety_penalty(do_sim, ret_sim, wit_sim) - operational_safety_penalty(do_now, ret_now, wit_now)
    do_opt = do_optimal_target_by_mlss(current_mlss=cur_mlss, mlss_q_low=mlss_q_low, mlss_q_high=mlss_q_high)
    delta += do_overaeration_penalty(do_sim, do_opt) - do_overaeration_penalty(do_now, do_opt)
    delta = float(np.clip(delta, -3.5, 3.5))

    growth = 1.0 / (1.0 + np.exp(-(x - 8.0) / 3.5))
    wave = 1.0 + (0.12 * np.sin((x / max(horizon_hours, 1)) * (2.0 * np.pi)))
    static_curve = delta * growth * wave
    delay_curve = delay_profile_effect_curve(
        horizon_hours=horizon_hours,
        do_delta=do_delta,
        ret_delta_pct=ret_delta,
        wit_delta=wit_delta,
        delay_profile=delay_profile,
        current_mlss=cur_mlss,
        mlss_q_low=mlss_q_low,
        mlss_q_high=mlss_q_high,
        cap=3.5,
    )
    sim = base + static_curve + delay_curve

    return base, sim

df = load_df(raw_path)
raw_mtime = Path(raw_path).stat().st_mtime if Path(raw_path).exists() else 0.0
model_preds = load_model_preds(raw_path, str(AUTOML_ROOT), raw_mtime)
response_model = load_response_model(raw_path, raw_mtime)
reports_dir = REPORT_ROOT
search_summary_path = reports_dir / "control_delay_multivar_search_summary.json"
delay_profile_path = reports_dir / "control_delay_multivar_AERB_DO_AERB_WIT_AERB_RET_max72h.json"
if not delay_profile_path.exists():
    alt_delay = reports_dir / "control_delay_multivar_AERB_DO_AERB_RET_AERB_WIT_max72h.json"
    if alt_delay.exists():
        delay_profile_path = alt_delay
if search_summary_path.exists():
    try:
        rows = json.loads(search_summary_path.read_text(encoding="utf-8"))
        cand = [r for r in rows if isinstance(r, dict) and isinstance(r.get("rmse_valid_dy"), (int, float)) and r.get("file")]
        if cand:
            cand = sorted(cand, key=lambda r: float(r["rmse_valid_dy"]))
            p = Path(str(cand[0]["file"]))
            if p.exists():
                delay_profile_path = p
    except Exception:
        pass

delay_profile = load_delay_profile(
    str(delay_profile_path),
    (delay_profile_path.stat().st_mtime if delay_profile_path.exists() else 0.0),
)

# latest current values (for delta sliders)
cur_do_base = float(df["AERB_DO"].dropna().iloc[-1]) if "AERB_DO" in df.columns and len(df["AERB_DO"].dropna()) else float("nan")
cur_ret_base = float(df["AERB_RET"].dropna().iloc[-1]) if "AERB_RET" in df.columns and len(df["AERB_RET"].dropna()) else float("nan")
cur_wit_base = float(df["AERB_WIT"].dropna().iloc[-1]) if "AERB_WIT" in df.columns and len(df["AERB_WIT"].dropna()) else float("nan")

with st.container(border=False):
    st.markdown('<div id="sim_controls_anchor"></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    do_cur_txt = f"{cur_do_base:.2f}" if np.isfinite(cur_do_base) else "-"
    ret_cur_txt = f"{cur_ret_base:.1f}" if np.isfinite(cur_ret_base) else "-"
    wit_cur_txt = f"{cur_wit_base:.1f}" if np.isfinite(cur_wit_base) else "-"
    do_delta = c1.slider(f"DO양 Δ (ppm, 현재 {do_cur_txt})", -2.0, 2.0, 0.0, 0.1)
    ret_delta_pct = c2.slider(f"반송량 Δ (%p, 현재 {ret_cur_txt}%)", -30.0, 30.0, 0.0, 1.0)
    wit_delta = c3.slider(f"인발량 Δ (t/h, 현재 {wit_cur_txt})", -10.0, 10.0, 0.0, 1.0)

st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

a, b = st.columns([2.5, 1], gap="large")

with a:
    with st.container(border=False):
        st.markdown('<div id="sim_chart_anchor"></div>', unsafe_allow_html=True)
        top_l, top_r = st.columns([2.5, 2.5], vertical_alignment="center")
        with top_l:
            st.markdown('<div class="trend-title">SIMULATION TREND</div>', unsafe_allow_html=True)
        with top_r:
            st.markdown('<div id="sim_range_anchor"></div>', unsafe_allow_html=True)
            sim_range_options = ["24h", "3d", "7d", "30d", "90d"]
            if hasattr(st, "segmented_control"):
                range_key = st.segmented_control(
                    "Range",
                    sim_range_options,
                    default="24h",
                    label_visibility="collapsed",
                    key="sim_date_range_selector",
                )
            else:
                range_key = st.radio(
                    "Range",
                    sim_range_options,
                    horizontal=True,
                    index=0,
                    label_visibility="collapsed",
                    key="sim_date_range_selector",
                )
            if range_key not in sim_range_options:
                range_key = st.session_state.get("sim_date_range_selector_last", "24h")
            st.session_state["sim_date_range_selector_last"] = range_key

        view = slice_by_range(df, range_key)
        base = view["pred_final"].copy()
        if base.isna().all():
            base = view["FINAL_TOC"].copy()
        cur_do = float(view["AERB_DO"].dropna().iloc[-1]) if "AERB_DO" in view.columns and len(view["AERB_DO"].dropna()) else float("nan")
        cur_ret = float(view["AERB_RET"].dropna().iloc[-1]) if "AERB_RET" in view.columns and len(view["AERB_RET"].dropna()) else float("nan")
        cur_wit = float(view["AERB_WIT"].dropna().iloc[-1]) if "AERB_WIT" in view.columns and len(view["AERB_WIT"].dropna()) else float("nan")
        cur_mlss = float(view["AERB_MLSS"].dropna().iloc[-1]) if "AERB_MLSS" in view.columns and len(view["AERB_MLSS"].dropna()) else float("nan")
        mlss_all = pd.to_numeric(df["AERB_MLSS"], errors="coerce") if "AERB_MLSS" in df.columns else pd.Series(dtype=float)
        mlss_q_low = float(mlss_all.quantile(0.33)) if len(mlss_all.dropna()) else float("nan")
        mlss_q_high = float(mlss_all.quantile(0.67)) if len(mlss_all.dropna()) else float("nan")
        sim = apply_simulation(
            base_pred=base,
            do_delta=do_delta,
            ret_delta_pct=ret_delta_pct,
            wit_delta=wit_delta,
            current_do=cur_do,
            current_ret=cur_ret,
            current_wit=cur_wit,
            response_model=response_model,
            current_mlss=cur_mlss,
            mlss_q_low=mlss_q_low,
            mlss_q_high=mlss_q_high,
        )

        last_actual = float(view["FINAL_TOC"].dropna().iloc[-1]) if "FINAL_TOC" in view.columns and len(view["FINAL_TOC"].dropna()) else float("nan")
        last_idx = view.index.max() if len(view.index) else None
        p6 = float(view["pred_t6"].dropna().iloc[-1]) if "pred_t6" in view.columns and len(view["pred_t6"].dropna()) else float("nan")
        p12 = float(view["pred_t12"].dropna().iloc[-1]) if "pred_t12" in view.columns and len(view["pred_t12"].dropna()) else float("nan")
        p24 = float(view["pred_t24"].dropna().iloc[-1]) if "pred_t24" in view.columns and len(view["pred_t24"].dropna()) else float("nan")
        if pd.notna(model_preds.get("pred_t6", float("nan"))):
            p6 = float(model_preds["pred_t6"])
        if pd.notna(model_preds.get("pred_t12", float("nan"))):
            p12 = float(model_preds["pred_t12"])
        if pd.notna(model_preds.get("pred_t24", float("nan"))):
            p24 = float(model_preds["pred_t24"])
        horizon_h = 12
        fc_base, fc_sim = _future_curve(
            last_actual=last_actual,
            p6=p6,
            p12=p12,
            p24=p24,
            do_delta=do_delta,
            ret_delta=ret_delta_pct,
            wit_delta=wit_delta,
            cur_do=cur_do,
            cur_ret=cur_ret,
            cur_wit=cur_wit,
            cur_mlss=cur_mlss,
            mlss_q_low=mlss_q_low,
            mlss_q_high=mlss_q_high,
            response_model=response_model,
            delay_profile=delay_profile,
            horizon_hours=horizon_h,
        )

        base_latest = float(fc_base[-1]) if fc_base is not None else float(base.dropna().iloc[-1]) if len(base.dropna()) else float("nan")
        sim_latest = float(fc_sim[-1]) if fc_sim is not None else float(sim.dropna().iloc[-1]) if len(sim.dropna()) else float("nan")
        sim_delta = sim_latest - base_latest
        ret_now = float(view["AERB_RET"].dropna().iloc[-1]) if "AERB_RET" in view.columns and len(view["AERB_RET"].dropna()) else float("nan")
        ret_sim = ret_now + ret_delta_pct if pd.notna(ret_now) else float("nan")
        if np.isfinite(cur_mlss) and np.isfinite(mlss_q_low) and np.isfinite(mlss_q_high):
            if cur_mlss <= mlss_q_low:
                mlss_band = "LOW"
            elif cur_mlss >= mlss_q_high:
                mlss_band = "HIGH"
            else:
                mlss_band = "MID"
        else:
            mlss_band = "UNKNOWN"

        sig_base = classify_signal(base_latest, ALARM_TOC, warn_ratio=(WARN_TOC / ALARM_TOC))
        sig_sim = classify_signal(sim_latest, ALARM_TOC, warn_ratio=(WARN_TOC / ALARM_TOC))

        fig = go.Figure()
        # Historical actual is fixed solid line.
        fig.add_trace(
            go.Scatter(
                x=view.index,
                y=view["FINAL_TOC"] if "FINAL_TOC" in view.columns else base,
                mode="lines",
                name="Actual",
                line=dict(color="#8B5CF6", width=3, shape="spline"),
            )
        )

        # Future forecast is dotted, with slider-driven curved simulation line.
        if fc_base is not None and fc_sim is not None and last_idx is not None:
            fx = [last_idx + pd.Timedelta(hours=int(h)) for h in range(0, horizon_h + 1)]
            fig.add_trace(
                go.Scatter(
                    x=fx,
                    y=fc_base,
                    mode="lines",
                    name="기준 예측",
                    line=dict(color="#22D3EE", width=2, dash="dot", shape="spline"),
                )
            )
            fx_mark = fx[::4]
            yb_mark = list(fc_base[::4])
            ys_mark = list(fc_sim[::4])
            m_sizes = [5.0 + 1.8 * np.sin((i / max(len(fx_mark) - 1, 1)) * (2.0 * np.pi)) for i in range(len(fx_mark))]
            fig.add_trace(
                go.Scatter(
                    x=fx_mark,
                    y=yb_mark,
                    mode="markers",
                    name="기준 포인트",
                    marker=dict(size=m_sizes, color="#22D3EE", symbol="circle-open"),
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=fx,
                    y=fc_sim,
                    mode="lines",
                    name="조정 예측",
                    line=dict(color="#F59E0B", width=3, dash="dot", shape="spline"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=fx_mark,
                    y=ys_mark,
                    mode="markers",
                    name="조정 포인트",
                    marker=dict(size=m_sizes, color="#F59E0B", symbol="circle-open-dot"),
                    showlegend=False,
                )
            )

        fig.add_hline(y=target_limit, line_dash="dash", line_color="#EF4444", line_width=2, opacity=0.6)
        fig.update_layout(
            height=500,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", color="#C4C4D8", size=11),
            xaxis=dict(gridcolor='rgba(139, 92, 246, 0.1)', showline=False, tickformat='%m/%d'),
            yaxis=dict(gridcolor='rgba(139, 92, 246, 0.1)', showline=False, title=""),
            showlegend=True,
            legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right", font=dict(color="#FFFFFF")),
        )
        if len(view.index):
            days_map = {"24h": 1, "3d": 3, "7d": 7, "30d": 30, "90d": 90}
            hist_days = days_map.get(range_key, 1)
            now_x = df.index.max()
            x_start = now_x - pd.Timedelta(days=hist_days)
            # Keep forecast area around 1/3 of visible span (future:past ~= 1:2),
            # while limiting future display to <=24h.
            past_span_hours = max(1.0, float((now_x - x_start).total_seconds() / 3600.0))
            future_h = min(float(horizon_h), max(1.0, past_span_hours / 2.0))
            x_end = now_x + pd.Timedelta(hours=future_h)
            span_ms = max(1, int((x_end - x_start).total_seconds() * 1000))
            tick_cfg = {
                "dtick": max(1, span_ms // 6),
                "tickformat": "%H:%M\n%m/%d" if range_key in {"24h", "3d"} else "%m/%d",
            }
            fig.update_xaxes(range=[x_start, x_end], **tick_cfg)
            now_ts = df.index.max()
            if pd.notna(now_ts):
                fig.add_vline(x=now_ts, line_dash="dot", line_color="rgba(255,255,255,0.35)", line_width=1)
                fig.add_vrect(
                    x0=now_ts,
                    x1=x_end,
                    fillcolor="rgba(34, 211, 238, 0.06)",
                    line_width=0,
                    layer="below",
                )
        if fc_base is not None and fc_sim is not None:
            y_join = np.concatenate([
                view["FINAL_TOC"].dropna().to_numpy() if "FINAL_TOC" in view.columns else np.array([]),
                np.asarray(fc_base),
                np.asarray(fc_sim),
            ])
            if len(y_join):
                y_min = float(np.nanmin(y_join))
                y_max = float(np.nanmax(y_join))
                pad = max(0.6, (y_max - y_min) * 0.16)
                fig.update_yaxes(range=[y_min - pad, y_max + pad])
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

with b:
    with st.container(border=False):
        st.markdown('<div id="sim_result_anchor"></div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">SIMULATION RESULTS</div>', unsafe_allow_html=True)
        st.markdown('<div id="sim_target_input_anchor"></div>', unsafe_allow_html=True)
        target_limit = st.number_input(
            "목표값 (Simulator)",
            min_value=0.0,
            max_value=100.0,
            value=target_limit,
            step=0.5,
            key="sim_target_limit",
            label_visibility="collapsed",
        )
        WARN_TOC = float(target_limit) * 0.60
        ALARM_TOC = float(target_limit) * 0.80
        sig_base = classify_signal(base_latest, ALARM_TOC, warn_ratio=(WARN_TOC / ALARM_TOC if ALARM_TOC > 0 else 0.75))
        sig_sim = classify_signal(sim_latest, ALARM_TOC, warn_ratio=(WARN_TOC / ALARM_TOC if ALARM_TOC > 0 else 0.75))
        st.markdown('<div style="height:0.6rem"></div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="glass-card" style="padding: 1rem; margin-bottom: 0.75rem;">
            <div class="metric-sub">기준 예측 / 조정 예측 TOC (12시간 뒤)</div>
            <div class="metric-value-md">{base_latest:.2f} / {sim_latest:.2f}</div>
            <div style="display:flex; gap:0.5rem; margin-top:.55rem;">
                {badge_pill_html(sig_base)}
                {badge_pill_html(sig_sim)}
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="glass-card" style="padding: 1rem; margin-bottom: 0.75rem;">
            <div class="metric-sub">예측 TOC 변화량</div>
            <div class="metric-value-md">{sim_delta:+.3f}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="glass-card" style="padding: 1rem; margin-bottom: 0.75rem;">
            <div class="metric-sub">현재 반송량</div>
            <div class="metric-value-md">{ret_now:.2f}% → {ret_sim:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="glass-card" style="padding: 1rem; margin-bottom: 0.75rem;">
            <div class="metric-sub">현재 인발량</div>
            <div class="metric-value-md">{cur_wit:.2f} → {cur_wit + wit_delta:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
