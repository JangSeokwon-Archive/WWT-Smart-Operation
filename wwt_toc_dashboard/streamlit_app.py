import os
from pathlib import Path
import json
import time
import re
import math
from textwrap import dedent
from concurrent.futures import ThreadPoolExecutor
from html import escape, unescape
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from core.data import load_raw_csv, pick_latest, slice_by_range, ensure_pred_columns
from core.signals import classify_signal, badge_pill_html, signal_label_ko
from core.automl_infer import predict_multi_leads_from_automl
from core.llm_advisor import run_llm_advisor, build_quick_report
from core.simulator import fit_quadratic_response_model, recommend_optimal_controls_24h, do_optimal_target_by_mlss
from core.settings_store import load_app_settings

DIAG_EXECUTOR = ThreadPoolExecutor(max_workers=1)

st.set_page_config(
    page_title="Main",
    layout="wide",
    initial_sidebar_state="expanded"
)


if "diag_drawer_open" not in st.session_state:
    st.session_state["diag_drawer_open"] = False
if "diag_pending_llm" not in st.session_state:
    st.session_state["diag_pending_llm"] = False
if "diag_context" not in st.session_state:
    st.session_state["diag_context"] = None
if "diag_progress" not in st.session_state:
    st.session_state["diag_progress"] = 0
if "diag_started_at" not in st.session_state:
    st.session_state["diag_started_at"] = 0.0
if "diag_future" not in st.session_state:
    st.session_state["diag_future"] = None

# Drop stale/corrupted diagnosis payloads early.
try:
    _diag_state = st.session_state.get("diagnosis_result")
    if isinstance(_diag_state, dict) and isinstance(_diag_state.get("payload"), dict):
        _p = _diag_state.get("payload", {})
        _raw_blob = " ".join(
            [
                str(_p.get("diagnosis", "")),
                str(_p.get("report", "")),
                json.dumps(_p.get("actions", []), ensure_ascii=False),
                json.dumps(_p.get("watchpoints", []), ensure_ascii=False),
            ]
        ).lower()
        if any(
            tok in _raw_blob
            for tok in [
                "<div",
                "&lt;div",
                "amp;lt;div",
                "class=",
                "diag-drawer",
                "diag-report-list",
                "</ul",
                "</li",
                "/div",
            ]
        ):
            st.session_state["diagnosis_result"] = None
            st.session_state["diag_pending_llm"] = False
except Exception:
    pass

# Load CSS
APP_DIR = Path(__file__).resolve().parent
APP_SETTINGS = load_app_settings(APP_DIR)
MAIN_TARGET_TOC = float(APP_SETTINGS.get("main_target_toc", 25.0))
MAIN_WARN_RATIO = float(APP_SETTINGS.get("main_warn_ratio", 0.60))
MAIN_ALARM_RATIO = float(APP_SETTINGS.get("main_alarm_ratio", 0.80))
WARN_TOC = MAIN_TARGET_TOC * MAIN_WARN_RATIO
ALARM_TOC = MAIN_TARGET_TOC * MAIN_ALARM_RATIO
OLLAMA_MODEL_NAME = str(APP_SETTINGS.get("ollama_model", "qwen2.5:7b-instruct"))
DIAG_DRAWER_WIDTH = int(APP_SETTINGS.get("diag_drawer_width_px", 420))
DIAG_MAIN_SCALE = float(APP_SETTINGS.get("diag_main_scale", 0.94))
EQ_TOC_WARN = float(APP_SETTINGS.get("eq_toc_warn", 12.0))
css_path = APP_DIR / "assets" / "style.css"
css_loaded = False
if css_path.exists():
    st.markdown(
        f"<style>{css_path.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True
    )
    css_loaded = True

raw_path = str(APP_DIR / "data" / "raw.csv")
automl_raw_path = APP_DIR.parent / "wwt-toc-automl" / "data" / "raw" / "raw.csv"
if automl_raw_path.exists():
    raw_path = str(automl_raw_path)
pred_cache_path = APP_DIR / ".cache" / "model_preds_main.json"
delay_profile_path = APP_DIR.parent / "wwt-toc-automl" / "outputs" / "reports" / "control_delay_multivar_AERB_DO_AERB_WIT_AERB_RET_max72h.json"
delay_search_summary_path = APP_DIR.parent / "wwt-toc-automl" / "outputs" / "reports" / "control_delay_multivar_search_summary.json"
PRED_CACHE_VERSION = 5


def _disp_target_name(t: str) -> str:
    t = str(t).upper()
    if t == "AERB_RET":
        return "폭기조 반송량"
    if t == "AERB_WIT":
        return "폭기조 인발량"
    if t == "AERB_DO":
        return "폭기조 DO양"
    return t


def _recent_slope(s: pd.Series, h: int = 12) -> float:
    try:
        v = pd.to_numeric(s, errors="coerce").dropna().tail(int(h) + 1)
        if len(v) < 2:
            return float("nan")
        return float((v.iloc[-1] - v.iloc[0]) / max(1, len(v) - 1))
    except Exception:
        return float("nan")


def _pick_first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _clean_diag_text(v: str) -> str:
    s = str(v or "")
    # Remove accidental HTML/template fragments from model output.
    for _ in range(3):
        t = unescape(s)
        if t == s:
            break
        s = t
    s = s.replace("\xa0", " ").replace("&nbsp;", " ")
    s = re.sub(r"&lt;[^&]*&gt;", " ", s)
    # Strip tags twice to catch nested/unescaped leftovers.
    s = re.sub(r"<[^>]*>", " ", s)
    s = re.sub(r"<[^>]*>", " ", s)
    # Remove broken/incomplete html-like residues.
    s = s.replace("<", " ").replace(">", " ")
    s = re.sub(r"\b/?(div|ul|li)\b", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\bclass\s*=\s*\"[^\"]*\"", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\bclass\s*=\s*'[^']*'", " ", s, flags=re.IGNORECASE)
    # Strip common html-like noise tokens that might survive malformed tags.
    s = re.sub(r"\b(diag-drawer-sec|diag-drawer-body|diag-report-list|diag-drawer-list)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _looks_like_html_noise(v: str) -> bool:
    s = str(v or "").lower()
    return any(
        tok in s
        for tok in [
            "<div",
            "</div",
            "&lt;div",
            "<ul",
            "</ul",
            "<li",
            "</li",
            "class=\"diag-",
            "class='diag-",
            "diag-drawer",
            "diag-report-list",
            "diag-drawer-sec",
            "diag-drawer-body",
            "diag-drawer-list",
            "/div",
            "/ul",
            "/li",
            "diag-",
            "class=",
            "&lt;",
            "&gt;",
            "amp;lt;",
            "amp;gt;",
        ]
    )


def _sanitize_diag_payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        return {}
    out = dict(payload)
    out["diagnosis"] = _clean_diag_text(out.get("diagnosis", ""))
    out["report"] = _clean_diag_text(out.get("report", ""))
    if isinstance(out.get("actions"), list):
        clean_actions = []
        for a in out["actions"][:3]:
            if not isinstance(a, dict):
                continue
            b = dict(a)
            b["name"] = _clean_diag_text(b.get("name", ""))
            b["reason"] = _clean_diag_text(b.get("reason", ""))
            b["eta_hours"] = _clean_diag_text(b.get("eta_hours", ""))
            clean_actions.append(b)
        out["actions"] = clean_actions
    else:
        out["actions"] = []
    if isinstance(out.get("watchpoints"), list):
        out["watchpoints"] = [_clean_diag_text(x) for x in out["watchpoints"][:4]]
    else:
        out["watchpoints"] = []
    # Hard-stop if any html-like residue remains.
    if _looks_like_html_noise(out.get("diagnosis", "")) or _looks_like_html_noise(out.get("report", "")):
        out["diagnosis"] = ""
        out["report"] = "진단 텍스트를 정리하지 못해 결과 표시를 생략했습니다. 진단하기를 다시 실행해 주세요."
        out["actions"] = []
        out["watchpoints"] = []
    return out


def _force_plain_or_fallback(text: str, fallback: str) -> str:
    s = _clean_diag_text(text)
    # If any html-ish residue still survives, drop it entirely.
    if _looks_like_html_noise(text) or "<" in str(text) or ">" in str(text):
        return fallback
    if _looks_like_html_noise(s) or "<" in s or ">" in s:
        return fallback
    if any(tok in s.lower() for tok in ["diag-drawer", "diag-report-list", "class=\"diag", "class='diag", "/div", "/ul", "/li"]):
        return fallback
    return s if s else fallback


def _build_local_diag_lines(ctx: dict, status: str) -> list[str]:
    cur = ctx.get("current_toc")
    p24 = ctx.get("pred_t24")
    flow = ctx.get("FLOW")
    do_now = ctx.get("AERB_DO")
    do_rec = ctx.get("mlss_recommended_do")
    lines: list[str] = []
    if isinstance(cur, (int, float)) and isinstance(p24, (int, float)):
        trend = "하락" if float(p24) < float(cur) else ("상승" if float(p24) > float(cur) else "보합")
        lines.append(f"현재 최종 TOC는 {float(cur):.2f}, 24시간 예측은 {float(p24):.2f}로 {trend} 추세입니다.")
    if isinstance(flow, (int, float)):
        lines.append(f"현재 유입량은 {float(flow):.0f}t/h 입니다.")
    if isinstance(do_now, (int, float)) and isinstance(do_rec, (int, float)):
        gap = float(do_now) - float(do_rec)
        lines.append(f"폭기조 DO양은 현재 {float(do_now):.2f}, 권장 {float(do_rec):.2f}로 편차 {gap:+.2f} 입니다.")
    if status == "양호":
        lines.append("현재는 급격한 조정보다 모니터링 중심 운전이 적절합니다.")
    return lines


def _split_sentences(text: str, limit: int = 2) -> list[str]:
    s = _force_plain_or_fallback(str(text or ""), "")
    if not s:
        return []
    chunks = [c.strip() for c in re.split(r"(?<=[\.\!\?])\s+|(?<=다\.)\s+", s) if c.strip()]
    return chunks[:max(0, int(limit))]


def _build_diag_sections(ctx: dict, status: str) -> list[tuple[str, list[str]]]:
    cur = ctx.get("current_toc")
    p12 = ctx.get("pred_t12")
    p24 = ctx.get("pred_t24")
    flow = ctx.get("FLOW")
    do_now = ctx.get("AERB_DO")
    do_rec = ctx.get("mlss_recommended_do")
    high_inflow = bool(ctx.get("high_inflow", False))

    sec12: list[str] = []
    sec24: list[str] = []

    if isinstance(cur, (int, float)) and isinstance(p12, (int, float)):
        d12 = float(p12) - float(cur)
        tr12 = "상승" if d12 > 0 else ("하락" if d12 < 0 else "보합")
        sec12.append(f"현재 최종 TOC {float(cur):.2f} 기준, 12시간 뒤 예측은 {float(p12):.2f}({tr12}, {d12:+.2f})입니다.")
    if isinstance(do_now, (int, float)) and isinstance(do_rec, (int, float)):
        gap = float(do_now) - float(do_rec)
        sec12.append(f"폭기조 DO양은 현재 {float(do_now):.2f}, 권장 {float(do_rec):.2f}로 편차 {gap:+.2f}입니다.")
    if status == "양호":
        sec12.append("즉시 대규모 조정보다는 모니터링 중심 운전이 적절합니다.")

    if isinstance(p24, (int, float)):
        if isinstance(p12, (int, float)):
            span = abs(float(p24) - float(p12))
            band = max(0.20, 0.35 * span)
            lo, hi = float(p24) - band, float(p24) + band
            sec24.append(f"24시간 뒤 예상 TOC는 {float(p24):.2f}, 리스크 범위는 {lo:.2f}~{hi:.2f}입니다.")
        else:
            sec24.append(f"24시간 뒤 예상 TOC는 {float(p24):.2f}입니다.")
    if isinstance(flow, (int, float)):
        if high_inflow:
            sec24.append(f"현재 유입량 {float(flow):.0f}t/h로 고유입 구간이며, 24시간 리스크를 보수적으로 모니터링해야 합니다.")
        else:
            sec24.append(f"현재 유입량 {float(flow):.0f}t/h로, 24시간 리스크는 운전값 미세 조정으로 관리 가능한 구간입니다.")

    return [
        ("2-1) 12시간 운전 진단", sec12[:4]),
        ("2-2) 24시간 리스크 전망", sec24[:4]),
    ]


def _build_local_actions(ctx: dict, status: str) -> list[dict]:
    if status == "양호":
        return []
    opt = ctx.get("sim_opt_24h")
    out: list[dict] = []
    if isinstance(opt, dict) and bool(opt.get("ok")) and isinstance(opt.get("best"), dict):
        b = opt.get("best", {})
        for target, key, eta in [
            ("AERB_DO", "do_delta", "6~24시간"),
            ("AERB_RET", "ret_delta_pct", "12~24시간"),
            ("AERB_WIT", "wit_delta", "8~24시간"),
        ]:
            try:
                d = float(b.get(key, 0.0))
            except Exception:
                d = 0.0
            if abs(d) > 1e-9:
                out.append(
                    {
                        "target": target,
                        "delta": d,
                        "eta_hours": eta,
                        "reason": "시뮬레이터 기준 24시간 TOC 최소화 조합",
                    }
                )
    return out[:3]


def _build_local_watchpoints(ctx: dict) -> list[str]:
    points = [
        "조정 후 6h/12h/24h TOC 추이 비교",
        "폭기조 DO양 과상승/과하강 여부",
        "반송량 급변 여부",
        "균등조/DAF 추세와 최종 TOC 동조 여부",
    ]
    return points


def _status_from_ctx(ctx: dict) -> str:
    th = ctx.get("thresholds", {}) if isinstance(ctx, dict) else {}
    warn = float(th.get("warn", WARN_TOC)) if isinstance(th, dict) else WARN_TOC
    alarm = float(th.get("alarm", ALARM_TOC)) if isinstance(th, dict) else ALARM_TOC
    p12 = ctx.get("pred_t12")
    p24 = ctx.get("pred_t24")
    cur = ctx.get("current_toc")
    ref = p12 if isinstance(p12, (int, float)) else (p24 if isinstance(p24, (int, float)) else cur)
    if not isinstance(ref, (int, float)):
        return "주의"
    if float(ref) >= float(alarm):
        return "주의"
    if float(ref) >= float(warn):
        return "경고"
    return "양호"


@st.cache_data(ttl=180)
def load_df(path):
    try:
        df = load_raw_csv(path)
        return ensure_pred_columns(df) if df is not None else None
    except Exception:
        return None

@st.cache_data(ttl=600, show_spinner=False)
def load_model_preds(raw_path: str, automl_root_path: str, raw_mtime: float):
    # Fast path: reuse disk cache if the same raw file is already inferred.
    try:
        if pred_cache_path.exists():
            cached = json.loads(pred_cache_path.read_text(encoding="utf-8"))
            if int(cached.get("version", -1)) == int(PRED_CACHE_VERSION) and float(cached.get("raw_mtime", -1.0)) == float(raw_mtime):
                preds = cached.get("preds", {})
                if isinstance(preds, dict):
                    return preds
    except Exception:
        pass
    try:
        preds = predict_multi_leads_from_automl(
            raw_path=raw_path,
            automl_root_path=automl_root_path,
            leads=(6, 12, 24),
        )
        try:
            pred_cache_path.parent.mkdir(parents=True, exist_ok=True)
            pred_cache_path.write_text(
                json.dumps({"version": int(PRED_CACHE_VERSION), "raw_mtime": float(raw_mtime), "generated_at": int(time.time()), "preds": preds}, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass
        return preds
    except Exception:
        return {"pred_t6": float("nan"), "pred_t12": float("nan"), "pred_t24": float("nan")}


@st.cache_data(ttl=600, show_spinner=False)
def load_delay_profile_summary(path: str):
    p = Path(path)
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj.get("summary", {}) if isinstance(obj, dict) else {}
    except Exception:
        return {}


@st.cache_data(ttl=900, show_spinner=False)
def load_response_model_for_diag(raw_path: str, raw_mtime: float):
    try:
        df0 = load_raw_csv(raw_path)
        return fit_quadratic_response_model(df0)
    except Exception:
        return {"ok": False}


@st.cache_data(ttl=900, show_spinner=False)
def load_delay_profile_for_diag(default_path: str, search_summary_path: str):
    p = Path(default_path)
    try:
        sp = Path(search_summary_path)
        if sp.exists():
            rows = json.loads(sp.read_text(encoding="utf-8"))
            cand = [
                r for r in rows
                if isinstance(r, dict)
                and isinstance(r.get("rmse_valid_dy"), (int, float))
                and r.get("file")
            ]
            if cand:
                cand = sorted(cand, key=lambda r: float(r["rmse_valid_dy"]))
                best = Path(str(cand[0]["file"]))
                if best.exists():
                    p = best
    except Exception:
        pass
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


@st.cache_data(ttl=900, show_spinner=False)
def load_reco_strength_profile(report_dir: str, last_days: int = 30):
    out = {}
    base = Path(report_dir)
    for lead in (6, 12, 24):
        p = base / f"recommendation_backtest_lead{lead}h_last{int(last_days)}d.json"
        if not p.exists():
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            mae_delta = float(d.get("mae_delta", 0.0))
            err_impr = float(d.get("abs_error_improvement_rate", 0.0))
            if mae_delta <= 0 and err_impr >= 0.10:
                mult = 1.00
                label = "표준"
            elif mae_delta <= 0:
                mult = 0.90
                label = "약보수"
            else:
                mult = 0.65
                label = "보수"
            out[str(lead)] = {
                "mae_delta": mae_delta,
                "abs_error_improvement_rate": err_impr,
                "multiplier": mult,
                "label": label,
            }
        except Exception:
            continue
    return out

df = load_df(raw_path)
if df is None:
    st.error(f"⚠️ Waiting for data... ({raw_path})")
    st.stop()
raw_mtime = Path(raw_path).stat().st_mtime if Path(raw_path).exists() else 0.0
model_preds = load_model_preds(raw_path, str(APP_DIR.parent / "wwt-toc-automl"), raw_mtime)
delay_summary = load_delay_profile_summary(str(delay_profile_path))

st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)

# CENTRAL CONTAINER WITH MARGINS
c_l, c_center, c_r = st.columns([0.2, 9.6, 0.2])

with c_center:
    col_kpi, col_chart = st.columns([2.5, 7.5], gap="large")
    with col_chart:
        with st.container(border=False):
            st.markdown('<div id="trend_card_anchor"></div>', unsafe_allow_html=True)
            head_l, head_r = st.columns([8.0, 2.0], vertical_alignment="center")
            with head_l:
                st.markdown('<div class="trend-title">TREND</div>', unsafe_allow_html=True)
                st.markdown('<div id="diag_btn_anchor"></div>', unsafe_allow_html=True)
                run_diag = st.button("진단하기", key="diagnose_btn")
            with head_r:
                st.markdown('<div id="trend_range_anchor"></div>', unsafe_allow_html=True)
                range_options = ["24h", "3d", "7d", "30d", "90d"]
                if hasattr(st, "segmented_control"):
                    range_key = st.segmented_control(
                        "Range",
                        range_options,
                        default="24h",
                        label_visibility="collapsed",
                        key="date_range_selector"
                    )
                else:
                    range_key = st.radio(
                        "Range",
                        range_options,
                        horizontal=True,
                        index=0,
                        label_visibility="collapsed",
                        key="date_range_selector"
                    )
                if range_key not in range_options:
                    range_key = st.session_state.get("date_range_selector_last", "24h")
                st.session_state["date_range_selector_last"] = range_key

            view = slice_by_range(df, range_key)
            # KPI는 범위 선택과 무관하게 "현재 최신값" 기준
            latest_toc = pick_latest(df, "FINAL_TOC")
            latest_flow = pick_latest(df, "FLOW")
            latest_temp = pick_latest(df, "TEMP")
            last_updated = df.index.max()
            pred_val = pick_latest(df, "pred_t12")
            pred_label = "12시간 뒤 예측"
            if pd.notna(model_preds.get("pred_t12", float("nan"))):
                pred_val = float(model_preds["pred_t12"])
            if not pd.notna(pred_val):
                pred_val = pick_latest(df, "pred_t24")
                pred_label = "24시간 뒤 예측"
            if pd.notna(model_preds.get("pred_t24", float("nan"))) and pred_label == "24시간 뒤 예측":
                pred_val = float(model_preds["pred_t24"])
            if not pd.notna(pred_val):
                pred_val = pick_latest(df, "pred_t36")
                pred_label = "36시간 뒤 예측"
            if pd.notna(model_preds.get("pred_t36", float("nan"))) and pred_label == "36시간 뒤 예측":
                pred_val = float(model_preds["pred_t36"])
            if not pd.notna(pred_val):
                pred_val = pick_latest(df, "pred_final")
                pred_label = "최신 예측"
            if not pd.notna(pred_val):
                pred_val = latest_toc
                pred_label = "최신값 대체"

            aerb_do = pick_latest(df, "AERB_DO")
            aerb_wit = pick_latest(df, "AERB_WIT")
            aerb_ret = pick_latest(df, "AERB_RET")
            aerb_mlss = pick_latest(df, "AERB_MLSS")
            flow_capacity = float(APP_SETTINGS.get("flow_capacity_tph", 500.0))
            flow_high_threshold = float(APP_SETTINGS.get("flow_high_threshold_tph", 400.0))
            flow_ratio = (float(latest_flow) / flow_capacity) if pd.notna(latest_flow) and flow_capacity > 0 else float("nan")
            flow_pct = (float(flow_ratio) * 100.0) if pd.notna(flow_ratio) else float("nan")
            high_inflow = bool(pd.notna(latest_flow) and float(latest_flow) >= flow_high_threshold)
            mlss_all = pd.to_numeric(df["AERB_MLSS"], errors="coerce") if "AERB_MLSS" in df.columns else pd.Series(dtype=float)
            mlss_q_low = float(mlss_all.quantile(0.33)) if len(mlss_all.dropna()) else float("nan")
            mlss_q_high = float(mlss_all.quantile(0.67)) if len(mlss_all.dropna()) else float("nan")
            rec_do = do_optimal_target_by_mlss(
                current_mlss=float(aerb_mlss) if pd.notna(aerb_mlss) else float("nan"),
                mlss_q_low=float(mlss_q_low) if pd.notna(mlss_q_low) else float("nan"),
                mlss_q_high=float(mlss_q_high) if pd.notna(mlss_q_high) else float("nan"),
            )
            if pd.notna(aerb_mlss) and pd.notna(mlss_q_low) and pd.notna(mlss_q_high):
                if float(aerb_mlss) <= mlss_q_low:
                    mlss_band = "LOW"
                elif float(aerb_mlss) >= mlss_q_high:
                    mlss_band = "HIGH"
                else:
                    mlss_band = "MID"
            else:
                mlss_band = "UNKNOWN"

            if run_diag:
                p6_ctx = pick_latest(df, "pred_t6")
                p12_ctx = pick_latest(df, "pred_t12")
                p24_ctx = pick_latest(df, "pred_t24")
                if pd.notna(model_preds.get("pred_t6", float("nan"))):
                    p6_ctx = float(model_preds["pred_t6"])
                if pd.notna(model_preds.get("pred_t12", float("nan"))):
                    p12_ctx = float(model_preds["pred_t12"])
                if pd.notna(model_preds.get("pred_t24", float("nan"))):
                    p24_ctx = float(model_preds["pred_t24"])
                d12 = (float(p12_ctx) - float(latest_toc)) if pd.notna(p12_ctx) and pd.notna(latest_toc) else float("nan")
                d24 = (float(p24_ctx) - float(p12_ctx)) if pd.notna(p24_ctx) and pd.notna(p12_ctx) else float("nan")
                if pd.notna(d12) and pd.notna(d24):
                    if d12 > 0 and d24 > 0 and d24 < d12:
                        pred_shape = "상승세이나 상승폭 둔화"
                    elif d12 > 0 and d24 > d12:
                        pred_shape = "상승 가속"
                    elif d12 < 0 and d24 < 0 and d24 > d12:
                        pred_shape = "하락세이나 하락폭 둔화"
                    elif d12 < 0 and d24 < d12:
                        pred_shape = "하락 가속"
                    else:
                        pred_shape = "보합/혼조"
                else:
                    if pd.notna(p24_ctx) and pd.notna(latest_toc):
                        if float(p24_ctx) > float(latest_toc):
                            pred_shape = "상승 추세"
                        elif float(p24_ctx) < float(latest_toc):
                            pred_shape = "하락 추세"
                        else:
                            pred_shape = "보합 추세"
                    else:
                        pred_shape = "보합/정보부족"

                eq_col = _pick_first_col(df, ["EQ_TOC", "EQ_TOC_raw", "EQ_OUT_TOC"])
                daf_col = _pick_first_col(df, ["DAF_TOC", "DAF_TOC_raw", "DAF_OUT_TOC"])
                eq_toc_latest = pick_latest(df, eq_col) if eq_col else float("nan")
                eq_slope = _recent_slope(df[eq_col], h=12) if eq_col else float("nan")
                daf_slope = _recent_slope(df[daf_col], h=12) if daf_col else float("nan")
                eq_daf_rise_room = bool((pd.notna(eq_slope) and eq_slope > 0.015) or (pd.notna(daf_slope) and daf_slope > 0.015))
                do_trend = _recent_slope(df["AERB_DO"], h=12) if "AERB_DO" in df.columns else float("nan")
                do_trend_text = "상승" if pd.notna(do_trend) and do_trend > 0.01 else ("하락" if pd.notna(do_trend) and do_trend < -0.01 else "보합")
                process_drop_risk = bool(
                    high_inflow
                    and pd.notna(latest_toc)
                    and pd.notna(p24_ctx)
                    and float(p24_ctx) >= float(latest_toc)
                )

                response_model_diag = load_response_model_for_diag(raw_path, raw_mtime)
                delay_profile_diag = load_delay_profile_for_diag(str(delay_profile_path), str(delay_search_summary_path))
                reco_strength = load_reco_strength_profile(str(APP_DIR.parent / "wwt-toc-automl" / "outputs" / "reports"), last_days=30)
                sim_opt_24h = recommend_optimal_controls_24h(
                    current_toc=float(latest_toc) if pd.notna(latest_toc) else float("nan"),
                    p6=float(p6_ctx) if pd.notna(p6_ctx) else float("nan"),
                    p12=float(p12_ctx) if pd.notna(p12_ctx) else float("nan"),
                    p24=float(p24_ctx) if pd.notna(p24_ctx) else float("nan"),
                    current_do=float(aerb_do) if pd.notna(aerb_do) else float("nan"),
                    current_ret=float(aerb_ret) if pd.notna(aerb_ret) else float("nan"),
                    current_wit=float(aerb_wit) if pd.notna(aerb_wit) else float("nan"),
                    response_model=response_model_diag,
                    delay_profile=delay_profile_diag,
                    current_mlss=float(aerb_mlss) if pd.notna(aerb_mlss) else float("nan"),
                    mlss_q_low=float(mlss_q_low) if pd.notna(mlss_q_low) else float("nan"),
                    mlss_q_high=float(mlss_q_high) if pd.notna(mlss_q_high) else float("nan"),
                )

                ctx = {
                    "timestamp": str(last_updated) if pd.notna(last_updated) else "",
                    "current_toc": float(latest_toc) if pd.notna(latest_toc) else None,
                    "pred_t6": float(p6_ctx) if pd.notna(p6_ctx) else None,
                    "pred_t12": float(p12_ctx) if pd.notna(p12_ctx) else None,
                    "pred_t24": float(p24_ctx) if pd.notna(p24_ctx) else None,
                    "FLOW": float(latest_flow) if pd.notna(latest_flow) else None,
                    "TEMP": float(latest_temp) if pd.notna(latest_temp) else None,
                    "AERB_DO": float(aerb_do) if pd.notna(aerb_do) else None,
                    "AERB_WIT": float(aerb_wit) if pd.notna(aerb_wit) else None,
                    "AERB_RET": float(aerb_ret) if pd.notna(aerb_ret) else None,
                    "AERB_MLSS": float(aerb_mlss) if pd.notna(aerb_mlss) else None,
                    "mlss_band": mlss_band,
                    "mlss_recommended_do": float(rec_do) if pd.notna(rec_do) else None,
                    "flow_capacity_tph": flow_capacity,
                    "flow_high_threshold_tph": flow_high_threshold,
                    "flow_load_pct": float(flow_pct) if pd.notna(flow_pct) else None,
                    "high_inflow": high_inflow,
                    "process_drop_risk": process_drop_risk,
                    "do_trend": do_trend_text,
                    "pred_shape": pred_shape,
                    "eq_toc_slope_12h": float(eq_slope) if pd.notna(eq_slope) else None,
                    "daf_toc_slope_12h": float(daf_slope) if pd.notna(daf_slope) else None,
                    "eq_daf_rise_room": eq_daf_rise_room,
                    "EQ_TOC": float(eq_toc_latest) if pd.notna(eq_toc_latest) else None,
                    "eq_toc_warn": EQ_TOC_WARN,
                    "delay_summary": delay_summary,
                    "sim_opt_24h": sim_opt_24h,
                    "reco_strength_profile": reco_strength,
                    "primary_reco_lead": 24,
                    "thresholds": {"warn": WARN_TOC, "alarm": ALARM_TOC},
                }
                st.session_state["diagnosis_result"] = None
                st.session_state["diag_drawer_open"] = True
                st.session_state["diag_context"] = ctx
                # Run diagnosis asynchronously and show progress in right drawer.
                try:
                    fut = DIAG_EXECUTOR.submit(run_llm_advisor, ctx, OLLAMA_MODEL_NAME, 40)
                except Exception:
                    fut = None
                st.session_state["diag_future"] = fut
                st.session_state["diag_pending_llm"] = True if fut is not None else False
                st.session_state["diag_progress"] = 1 if fut is not None else 100
                st.session_state["diag_started_at"] = time.time()
                if fut is None:
                    quick = build_quick_report(ctx)
                    st.session_state["diagnosis_result"] = {
                        "source": quick.source,
                        "error": quick.error,
                        "payload": _sanitize_diag_payload(quick.payload if isinstance(quick.payload, dict) else {}),
                    }
                st.rerun()

            fig = go.Figure()
            x_pts = []
            if "FINAL_TOC" in view.columns:
                fig.add_trace(go.Scatter(
                    x=view.index,
                    y=view["FINAL_TOC"],
                    mode="lines",
                    name="Actual",
                    line=dict(color="#8B5CF6", width=3, shape="spline"),
                    fill="tozeroy",
                    fillcolor="rgba(139, 92, 246, 0.22)",
                    hovertemplate="<b>TOC</b>: %{y:.2f}<br>%{x}<extra></extra>"
                ))

            # 최신 기준 예측 점선(+24h 범위)
            if len(df) and pd.notna(df.index.max()):
                last_idx = df.index.max()
                last_actual = pick_latest(df, "FINAL_TOC")
                p6 = pick_latest(df, "pred_t6")
                p12 = pick_latest(df, "pred_t12")
                p24 = pick_latest(df, "pred_t24")
                if pd.notna(model_preds.get("pred_t6", float("nan"))):
                    p6 = float(model_preds["pred_t6"])
                if pd.notna(model_preds.get("pred_t12", float("nan"))):
                    p12 = float(model_preds["pred_t12"])
                if pd.notna(model_preds.get("pred_t24", float("nan"))):
                    p24 = float(model_preds["pred_t24"])
                candidates = [
                    (last_idx + pd.Timedelta(hours=6), p6),
                    (last_idx + pd.Timedelta(hours=12), p12),
                    (last_idx + pd.Timedelta(hours=24), p24),
                ]
                valid = [(x, y) for x, y in candidates if pd.notna(y)]
                x_pts = [x for x, _ in valid]
                y_pts = [y for _, y in valid]
                if len(y_pts) == 0 and pd.notna(last_actual):
                    x_pts = [last_idx + pd.Timedelta(hours=12), last_idx + pd.Timedelta(hours=24)]
                    y_pts = [last_actual, last_actual]

                if pd.notna(last_actual) and len(y_pts):
                    # Forecast window is always +24h (model support horizon).
                    display_end = last_idx + pd.Timedelta(hours=24)
                    if len(y_pts):
                        a_x = np.array([0.0], dtype=float)
                        a_y = np.array([float(last_actual)], dtype=float)
                        if pd.notna(p6):
                            a_x = np.append(a_x, 6.0)
                            a_y = np.append(a_y, float(p6))
                        if pd.notna(p12):
                            a_x = np.append(a_x, 12.0)
                            a_y = np.append(a_y, float(p12))
                        if pd.notna(p24):
                            a_x = np.append(a_x, 24.0)
                            a_y = np.append(a_y, float(p24))
                        order = np.argsort(a_x)
                        a_x, a_y = a_x[order], a_y[order]
                        h = np.arange(0.0, 25.0, 1.0)
                        y_h = np.interp(h, a_x, a_y)
                        x_plot = [last_idx + pd.Timedelta(hours=int(v)) for v in h]
                        y_plot = list(y_h)
                    else:
                        x_plot = [last_idx, display_end]
                        y_plot = [last_actual, last_actual]
                    fig.add_trace(go.Scatter(
                        x=x_plot,
                        y=y_plot,
                        mode="lines",
                        name="Forecast",
                        line=dict(color="#22D3EE", width=3, dash="dot", shape="spline"),
                        hovertemplate="<b>Forecast</b>: %{y:.2f}<br>%{x}<extra></extra>"
                    ))
                    mk_step = 4
                    mk_x = x_plot[::mk_step]
                    mk_y = y_plot[::mk_step]
                    mk_sizes = [6.0 + 1.6 * np.sin((i / max(len(mk_x) - 1, 1)) * (2.0 * np.pi)) for i in range(len(mk_x))]
                    fig.add_trace(go.Scatter(
                        x=mk_x,
                        y=mk_y,
                        mode="markers",
                        name="Forecast Points",
                        marker=dict(size=mk_sizes, color="#22D3EE", line=dict(width=0)),
                        showlegend=False,
                        hovertemplate="<b>Forecast</b>: %{y:.2f}<br>%{x}<extra></extra>"
                    ))

            fig.update_layout(
                height=470,
                margin=dict(l=12, r=12, t=10, b=32),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(
                    gridcolor="rgba(139,92,246,0.10)",
                    zeroline=False,
                    showline=False,
                    tickfont=dict(size=10, color="#888"),
                ),
                yaxis=dict(
                    gridcolor="rgba(139,92,246,0.10)",
                    zeroline=False,
                    showline=False,
                    tickfont=dict(size=10, color="#888"),
                ),
                showlegend=True,
                legend=dict(font=dict(color="#FFFFFF")),
                hovermode="x unified",
            )
            if len(view.index):
                days_map = {"24h": 1, "3d": 3, "7d": 7, "30d": 30, "90d": 90}
                now = df.index.max()
                if range_key == "24h":
                    x_start = now - pd.Timedelta(hours=24)
                else:
                    x_start = now - pd.Timedelta(days=days_map.get(range_key, 1))
                # Keep forecast window to roughly right 1/3 of total visible span (future:past ~= 1:2),
                # while still limiting forecast display to <=24h.
                past_span_hours = max(1.0, float((now - x_start).total_seconds() / 3600.0))
                fut_hours = min(24.0, max(1.0, past_span_hours / 2.0))
                x_end = now + pd.Timedelta(hours=fut_hours)
                span_ms = max(1, int((x_end - x_start).total_seconds() * 1000))
                tick_cfg = {
                    "dtick": max(1, span_ms // 6),
                    "tickformat": "%H:%M\n%m/%d" if range_key in {"24h", "3d"} else "%m/%d",
                }
                fig.update_xaxes(range=[x_start, x_end], **tick_cfg)
                if pd.notna(now):
                    fig.add_vline(x=now, line_dash="dot", line_color="rgba(255,255,255,0.35)", line_width=1)
                    fig.add_vrect(
                        x0=now,
                        x1=x_end,
                        fillcolor="rgba(34, 211, 238, 0.06)",
                        line_width=0,
                        layer="below",
                    )

            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            # NOTE: Diagnosis is displayed only in the right drawer.

    if st.session_state.get("diag_drawer_open", False) and st.session_state.get("diag_pending_llm", False):
        future = st.session_state.get("diag_future")
        started = float(st.session_state.get("diag_started_at") or time.time())
        if future is not None:
            loading_slot = st.empty()
            while not getattr(future, "done", lambda: True)():
                elapsed = max(0.0, time.time() - started)
                # Smooth in-run progress without rerun layout flicker.
                pct = min(99, max(3, int(99.0 * (1.0 - math.exp(-elapsed / 12.0)))))
                st.session_state["diag_progress"] = pct
                drawer_loading = dedent(
                    f"""
                    <div class="diag-drawer open" style="position:fixed;top:0;right:0;width:{DIAG_DRAWER_WIDTH}px;height:100dvh;z-index:1200;">
                        <div class="diag-drawer-head">AI 운전 진단</div>
                        <div class="diag-drawer-source">모델 분석 준비 중</div>
                        <div class="diag-loading-wrap">
                            <div class="diag-loading-spinner"></div>
                            <div class="diag-loading-text">Load 중 ({pct}%)</div>
                        </div>
                    </div>
                    """
                ).strip()
                if hasattr(loading_slot, "html"):
                    loading_slot.html(drawer_loading)
                else:
                    loading_slot.markdown(drawer_loading, unsafe_allow_html=True)
                time.sleep(0.15)
            try:
                diag = future.result()
            except Exception as e:
                diag = run_llm_advisor(st.session_state.get("diag_context", {}) or {}, model=OLLAMA_MODEL_NAME)
                if not getattr(diag, "error", ""):
                    diag.error = str(e)
            st.session_state["diag_progress"] = 100
            sanitized_payload = _sanitize_diag_payload(diag.payload if isinstance(diag.payload, dict) else {})
            st.session_state["diagnosis_result"] = {
                "source": diag.source,
                "error": diag.error,
                "payload": sanitized_payload,
            }
            st.session_state["diag_pending_llm"] = False
            st.session_state["diag_future"] = None
            st.rerun()

    diag_state = st.session_state.get("diagnosis_result")
    if st.session_state.get("diag_drawer_open", False) and not st.session_state.get("diag_pending_llm", False):
        cctx = st.session_state.get("diag_context") if isinstance(st.session_state.get("diag_context"), dict) else {}
        if not cctx:
            cctx = {}
        # Render drawer with local deterministic content only.
        # This hard-blocks any malformed HTML text leakage from LLM payloads.
        p = {}
        status = _status_from_ctx(cctx)
        color = "#10B981" if status == "양호" else ("#F59E0B" if status == "경고" else "#EF4444")
        diagnosis = "" if status == "양호" else "운전 안정성 점검이 필요합니다."
        source = escape(str(diag_state.get("source", "rule_local"))) if isinstance(diag_state, dict) else "rule_local"
        source = f"{source} · ui-v3"
        # Never render raw payload body lists directly (prevents HTML-noise leakage).
        actions = _build_local_actions(cctx, status)
        watchpoints = _build_local_watchpoints(cctx)

        actions_html = ""
        if isinstance(actions, list):
            for a in actions[:3]:
                if not isinstance(a, dict):
                    continue
                try:
                    dlt = float(a.get("delta", 0.0))
                except Exception:
                    dlt = 0.0
                target_name = str(a.get("target", "")).upper()
                dlt_txt = f"{dlt:+.2f}%p" if target_name == "AERB_RET" else f"{dlt:+.2f}"
                actions_html += (
                    f'<div class="diag-action-card">'
                    f'<div class="diag-action-head">'
                    f'<b>{escape(_disp_target_name(a.get("target","")))} {dlt_txt}</b>'
                    f'<span>{escape(str(a.get("eta_hours","")))}</span>'
                    f"</div>"
                    f'<div class="diag-action-reason">{escape(_clean_diag_text(str(a.get("reason",""))))}</div>'
                    f"</div>"
                )
        wp_html = ""
        if isinstance(watchpoints, list):
            for w in watchpoints[:4]:
                w_clean = _force_plain_or_fallback(str(w), "")
                if w_clean:
                    wp_html += f"<li>{escape(w_clean)}</li>"
        flow_now = cctx.get("FLOW")
        flow_pct = cctx.get("flow_load_pct")
        do_now = cctx.get("AERB_DO")
        do_rec = cctx.get("mlss_recommended_do")
        # Build section lines directly from numeric context only (no payload text usage).
        sec12_raw: list[str] = []
        sec24_raw: list[str] = []
        cur_v = cctx.get("current_toc")
        p12_v = cctx.get("pred_t12")
        p24_v = cctx.get("pred_t24")
        if isinstance(cur_v, (int, float)) and isinstance(p12_v, (int, float)):
            d12 = float(p12_v) - float(cur_v)
            tr12 = "상승" if d12 > 0 else ("하락" if d12 < 0 else "보합")
            sec12_raw.append(f"현재 최종 TOC {float(cur_v):.2f} 기준, 12시간 뒤 예측은 {float(p12_v):.2f}({tr12}, {d12:+.2f})입니다.")
        if isinstance(do_now, (int, float)) and isinstance(do_rec, (int, float)):
            gap = float(do_now) - float(do_rec)
            sec12_raw.append(f"폭기조 DO양은 현재 {float(do_now):.2f}, 권장 {float(do_rec):.2f}로 편차 {gap:+.2f}입니다.")
        if status == "양호":
            sec12_raw.append("즉시 대규모 조정보다는 모니터링 중심 운전이 적절합니다.")

        if isinstance(p24_v, (int, float)):
            if isinstance(p12_v, (int, float)):
                span = abs(float(p24_v) - float(p12_v))
                band = max(0.20, 0.35 * span)
                lo, hi = float(p24_v) - band, float(p24_v) + band
                sec24_raw.append(f"24시간 뒤 예상 TOC는 {float(p24_v):.2f}, 리스크 범위는 {lo:.2f}~{hi:.2f}입니다.")
            else:
                sec24_raw.append(f"24시간 뒤 예상 TOC는 {float(p24_v):.2f}입니다.")
        if isinstance(flow_now, (int, float)):
            if bool(cctx.get("high_inflow", False)):
                sec24_raw.append(f"현재 유입량 {float(flow_now):.0f}t/h로 고유입 구간이며, 24시간 리스크를 보수적으로 모니터링해야 합니다.")
            else:
                sec24_raw.append(f"현재 유입량 {float(flow_now):.0f}t/h로, 24시간 리스크는 운전값 미세 조정으로 관리 가능한 구간입니다.")

        def _safe_list_html(lines: list[str]) -> str:
            out: list[str] = []
            for x in lines[:4]:
                s = _force_plain_or_fallback(str(x), "")
                if not s:
                    continue
                low = s.lower()
                if any(tok in low for tok in ["<div", "&lt;div", "class=", "diag-drawer", "/div", "/ul", "/li"]):
                    continue
                out.append(f"<li>{escape(s)}</li>")
            return "".join(out) if out else "<li>표시 가능한 분석 문장이 없습니다.</li>"

        sec12_lis = _safe_list_html(sec12_raw)
        sec24_lis = _safe_list_html(sec24_raw)
        diagnosis_html = "" if status == "양호" else f'<div class="diag-drawer-source" style="margin-top:-.35rem;">{escape(diagnosis)}</div>'
        flow_text = (
            f"{float(flow_now):.0f} t/h"
            if isinstance(flow_now, (int, float)) and isinstance(flow_pct, (int, float))
            else "-"
        )
        do_text = (
            f"{float(do_now):.2f} / 권장 {float(do_rec):.2f}"
            if isinstance(do_now, (int, float)) and isinstance(do_rec, (int, float))
            else "-"
        )

        drawer_html = dedent(
            f"""
            <div class="diag-drawer open" style="position:fixed;top:0;right:0;width:{DIAG_DRAWER_WIDTH}px;height:100dvh;z-index:1200;">
                <div class="diag-drawer-head">AI 운전 진단</div>
                <div class="diag-drawer-source">{source}</div>
                <div class="diag-kpi-grid">
                    <div class="diag-kpi-card"><span>유입부하</span><b>{escape(flow_text)}</b></div>
                    <div class="diag-kpi-card"><span>폭기조 DO양</span><b>{escape(do_text)}</b></div>
                </div>
                <div class="diag-drawer-sec">1) 상태 요약</div>
                <div class="diag-drawer-status" style="color:{color};">{status}</div>
                {diagnosis_html}
                <div class="diag-drawer-sec">2) 상세 분석</div>
                <div class='diag-drawer-sec'>2-1) 12시간 운전 진단</div>
                <div class='diag-drawer-body'><ul class='diag-report-list'>{sec12_lis}</ul></div>
                <div class='diag-drawer-sec'>2-2) 24시간 리스크 전망</div>
                <div class='diag-drawer-body'><ul class='diag-report-list'>{sec24_lis}</ul></div>
                {"<div class='diag-drawer-sec'>3) 권장 조치</div><div class='diag-action-wrap'>" + actions_html + "</div>" if actions_html else ""}
                <div class="diag-drawer-sec">4) 모니터링 포인트</div>
                <ul class="diag-drawer-list">{wp_html if wp_html else "<li>포인트 없음</li>"}</ul>
            </div>
            """
        ).strip()
        if hasattr(st, "html"):
            st.html(drawer_html)
        else:
            st.markdown(drawer_html, unsafe_allow_html=True)
        st.markdown('<div id="diag_close_btn_anchor"></div>', unsafe_allow_html=True)
        if st.button("닫기", key="diag_close_btn"):
            st.session_state["diag_drawer_open"] = False
            st.session_state["diag_pending_llm"] = False
            st.session_state["diag_future"] = None
            st.rerun()

    if st.session_state.get("diag_drawer_open", False):
        st.markdown(
            f"""
            <style>
            [data-testid="stAppViewContainer"] .main {{
                box-sizing: border-box !important;
                width: calc(100% - {DIAG_DRAWER_WIDTH + 16}px) !important;
                max-width: calc(100% - {DIAG_DRAWER_WIDTH + 16}px) !important;
                margin-right: {DIAG_DRAWER_WIDTH + 16}px !important;
                padding-right: 0 !important;
                transition: none !important;
            }}
            [data-testid="stAppViewContainer"] .main .block-container {{
                max-width: 1400px !important;
                width: 100% !important;
                transform: none !important;
                transition: none !important;
            }}
            .diag-drawer {{
                width: {DIAG_DRAWER_WIDTH}px !important;
            }}
            div[data-testid="stVerticalBlock"]:has(#diag_close_btn_anchor) [data-testid="stButton"] > button {{
                width: {max(260, DIAG_DRAWER_WIDTH - 32)}px !important;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )

    # =========================
    # KPI COLUMN
    # =========================
    with col_kpi:
        def _fmt1(v):
            return f"{float(v):.1f}" if pd.notna(v) else "-"

        toc_signal = classify_signal(latest_toc, ALARM_TOC, warn_ratio=(WARN_TOC / ALARM_TOC))

        st.markdown(f"""
        <div class="glass-card" style="margin-bottom: 1.5rem; padding: 1.5rem;">
            <div class="metric-label">CURRENT TOC</div>
            <div class="metric-value">{_fmt1(latest_toc)}</div>
            <div style="margin-top: 1rem;">{badge_pill_html(toc_signal)}</div>
            <div class="metric-sub" style="margin-top: 0.5rem;">Updated: {last_updated.strftime('%H:%M') if pd.notna(last_updated) else '-'}</div>
        </div>
        """, unsafe_allow_html=True)

        pred_sig = classify_signal(pred_val, ALARM_TOC, warn_ratio=(WARN_TOC / ALARM_TOC))
        pred_jump = (float(pred_val) - float(latest_toc)) if (pd.notna(pred_val) and pd.notna(latest_toc)) else float("nan")
        jump_warn = pd.notna(pred_jump) and abs(pred_jump) >= 3.0
        st.markdown(f"""
        <div class="glass-card" style="margin-bottom: 1.5rem; padding: 1.5rem;">
            <div class="metric-label">예측 ({pred_label})</div>
            <div class="metric-value-md">{_fmt1(pred_val)}</div>
            <div style="margin-top: 1rem;">{badge_pill_html(pred_sig)}</div>
        </div>
        """, unsafe_allow_html=True)
        if jump_warn:
            st.markdown(
                """
                <div class="metric-sub" style="margin:-0.9rem 0 1.2rem; color:#F59E0B;">
                    24시간 예측 급변 구간입니다. 최근 입력값/센서 이상 여부와 12시간 예측을 함께 확인하세요.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(f"""
        <div class="glass-card" style="padding: 1.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div class="metric-label">FLOW</div>
                    <div class="metric-value-md" style="font-size: 1.5rem;">{latest_flow:.0f}</div>
                </div>
                <div>
                    <div class="metric-label">TEMP</div>
                    <div class="metric-value-md" style="font-size: 1.5rem;">{latest_temp:.0f}°</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # =========================
    # BOTTOM SECTION
    # =========================
    st.markdown('<div style="height: 2rem;"></div>', unsafe_allow_html=True)

    with st.container(border=False):
        st.markdown('<div id="recent_signals_anchor"></div>', unsafe_allow_html=True)
        st.markdown("### Recent Signals")
        recent_view = view.tail(5).sort_index(ascending=False)

        st.markdown("""
        <div class="recent-signals-head">
            <div>TIMESTAMP</div><div>TOC</div><div>FLOW</div><div>TEMP</div><div>STATUS</div>
        </div>
        """, unsafe_allow_html=True)

        for idx, row in recent_view.iterrows():
            r_toc = row.get("FINAL_TOC", 0)
            r_flow = row.get("FLOW", 0)
            r_temp = row.get("TEMP", 0)
            r_sig = classify_signal(r_toc, ALARM_TOC, warn_ratio=(WARN_TOC / ALARM_TOC))
            r_sig_text = signal_label_ko(r_sig)
            status_color = "#10B981" if r_sig == "OK" else "#F59E0B" if r_sig == "WARN" else "#EF4444"

            st.markdown(f"""
            <div class="recent-signals-row">
                <div style="color: #AAA;">{idx.strftime('%Y-%m-%d %H:%M')}</div>
                <div style="color: #FFF; font-weight: 600;">{r_toc:.2f}</div>
                <div style="color: #CCC;">{r_flow:.0f}</div>
                <div style="color: #CCC;">{r_temp:.0f}</div>
                <div style="color: {status_color}; font-weight: 700;">{r_sig_text}</div>
            </div>
            """, unsafe_allow_html=True)
