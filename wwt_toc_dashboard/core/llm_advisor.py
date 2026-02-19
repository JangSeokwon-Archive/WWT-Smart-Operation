from __future__ import annotations

import json
import os
import re
import subprocess
from html import unescape
from dataclasses import dataclass
from typing import Any


WARN_TOC = 15.0
ALARM_TOC = 20.0


@dataclass
class AdvisorResult:
    ok: bool
    source: str
    payload: dict[str, Any]
    raw_text: str = ""
    error: str = ""


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _status_from_toc(v: float) -> str:
    if not (v == v):
        return "주의"
    if v >= ALARM_TOC:
        return "주의"
    if v >= WARN_TOC:
        return "경고"
    return "양호"


def _status_from_toc_with_thresholds(v: float, warn: float, alarm: float) -> str:
    if not (v == v):
        return "주의"
    if alarm == alarm and v >= alarm:
        return "주의"
    if warn == warn and v >= warn:
        return "경고"
    return "양호"


def _humanize_terms(text: str) -> str:
    s = str(text or "")
    rep = {
        "AERB_DO": "폭기조 DO양",
        "AERB_RET": "폭기조 반송량",
        "AERB_WIT": "폭기조 인발량",
        "AERB_MLSS": "폭기조 MLSS",
        "EQ_TOC": "균등조 TOC",
        "DAF_TOC": "DAF TOC",
        "FINAL_TOC": "최종 TOC",
    }
    for k, v in rep.items():
        s = s.replace(k, v)
    return s


def _actions_from_sim_opt(ctx: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    opt = ctx.get("sim_opt_24h")
    if not isinstance(opt, dict) or not opt.get("ok"):
        return [], ""
    best = opt.get("best")
    if not isinstance(best, dict):
        return [], ""

    do_d = _safe_float(best.get("do_delta"), 0.0)
    ret_d = _safe_float(best.get("ret_delta_pct"), 0.0)
    wit_d = _safe_float(best.get("wit_delta"), 0.0)
    imp = _safe_float(best.get("improve_vs_base"), 0.0)
    base24 = _safe_float(opt.get("base_t24"))
    sim24 = _safe_float(best.get("sim_t24"))

    actions = []
    if abs(do_d) > 1e-6:
        actions.append(
            {
                "name": "DO 최적화",
                "target": "AERB_DO",
                "delta": float(do_d),
                "eta_hours": "6~24시간",
                "reason": f"시뮬레이터 최적안 기준 24시간 TOC 최소화(개선 {imp:+.2f})",
            }
        )
    if abs(ret_d) > 1e-6:
        actions.append(
            {
                "name": "반송 최적화",
                "target": "AERB_RET",
                "delta": float(ret_d),
                "eta_hours": "12~24시간",
                "reason": "반송(%p) 조정의 지연효과 반영",
            }
        )
    if abs(wit_d) > 1e-6:
        actions.append(
            {
                "name": "인발 최적화",
                "target": "AERB_WIT",
                "delta": float(wit_d),
                "eta_hours": "8~24시간",
                "reason": "인발 조정으로 변동성 완화/부하 밸런스 보정",
            }
        )
    summary = ""
    if sim24 == sim24:
        if base24 == base24:
            summary = f"권장 조치를 적용하면 24시간 뒤 TOC는 {base24:.2f}에서 {sim24:.2f} 수준으로 조정될 것으로 예상됩니다(개선량 {abs(imp):.2f})."
        else:
            summary = f"권장 조치를 적용하면 24시간 뒤 TOC는 약 {sim24:.2f} 수준으로 예상됩니다."
    return actions[:3], summary


def _eq_warn_message(ctx: dict[str, Any]) -> str:
    eq_v = _safe_float(ctx.get("EQ_TOC"))
    cur = _safe_float(ctx.get("current_toc"))
    p24 = _safe_float(ctx.get("pred_t24"))
    if eq_v == eq_v and p24 == p24 and cur == cur and p24 >= cur:
        return f"균등조 TOC가 높은 구간({eq_v:.1f})으로 유지되어, 최종 TOC는 현재({cur:.0f}) 대비 +1 내외 상승 가능성에 주의하세요."
    return ""


def _target_label(target: str) -> str:
    t = str(target).upper()
    if t == "AERB_RET":
        return "폭기조 반송량"
    if t == "AERB_WIT":
        return "폭기조 인발량"
    if t == "AERB_DO":
        return "폭기조 DO양"
    return t


def _priority_action_line(actions: list[dict[str, Any]], ctx: dict[str, Any]) -> str:
    if not actions:
        return ""
    a = actions[0]
    target = str(a.get("target", "")).upper()
    delta = _safe_float(a.get("delta"), 0.0)
    if target == "AERB_RET":
        cur = _safe_float(ctx.get("AERB_RET"))
        if cur == cur:
            nxt = cur + delta
            return f"운전자가 바로 실행할 우선순위 조치: 폭기조 반송량 {delta:+.1f}%p ({cur:.1f}% -> {nxt:.1f}%)로 조정합니다."
        return f"운전자가 바로 실행할 우선순위 조치: 폭기조 반송량 {delta:+.1f}%p로 조정합니다."
    if target == "AERB_WIT":
        cur = _safe_float(ctx.get("AERB_WIT"))
        if cur == cur:
            nxt = cur + delta
            return f"운전자가 바로 실행할 우선순위 조치: 폭기조 인발량 {delta:+.1f} ({cur:.1f} -> {nxt:.1f})로 조정합니다."
        return f"운전자가 바로 실행할 우선순위 조치: 폭기조 인발량 {delta:+.1f}로 조정합니다."
    if target == "AERB_DO":
        cur = _safe_float(ctx.get("AERB_DO"))
        if cur == cur:
            nxt = cur + delta
            return f"운전자가 바로 실행할 우선순위 조치: 폭기조 DO양 {delta:+.2f} ({cur:.2f} -> {nxt:.2f})로 조정합니다."
        return f"운전자가 바로 실행할 우선순위 조치: 폭기조 DO양 {delta:+.2f}로 조정합니다."
    return f"운전자가 바로 실행할 우선순위 조치: {_target_label(target)} {delta:+.2f} 조정입니다."


def _normalize_report_text(report: str, actions: list[dict[str, Any]], ctx: dict[str, Any]) -> str:
    r = _humanize_terms(report or "")
    r = re.sub(r"운전자가\s*바로\s*실행할\s*우선순위\s*조치\s*:[^\.]*(?:\.)?", "", r)
    r = re.sub(r"예측\s*형태는\s*'[^']*'\s*입니다\.?", "", r)
    r = r.replace("보합/정보부족", "정보 부족")
    pr = _priority_action_line(actions, ctx)
    r = re.sub(r"\s+", " ", r).strip()
    if r and pr:
        return f"{r} {pr}"
    if r:
        return r
    return pr


def _normalize_watchpoint_text(text: str) -> str:
    s = _humanize_terms(str(text or "").strip())
    s = re.sub(r"균등조\s*TOC가?\s*\d+(\.\d+)?\s*(ppm)?\s*이상", "균등조 TOC 고수준 구간", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _sanitize_plain_text(text: str) -> str:
    s = str(text or "")
    for _ in range(3):
        t = unescape(s)
        if t == s:
            break
        s = t
    s = s.replace("\xa0", " ").replace("&nbsp;", " ")
    s = re.sub(r"&lt;[^&]*&gt;", " ", s)
    s = re.sub(r"<[^>]*>", " ", s)
    s = re.sub(r"<[^>]*>", " ", s)
    s = re.sub(r"\b(diag-drawer-sec|diag-drawer-body|diag-report-list|diag-drawer-list)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _apply_strength_profile(actions: list[dict[str, Any]], ctx: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    prof = ctx.get("reco_strength_profile")
    lead = str(int(_safe_float(ctx.get("primary_reco_lead"), 24)))
    if not isinstance(prof, dict) or lead not in prof:
        return actions, "표준"
    meta = prof.get(lead, {})
    mult = _safe_float(meta.get("multiplier"), 1.0)
    label = str(meta.get("label", "표준"))
    if not (mult == mult):
        mult = 1.0
    scaled = []
    for a in actions:
        b = dict(a)
        b["delta"] = float(_safe_float(a.get("delta"), 0.0) * mult)
        scaled.append(b)
    return scaled, label


def _extract_json_block(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    block = m.group(0)
    try:
        obj = json.loads(block)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _validate_and_guardrail(payload: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    pred24 = _safe_float(ctx.get("pred_t24"))
    pred12 = _safe_float(ctx.get("pred_t12"))
    current_toc = _safe_float(ctx.get("current_toc"))
    th = ctx.get("thresholds", {})
    warn_th = _safe_float(th.get("warn"), WARN_TOC) if isinstance(th, dict) else WARN_TOC
    alarm_th = _safe_float(th.get("alarm"), ALARM_TOC) if isinstance(th, dict) else ALARM_TOC
    ref_toc = pred12 if pred12 == pred12 else (pred24 if pred24 == pred24 else current_toc)
    status_rule = _status_from_toc_with_thresholds(ref_toc, warn_th, alarm_th)
    status = status_rule

    # If trend is clearly decreasing, prevent over-severe label.
    if pred24 == pred24 and pred12 == pred12 and current_toc == current_toc:
        down_clear = (pred24 <= pred12) and (pred12 <= current_toc)
        if down_clear and pred24 < alarm_th:
            if pred24 < warn_th:
                status = "양호"
            elif status == "주의":
                status = "경고"

    diagnosis = str(payload.get("diagnosis", "")).strip() or "현재 데이터 기준으로 운전 안정성 점검이 필요합니다."
    diagnosis = _sanitize_plain_text(_humanize_terms(diagnosis))
    confidence = _safe_float(payload.get("confidence", 0.55))
    confidence = max(0.0, min(1.0, confidence))

    actions_raw = payload.get("actions", [])
    if not isinstance(actions_raw, list):
        actions_raw = []
    actions = []
    for item in actions_raw[:3]:
        if not isinstance(item, dict):
            continue
        target = str(item.get("target", "")).strip().upper()
        delta = _safe_float(item.get("delta", 0.0), 0.0)
        # hard guardrails
        if target == "AERB_DO":
            delta = max(-0.8, min(0.8, delta))
        elif target == "AERB_RET":
            delta = max(-10.0, min(10.0, delta))
        elif target == "AERB_WIT":
            delta = max(-20.0, min(20.0, delta))
        name = _humanize_terms(str(item.get("name", "")).strip() or f"{target} 조정")
        eta = str(item.get("eta_hours", "")).strip() or "6~24시간"
        reason = str(item.get("reason", "")).strip() or "지연효과와 현재 시그널 기준"
        reason = _sanitize_plain_text(_humanize_terms(reason))
        if target == "AERB_RET":
            name = name.replace("인발", "반송")
            reason = reason.replace("인발", "반송")
        elif target == "AERB_WIT":
            name = name.replace("반송", "인발")
            reason = reason.replace("반송", "인발")
        actions.append(
            {
                "name": name,
                "target": target if target else "AERB_DO",
                "delta": float(delta),
                "eta_hours": eta,
                "reason": reason,
            }
        )
    if status == "양호":
        actions = []
    if not actions and status != "양호":
        opt_actions, _ = _actions_from_sim_opt(ctx)
        actions = opt_actions
    actions, strength_label = _apply_strength_profile(actions, ctx)
    if not actions and status != "양호":
        actions = [
            {
                "name": "운전 안정화 모니터링",
                "target": "AERB_DO",
                "delta": 0.0,
                "eta_hours": "6~24시간",
                "reason": "현재는 대규모 조정보다 추세/부하 변화를 우선 모니터링",
            }
        ]

    watchpoints = payload.get("watchpoints", [])
    if not isinstance(watchpoints, list):
        watchpoints = []
    watchpoints = [_sanitize_plain_text(_normalize_watchpoint_text(str(x))) for x in watchpoints[:4] if str(x).strip()]
    eq_warn_msg = _eq_warn_message(ctx)
    if eq_warn_msg:
        watchpoints = [eq_warn_msg] + watchpoints

    report = _sanitize_plain_text(_humanize_terms(str(payload.get("report", "")).strip()))
    if not report:
        cur = _safe_float(ctx.get("current_toc"))
        p12 = _safe_float(ctx.get("pred_t12"))
        p24 = _safe_float(ctx.get("pred_t24"))
        mlss_band = str(ctx.get("mlss_band", "UNKNOWN"))
        do_v = _safe_float(ctx.get("AERB_DO"))
        wit_v = _safe_float(ctx.get("AERB_WIT"))
        ret_v = _safe_float(ctx.get("AERB_RET"))
        mlss_do_rec = _safe_float(ctx.get("mlss_recommended_do"))
        flow_v = _safe_float(ctx.get("FLOW"))
        flow_pct = _safe_float(ctx.get("flow_load_pct"))
        flow_cap = _safe_float(ctx.get("flow_capacity_tph"), 500.0)
        flow_high = _safe_float(ctx.get("flow_high_threshold_tph"), 400.0)
        high_inflow = bool(ctx.get("high_inflow", False))
        process_drop_risk = bool(ctx.get("process_drop_risk", False))
        do_trend = str(ctx.get("do_trend", "보합"))
        eq_slope = _safe_float(ctx.get("eq_toc_slope_12h"))
        daf_slope = _safe_float(ctx.get("daf_toc_slope_12h"))
        eq_daf_rise_room = bool(ctx.get("eq_daf_rise_room", False))
        trend = "상승" if (p24 == p24 and cur == cur and p24 > cur) else ("하락" if (p24 == p24 and cur == cur and p24 < cur) else "보합")
        lead_line = (
            f"현재 FINAL TOC는 {cur:.2f}이며 12시간 뒤/24시간 뒤 예측은 {p12:.2f}/{p24:.2f}로 {trend} 추세입니다."
            if (cur == cur and p12 == p12 and p24 == p24)
            else "현재 FINAL TOC 및 단기 예측 추세를 기준으로 상태를 평가했습니다."
        )
        proc_line = (
            f"폭기조 지표는 DO양={do_v:.2f}(추세 {do_trend}), 인발량={wit_v:.2f}, 반송량={ret_v:.2f}%, MLSS 구간={mlss_band}이며, "
            f"과폭기 또는 과반송 시 처리효율 저하와 변동성 확대 가능성을 함께 점검해야 합니다."
            if (do_v == do_v and wit_v == wit_v and ret_v == ret_v)
            else "폭기조 미생물/운전 지표를 함께 고려해 과폭기 및 반송 불안정 리스크를 점검했습니다."
        )
        load_line = (
            f"현재 유입수는 {flow_v:.0f}t/h이며, "
            f"{'고유입 구간(' + f'{flow_high:.0f}t/h 이상)' if high_inflow else '보통 유입 구간'}으로 판단됩니다."
            if (flow_v == flow_v)
            else "유입수 유량 대비 캐파 비율을 함께 점검해 부하 리스크를 평가했습니다."
        )
        mlss_line = (
            f"현재 MLSS 구간 기준 권장 DO양은 약 {mlss_do_rec:.2f}이며, 현재 DO양과의 편차를 단계적으로 보정하는 전략이 유효합니다."
            if mlss_do_rec == mlss_do_rec
            else "MLSS 구간별 권장 DO양 대비 편차를 점검하며 조정하는 전략이 필요합니다."
        )
        front_line = (
            f"균등조/DAF 최근 기울기(12h)는 {eq_slope:+.3f}/{daf_slope:+.3f}이며, "
            f"{'전단 곡선이 아직 완만하지 않아 추가 상승 여력이 있어 보입니다.' if eq_daf_rise_room else '전단 곡선이 완만해 추가 급상승 리스크는 제한적입니다.'}"
            if (eq_slope == eq_slope and daf_slope == daf_slope)
            else "균등조/DAF 곡선 완만화 여부를 함께 추적해야 후단 TOC 상승여력을 판단할 수 있습니다."
        )
        if status == "양호":
            inflow_ctrl_line = "현재는 즉시 운전값을 크게 바꾸기보다 추세 모니터링 중심 운전이 적절합니다."
        else:
            inflow_ctrl_line = (
                "현재 고유입 상태에서 폭기조 처리능 저하 신호가 보여 유입량 단계 감량 또는 유량 완충 운전을 함께 검토해야 합니다."
                if (high_inflow and process_drop_risk)
                else "현재 유입량은 즉시 감량보다는 폭기조 조정 효과를 우선 확인하는 구간으로 판단됩니다."
            )
        eq_warn_line = _eq_warn_message(ctx)
        _, opt_summary = _actions_from_sim_opt(ctx)
        if status == "양호":
            action_line = "현재는 폭기조 인발량/DO 즉시 조정 필요성은 낮으며, 현 수준 유지 + 이상 징후 조기 감지를 권장합니다."
        elif actions:
            action_line = f"권장 조치는 단계적 소폭 조정({strength_label})으로, "
            def _disp_target(t: str) -> str:
                return "DO" if t == "AERB_DO" else ("반송" if t == "AERB_RET" else ("인발" if t == "AERB_WIT" else t))
            action_line += ", ".join([f"{_disp_target(a['target'])} {a['delta']:+.2f}({a['eta_hours']})" for a in actions[:3]])
            action_line += " 순으로 우선 적용하는 것입니다."
        else:
            action_line = "현재는 즉시 대규모 조정보다 추세 모니터링을 우선 권고합니다."
        eta_line = "조치 후 6~12시간 1차 반응, 12~24시간 추세 확인, 24~48시간 누적 개선 여부를 재평가하세요."
        report = " ".join([lead_line, load_line, proc_line, mlss_line, front_line, eq_warn_line, inflow_ctrl_line, action_line, opt_summary, eta_line])

    report = _sanitize_plain_text(_normalize_report_text(report, actions, ctx))

    return {
        "status": status,
        "diagnosis": diagnosis,
        "report": report,
        "confidence": confidence,
        "actions": actions,
        "watchpoints": watchpoints,
        "reco_strength_label": strength_label,
    }


def _build_prompt(ctx: dict[str, Any]) -> str:
    return f"""
너는 대산 WWT FINAL TOC 운영 진단 보조다.
반드시 JSON만 출력해라. 설명문, 마크다운, 코드블록 금지.

[WWT 배경지식]
- FINAL TOC는 낮을수록 유리하다.
- 과폭기(과도한 DO)는 미생물 환경 불안정/에너지 과소비/처리효율 변동을 유발할 수 있다.
- RET(반송) 과도 증감은 슬러지 체류/부하 밸런스 악화로 TOC 변동을 키울 수 있다.
- 변수 조정 효과는 즉시가 아니라 지연되어 나타난다(수 시간~수십 시간).
- 조치 제안은 안전한 소폭 조정 중심이어야 하며 한번에 1~2개 변수 우선.
- 설비 캐파/고유입 기준은 입력값(flow_capacity_tph, flow_high_threshold_tph)을 따른다.
- MLSS 구간에 따라 권장 DO양이 달라지며, 현재 DO와 권장 DO 편차를 진단에 반영한다.
- 전단(균등조/EQ, DAF) TOC 곡선이 완만하지 않으면 후단 FINAL TOC 추가상승 여력이 있을 수 있다.
- 예측곡선이 상승하되 상승분이 둔화되면 "상승세이나 상승폭 둔화"로 해석한다.

[가드레일]
- AERB_DO delta 범위: [-0.8, +0.8]
- AERB_RET delta 범위(%p): [-10, +10]
- AERB_WIT delta 범위: [-20, +20]
- 상태는 양호/경고/주의 중 하나.

[현재 입력]
{json.dumps(ctx, ensure_ascii=False, indent=2)}

[추가 지시]
- 입력의 sim_opt_24h가 있으면, 해당 최적안(24시간 TOC 최소)을 우선 반영해 actions/reason/report를 작성해라.
- actions는 가능한 한 sim_opt_24h의 best 조합과 일치시켜라.
- report에 "24시간 뒤 예상 TOC"와 "기준 대비 개선량"을 숫자로 반드시 포함해라.
- report는 "12시간 운전 진단" 문장을 먼저 쓰고, "24시간 리스크 전망" 문장을 별도로 작성해라.
- AERB_DO/AERB_RET/AERB_WIT 같은 칼럼명을 그대로 쓰지 말고 "폭기조 DO양/폭기조 반송량/폭기조 인발량"으로 써라.
- EQ_TOC/DAF_TOC/FINAL_TOC 같은 컬럼명도 그대로 쓰지 말고 "균등조 TOC/DAF TOC/최종 TOC"로 써라.
- RET는 반드시 "반송량(%p)" 기준 절대값 조정으로 표현하고, WIT는 인발량으로만 표현해라(반송/인발 혼동 금지).
- status가 "양호"이면 "조정이 필요합니다/시급합니다" 같은 표현은 쓰지 말고 모니터링 중심으로 써라.
- status가 "양호"이면 diagnosis는 빈 문자열로 두고, actions는 빈 배열로 반환해라.
- 동일 의미 문장을 반복하지 말고, "권장조치 없음"과 "우선순위 조치 존재" 같은 모순을 만들지 마라.
- 현재 시점 모니터링 관점으로 작성하되, 컬럼명(EQ_TOC/DAF_TOC/FINAL_TOC/AERB_*)은 그대로 쓰지 마라.
- report는 반드시 아래를 포함해라:
  1) 12시간 운전 진단(현재 대비 변화량)
  2) 24시간 리스크 전망(단일값 + 범위형 표현)
  3) 유입부하 해석(400/500 기준)
  4) MLSS 기반 권장 DO와 현재 편차
  5) 조치 후 예상 반응시간과 예상 TOC
- 경고/주의 단계에서 고유입(400t/h 이상) + 처리능 저하 징후가 있으면 유입량 단계 감량 검토를 포함해라.

[출력 스키마(JSON)]
{{
  "status": "양호|경고|주의",
  "diagnosis": "한 문장 요약",
  "report": "현재상황, 추세, 폭기조 지표 해석, 조치, 몇시간 후 전망을 포함한 4~6문장",
  "confidence": 0.0,
  "actions": [
    {{
      "name": "조치명",
      "target": "AERB_DO|AERB_RET|AERB_WIT",
      "delta": 0.0,
      "eta_hours": "예: 6~24시간",
      "reason": "근거(지연시간/예측/리스크)"
    }}
  ],
  "watchpoints": ["모니터링 포인트1", "포인트2"]
}}
""".strip()


def _fallback_rule_based(ctx: dict[str, Any], error: str = "") -> AdvisorResult:
    cur = _safe_float(ctx.get("current_toc"))
    p12 = _safe_float(ctx.get("pred_t12"))
    p24 = _safe_float(ctx.get("pred_t24"))
    do_v = _safe_float(ctx.get("AERB_DO"))
    ret_v = _safe_float(ctx.get("AERB_RET"))
    mlss_band = str(ctx.get("mlss_band", "UNKNOWN"))
    mlss_do_rec = _safe_float(ctx.get("mlss_recommended_do"))
    flow_v = _safe_float(ctx.get("FLOW"))
    flow_high = _safe_float(ctx.get("flow_high_threshold_tph"), 400.0)
    high_inflow = bool(ctx.get("high_inflow", False))
    process_drop_risk = bool(ctx.get("process_drop_risk", False))
    eq_slope = _safe_float(ctx.get("eq_toc_slope_12h"))
    daf_slope = _safe_float(ctx.get("daf_toc_slope_12h"))
    eq_daf_rise_room = bool(ctx.get("eq_daf_rise_room", False))

    ref = p12 if p12 == p12 else (p24 if p24 == p24 else cur)
    status = _status_from_toc(ref)

    actions = []
    opt_actions, opt_summary = _actions_from_sim_opt(ctx)
    if status in {"경고", "주의"}:
        if opt_actions:
            actions = opt_actions
        else:
            actions.append(
                {
                    "name": "DO 미세 조정",
                    "target": "AERB_DO",
                    "delta": -0.2 if do_v > 2.8 else 0.2,
                    "eta_hours": "6~24시간",
                    "reason": "과폭기/저DO 리스크를 피하는 범위에서 소폭 조정",
                }
            )
            actions.append(
                {
                    "name": "반송 안정화",
                    "target": "AERB_RET",
                    "delta": -3.0 if ret_v > 165 else (3.0 if ret_v < 100 else 0.0),
                    "eta_hours": "12~36시간",
                    "reason": "반송 과도 변동으로 인한 TOC 변동성 완화",
                }
            )
    elif opt_actions:
        actions = []
    actions, strength_label = _apply_strength_profile(actions, ctx)
    eq_warn_msg = _eq_warn_message(ctx)

    payload = {
        "status": status,
        "diagnosis": f"현재 TOC={cur:.2f}, 12시간 뒤 예측 TOC={ref:.2f}, MLSS={mlss_band} 기준 진단입니다.",
        "report": (
            f"현재 최종 TOC는 {cur:.2f}이고 12시간 뒤 예측은 {(p12 if p12 == p12 else ref):.2f}입니다. "
            f"24시간 뒤 리스크 기준 예측은 {(p24 if p24 == p24 else ref):.2f}입니다. "
            f"현재 유입수는 {flow_v:.0f}t/h로 {'고유입(' + f'{flow_high:.0f} 이상)' if high_inflow else '보통 유입'} 상태입니다. "
            f"폭기조는 DO양={do_v:.2f}, 반송량={ret_v:.2f}%, MLSS={mlss_band}이며 MLSS 기준 권장 DO양은 {mlss_do_rec:.2f}입니다. "
            f"균등조/DAF 기울기(12h)는 {eq_slope:+.3f}/{daf_slope:+.3f}로, {'추가 상승 여력이 남아있습니다.' if eq_daf_rise_room else '추가 급상승 리스크는 제한적입니다.'} "
            f"{eq_warn_msg + ' ' if eq_warn_msg else ''}"
            f"{'고유입과 처리능 저하가 동시에 나타나 유입량 단계 감량을 함께 검토하세요. ' if (high_inflow and process_drop_risk) else ''}"
            f"{opt_summary if status in {'경고', '주의'} else '현재는 즉시 조정보다 모니터링 중심 운전이 적절합니다.'} "
            f"조치 후 6~12시간 1차 반응, 12~24시간 추세, 24~48시간 누적 개선을 순차 확인하세요."
        ),
        "confidence": 0.45,
        "actions": actions[:3],
        "watchpoints": [
            *([eq_warn_msg] if eq_warn_msg else []),
            "조정 후 6h/12h/24h TOC 추이 비교",
            "DO 과상승(과폭기) 및 반송 급변 모니터링",
            "유입수 400t/h 초과 구간의 부하 지속시간 확인",
            "균등조/DAF 곡선 완만화 여부와 최종 TOC 동조 여부 확인",
        ],
        "reco_strength_label": strength_label,
    }
    payload["report"] = _normalize_report_text(str(payload.get("report", "")), actions[:3], ctx)
    return AdvisorResult(ok=True, source="rule_fallback", payload=payload, error=error)


def build_quick_report(context: dict[str, Any]) -> AdvisorResult:
    """Immediate deterministic report for instant UI feedback."""
    return _fallback_rule_based(context, error="")


def run_llm_advisor(context: dict[str, Any], model: str | None = None, timeout_sec: int = 40) -> AdvisorResult:
    model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
    prompt = _build_prompt(context)

    try:
        cp = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            timeout=int(timeout_sec),
            check=False,
        )
        raw = (cp.stdout or "").strip()
        if cp.returncode != 0:
            err = (cp.stderr or "").strip() or f"ollama exit code={cp.returncode}"
            return _fallback_rule_based(context, error=err)

        obj = _extract_json_block(raw)
        if not obj:
            return _fallback_rule_based(context, error="llm output is not valid json")
        guarded = _validate_and_guardrail(obj, context)
        return AdvisorResult(ok=True, source=f"ollama:{model}", payload=guarded, raw_text=raw)
    except Exception as e:
        return _fallback_rule_based(context, error=str(e))
