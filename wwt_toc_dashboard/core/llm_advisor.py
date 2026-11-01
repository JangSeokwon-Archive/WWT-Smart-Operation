"""Evidence-bound Ollama explanations; numerical decisions are deterministic."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import subprocess
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


def _number(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def rule_status(ctx: dict) -> str:
    values = [_number(ctx.get(key)) for key in ("current_toc", "pred_t6", "pred_t12", "pred_t24")]
    finite = [value for value in values if math.isfinite(value)]
    thresholds = ctx.get("thresholds") or {}
    warn = _number(thresholds.get("warn", WARN_TOC))
    alarm = _number(thresholds.get("alarm", ALARM_TOC))
    if not finite or not (math.isfinite(warn) and math.isfinite(alarm) and 0 <= warn < alarm):
        return "주의"
    if max(finite) >= alarm:
        return "주의"
    if max(finite) >= warn:
        return "경고"
    if len(finite) != len(values) or ctx.get("forecast_meta", {}).get("quality", {}).get("status") == "degraded":
        return "주의"
    return "양호"


def _evidence(ctx: dict) -> dict[str, str]:
    result = {"validation_needed": "예측 모델의 검증 상태와 기준 시각을 확인하세요. 시뮬레이션 추정값의 실제 제어 효과는 미검증입니다."}
    current, pred = _number(ctx.get("current_toc")), _number(ctx.get("pred_t24"))
    if math.isfinite(current) and math.isfinite(pred):
        result["forecast_rising" if pred > current else "forecast_nonrising"] = "예측 TOC가 현재보다 상승합니다. 전단 부하와 후단 추세를 함께 관찰하세요." if pred > current else "예측 TOC가 현재 이하입니다. 관측값으로 추세를 계속 확인하세요."
    if ctx.get("high_inflow") is True:
        result["high_inflow"] = "설정된 고유입 기준에 해당합니다. 유량 및 유입 TOC 추세를 확인하세요."
    if any(not math.isfinite(_number(ctx.get(key))) for key in ("current_toc", "pred_t6", "pred_t12", "pred_t24")):
        result["missing_data"] = "일부 관측 또는 예측값이 없어 판단이 제한됩니다. 계측과 데이터 갱신 상태를 확인하세요."
    return result


def _validate_and_guardrail(payload: dict, ctx: dict) -> dict:
    evidence = _evidence(ctx)
    requested = payload.get("evidence_codes", [])
    if not isinstance(requested, list):
        requested = []
    codes = [code for code in requested if isinstance(code, str) and code in evidence]
    codes = list(dict.fromkeys(["validation_needed", *codes]))
    lines = []
    for key, label in (("current_toc", "현재 TOC"), ("pred_t6", "6h 모델"), ("pred_t12", "12h 모델"), ("pred_t24", "24h 모델")):
        value = _number(ctx.get(key))
        lines.append(f"{label}: {value:.2f}" if math.isfinite(value) else f"{label}: 데이터 부족")
    return {
        "status": rule_status(ctx),
        "diagnosis": "관측 및 예측값에 명시적 기준을 적용한 운전 점검 결과입니다.",
        "report": "; ".join(lines) + ". " + " ".join(evidence[code] for code in codes),
        "confidence": None,
        "confidence_status": "미검증",
        "actions": [],
        "watchpoints": [evidence[code] for code in codes],
        "reco_strength_label": "시뮬레이션 추정값 · 실제 효과 미검증",
    }


def build_quick_report(context: dict) -> AdvisorResult:
    return AdvisorResult(True, "rule_local_v2", _validate_and_guardrail({"evidence_codes": list(_evidence(context))}, context))


def _build_prompt(ctx: dict) -> str:
    # Only evidence codes cross the LLM boundary; arbitrary claims and numbers
    # are not accepted by the renderer, even when present in valid JSON.
    return "주어진 근거 중 운전자 해설에 필요한 항목을 선택하세요. 수치 예측이나 운전 조작값을 생성하지 마세요. JSON 형식 {\"evidence_codes\": [코드]}만 반환하세요.\n" + json.dumps(_evidence(ctx), ensure_ascii=False)


def run_llm_advisor(context: dict, model: str | None = None, timeout_sec: int = 40) -> AdvisorResult:
    model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
    try:
        result = subprocess.run(["ollama", "run", model, _build_prompt(context)], capture_output=True, text=True,
                                timeout=timeout_sec, check=False)
        if result.returncode:
            raise RuntimeError("Ollama did not complete successfully")
        payload = json.loads(result.stdout.strip())
        if not isinstance(payload, dict):
            raise ValueError("Ollama response must be a JSON object")
        return AdvisorResult(True, f"ollama:{model}", _validate_and_guardrail(payload, context))
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        report = build_quick_report(context)
        report.error = str(error)
        return report
