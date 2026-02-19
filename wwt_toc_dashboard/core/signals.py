import numpy as np
import pandas as pd

def classify_signal(value: float, limit: float, warn_ratio: float = 0.9) -> str:
    """Classifies the signal based on the limit."""
    if not np.isfinite(value):
        return "UNKNOWN"
    if value >= limit:
        return "ALARM"
    if value >= limit * warn_ratio:
        return "WARN"
    return "OK"

def badge_pill_html(signal: str) -> str:
    """Generates a detailed pill badge HTML."""
    if signal == "OK":
        return '<span class="status-indicator status-ok"><span class="status-dot"></span> 양호</span>'
    if signal == "WARN":
        return '<span class="status-indicator status-warn"><span class="status-dot"></span> 경고</span>'
    if signal == "ALARM":
        return '<span class="status-indicator status-alarm"><span class="status-dot"></span> 주의</span>'
    return '<span class="status-indicator status-unknown"><span class="status-dot"></span> UNKNOWN</span>'

def signal_label_ko(signal: str) -> str:
    if signal == "OK":
        return "양호"
    if signal == "WARN":
        return "경고"
    if signal == "ALARM":
        return "주의"
    return "미정"

def simple_badge_html(signal: str) -> str:
    """Generates a small dot/text badge."""
    if signal == "OK":
        return '<span class="text-success">●</span>'
    if signal == "WARN":
        return '<span class="text-warning">●</span>'
    if signal == "ALARM":
        return '<span class="text-danger">●</span>'
    return '<span class="text-muted">●</span>'

def future_points(index_last: pd.Timestamp):
    return [
        index_last + pd.Timedelta(hours=12),
        index_last + pd.Timedelta(hours=24),
        index_last + pd.Timedelta(hours=36),
    ]
