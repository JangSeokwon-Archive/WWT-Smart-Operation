import os
import uuid
import json
from pathlib import Path
import streamlit as st
from datetime import datetime

from core.data import load_raw_csv, ensure_pred_columns
from core.signals import classify_signal
from core.notes_db import (
    init_db, insert_note, list_notes, get_note, save_media, delete_notes, update_note_memo
)

st.set_page_config(
    page_title="Notes | WWT TOC",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# CSS
APP_DIR = Path(__file__).resolve().parents[1]
css_path = APP_DIR / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

init_db()

raw_path = str(APP_DIR / "data" / "raw.csv")
automl_raw_path = APP_DIR.parent / "wwt-toc-automl" / "data" / "raw" / "raw.csv"
if automl_raw_path.exists():
    raw_path = str(automl_raw_path)
limit = 13.0

@st.cache_data(ttl=60)
def load_data(path):
    try:
        return ensure_pred_columns(load_raw_csv(path))
    except:
        return None

df = load_data(raw_path)

# ---------------------------
# State
# ---------------------------
if "modal" not in st.session_state:
    st.session_state.modal = None      # "create" | "view" | "delete" | None
if "modal_note_id" not in st.session_state:
    st.session_state.modal_note_id = None
if "notes_search" not in st.session_state:
    st.session_state.notes_search = ""

def open_modal(name, note_id=None):
    st.session_state.modal = name
    st.session_state.modal_note_id = note_id

def metric_box(label, value, color=None):
    style = f"color:{color};" if color else ""
    return f"""
    <div class="metric-box">
        <div class="metric-box-label">{label}</div>
        <div class="metric-box-value" style="{style}">{value}</div>
    </div>
    """


def _risk_ko(v: str) -> str:
    t = str(v or "").upper()
    if t == "OK":
        return "양호"
    if t == "WARN":
        return "경고"
    if t == "ALARM":
        return "주의"
    return "미정"

# ---------------------------
# Layout wrapper
# ---------------------------
st.markdown('<div style="height: 1.25rem;"></div>', unsafe_allow_html=True)
pad_l, main, pad_r = st.columns([0.9, 8.8, 1.3])

with main:
    st.markdown('<div id="notes_header_anchor"></div>', unsafe_allow_html=True)

    # ✅ rows 먼저 가져와야 "체크 즉시 Delete 활성화"가 됨
    rows = list_notes(limit=300)

    # ✅ 현재 체크 상태는 session_state의 sel_* 를 바로 읽는다 (rerun 즉시 반영)
    selected_ids_now = []
    for r in rows:
        nid = r[0]
        if st.session_state.get(f"sel_{nid}", False):
            selected_ids_now.append(nid)

    # Header
    left, mid, right = st.columns([2.2, 5.6, 2.2], vertical_alignment="center", gap="small")

    with left:
        if st.button("＋ New Note", type="primary", key="new_note_btn"):
            open_modal("create")

    with mid:
        search_query = st.text_input("Search", placeholder="Search...", label_visibility="collapsed", key="notes_search")

    with right:
        if st.button(
            f"Delete ({len(selected_ids_now)})",
            disabled=(len(selected_ids_now) == 0),
            key="delete_selected_btn",
        ):
            open_modal("delete")

    st.markdown('<div style="height: 0.9rem;"></div>', unsafe_allow_html=True)

    # List
    st.markdown('<div id="notes_list_anchor"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="notes-head">
        <div></div>
        <div>Date</div>
        <div>Risk</div>
        <div>TOC</div>
        <div>Memo</div>
        <div></div>
    </div>
    """, unsafe_allow_html=True)

    if not rows:
        st.info("No notes yet.")
    else:
        for r in rows:
            nid, r_date, r_toc, _, _, r_risk, r_memo, _ = r

            if search_query:
                if (search_query.lower() not in (r_memo or "").lower()) and (search_query not in (r_date or "")):
                    continue

            badge_color = "#9CA3AF"
            if r_risk == "OK": badge_color = "#10B981"
            elif r_risk == "WARN": badge_color = "#F59E0B"
            elif r_risk == "ALARM": badge_color = "#EF4444"
            r_risk_ko = _risk_ko(r_risk)

            row = st.columns([0.45, 1.6, 1.0, 1.0, 5.0, 1.1], gap="small")

            with row[0]:
                st.checkbox(
                    "sel",
                    value=bool(st.session_state.get(f"sel_{nid}", False)),
                    key=f"sel_{nid}",
                    label_visibility="collapsed"
                )

            with row[1]:
                st.markdown(f'<span class="note-date">{(r_date or "")[:16]}</span>', unsafe_allow_html=True)

            with row[2]:
                st.markdown(f'<span class="note-risk" style="color:{badge_color};">● {r_risk_ko}</span>', unsafe_allow_html=True)

            with row[3]:
                try:
                    toc_val = float(r_toc) if r_toc is not None else None
                    toc_txt = f"{toc_val:.2f}" if toc_val is not None else "-"
                except:
                    toc_txt = "-"
                st.markdown(f'<span class="note-toc">{toc_txt}</span>', unsafe_allow_html=True)

            with row[4]:
                safe_memo = (r_memo or "").replace("\n", " ")
                st.markdown(f'<span class="note-memo">{safe_memo}</span>', unsafe_allow_html=True)

            with row[5]:
                if st.button("View", key=f"view_{nid}"):
                    open_modal("view", note_id=nid)

            st.markdown('<div class="note-divider"></div>', unsafe_allow_html=True)

# ---------------------------
# Dialogs
# ---------------------------
@st.dialog("Confirm Delete", width="small")
def dialog_delete():
    # ✅ dialog 안에서도 최신 체크 상태를 다시 읽는다
    rows = list_notes(limit=300)
    ids = [r[0] for r in rows if st.session_state.get(f"sel_{r[0]}", False)]

    st.markdown(f"Delete **{len(ids)}** selected notes? This cannot be undone.")

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Cancel", use_container_width=True):
            st.rerun()
    with c2:
        if st.button("Delete", type="primary", use_container_width=True):
            deleted = delete_notes(ids)
            # clear checkboxes
            for nid in ids:
                if f"sel_{nid}" in st.session_state:
                    st.session_state[f"sel_{nid}"] = False
            st.success(f"Deleted {deleted} notes.")
            st.rerun()

@st.dialog("New Note", width="large")
def dialog_create():
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    current_toc, current_flow, current_temp, current_risk = None, None, None, "UNKNOWN"
    if df is not None and not df.empty:
        latest = df.iloc[-1]
        current_toc = float(latest.get("FINAL_TOC", 0))
        current_flow = float(latest.get("FLOW", 0))
        current_temp = float(latest.get("TEMP", 0))
        current_risk = classify_signal(current_toc, limit)
        now_str = str(df.index.max())

    st.markdown("### New Note Entry")
    st.caption(f"Time Reference: {now_str}")
    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(metric_box("TOC", f"{current_toc:.2f}" if current_toc is not None else "-"), unsafe_allow_html=True)
    with c2: st.markdown(metric_box("FLOW", f"{current_flow:.0f}" if current_flow is not None else "-"), unsafe_allow_html=True)
    with c3: st.markdown(metric_box("TEMP", f"{current_temp:.0f}" if current_temp is not None else "-"), unsafe_allow_html=True)
    with c4:
        risk_color = "#FFFFFF"
        if current_risk == "OK": risk_color = "#10B981"
        elif current_risk == "WARN": risk_color = "#F59E0B"
        elif current_risk == "ALARM": risk_color = "#EF4444"
        st.markdown(metric_box("RISK", _risk_ko(current_risk), risk_color), unsafe_allow_html=True)

    st.markdown('<div style="height:1.25rem"></div>', unsafe_allow_html=True)

    st.markdown('<div class="memo-dark-wrap">', unsafe_allow_html=True)
    memo_input = st.text_area("Analysis Memo", height=160, placeholder="Write your observation here...", key="memo_dark_create")
    st.markdown('</div>', unsafe_allow_html=True)
    files = st.file_uploader("Attachments", accept_multiple_files=True, key="create_files")

    st.markdown('<div style="height:0.75rem"></div>', unsafe_allow_html=True)

    col_x, col_y = st.columns([3, 1])
    with col_y:
        if st.button("Save Entry", type="primary", use_container_width=True):
            if not memo_input.strip():
                st.warning("Memo required.")
                return

            note_id = str(uuid.uuid4())
            media_paths = []
            if files:
                for f in files:
                    media_paths.append(save_media(f.name, f.getvalue()))

            insert_note(note_id, now_str, current_toc, current_flow, current_temp, current_risk, memo_input, media_paths)
            st.success("Entry Saved")
            st.rerun()

@st.dialog("Note Detail", width="large")
def dialog_view(note_id):
    note = get_note(note_id)
    if not note:
        st.error("Note not found.")
        return

    _, created_at, ftoc, flow, temp, risk, memo, mjson = note

    st.markdown("### Note Detail")
    st.caption(f"Created: {created_at}")
    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1: st.markdown(metric_box("TOC", f"{ftoc:.2f}" if ftoc is not None else "-"), unsafe_allow_html=True)
    with m2: st.markdown(metric_box("FLOW", f"{flow:.0f}" if flow is not None else "-"), unsafe_allow_html=True)
    with m3: st.markdown(metric_box("TEMP", f"{temp:.0f}" if temp is not None else "-"), unsafe_allow_html=True)
    with m4:
        risk_color = "#FFFFFF"
        if risk == "OK": risk_color = "#10B981"
        elif risk == "WARN": risk_color = "#F59E0B"
        elif risk == "ALARM": risk_color = "#EF4444"
        st.markdown(metric_box("RISK", _risk_ko(risk), risk_color), unsafe_allow_html=True)

    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

    # memo + attachment 추가 저장
    st.markdown('<div class="memo-dark-wrap">', unsafe_allow_html=True)
    edit = st.text_area("Memo", value=memo or "", height=180, key=f"memo_dark_edit_{note_id}")
    st.markdown('</div>', unsafe_allow_html=True)

    existing_paths = json.loads(mjson) if mjson else []
    if existing_paths:
        st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
        st.caption("Attachments")
        with st.container(height=460, border=True):
            for p in existing_paths:
                if p and os.path.exists(p):
                    st.image(p, use_container_width=True)

    st.markdown('<div style="height:0.6rem"></div>', unsafe_allow_html=True)
    new_files = st.file_uploader(
        "Add Attachments",
        accept_multiple_files=True,
        key=f"edit_files_{note_id}",
    )

    c1, c2 = st.columns([3, 1])
    with c2:
        if st.button("Save Memo", type="primary", use_container_width=True):
            merged_paths = list(existing_paths)
            if new_files:
                for f in new_files:
                    merged_paths.append(save_media(f"{note_id}_{f.name}", f.getvalue()))
            update_note_memo(note_id, edit, merged_paths)
            st.success("Saved.")
            st.rerun()

# ---------------------------
# One-shot modal trigger
# ---------------------------
modal = st.session_state.modal
note_id = st.session_state.modal_note_id
if modal is not None:
    st.session_state.modal = None
    st.session_state.modal_note_id = None

    if modal == "create":
        dialog_create()
    elif modal == "view" and note_id:
        dialog_view(note_id)
    elif modal == "delete":
        dialog_delete()
