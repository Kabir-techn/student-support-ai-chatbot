"""
frontend/streamlit_app.py
==========================
Streamlit frontend for the AI Student Support Services Chatbot.

Pages:
  - Home:  project overview + "Start Chat" button
  - Chat:  chat interface, sidebar (recent conversations, suggested questions),
           feedback buttons, source citations, confidence display
  - Admin: upload/delete documents, rebuild FAISS index, view analytics & logs

Run:
    streamlit run frontend/streamlit_app.py

Requires the FastAPI backend to be running (default: http://localhost:8000).
"""

from __future__ import annotations

import os
import sys
import time

import requests
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings  # noqa: E402

API_BASE = os.environ.get("API_BASE_URL", f"http://localhost:{settings.API_PORT}")

st.set_page_config(
    page_title=settings.APP_NAME,
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Session state initialization
# --------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{role, content, confidence, sources, message_id}]
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False


# --------------------------------------------------------------------------
# Styling (light/dark mode + chat bubble polish)
# --------------------------------------------------------------------------
def inject_css() -> None:
    dark = st.session_state.dark_mode
    bg = "#0E1117" if dark else "#FFFFFF"
    text = "#F0F2F6" if dark else "#1A1A1A"
    bubble_user = "#2C3E50" if dark else "#DCEEFB"
    bubble_bot = "#1E2530" if dark else "#F4F6F7"
    user_text = "#FFFFFF" if dark else "#000000"

    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: {bg}; color: {text}; }}
        .chat-bubble-user {{
            background-color: {bubble_user}; color: {user_text};
            padding: 12px 16px; border-radius: 14px 14px 2px 14px;
            margin: 6px 0; max-width: 80%; margin-left: auto;
        }}
        .chat-bubble-bot {{
            background-color: {bubble_bot}; color: {text};
            padding: 12px 16px; border-radius: 14px 14px 14px 2px;
            margin: 6px 0; max-width: 80%;
        }}
        .source-tag {{
            font-size: 0.75rem; color: #888; margin-top: 4px;
        }}
        .confidence-badge {{
            display: inline-block; font-size: 0.7rem; padding: 2px 8px;
            border-radius: 10px; background: #E8F5E9; color: #2E7D32; margin-left: 6px;
        }}
        .confidence-badge.low {{ background: #FFF3E0; color: #E65100; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# API helpers
# --------------------------------------------------------------------------
def api_post(path: str, json_body: dict | None = None, files=None) -> dict | None:
    try:
        resp = requests.post(f"{API_BASE}{path}", json=json_body, files=files, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        st.error(f"Could not reach the backend API ({API_BASE}). Is it running? Details: {exc}")
        return None


def api_get(path: str, params: dict | None = None):
    try:
        resp = requests.get(f"{API_BASE}{path}", params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        st.error(f"Could not reach the backend API ({API_BASE}). Is it running? Details: {exc}")
        return None


def api_delete(path: str):
    try:
        resp = requests.delete(f"{API_BASE}{path}", timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        st.error(f"Request failed: {exc}")
        return None


# --------------------------------------------------------------------------
# Sidebar (shared across pages)
# --------------------------------------------------------------------------
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(f"## 🎓 {settings.APP_NAME}")
        st.session_state.dark_mode = st.toggle("🌙 Dark mode", value=st.session_state.dark_mode)

        st.divider()
        nav = st.radio("Navigate", ["Home", "Chat", "Admin"], index=["Home", "Chat", "Admin"].index(st.session_state.page))
        st.session_state.page = nav

        if st.session_state.page == "Chat":
            st.divider()
            st.markdown("### 💡 Suggested Questions")
            suggestions = api_get("/chat/suggested-questions") or []
            for q in suggestions[:6]:
                if st.button(q, key=f"sugg-{q}", use_container_width=True):
                    send_message(q)

            st.divider()
            if st.button("🗑️ New conversation", use_container_width=True):
                st.session_state.session_id = None
                st.session_state.messages = []
                st.rerun()


# --------------------------------------------------------------------------
# Chat logic
# --------------------------------------------------------------------------
def send_message(text: str) -> None:
    st.session_state.messages.append({"role": "user", "content": text})
    with st.spinner("Thinking..."):
        result = api_post(
            "/chat", json_body={"message": text, "session_id": st.session_state.session_id}
        )
    if result:
        st.session_state.session_id = result["session_id"]
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "confidence": result["confidence"],
                "sources": result["sources"],
                "message_id": result["message_id"],
                "answered_by": result["answered_by"],
            }
        )
    st.rerun()


def render_message(msg: dict, idx: int) -> None:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bubble-bot">{msg["content"]}</div>', unsafe_allow_html=True)

        conf = msg.get("confidence")
        if conf is not None:
            badge_class = "confidence-badge" if conf >= settings.CONFIDENCE_THRESHOLD else "confidence-badge low"
            st.markdown(
                f'<span class="{badge_class}">Confidence: {conf * 100:.0f}%</span>'
                f'<span class="source-tag">&nbsp;&nbsp;via {msg.get("answered_by", "rag").upper()}</span>',
                unsafe_allow_html=True,
            )

        sources = msg.get("sources") or []
        if sources:
            src_str = "; ".join(
                f"{s['source']}" + (f" (p. {s['page']})" if s.get("page") else "") for s in sources
            )
            st.caption(f"📄 Source: {src_str}")

        # Feedback + copy row
        cols = st.columns([1, 1, 1, 8])
        message_id = msg.get("message_id")
        if message_id:
            if cols[0].button("👍", key=f"up-{idx}"):
                api_post("/chat/feedback", {"message_id": message_id, "is_helpful": True})
                st.toast("Thanks for your feedback!")
            if cols[1].button("👎", key=f"down-{idx}"):
                api_post("/chat/feedback", {"message_id": message_id, "is_helpful": False})
                st.toast("Thanks — we'll use this to improve.")


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def page_home() -> None:
    st.title(f"🎓 {settings.APP_NAME}")
    st.markdown(
        """
        Welcome! This assistant can help you with:

        - 🎓 **Admissions** — process, eligibility, required documents
        - 💰 **Fees & Payments** — fee structure, online payment, due dates
        - 🏆 **Scholarships** — eligibility, how to apply, renewal
        - 🏠 **Hostel** — availability, fees, facilities
        - 📚 **Library** — timings, resources
        - 💼 **Placements** — companies, packages, eligibility
        - 📝 **Examinations** — schedules, attendance rules, revaluation
        - 📅 **Academic Calendar** — semester dates, holidays, events

        Answers are grounded in official college documents using Retrieval-Augmented
        Generation (RAG), with source citations and a confidence score on every answer.
        """
    )
    if st.button("🚀 Start Chat", type="primary"):
        st.session_state.page = "Chat"
        st.rerun()

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Response time target", "< 3 sec")
    c2.metric("Knowledge base formats", "PDF · DOCX · TXT · CSV")
    c3.metric("Hallucination control", "Confidence-gated")


def page_chat() -> None:
    st.title("💬 Student Support Chat")

    for i, msg in enumerate(st.session_state.messages):
        render_message(msg, i)

    prompt = st.chat_input("Ask about admissions, fees, hostel, exams...")
    if prompt:
        send_message(prompt)


def page_admin() -> None:
    st.title("🛠️ Admin Dashboard")

    tab_docs, tab_analytics, tab_logs = st.tabs(["📁 Documents", "📊 Analytics", "📜 Logs"])

    with tab_docs:
        st.subheader("Upload a new document")
        uploaded = st.file_uploader("PDF, DOCX, TXT, or CSV", type=["pdf", "docx", "txt", "csv"])
        if uploaded is not None and st.button("Upload"):
            files = {"file": (uploaded.name, uploaded.getvalue())}
            result = api_post("/admin/upload", files=files)
            if result:
                st.success(f"Uploaded: {result['filename']}")

        st.subheader("Knowledge base documents")
        docs = api_get("/admin/documents") or []
        if docs:
            for d in docs:
                c1, c2, c3 = st.columns([4, 2, 1])
                c1.write(d["filename"])
                c2.write("✅ Indexed" if d["indexed"] else "⏳ Not indexed")
                if c3.button("Delete", key=f"del-{d['filename']}"):
                    api_delete(f"/admin/documents/{d['filename']}")
                    st.rerun()
        else:
            st.info("No documents registered yet. Upload one above, or add files directly to the `documents/` folder.")

        st.divider()
        if st.button("🔄 Rebuild FAISS Vector Database", type="primary"):
            with st.spinner("Re-embedding documents and rebuilding index..."):
                result = api_post("/admin/rebuild-index")
            if result:
                st.success(f"Rebuilt index with {result['chunks_indexed']} chunks.")

    with tab_analytics:
        stats = api_get("/admin/analytics") or {}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total messages", stats.get("total_messages", 0))
        c2.metric("Total sessions", stats.get("total_sessions", 0))
        c3.metric("👍 Helpful", stats.get("helpful_feedback", 0))
        c4.metric("👎 Not helpful", stats.get("not_helpful_feedback", 0))
        if stats.get("average_confidence") is not None:
            st.metric("Average confidence", f"{stats['average_confidence'] * 100:.1f}%")

        st.subheader("Most common questions")
        common = api_get("/admin/common-questions") or []
        if common:
            st.table(common)
        else:
            st.info("No chat data yet.")

    with tab_logs:
        st.subheader("Export chat logs")
        if st.button("⬇️ Download CSV"):
            try:
                resp = requests.get(f"{API_BASE}/admin/export-logs", timeout=15)
                resp.raise_for_status()
                st.download_button(
                    "Save chat_logs.csv", data=resp.content, file_name="chat_logs.csv", mime="text/csv"
                )
            except requests.RequestException as exc:
                st.error(f"Export failed: {exc}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
inject_css()
render_sidebar()

if st.session_state.page == "Home":
    page_home()
elif st.session_state.page == "Chat":
    page_chat()
elif st.session_state.page == "Admin":
    page_admin()
