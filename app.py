"""
Citation Desk — Streamlit UI.

Hybrid-retrieval RAG over your PDFs, with cross-encoder reranking,
grounded citations, auto-summaries, and suggested follow-ups.
"""
from __future__ import annotations

import io
import os
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from pipeline import Answer, CitationDesk, MAX_FILES

MAX_FILE_BYTES = 20 * 1024 * 1024
APP_NAME = "Citation Desk"

load_dotenv()

st.set_page_config(
    page_title="Citation Desk — Research workspace",
    page_icon="🗂️",
    layout="wide",
)

st.markdown(
    """<style>
    .stApp {
        background-color: #f8fafc;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    /* High contrast text in main workspace */
    [data-testid="stMainBlockContainer"],
    [data-testid="stMainBlockContainer"] h1,
    [data-testid="stMainBlockContainer"] h2,
    [data-testid="stMainBlockContainer"] h3,
    [data-testid="stMainBlockContainer"] h4,
    [data-testid="stMainBlockContainer"] p,
    [data-testid="stMainBlockContainer"] span,
    [data-testid="stMainBlockContainer"] label,
    [data-testid="stMainBlockContainer"] .stMarkdown {
        color: #0f172a !important;
    }
    [data-testid="stMainBlockContainer"] [data-testid="stCaptionContainer"] p {
        color: #475569 !important;
    }
    /* Sidebar high contrast */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
    [data-testid="stSidebar"] * {
        color: #f1f5f9;
    }
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea {
        color: #f8fafc !important;
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
    }
    [data-testid="stSidebar"] input::placeholder,
    [data-testid="stSidebar"] textarea::placeholder {
        color: #94a3b8 !important;
    }
    .research-kicker {
        color: #0284c7 !important;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }
    .author-badge {
        display: inline-block;
        background: rgba(14, 165, 233, 0.15);
        color: #38bdf8 !important;
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 9999px;
        padding: 3px 12px;
        font-size: 0.76rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    /* Alert / info banner contrast */
    [data-testid="stAlert"] {
        background-color: #e0f2fe !important;
        border: 1px solid #bae6fd !important;
    }
    [data-testid="stAlert"] * {
        color: #0369a1 !important;
    }
    .citation-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #0284c7;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 8px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .citation-card * {
        color: #0f172a !important;
    }
    .citation-card blockquote {
        border-left: none;
        margin: 4px 0 0 0;
        color: #334155 !important;
        font-style: italic;
        font-size: 0.93rem;
    }
    </style>""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state setup
# ---------------------------------------------------------------------------

if "engine" not in st.session_state:
    st.session_state.engine = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # list[{role, content, answer?}]
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "collection_name" not in st.session_state:
    st.session_state.collection_name = "Untitled reading collection"
if "research_objective" not in st.session_state:
    st.session_state.research_objective = ""
if "research_notes" not in st.session_state:
    st.session_state.research_notes = ""


def _api_key() -> str | None:
    """Read API key from environment (.env) or Streamlit secrets without triggering missing secrets warnings."""
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if key:
        return key

    local_secrets = os.path.join(os.path.dirname(__file__), ".streamlit", "secrets.toml")
    user_secrets = os.path.expanduser("~/.streamlit/secrets.toml")
    if os.path.exists(local_secrets) or os.path.exists(user_secrets):
        try:
            return st.secrets.get("GOOGLE_API_KEY") or st.secrets.get("GEMINI_API_KEY")
        except Exception:
            pass
    return None


def _history_text(limit: int = 6) -> str:
    msgs = st.session_state.messages[-limit:]
    lines = []
    for m in msgs:
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)


def _markdown_export() -> str:
    lines = [
        f"# {APP_NAME} research conversation",
        f"**Author / Workspace:** Anand ([@anand15-og](https://github.com/anand15-og))",
        f"**Collection:** {st.session_state.collection_name}",
        f"**Research objective:** {st.session_state.research_objective or '(not set)'}",
        f"_Exported {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "",
        f"**Documents:** {', '.join(st.session_state.ingested_files) or '(none)'}",
        "",
        "---",
        "",
    ]
    for m in st.session_state.messages:
        if m["role"] == "user":
            lines.append(f"### 🧑 Question\n{m['content']}\n")
        else:
            lines.append(f"### 🤖 Answer\n{m['content']}\n")
            ans: Answer | None = m.get("answer")
            if ans and ans.citations:
                lines.append("**Citations**")
                for c in ans.citations:
                    lines.append(
                        f"- [{c.marker}] {c.source} · p.{c.page} — "
                        f"\"{c.excerpt}\""
                    )
                lines.append("")
    return "\n".join(lines)


def _research_brief() -> str:
    """Create a compact, editable research handoff from the current session."""
    engine: CitationDesk = st.session_state.engine
    lines = [
        f"# {st.session_state.collection_name}",
        f"_Prepared with Citation Desk by Anand (@anand15-og)_",
        "",
        "## Research objective",
        st.session_state.research_objective or "Not set.",
        "",
        "## Collection overview",
    ]
    for name, stats in engine.document_stats.items():
        lines.append(
            f"- **{name}** — {stats['pages']} readable pages, "
            f"{stats['characters']:,} extracted characters"
        )
    lines.extend(["", "## Document summaries"])
    for name, summary in engine.summaries.items():
        lines.extend([f"### {name}", summary, ""])
    lines.extend([
        "## Research notes",
        st.session_state.research_notes or "No notes recorded yet.",
        "",
        "## Citation reminder",
        "Verify claims against the cited source pages before using this brief in academic or professional work.",
    ])
    return "\n".join(lines)


def _render_answer(answer: Answer, key_prefix: str) -> None:
    """Render the answer body, citations, and follow-up buttons."""
    st.markdown(answer.text)

    if answer.citations:
        with st.expander(f"🔎 {len(answer.citations)} cited sources"):
            for c in answer.citations:
                st.markdown(
                    f"<div class='citation-card'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;'>"
                    f"<span style='font-weight:600;'>[{c.marker}] {c.source} · Page {c.page}</span>"
                    f"<span style='background:#e0f2fe;color:#0284c7;font-size:0.75rem;padding:2px 8px;border-radius:12px;font-weight:600;'>relevance {c.score:+.2f}</span>"
                    f"</div>"
                    f"<blockquote>“{c.excerpt}”</blockquote>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    if answer.followups:
        cols = st.columns(len(answer.followups))
        for idx, (col, q) in enumerate(zip(cols, answer.followups)):
            if col.button(q, key=f"{key_prefix}_{idx}"):
                st.session_state.pending_question = q
                st.rerun()


# ---------------------------------------------------------------------------
# Sidebar — upload, summaries, stats, export
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        f"""
        <div style='margin-bottom: 8px;'>
            <h2 style='margin: 0; font-size: 1.45rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.02em;'>
                🗂️ {APP_NAME}
            </h2>
            <p style='margin: 2px 0 8px 0; font-size: 0.8rem; color: #94a3b8;'>
                AI-Powered PDF Research Workspace
            </p>
            <span class='author-badge'>by Anand</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("A focused workspace for reading, questioning, and citing PDFs.")
    st.divider()

    st.session_state.collection_name = st.text_input(
        "Collection name",
        value=st.session_state.collection_name,
        help="Use a meaningful name for this reading session and its exports.",
    )
    st.session_state.research_objective = st.text_area(
        "Research objective",
        value=st.session_state.research_objective,
        placeholder="What question should this collection help you answer?",
        height=86,
    )

    api_key = _api_key()
    if not api_key:
        api_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            placeholder="Paste Gemini API key (AQ... or AIza...)",
            help="Get your API key from Google AI Studio: https://aistudio.google.com/apikey",
        )

    pdf_files = st.file_uploader(
        "Add research PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    oversized_files = [
        file.name
        for file in pdf_files or []
        if file.size > MAX_FILE_BYTES
    ]
    too_many_files = len(pdf_files or []) > MAX_FILES
    if too_many_files:
        st.error(f"Upload at most {MAX_FILES} PDFs at a time.")
    if oversized_files:
        st.error(
            "Each PDF must be 20 MB or smaller: "
            + ", ".join(oversized_files)
        )

    st.info(
        "Data disclosure: extracted PDF text is sent to Google Gemini for "
        "summaries, embeddings, answers, and follow-up generation. A local "
        "cross-encoder reranks retrieved chunks."
    )
    data_consent = st.checkbox(
        "I confirm these PDFs contain no confidential, regulated, or personal data."
    )

    if st.button(
        "Build collection",
        type="primary",
        use_container_width=True,
        disabled=(
            not pdf_files
            or not api_key
            or not data_consent
            or too_many_files
            or bool(oversized_files)
        ),
    ):
        with st.spinner("Reading, chunking, embedding, summarizing…"):
            try:
                engine = CitationDesk(api_key=api_key)
                engine.ingest(
                    (f.name, io.BytesIO(f.getvalue())) for f in pdf_files
                )
                st.session_state.engine = engine
                st.session_state.ingested_files = [f.name for f in pdf_files]
                st.session_state.messages = []
                st.success(f"Collection ready: {len(pdf_files)} document(s) indexed.")
            except Exception as e:
                st.error(f"Failed: {e}")

    if st.session_state.engine:
        st.divider()
        st.markdown("**📚 Document summaries**")
        for name, summary in st.session_state.engine.summaries.items():
            with st.expander(name, expanded=False):
                st.write(summary)

        st.markdown("**📁 In this collection**")
        for file_name in st.session_state.ingested_files:
            st.caption(f"• {file_name}")

        st.markdown("**🔬 Collection diagnostics**")
        total_pages = sum(
            stats["pages"] for stats in eng.document_stats.values()
        )
        total_chars = sum(
            stats["characters"] for stats in eng.document_stats.values()
        )
        diag_cols = st.columns(2)
        diag_cols[0].metric("Readable pages", total_pages)
        diag_cols[1].metric("Extracted chars", f"{total_chars:,}")

        st.divider()
        eng = st.session_state.engine
        st.markdown("**📊 Session stats**")
        cols = st.columns(2)
        cols[0].metric("Tokens in", f"{eng.total_tokens_in:,}")
        cols[1].metric("Tokens out", f"{eng.total_tokens_out:,}")

        st.divider()
        st.download_button(
            "⬇️ Export chat (Markdown)",
            data=_markdown_export(),
            file_name="citation_desk_notes.md",
            mime="text/markdown",
            use_container_width=True,
            disabled=not st.session_state.messages,
        )
        st.download_button(
            "⬇️ Export research brief",
            data=_research_brief(),
            file_name="citation_desk_research_brief.md",
            mime="text/markdown",
            use_container_width=True,
        )
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.markdown(
        """
        <div style='margin-top: 2rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.12); font-size: 0.76rem; color: #94a3b8; text-align: center;'>
            Citation Desk · Built by <a href='https://github.com/anand15-og' target='_blank' style='color: #38bdf8; text-decoration: none; font-weight: 600;'>@anand15-og</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

st.markdown("<p class='research-kicker'>Your reading workspace</p>", unsafe_allow_html=True)
st.markdown(f"## {APP_NAME}")
st.caption(
    f"{st.session_state.collection_name} · Ask grounded, cited questions about your PDFs. "
    "Hybrid retrieval (semantic + BM25) → cross-encoder reranking → Gemini."
)

if st.session_state.research_objective:
    st.info(f"**Research objective:** {st.session_state.research_objective}")

if st.session_state.engine is None:
    st.info(
        "👈 Add PDFs in the sidebar, name the collection, and click **Build collection** to begin."
    )
    st.stop()

# Quick exploration starters for newly built collections
if not st.session_state.messages:
    st.markdown("##### 💡 Suggested Questions")
    q_cols = st.columns(3)
    starters = [
        ("📌 Key Findings", "What are the core conclusions and key takeaways?"),
        ("🔬 Methodology", "What methodology, models, or datasets are used?"),
        ("⚖️ Limitations", "What limitations, assumptions, or future work are discussed?"),
    ]
    for col, (label, query) in zip(q_cols, starters):
        if col.button(label, use_container_width=True, help=query):
            st.session_state.pending_question = query
            st.rerun()

# Render past messages
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            ans: Answer | None = msg.get("answer")
            if ans:
                _render_answer(ans, key_prefix=f"past_{i}")
            else:
                st.markdown(msg["content"])

st.markdown("### Working notes")
st.session_state.research_notes = st.text_area(
    "Capture ideas, evidence, and next steps while you read.",
    value=st.session_state.research_notes,
    height=140,
    label_visibility="collapsed",
    placeholder="Add your own observations here. They will be included in the research-brief export.",
)

# Handle queued follow-up click OR new chat input
prompt = st.session_state.pending_question
st.session_state.pending_question = None
if not prompt:
    prompt = st.chat_input("Ask a question about this collection…")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving, reranking, generating…"):
            try:
                answer = st.session_state.engine.ask(
                    prompt, history=_history_text(limit=6)
                )
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        _render_answer(answer, key_prefix=f"new_{len(st.session_state.messages)}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer.text, "answer": answer}
    )
