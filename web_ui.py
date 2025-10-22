#!/usr/bin/env python3
"""
Clinical Evidence Navigator - Web UI
Built with Streamlit for easy web access
"""

import os
import json
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from clinical_evidence import ClinicalEvidenceNavigator

# --- Page config ---
st.set_page_config(
    page_title="Clinical Evidence Navigator",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Styles ---
st.markdown(
    """
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.search-box {
    padding: 1rem;
    border-radius: 10px;
    border: 2px solid #1f77b4;
    background-color: #f0f8ff;
}
.result-card {
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 5px solid #1f77b4;
    background-color: #ffffff;
    margin: 1rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.evidence-grade-A { border-left-color: #28a745; }
.evidence-grade-B { border-left-color: #ffc107; }
.evidence-grade-C { border-left-color: #fd7e14; }
.evidence-grade-D { border-left-color: #dc3545; }
</style>
""",
    unsafe_allow_html=True,
)

# --- Cache the navigator (one per process) ---
@st.cache_resource(show_spinner=False)
def get_navigator() -> ClinicalEvidenceNavigator:
    return ClinicalEvidenceNavigator()

# --- Session state ---
st.session_state.setdefault("navigator_ready", False)
st.session_state.setdefault("search_results", None)
st.session_state.setdefault("search_history", [])
st.session_state.setdefault("current_query", "")

# --- Helpers ---
def ensure_navigator() -> ClinicalEvidenceNavigator | None:
    try:
        nav = get_navigator()
        st.session_state.navigator_ready = True
        return nav
    except Exception as e:
        st.session_state.navigator_ready = False
        st.error(f"❌ Failed to initialize navigator: {e}")
        return None

def perform_search(query: str):
    nav = ensure_navigator()
    if not nav:
        return None
    with st.spinner(f"🔍 Searching for evidence on: {query}"):
        results = nav.query_evidence(query)
        st.session_state.search_results = results
        st.session_state.search_history.append(
            {
                "query": query,
                "timestamp": datetime.now(),
                "evidence_count": results.get("evidence_count", 0),
            }
        )
        return results

def display_results(results: dict):
    if not results:
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Query", results.get("query", ""))
    with col2:
        st.metric("Evidence Count", results.get("evidence_count", 0))
    with col3:
        ts = results.get("timestamp", "")
        st.metric("Search Time", ts.split("T")[1][:8] if "T" in ts else ts)

    st.markdown("## 📄 Clinical Evidence Report")
    results_text = results.get("results", "")

    with st.expander("📊 Executive Summary", expanded=True):
        head = results_text[:1000]
        st.markdown(head + ("..." if len(results_text) > len(head) else ""))

    with st.expander("🔬 Detailed Analysis"):
        # Use code block for readability regardless of formatting style returned
        st.code(results_text, language="markdown")

    with st.expander("📈 Evidence Quality Assessment (illustrative)"):
        # Example chart (replace with real grading if you compute it)
        df = pd.DataFrame(
            {
                "Evidence Level": ["High", "Moderate", "Low", "Very Low"],
                "Study Count": [3, 5, 8, 2],
                "Grade": ["A", "B", "C", "D"],
            }
        )
        fig = px.bar(
            df,
            x="Evidence Level",
            y="Study Count",
            color="Grade",
            title="Evidence Quality Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

def display_search_history():
    if not st.session_state["search_history"]:
        return
    st.sidebar.markdown("## 🕒 Recent Searches")
    for i, search in enumerate(reversed(st.session_state["search_history"][-5:])):
        with st.sidebar.expander(f"🔍 {search['query'][:30]}...", expanded=False):
            st.write(f"**Time:** {search['timestamp'].strftime('%H:%M:%S')}")
            st.write(f"**Evidence:** {search['evidence_count']} pieces")
            if st.button("Repeat Search", key=f"repeat_{i}"):
                st.session_state["current_query"] = search["query"]
                st.experimental_rerun()

def display_database_stats():
    nav = get_navigator() if st.session_state.navigator_ready else None
    if not nav:
        return
    try:
        cursor = nav.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM studies")
        studies_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM evidence_gaps")
        queries_count = cursor.fetchone()[0]
        st.sidebar.markdown("## 📊 Database Stats")
        st.sidebar.metric("Total Studies", studies_count)
        st.sidebar.metric("Total Queries", queries_count)
    except Exception as e:
        st.sidebar.error(f"Database stats unavailable: {e}")

def export_results():
    results = st.session_state.get("search_results")
    if not results:
        return
    st.markdown("## 📥 Export Results")
    c1, c2 = st.columns(2)

    with c1:
        st.download_button(
            label="📄 Download JSON",
            data=json.dumps(results, indent=2),
            file_name=f"evidence_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )

    with c2:
        report_text = f"""CLINICAL EVIDENCE REPORT
========================
Query: {results.get('query','')}
Timestamp: {results.get('timestamp','')}
Evidence Count: {results.get('evidence_count',0)}

RESULTS:
{results.get('results','')}
"""
        st.download_button(
            label="📝 Download Report",
            data=report_text,
            file_name=f"clinical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
        )

# --- UI ---
st.markdown('<h1 class="main-header">🏥 Clinical Evidence Navigator</h1>', unsafe_allow_html=True)
st.markdown("### AI-Powered Clinical Research Analysis & Evidence Synthesis")

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ System Status")
    if st.button("🚀 Initialize System", type="primary"):
        ensure_navigator()
    if st.session_state.navigator_ready:
        nav = get_navigator()
        st.success("✅ Navigator: Ready")
        st.success(f"✅ Model: {nav.model_name or '— not detected —'}")
        st.write("**Data dir:**", nav.data_dir)
        st.write("**DB path:**", os.path.join(nav.data_dir, "clinical_evidence.db"))
        st.write("**Chroma collection:** clinical_evidence")
    else:
        st.warning("⚠️ Navigator: Not initialized")
    display_database_stats()
    display_search_history()

# If not initialized, encourage init but still allow search (auto-init on run)
if not st.session_state.navigator_ready:
    st.info("👆 Click **Initialize System** (or just run a search — we’ll auto-init).")

# Search interface
st.markdown("## 🔍 Clinical Evidence Search")
with st.form("search_form"):
    c1, c2 = st.columns([4, 1])
    with c1:
        query = st.text_input(
            "Enter your clinical query:",
            value=st.session_state.get("current_query", ""),
            placeholder="e.g., diabetes treatment metformin elderly patients",
            help="Be specific for better results. Include condition, intervention, population.",
        )
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_button = st.form_submit_button("🔍 Search Evidence", type="primary")

# Example queries
st.markdown("**💡 Example Queries:**")
e1, e2, e3 = st.columns(3)
if e1.button("🫀 Hypertension ACE Inhibitors"):
    st.session_state["current_query"] = "hypertension ACE inhibitors elderly patients"
    perform_search(st.session_state["current_query"])
    st.success(f"✅ Search completed for: {st.session_state['current_query']}")
if e2.button("🧠 Depression CBT vs Medication"):
    st.session_state["current_query"] = "depression cognitive behavioral therapy vs medication"
    perform_search(st.session_state["current_query"])
    st.success(f"✅ Search completed for: {st.session_state['current_query']}")
if e3.button("💉 COVID-19 Vaccine Effectiveness"):
    st.session_state["current_query"] = "COVID-19 vaccine effectiveness immunocompromised"
    perform_search(st.session_state["current_query"])
    st.success(f"✅ Search completed for: {st.session_state['current_query']}")

# Run search from form
if search_button and query:
    st.session_state["current_query"] = query
    out = perform_search(query)
    if out:
        st.success(f"✅ Search completed for: {query}")

# Results + export
if st.session_state["search_results"]:
    st.markdown("---")
    display_results(st.session_state["search_results"])
    export_results()

# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #666; margin-top: 2rem;'>
  <p>🏥 <strong>Clinical Evidence Navigator</strong> | AI-Powered Evidence-Based Medicine</p>
  <p>Built with CrewAI, Ollama, ChromaDB | 100% Free & Open Source</p>
</div>
""",
    unsafe_allow_html=True,
)
