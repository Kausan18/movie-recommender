"""
app.py — Phase 7 Streamlit entry point.

Run from the project root:
    streamlit run phase7_frontend/app.py

Both servers must be running simultaneously:
  Terminal 1: uvicorn phase6_api.main:app --reload --port 8000
  Terminal 2: streamlit run phase7_frontend/app.py
"""

import sys
import os

# Ensure project root is on the path so cross-phase imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Minimal global style tweaks ────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* Tighten the default top padding */
        .block-container { padding-top: 1.5rem; }
        /* Subtler tab underline */
        .stTabs [data-baseweb="tab-highlight"] { background-color: #1D9E75; }
        .stTabs [data-baseweb="tab"]:focus { box-shadow: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

from phase7_frontend.pages import home, dashboard

tab1, tab2 = st.tabs(["🎬  Recommendations", "📊  Dashboard"])

with tab1:
    home.render()

with tab2:
    dashboard.render()