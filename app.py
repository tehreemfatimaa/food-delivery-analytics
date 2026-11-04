import os
import textwrap
import streamlit as st
import pandas as pd
import numpy as np

# Optional Plotly import for interactive charts
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    import matplotlib.pyplot as plt

# Safe check for statsmodels (required by Plotly for OLS trendlines)
try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# Import project modules
from src.data_loader import load_raw_data, inspect_raw_data
from src.data_cleaning import clean_dataset, clean_dataset_with_report
from src.analysis import (
    calculate_basic_metrics,
    answer_q1_traffic_impact,
    answer_q2_distance_impact,
    answer_q3_combined_conditions,
    generate_business_insights,
)
from src.ai_explanation import generate_ai_explanation, answer_user_question_ai
from src.visualization import plot_traffic_impact, plot_distance_vs_time, plot_weather_impact


def render_html(html_code):
    """
    Renders pure HTML cleanly without allowing Markdown parsers to convert
    indented lines into raw <pre><code> code blocks.
    """
    clean_html = textwrap.dedent(html_code).strip()
    if hasattr(st, "html"):
        st.html(clean_html)
    else:
        st.markdown(clean_html, unsafe_allow_html=True)


# ==============================================================================
# 1. STREAMLIT PAGE CONFIGURATION & LUXURY EXECUTIVE CSS STYLING
# ==============================================================================
st.set_page_config(
    page_title="SwiftPulse | Food Delivery Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Styling: Glassmorphism, Google Fonts, Vibrant Gradients, Micro-animations
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* =========================================================================
       MATCH TOP HEADER & CONTROLS TO THE DARK THEME
       ========================================================================= */
    /* 1. Top rainbow line styled with theme accent gradient */
    [data-testid="stDecoration"] {
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc, #f43f5e) !important;
        height: 3px !important;
    }

    /* 2. Header container turned seamless dark */
    header,
    header[data-testid="stHeader"],
    .stAppHeader,
    [data-testid="stHeader"] {
        background-color: #0b0f19 !important;
        background: #0b0f19 !important;
        color: #f8fafc !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    /* 3. Toolbar & action buttons in header */
    [data-testid="stToolbar"],
    [data-testid="stToolbar"] *,
    header[data-testid="stHeader"] button,
    header[data-testid="stHeader"] a,
    header[data-testid="stHeader"] svg,
    header[data-testid="stHeader"] span {
        color: #94a3b8 !important;
        fill: #94a3b8 !important;
        background: transparent !important;
    }

    /* 4. Status widget ("Source file changed", "Rerun", "Always rerun") */
    [data-testid="stStatusWidget"],
    .stStatusWidget,
    div[data-testid="stStatusWidget"] {
        background-color: #1e293b !important;
        background: #1e293b !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 10px !important;
    }

    div[data-testid="stStatusWidget"] * {
        color: #cbd5e1 !important;
    }

    div[data-testid="stStatusWidget"] button {
        background: rgba(99, 102, 241, 0.25) !important;
        color: #c7d2fe !important;
        border: 1px solid rgba(129, 140, 248, 0.4) !important;
        border-radius: 8px !important;
    }

    div[data-testid="stStatusWidget"] button:hover {
        background: rgba(99, 102, 241, 0.5) !important;
        color: #ffffff !important;
    }

    /* 5. Overall App Background Glow */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background: radial-gradient(circle at 15% 10%, rgba(99, 102, 241, 0.08) 0%, transparent 40%),
                    radial-gradient(circle at 85% 90%, rgba(6, 182, 212, 0.08) 0%, transparent 40%),
                    #0b0f19 !important;
        background-color: #0b0f19 !important;
        color: #f8fafc;
    }

    /* Top Brand Hero Banner */
    .hero-container {
        position: relative;
        padding: 24px 30px;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        backdrop-filter: blur(20px);
        box-shadow: 0 16px 36px -10px rgba(0, 0, 0, 0.5), inset 0 1px 1px rgba(255, 255, 255, 0.1);
        margin-bottom: 24px;
        overflow: hidden;
    }
    .hero-container::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc, #f43f5e);
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(129, 140, 248, 0.3);
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        color: #a5b4fc;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 10px #10b981;
        animation: pulseAnimation 2s infinite;
    }
    @keyframes pulseAnimation {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1.1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.8px;
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.96rem;
        font-weight: 400;
        margin-top: 6px;
    }

    /* KPI Cards Styling */
    .kpi-card {
        position: relative;
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.75) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 16px;
        backdrop-filter: blur(14px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        overflow: hidden;
        min-height: 125px;
    }
    .kpi-card:hover {
        transform: translateY(-4px);
        border-color: rgba(129, 140, 248, 0.4);
        box-shadow: 0 12px 24px -6px rgba(0, 0, 0, 0.5), 0 0 16px rgba(99, 102, 241, 0.2);
    }
    .kpi-top-bar {
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
    }
    .kpi-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
    }
    .kpi-label {
        font-size: 0.73rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #94a3b8;
    }
    .kpi-icon-box {
        width: 30px;
        height: 30px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
    }
    .kpi-value {
        font-size: 1.55rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.5px;
        line-height: 1.1;
        margin-bottom: 6px;
    }
    .kpi-pill {
        display: inline-flex;
        align-items: center;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.2px;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 23, 42, 0.6);
        padding: 8px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        white-space: nowrap;
        background: transparent;
        border-radius: 10px;
        color: #94a3b8;
        font-size: 0.88rem;
        font-weight: 600;
        padding: 0 16px;
        border: none;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.25) 0%, rgba(129, 140, 248, 0.15) 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(129, 140, 248, 0.5) !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25);
    }

    /* Card Containers */
    .executive-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.65) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        backdrop-filter: blur(14px);
        margin-bottom: 18px;
        box-shadow: 0 8px 24px -6px rgba(0, 0, 0, 0.35);
    }
    .card-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f8fafc;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 12px;
    }

    /* Highlight Banner Cards */
    .highlight-card {
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 16px;
        position: relative;
        overflow: hidden;
    }
    .highlight-card.cyan {
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.1) 0%, rgba(15, 23, 42, 0.8) 100%);
        border-left: 4px solid #06b6d4;
    }
    .highlight-card.amber {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(15, 23, 42, 0.8) 100%);
        border-left: 4px solid #f59e0b;
    }
    .highlight-card.rose {
        background: linear-gradient(135deg, rgba(244, 63, 94, 0.1) 0%, rgba(15, 23, 42, 0.8) 100%);
        border-left: 4px solid #f43f5e;
    }
    .highlight-card.indigo {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(15, 23, 42, 0.8) 100%);
        border-left: 4px solid #6366f1;
    }

    /* Sidebar Customization */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b0f19 0%, #111827 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    .sidebar-brand-box {
        text-align: center;
        padding: 10px 0 14px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 16px;
    }
    .sidebar-brand-title {
        font-size: 1.25rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 6px;
    }
    .sidebar-stat-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 14px;
        font-size: 0.82rem;
    }

    /* Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: #ffffff;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 10px;
        padding: 10px 20px;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35);
        transition: all 0.25s ease;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #818cf8 0%, #6366f1 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5);
        border-color: rgba(255, 255, 255, 0.3);
    }

    /* Dataframe Polish */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# Helper function for unified Plotly dark theme
def apply_dark_plotly_theme(fig, title="", height=400):
    fig.update_layout(
        title=f"<b>{title}</b>" if title else None,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0", family="Plus Jakarta Sans, sans-serif", size=12),
        margin=dict(l=40, r=40, t=60 if title else 30, b=40),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.06)",
            zerolinecolor="rgba(255, 255, 255, 0.1)",
            linecolor="rgba(255, 255, 255, 0.1)",
            tickfont=dict(color="#94a3b8"),
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.06)",
            zerolinecolor="rgba(255, 255, 255, 0.1)",
            linecolor="rgba(255, 255, 255, 0.1)",
            tickfont=dict(color="#94a3b8"),
        ),
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.7)",
            bordercolor="rgba(255, 255, 255, 0.1)",
            borderwidth=1,
            font=dict(color="#cbd5e1"),
        ),
    )
    return fig


# Cache data loading & cleaning for lightning-fast responsiveness
@st.cache_data
def get_dataset():
    raw_df, dataset_path = load_raw_data()
    raw_info = inspect_raw_data(raw_df)
    cleaned_df, audit_report = clean_dataset_with_report(raw_df)
    return raw_df, cleaned_df, dataset_path, raw_info, audit_report


try:
    raw_df, cleaned_df, dataset_path, raw_info, audit_report = get_dataset()
    data_loaded = True
except Exception as e:
    data_loaded = False
    st.error(f"⚠️ Error loading dataset: {e}. Please ensure CSV dataset is placed in the 'data/' directory.")

if data_loaded:
    # ==============================================================================
    # 2. SIDEBAR NAVIGATION & DYNAMIC FILTERS
    # ==============================================================================
    with st.sidebar:
        # Brand Logo & Title Box
        if os.path.exists("assets/logo.jpg"):
            st.image("assets/logo.jpg", use_container_width=True)

        render_html(
            """
            <div class="sidebar-brand-box">
                <div class="sidebar-brand-title">SWIFTPULSE ANALYTICS</div>
                <div style="font-size: 0.78rem; color: #94a3b8;">Fleet Intelligence & Delivery Optimization</div>
            </div>
            """
        )

        # Dataset Status Capsule
        render_html(
            f"""
            <div class="sidebar-stat-card">
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="color: #64748b; font-weight: 600;">SOURCE FILE</span>
                    <span style="color: #38bdf8; font-weight: 700;">{os.path.basename(dataset_path)}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="color: #64748b; font-weight: 600;">RAW RECORDS</span>
                    <span style="color: #f8fafc; font-weight: 700;">{raw_info['num_rows']:,}</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #64748b; font-weight: 600;">CLEANED ACTIVE</span>
                    <span style="color: #10b981; font-weight: 700;">{len(cleaned_df):,}</span>
                </div>
            </div>
            """
        )

        st.markdown("**🎯 Operational Filters**")

        # 1. City Filter
        city_options = ["All Cities"]
        if "City" in cleaned_df.columns:
            city_options += sorted([str(x) for x in cleaned_df["City"].dropna().unique() if str(x) != "nan"])
        selected_city = st.selectbox("City Region", options=city_options)

        # 2. Traffic Density Filter
        traffic_options = ["All Densities"]
        if "Road_traffic_density" in cleaned_df.columns:
            traffic_options += sorted([str(x) for x in cleaned_df["Road_traffic_density"].dropna().unique() if str(x) != "nan"])
        selected_traffic = st.selectbox("Road Traffic Density", options=traffic_options)

        # 3. Weather Condition Filter
        weather_options = ["All Weather"]
        if "Weather_conditions" in cleaned_df.columns:
            weather_options += sorted([str(x) for x in cleaned_df["Weather_conditions"].dropna().unique() if str(x) != "nan"])
        selected_weather = st.selectbox("Weather Condition", options=weather_options)

        # 4. Vehicle Condition Filter
        vehicle_options = ["All Conditions"]
        if "Vehicle_condition" in cleaned_df.columns:
            vehicle_options += sorted([str(x) for x in cleaned_df["Vehicle_condition"].dropna().unique() if str(x) != "nan"])
        selected_vehicle = st.selectbox("Vehicle Rating / Condition", options=vehicle_options)

        # Reset Filters
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Reset All Filters", use_container_width=True):
            if hasattr(st, "rerun"):
                st.rerun()
            elif hasattr(st, "experimental_rerun"):
                st.experimental_rerun()

        st.markdown("---")
        st.markdown(
            """
            <div style="font-size: 0.74rem; color: #64748b; text-align: center;">
                Powered by Python, Streamlit, Plotly & LLM Intelligence.<br>Built for Data Hackathons.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Filter execution
    filtered_df = cleaned_df.copy()
    if selected_city != "All Cities" and "City" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["City"].astype(str) == selected_city]
    if selected_traffic != "All Densities" and "Road_traffic_density" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Road_traffic_density"].astype(str) == selected_traffic]
    if selected_weather != "All Weather" and "Weather_conditions" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Weather_conditions"].astype(str) == selected_weather]
    if selected_vehicle != "All Conditions" and "Vehicle_condition" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Vehicle_condition"].astype(str) == selected_vehicle]

    is_empty = len(filtered_df) == 0
    basic_metrics = calculate_basic_metrics(filtered_df)

    def fmt(value, suffix=""):
        if value is None or value == "N/A" or (isinstance(value, float) and (np.isnan(value))):
            return "N/A"
        return f"{value}{suffix}"

    # ==============================================================================
    # 3. EXECUTIVE HERO BANNER & KPI METRICS
    # ==============================================================================
    render_html(
        f"""
        <div class="hero-container">
            <div class="hero-badge">
                <div class="pulse-dot"></div>
                SWIFTPULSE FLEET ANALYTICS ENGINE &bull; {len(filtered_df):,} ACTIVE TRIPS
            </div>
            <div class="hero-title">Food Delivery Operational Performance</div>
            <div class="hero-subtitle">
                High-resolution telemetry analysis uncovering delivery duration bottlenecks, Haversine spatial correlation, and compounding weather cascades.
            </div>
        </div>
        """
    )

    if is_empty:
        st.warning("⚠️ No delivery records match the selected filters. Please adjust or reset your filter combination.")

    # High-End Custom KPI Grid using st.columns to prevent Markdown indentation issues
    tot_val = "0" if is_empty else f"{basic_metrics['total_deliveries']:,}"
    time_val = fmt(basic_metrics["avg_delivery_time"], " m")
    dist_val = fmt(basic_metrics["avg_distance"], " km")
    spd_val = fmt(basic_metrics["avg_speed_kmh"], " km/h")
    rat_val = fmt(basic_metrics["avg_rating"], " ★")
    age_val = fmt(basic_metrics["avg_age"], " yrs")

    k_col1, k_col2, k_col3, k_col4, k_col5, k_col6 = st.columns(6)

    with k_col1:
        render_html(f"""
        <div class="kpi-card">
            <div class="kpi-top-bar" style="background: linear-gradient(90deg, #38bdf8, #0284c7);"></div>
            <div class="kpi-header">
                <span class="kpi-label">TOTAL DELIVERIES</span>
                <div class="kpi-icon-box" style="background: rgba(56, 189, 248, 0.15); color: #38bdf8;">📦</div>
            </div>
            <div class="kpi-value">{tot_val}</div>
            <span class="kpi-pill" style="background: rgba(56, 189, 248, 0.12); color: #38bdf8;">Verified Records</span>
        </div>
        """)

    with k_col2:
        render_html(f"""
        <div class="kpi-card">
            <div class="kpi-top-bar" style="background: linear-gradient(90deg, #f59e0b, #d97706);"></div>
            <div class="kpi-header">
                <span class="kpi-label">AVG DELIVERY TIME</span>
                <div class="kpi-icon-box" style="background: rgba(245, 158, 11, 0.15); color: #f59e0b;">⏱️</div>
            </div>
            <div class="kpi-value">{time_val}</div>
            <span class="kpi-pill" style="background: rgba(245, 158, 11, 0.12); color: #f59e0b;">Mean Duration</span>
        </div>
        """)

    with k_col3:
        render_html(f"""
        <div class="kpi-card">
            <div class="kpi-top-bar" style="background: linear-gradient(90deg, #10b981, #059669);"></div>
            <div class="kpi-header">
                <span class="kpi-label">AVG DISTANCE</span>
                <div class="kpi-icon-box" style="background: rgba(16, 185, 129, 0.15); color: #10b981;">📍</div>
            </div>
            <div class="kpi-value">{dist_val}</div>
            <span class="kpi-pill" style="background: rgba(16, 185, 129, 0.12); color: #10b981;">Haversine Math</span>
        </div>
        """)

    with k_col4:
        render_html(f"""
        <div class="kpi-card">
            <div class="kpi-top-bar" style="background: linear-gradient(90deg, #818cf8, #4f46e5);"></div>
            <div class="kpi-header">
                <span class="kpi-label">AVG FLEET SPEED</span>
                <div class="kpi-icon-box" style="background: rgba(129, 140, 248, 0.15); color: #818cf8;">⚡</div>
            </div>
            <div class="kpi-value">{spd_val}</div>
            <span class="kpi-pill" style="background: rgba(129, 140, 248, 0.12); color: #818cf8;">Transit Velocity</span>
        </div>
        """)

    with k_col5:
        render_html(f"""
        <div class="kpi-card">
            <div class="kpi-top-bar" style="background: linear-gradient(90deg, #f43f5e, #e11d48);"></div>
            <div class="kpi-header">
                <span class="kpi-label">PARTNER RATING</span>
                <div class="kpi-icon-box" style="background: rgba(244, 63, 94, 0.15); color: #f43f5e;">⭐</div>
            </div>
            <div class="kpi-value">{rat_val}</div>
            <span class="kpi-pill" style="background: rgba(244, 63, 94, 0.12); color: #f43f5e;">Out of 5.0</span>
        </div>
        """)

    with k_col6:
        render_html(f"""
        <div class="kpi-card">
            <div class="kpi-top-bar" style="background: linear-gradient(90deg, #c084fc, #9333ea);"></div>
            <div class="kpi-header">
                <span class="kpi-label">PARTNER AGE</span>
                <div class="kpi-icon-box" style="background: rgba(192, 132, 252, 0.15); color: #c084fc;">👤</div>
            </div>
            <div class="kpi-value">{age_val}</div>
            <span class="kpi-pill" style="background: rgba(192, 132, 252, 0.12); color: #c084fc;">Fleet Cohort</span>
        </div>
        """)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # ==============================================================================
    # 4. LUXURY TABS NAVIGATION
    # ==============================================================================
    tab_overview, tab_cleaning, tab_comp, tab_viz, tab_insights, tab_ai = st.tabs([
        "📊 Telemetry Overview",
        "🧹 Pipeline & Data Audit",
        "🏆 Competition Questions",
        "📈 Interactive Analytics",
        "💡 Business Insights",
        "🤖 AI Intelligence & Chat",
    ])

    # ------------------------------------------------------------------------------
    # TAB 1: TELEMETRY OVERVIEW
    # ------------------------------------------------------------------------------
    with tab_overview:
        c_left, c_right = st.columns([1, 1])

        with c_left:
            render_html(
                """
                <div class="executive-card">
                    <div class="card-title">📋 Executive Dataset Summary</div>
                </div>
                """
            )
            if is_empty:
                st.info("No data available for the selected filters.")
            else:
                summary_table = pd.DataFrame({
                    "Operational Parameter": [
                        "Total Delivery Trips",
                        "Shortest Trip Duration",
                        "Longest Trip Duration",
                        "Average Delivery Duration",
                        "Average Travel Distance",
                        "Estimated Transit Velocity",
                        "Mean Rider Customer Rating",
                        "Average Delivery Rider Age"
                    ],
                    "Observed Metric": [
                        f"{basic_metrics['total_deliveries']:,}",
                        fmt(basic_metrics["min_delivery_time"], " min"),
                        fmt(basic_metrics["max_delivery_time"], " min"),
                        fmt(basic_metrics["avg_delivery_time"], " min"),
                        fmt(basic_metrics["avg_distance"], " km"),
                        fmt(basic_metrics["avg_speed_kmh"], " km/h"),
                        fmt(basic_metrics["avg_rating"], " / 5.0"),
                        fmt(basic_metrics["avg_age"], " years"),
                    ]
                })
                st.dataframe(summary_table, use_container_width=True, hide_index=True)

        with c_right:
            render_html(
                """
                <div class="executive-card">
                    <div class="card-title">📈 Statistical Spread & Distribution</div>
                </div>
                """
            )
            if is_empty:
                st.info("No data available for the selected filters.")
            elif "Time_taken (min)" in filtered_df.columns and "distance_km" in filtered_df.columns:
                desc = filtered_df[["Time_taken (min)", "distance_km"]].describe().T
                desc.rename(columns={
                    "count": "Trips", "mean": "Mean", "std": "Std Dev",
                    "min": "Min", "25%": "25th %", "50%": "Median",
                    "75%": "75th %", "max": "Max"
                }, inplace=True)
                st.dataframe(desc.round(2), use_container_width=True)

    # ------------------------------------------------------------------------------
    # TAB 2: PIPELINE & DATA CLEANING AUDIT
    # ------------------------------------------------------------------------------
    with tab_cleaning:
        render_html(
            """
            <div class="highlight-card cyan">
                <h4 style="margin:0 0 6px 0; color: #38bdf8;">🛡️ Programmatic Data Integrity & Transformation Audit</h4>
                <div style="font-size: 0.88rem; color: #cbd5e1;">
                    Every cleaning step is calculated programmatically without lossy heuristic imputations. 
                    Valid missing entries in Rider Age and Rating are preserved as strictly statistical <code>NaN</code>s 
                    to avoid biasing regression models or operational KPIs.
                </div>
            </div>
            """
        )

        # 4 Audit Summary Cards
        ac1, ac2, ac3, ac4 = st.columns(4)
        with ac1:
            render_html(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">RAW INGESTION</div>
                    <div class="kpi-value" style="font-size: 1.35rem; color: #38bdf8;">{audit_report['raw_rows']:,} rows</div>
                    <div style="font-size: 0.76rem; color: #94a3b8;">{audit_report['raw_cols']} original columns</div>
                </div>
                """
            )
        with ac2:
            render_html(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">CLEANED VALID POOL</div>
                    <div class="kpi-value" style="font-size: 1.35rem; color: #10b981;">{audit_report['cleaned_rows']:,} rows</div>
                    <div style="font-size: 0.76rem; color: #94a3b8;">{audit_report['cleaned_cols']} standardized schema</div>
                </div>
                """
            )
        with ac3:
            render_html(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">DUPLICATES PURGED</div>
                    <div class="kpi-value" style="font-size: 1.35rem; color: #f59e0b;">{audit_report['duplicates_removed']}</div>
                    <div style="font-size: 0.76rem; color: #94a3b8;">Exact row matches dropped</div>
                </div>
                """
            )
        with ac4:
            render_html(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">MISSING CELLS HANDLED</div>
                    <div class="kpi-value" style="font-size: 1.35rem; color: #c084fc;">{audit_report['total_missing_before']:,} → {audit_report['total_missing_after']:,}</div>
                    <div style="font-size: 0.76rem; color: #94a3b8;">Preserved true statistical NaNs</div>
                </div>
                """
            )

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # Before vs After Health Table
        render_html("<div class='card-title'>📋 Before vs. After System Transformation Health</div>")
        health_comparison = pd.DataFrame({
            "Dataset Dimension": [
                "Total Record Rows",
                "Total Feature Columns",
                "Exact Duplicate Rows",
                "Total Missing Values",
                "Delivery Time Column Format",
                "Weather Categorical Format",
                "Delivery Distance",
                "Missing Age / Ratings Strategy"
            ],
            "Before Cleaning (Raw Data)": [
                f"{audit_report['raw_rows']:,}",
                f"{audit_report['raw_cols']} columns",
                f"{audit_report['duplicates_removed']} rows",
                f"{audit_report['total_missing_before']:,} missing cells",
                "String format (e.g., '(min) 24', '24 min')",
                "Prefix noise (e.g., 'conditions Fog', 'conditions Sunny')",
                "Lat/Long coordinates only (no direct km distance)",
                "Contains raw NaN and literal string 'NaN' / 'nan'"
            ],
            "After Cleaning (Standardized)": [
                f"{audit_report['cleaned_rows']:,}",
                f"{audit_report['cleaned_cols']} standardized columns",
                "0 duplicate rows",
                f"{audit_report['total_missing_after']:,} preserved valid NaNs",
                "Pure numeric float (minutes)",
                "Standardized categories ('Fog', 'Sunny', etc.)",
                "Haversine great-circle distance computed in km",
                "Preserved as true np.nan (never converted to artificial 0)"
            ],
            "Action / Transformation Applied": [
                "Filtered non-positive duration values safely",
                "Dynamically canonicalized column names via fuzzy mapping",
                "Dropped duplicate rows with .drop_duplicates()",
                "Replaced string 'nan' with np.nan; extracted numeric digits",
                "Applied regex r'(\\d+\\.?\\d*)' extraction",
                "Stripped whitespace and removed 'conditions ' prefix",
                "Computed distance (km) using Haversine formula from coords",
                "Maintained statistical integrity for unbiased metrics"
            ]
        })
        st.dataframe(health_comparison, use_container_width=True, hide_index=True)

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # Missing Values Breakdown Chart
        missing_before_dict = audit_report["missing_before"]
        missing_after_dict = audit_report["missing_after"]

        missing_chart_data = []
        for col, count in missing_before_dict.items():
            if count > 0 or missing_after_dict.get(col, 0) > 0:
                missing_chart_data.append({"Column": col, "Stage": "Raw Before", "Missing Count": count})
                missing_chart_data.append({"Column": col, "Stage": "Cleaned After", "Missing Count": missing_after_dict.get(col, 0)})

        if missing_chart_data and HAS_PLOTLY:
            missing_df = pd.DataFrame(missing_chart_data)
            fig_miss = px.bar(
                missing_df,
                x="Missing Count",
                y="Column",
                color="Stage",
                barmode="group",
                orientation="h",
                color_discrete_map={"Raw Before": "#f43f5e", "Cleaned After": "#10b981"},
                text="Missing Count"
            )
            fig_miss.update_traces(textposition="outside")
            apply_dark_plotly_theme(fig_miss, title="Missing Value Audit per Feature (Raw vs Cleaned)", height=320)
            st.plotly_chart(fig_miss, use_container_width=True)

    # ------------------------------------------------------------------------------
    # TAB 3: COMPETITION QUESTIONS (Q1, Q2, Q3)
    # ------------------------------------------------------------------------------
    with tab_comp:
        if is_empty:
            st.info("No data available for the selected filters.")
        else:
            q1_res = answer_q1_traffic_impact(filtered_df)
            q2_res = answer_q2_distance_impact(filtered_df)
            q3_res = answer_q3_combined_conditions(filtered_df)

            # Q1 Hero Card
            render_html(
                f"""
                <div class="highlight-card amber">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <h4 style="margin: 0; color: #fbbf24;">🚦 Question 1: Road Traffic Bottleneck</h4>
                        <span class="kpi-pill" style="background: rgba(245, 158, 11, 0.2); color: #fbbf24; font-size: 0.8rem;">
                            CRITICAL DELAY DRIVER
                        </span>
                    </div>
                    <div style="font-size: 0.95rem; color: #e2e8f0; margin-bottom: 12px;">
                        <b>Prompt:</b> Which road traffic condition has the highest average delivery time?<br>
                        <b>Finding:</b> Deliveries during <b>'{q1_res['highest_condition']}'</b> traffic encounter the highest delay, 
                        averaging <b>{q1_res['highest_avg_time']} minutes</b>.
                    </div>
                </div>
                """
            )

            q1_c1, q1_c2 = st.columns([1, 1])
            with q1_c1:
                t_df = pd.DataFrame([
                    {"Traffic Density": k, "Avg Delivery Time (min)": v}
                    for k, v in q1_res["traffic_table"].items()
                ])
                st.dataframe(t_df, use_container_width=True, hide_index=True)
            with q1_c2:
                if HAS_PLOTLY and q1_res["traffic_table"]:
                    fig_q1 = px.bar(
                        t_df,
                        x="Traffic Density",
                        y="Avg Delivery Time (min)",
                        text="Avg Delivery Time (min)",
                        color="Traffic Density",
                        color_discrete_map={"Jam": "#ef4444", "High": "#f97316", "Medium": "#f59e0b", "Low": "#10b981"},
                    )
                    fig_q1.update_traces(texttemplate="%{text:.2f} m", textposition="outside")
                    apply_dark_plotly_theme(fig_q1, title="Delivery Time by Traffic Density", height=280)
                    st.plotly_chart(fig_q1, use_container_width=True)

            st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

            # Q2 Hero Card
            render_html(
                f"""
                <div class="highlight-card cyan">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <h4 style="margin: 0; color: #38bdf8;">📏 Question 2: Spatial Distance vs. Time Correlation</h4>
                        <span class="kpi-pill" style="background: rgba(56, 189, 248, 0.2); color: #38bdf8; font-size: 0.8rem;">
                            PEARSON r = {q2_res['correlation']}
                        </span>
                    </div>
                    <div style="font-size: 0.95rem; color: #e2e8f0; margin-bottom: 12px;">
                        <b>Prompt:</b> How does delivery distance affect delivery time?<br>
                        <b>Finding:</b> Pearson correlation is <b>{q2_res['correlation']}</b> ({q2_res['interpretation']}).
                    </div>
                </div>
                """
            )

            q2_c1, q2_c2 = st.columns([1, 1])
            with q2_c1:
                bin_df = pd.DataFrame([
                    {"Distance Tier": k, "Avg Delivery Time (min)": v}
                    for k, v in q2_res["distance_bins_table"].items()
                ])
                st.dataframe(bin_df, use_container_width=True, hide_index=True)
            with q2_c2:
                if HAS_PLOTLY and q2_res["distance_bins_table"]:
                    fig_q2 = px.bar(
                        bin_df,
                        x="Distance Tier",
                        y="Avg Delivery Time (min)",
                        text="Avg Delivery Time (min)",
                        color="Avg Delivery Time (min)",
                        color_continuous_scale="Viridis",
                    )
                    fig_q2.update_traces(texttemplate="%{text:.2f} m", textposition="outside")
                    apply_dark_plotly_theme(fig_q2, title="Duration Across Distance Tiers", height=280)
                    st.plotly_chart(fig_q2, use_container_width=True)

            st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

            # Q3 Hero Card
            render_html(
                f"""
                <div class="highlight-card rose">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <h4 style="margin: 0; color: #fb7185;">🌧️ Question 3: Compounding Weather + Traffic Bottlenecks</h4>
                        <span class="kpi-pill" style="background: rgba(244, 63, 94, 0.2); color: #fb7185; font-size: 0.8rem;">
                            WORST: {q3_res['highest_avg_time']} MIN
                        </span>
                    </div>
                    <div style="font-size: 0.95rem; color: #e2e8f0; margin-bottom: 12px;">
                        <b>Prompt:</b> Which combination of weather condition and traffic density has the highest average delivery time?<br>
                        <b>Finding:</b> The worst combination is <b>'{q3_res['worst_weather']}' Weather</b> combined with <b>'{q3_res['worst_traffic']}' Traffic</b>, 
                        resulting in an average delivery time of <b>{q3_res['highest_avg_time']} minutes</b>.
                    </div>
                </div>
                """
            )

            comb_df = pd.DataFrame([
                {"Weather + Traffic Combination": k, "Avg Delivery Time (min)": v}
                for k, v in q3_res["combined_table"].items()
            ])
            st.dataframe(comb_df.head(10), use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------------------
    # TAB 4: INTERACTIVE VISUALIZATIONS
    # ------------------------------------------------------------------------------
    with tab_viz:
        if is_empty:
            st.info("No data available to display for the selected filters.")
        else:
            q1_res = answer_q1_traffic_impact(filtered_df)
            q2_res = answer_q2_distance_impact(filtered_df)

            viz_engine = st.radio(
                "Visualization Engine:",
                options=["✨ Plotly Interactive High-DPI Charts", "📊 High-Resolution Matplotlib PNGs"],
                horizontal=True,
                key="viz_engine_toggle"
            )
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

            if "Plotly" in viz_engine and HAS_PLOTLY:
                row1_c1, row1_c2 = st.columns(2)

                # Chart 1: Traffic
                with row1_c1:
                    if q1_res["traffic_table"]:
                        traffic_df = pd.DataFrame({
                            "Traffic Density": list(q1_res["traffic_table"].keys()),
                            "Avg Delivery Time (min)": list(q1_res["traffic_table"].values()),
                        })
                        fig1 = px.bar(
                            traffic_df,
                            x="Traffic Density",
                            y="Avg Delivery Time (min)",
                            text="Avg Delivery Time (min)",
                            color="Traffic Density",
                            color_discrete_map={"Jam": "#ef4444", "High": "#f97316", "Medium": "#f59e0b", "Low": "#10b981"},
                        )
                        fig1.update_traces(texttemplate="%{text:.2f} min", textposition="outside", cliponaxis=False)
                        fig1.update_layout(yaxis=dict(range=[0, max(traffic_df["Avg Delivery Time (min)"]) * 1.25]), showlegend=False)
                        apply_dark_plotly_theme(fig1, title="Average Delivery Time by Traffic Density", height=380)
                        st.plotly_chart(fig1, use_container_width=True)

                # Chart 2: Distance Scatter
                with row1_c2:
                    valid_scatter = filtered_df[(filtered_df["distance_km"] > 0) & (filtered_df["Time_taken (min)"] > 0)].dropna(subset=["distance_km", "Time_taken (min)"])
                    if len(valid_scatter) > 0:
                        sample_scatter = valid_scatter.sample(min(2500, len(valid_scatter)), random_state=42)
                        fig2 = px.scatter(
                            sample_scatter,
                            x="distance_km",
                            y="Time_taken (min)",
                            opacity=0.4,
                            trendline="ols" if (HAS_STATSMODELS and len(sample_scatter) > 10) else None,
                            trendline_color_override="#ef4444",
                            color_discrete_sequence=["#38bdf8"],
                            labels={"distance_km": "Distance (km)", "Time_taken (min)": "Delivery Time (min)"}
                        )
                        apply_dark_plotly_theme(fig2, title="Spatial Distance (km) vs. Delivery Time (min)", height=380)
                        st.plotly_chart(fig2, use_container_width=True)

                st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
                row2_c1, row2_c2 = st.columns(2)

                # Chart 3: Weather
                with row2_c1:
                    if "Weather_conditions" in filtered_df.columns and "Time_taken (min)" in filtered_df.columns:
                        weather_avg = (
                            filtered_df.groupby("Weather_conditions")["Time_taken (min)"]
                            .mean()
                            .round(2)
                            .reset_index()
                            .sort_values(by="Time_taken (min)", ascending=False)
                        )
                        if len(weather_avg) > 0:
                            fig3 = px.bar(
                                weather_avg,
                                x="Weather_conditions",
                                y="Time_taken (min)",
                                text="Time_taken (min)",
                                color="Weather_conditions",
                                color_discrete_sequence=px.colors.qualitative.Pastel,
                            )
                            fig3.update_traces(texttemplate="%{text:.2f} min", textposition="outside", cliponaxis=False)
                            fig3.update_layout(yaxis=dict(range=[0, max(weather_avg["Time_taken (min)"]) * 1.25]), showlegend=False)
                            apply_dark_plotly_theme(fig3, title="Average Delivery Duration by Weather Condition", height=380)
                            st.plotly_chart(fig3, use_container_width=True)

                # Chart 4: Distance Bins
                with row2_c2:
                    if q2_res["distance_bins_table"]:
                        bin_df = pd.DataFrame({
                            "Distance Range": list(q2_res["distance_bins_table"].keys()),
                            "Avg Delivery Time (min)": list(q2_res["distance_bins_table"].values()),
                        })
                        fig4 = px.bar(
                            bin_df,
                            x="Distance Range",
                            y="Avg Delivery Time (min)",
                            text="Avg Delivery Time (min)",
                            color="Distance Range",
                            color_discrete_sequence=px.colors.sequential.Teal,
                        )
                        fig4.update_traces(texttemplate="%{text:.2f} min", textposition="outside", cliponaxis=False)
                        fig4.update_layout(yaxis=dict(range=[0, max(bin_df["Avg Delivery Time (min)"]) * 1.25]), showlegend=False)
                        apply_dark_plotly_theme(fig4, title="Delivery Duration by Distance Tier (km)", height=380)
                        st.plotly_chart(fig4, use_container_width=True)

            else:
                c_img1, c_img2 = st.columns(2)
                with c_img1:
                    img1_path = plot_traffic_impact(q1_res["traffic_table"])
                    if img1_path and os.path.exists(img1_path):
                        st.image(img1_path, use_container_width=True)
                with c_img2:
                    img2_path = plot_distance_vs_time(filtered_df)
                    if img2_path and os.path.exists(img2_path):
                        st.image(img2_path, use_container_width=True)

    # ------------------------------------------------------------------------------
    # TAB 5: BUSINESS INSIGHTS
    # ------------------------------------------------------------------------------
    with tab_insights:
        if is_empty:
            st.info("No data available to formulate business insights.")
        else:
            q1_res = answer_q1_traffic_impact(filtered_df)
            q2_res = answer_q2_distance_impact(filtered_df)
            q3_res = answer_q3_combined_conditions(filtered_df)
            insights = generate_business_insights(basic_metrics, q1_res, q2_res, q3_res)

            for ins in insights:
                num = ins['insight_num']
                card_class = "rose" if num == 1 else ("amber" if num == 2 else "indigo")
                render_html(
                    f"""
                    <div class="highlight-card {card_class}">
                        <div style="font-size: 1.15rem; font-weight: 700; color: #ffffff; margin-bottom: 8px;">
                            Strategic Insight #{num}: {ins['title']}
                        </div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 10px;">
                            <div>
                                <div style="font-size: 0.82rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">FINDING & EVIDENCE</div>
                                <div style="color: #f1f5f9; font-size: 0.92rem; margin-top: 4px;">{ins['finding']}</div>
                                <div style="color: #38bdf8; font-size: 0.85rem; margin-top: 4px; font-family: 'JetBrains Mono', monospace;">Evidence: {ins['evidence']}</div>
                            </div>
                            <div>
                                <div style="font-size: 0.82rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">IMPACT & RECOMMENDED ACTION</div>
                                <div style="color: #f1f5f9; font-size: 0.92rem; margin-top: 4px;">{ins['business_meaning']}</div>
                                <div style="color: #10b981; font-weight: 600; font-size: 0.88rem; margin-top: 6px;">Action: {ins['recommended_action']}</div>
                            </div>
                        </div>
                    </div>
                    """
                )

    # ------------------------------------------------------------------------------
    # TAB 6: AI BUSINESS INTELLIGENCE & CHATBOT
    # ------------------------------------------------------------------------------
    with tab_ai:
        ai_subtab_summary, ai_subtab_chat = st.tabs([
            "📊 Executive Briefing Report",
            "💬 Conversational Operations Assistant"
        ])

        summary_dict = {
            "total_deliveries": basic_metrics["total_deliveries"],
            "avg_delivery_time": basic_metrics["avg_delivery_time"],
            "min_delivery_time": basic_metrics["min_delivery_time"],
            "max_delivery_time": basic_metrics["max_delivery_time"],
            "avg_distance": basic_metrics["avg_distance"],
            "avg_speed_kmh": basic_metrics["avg_speed_kmh"],
            "avg_rating": basic_metrics["avg_rating"],
            "avg_age": basic_metrics["avg_age"],
            "highest_traffic_condition": q1_res["highest_condition"] if not is_empty else "N/A",
            "highest_traffic_avg_time": q1_res["highest_avg_time"] if not is_empty else "N/A",
            "distance_correlation": q2_res["correlation"] if not is_empty else "N/A",
            "worst_weather": q3_res["worst_weather"] if not is_empty else "N/A",
            "worst_traffic": q3_res["worst_traffic"] if not is_empty else "N/A",
            "worst_combined_avg_time": q3_res["highest_avg_time"] if not is_empty else "N/A",
            "selected_city": selected_city,
            "selected_traffic": selected_traffic,
            "selected_weather": selected_weather,
            "selected_vehicle": selected_vehicle,
        }

        # Subtab 1: AI Summary
        with ai_subtab_summary:
            render_html(
                """
                <div class="executive-card">
                    <div class="card-title">🤖 AI-Generated Strategic Briefing</div>
                    <div style="font-size: 0.88rem; color: #94a3b8;">
                        Synthesizes pre-calculated operational metrics into an executive-ready presentation briefing.
                    </div>
                </div>
                """
            )

            if "ai_summary_output" not in st.session_state:
                st.session_state["ai_summary_output"] = None

            if st.button("✨ Synthesize Executive Briefing", key="btn_gen_ai_brief"):
                with st.spinner("Synthesizing telemetry data into executive insights..."):
                    st.session_state["ai_summary_output"] = generate_ai_explanation(summary_dict)

            if st.session_state["ai_summary_output"]:
                # Action header bar with status badge & download button
                top_c1, top_c2 = st.columns([3, 1])
                with top_c1:
                    render_html(
                        """
                        <div style="display: flex; align-items: center; gap: 8px; margin-top: 14px; margin-bottom: 10px;">
                            <span class="hero-badge" style="margin: 0; background: rgba(16, 185, 129, 0.15); border-color: rgba(16, 185, 129, 0.4); color: #34d399;">
                                ✔️ AUDIT VERIFIED EXECUTIVE REPORT
                            </span>
                        </div>
                        """
                    )
                with top_c2:
                    st.download_button(
                        label="📥 Download Briefing (.md)",
                        data=st.session_state["ai_summary_output"],
                        file_name="SwiftPulse_Executive_Briefing.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )

                # Render the executive briefing cleanly as GitHub-flavored Markdown
                st.markdown(st.session_state["ai_summary_output"])
            else:
                st.info("Click 'Synthesize Executive Briefing' to generate a live business analysis report.")

        # Subtab 2: AI Chatbot
        with ai_subtab_chat:
            render_html(
                """
                <div class="executive-card">
                    <div class="card-title">💬 Interactive Operations Copilot</div>
                    <div style="font-size: 0.88rem; color: #94a3b8;">
                        Ask natural language questions regarding fleet velocity, traffic bottlenecks, weather impacts, or dispatch optimizations.
                    </div>
                </div>
                """
            )

            if "chatbot_messages" not in st.session_state:
                st.session_state["chatbot_messages"] = [
                    {
                        "role": "assistant",
                        "content": (
                            "👋 Welcome to the **SwiftPulse Fleet Assistant**!\n\n"
                            "I can analyze operational delays, weather bottlenecks, distance-to-time correlations, and driver performance.\n\n"
                            "**Quick Queries:**\n"
                            "- *Which road traffic causes the worst bottlenecks?*\n"
                            "- *How does distance correlate with transit duration?*\n"
                            "- *What strategic actions reduce delivery times during rainy weather?*"
                        )
                    }
                ]

            # Quick Prompt Buttons
            q_c1, q_c2, q_c3, q_c4 = st.columns(4)
            preset_prompt = None
            with q_c1:
                if st.button("🚦 Traffic Delays", key="btn_fast_traffic", use_container_width=True):
                    preset_prompt = "What road traffic condition causes the worst delivery delays?"
            with q_c2:
                if st.button("🌧️ Weather Cascades", key="btn_fast_weather", use_container_width=True):
                    preset_prompt = "Which weather condition slows down deliveries the most?"
            with q_c3:
                if st.button("📏 Distance Correlation", key="btn_fast_dist", use_container_width=True):
                    preset_prompt = "How does delivery distance correlate with delivery duration and fleet speed?"
            with q_c4:
                if st.button("💡 Strategic Actions", key="btn_fast_strat", use_container_width=True):
                    preset_prompt = "What are the top 3 operational recommendations to optimize fulfillment?"

            # Message Stream
            for msg in st.session_state["chatbot_messages"]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            user_query = st.chat_input("Ask a question about delivery operations...") or preset_prompt

            if user_query:
                st.session_state["chatbot_messages"].append({"role": "user", "content": user_query})
                with st.chat_message("user"):
                    st.markdown(user_query)

                with st.chat_message("assistant"):
                    with st.spinner("Analyzing operational telemetry..."):
                        reply = answer_user_question_ai(
                            user_query=user_query,
                            summary_results=summary_dict,
                            conversation_history=st.session_state["chatbot_messages"]
                        )
                        st.markdown(reply)

                st.session_state["chatbot_messages"].append({"role": "assistant", "content": reply})

else:
    st.error("Failed to load dataset. Please ensure the CSV file is located in the 'data/' directory.")
