import os
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

# ==============================================================================
# 1. STREAMLIT PAGE CONFIGURATION & CUSTOM CSS STYLING
# ==============================================================================
st.set_page_config(
    page_title="Food Delivery Analytics Dashboard",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for executive aesthetic, polished cards, and status pills
st.markdown(
    """
    <style>
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    div[data-testid="stMetric"] label {
        font-weight: 600;
        color: #495057;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #1e293b;
        font-weight: 700;
        font-size: 1.4rem;
    }

    /* Badge & Status Pills */
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-success { background-color: #d1fae5; color: #065f46; }
    .badge-warning { background-color: #fef3c7; color: #92400e; }
    .badge-info { background-color: #dbeafe; color: #1e40af; }

    /* Custom Header Styling */
    .main-header {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 2px;
    }
    .sub-header {
        color: #64748b;
        font-size: 0.95rem;
        margin-bottom: 20px;
    }

    /* Custom Container Boxes */
    .card-box {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Cache data loading & cleaning for fast interaction
@st.cache_data
def get_dataset():
    """
    Load raw dataset, inspect metadata, and execute clean_dataset_with_report.
    """
    raw_df, dataset_path = load_raw_data()
    raw_info = inspect_raw_data(raw_df)
    cleaned_df, audit_report = clean_dataset_with_report(raw_df)
    return raw_df, cleaned_df, dataset_path, raw_info, audit_report


# Load dataset gracefully with error handling
try:
    raw_df, cleaned_df, dataset_path, raw_info, audit_report = get_dataset()
    data_loaded = True
except Exception as e:
    data_loaded = False
    st.error(f"⚠️ Error loading dataset: {e}. Please ensure CSV dataset is placed in the 'data/' directory.")

if data_loaded:
    # ==============================================================================
    # 2. SIDEBAR & DYNAMIC FILTERS
    # ==============================================================================
    st.sidebar.title("🍔 Food Delivery Analytics")
    st.sidebar.markdown(f"**Dataset**: `{os.path.basename(dataset_path)}`")
    st.sidebar.markdown(f"**Raw Rows**: {raw_info['num_rows']:,} | **Cols**: {raw_info['num_cols']}")
    st.sidebar.markdown(f"**Cleaned Records**: {len(cleaned_df):,}")
    st.sidebar.markdown("---")

    st.sidebar.header("🔍 Analysis Filters")

    # City Filter
    city_options = ["All"]
    if "City" in cleaned_df.columns:
        city_options += sorted([str(x) for x in cleaned_df["City"].dropna().unique() if str(x) != "nan"])
    selected_city = st.sidebar.selectbox("Filter by City", options=city_options)

    # Traffic Density Filter
    traffic_options = ["All"]
    if "Road_traffic_density" in cleaned_df.columns:
        traffic_options += sorted([str(x) for x in cleaned_df["Road_traffic_density"].dropna().unique() if str(x) != "nan"])
    selected_traffic = st.sidebar.selectbox("Filter by Traffic Density", options=traffic_options)

    # Weather Condition Filter
    weather_options = ["All"]
    if "Weather_conditions" in cleaned_df.columns:
        weather_options += sorted([str(x) for x in cleaned_df["Weather_conditions"].dropna().unique() if str(x) != "nan"])
    selected_weather = st.sidebar.selectbox("Filter by Weather Condition", options=weather_options)

    # Vehicle Condition Filter
    vehicle_options = ["All"]
    if "Vehicle_condition" in cleaned_df.columns:
        vehicle_options += sorted([str(x) for x in cleaned_df["Vehicle_condition"].dropna().unique() if str(x) != "nan"])
    selected_vehicle = st.sidebar.selectbox("Filter by Vehicle Condition", options=vehicle_options)

    # Reset Filters Button
    if st.sidebar.button("🔄 Reset Filters"):
        if hasattr(st, "rerun"):
            st.rerun()
        elif hasattr(st, "experimental_rerun"):
            st.experimental_rerun()

    # Apply filters dynamically to DataFrame
    filtered_df = cleaned_df.copy()
    if selected_city != "All" and "City" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["City"].astype(str) == selected_city]
    if selected_traffic != "All" and "Road_traffic_density" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Road_traffic_density"].astype(str) == selected_traffic]
    if selected_weather != "All" and "Weather_conditions" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Weather_conditions"].astype(str) == selected_weather]
    if selected_vehicle != "All" and "Vehicle_condition" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Vehicle_condition"].astype(str) == selected_vehicle]

    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Hackathon Project**: Built with Python, Pandas, Streamlit, Plotly & OpenAI/Gemini LLM API.")

    # ==============================================================================
    # 3. MAIN DASHBOARD BANNER & KPI METRICS
    # ==============================================================================
    st.markdown('<div class="main-header">🚚 FOOD DELIVERY ANALYTICS DASHBOARD</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Data-Driven Operational Performance Analysis & Optimization System</div>', unsafe_allow_html=True)

    # Detect empty filtered dataset (e.g., incompatible filter combination)
    is_empty = len(filtered_df) == 0

    if is_empty:
        # Show exact prominent warning requested
        st.warning("⚠️ No delivery records match the selected filters. Please choose a different filter combination.")

    # Compute metrics for filtered dataset
    basic_metrics = calculate_basic_metrics(filtered_df)

    # Helper to safely format metric values — avoids showing 0.0 or NaN when data is absent
    def fmt(value, suffix=""):
        if value is None or value == "N/A" or (isinstance(value, float) and (np.isnan(value))):
            return "N/A"
        return f"{value}{suffix}"

    # Display KPI Cards — always visible so the layout stays stable
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Total Deliveries", "0" if is_empty else f"{basic_metrics['total_deliveries']:,}")
    with col2:
        st.metric("Avg Delivery Time", fmt(basic_metrics["avg_delivery_time"], " min"))
    with col3:
        st.metric("Avg Distance", fmt(basic_metrics["avg_distance"], " km"))
    with col4:
        st.metric("Avg Speed", fmt(basic_metrics["avg_speed_kmh"], " km/h"))
    with col5:
        st.metric("Avg Rating", fmt(basic_metrics["avg_rating"], " / 5.0"))
    with col6:
        st.metric("Avg Driver Age", fmt(basic_metrics["avg_age"], " yrs"))

    st.markdown("---")

    # ==============================================================================
    # 4. TABS NAVIGATION
    # ==============================================================================
    tab_overview, tab_cleaning, tab_comp, tab_viz, tab_insights, tab_ai = st.tabs([
        "📊 Overview",
        "🧹 Data Cleaning Pipeline",
        "🏆 Competition Questions",
        "📈 Visualizations",
        "💡 Business Insights",
        "🤖 AI Business Explanation",
    ])

    # ------------------------------------------------------------------------------
    # TAB 1: OVERVIEW
    # ------------------------------------------------------------------------------
    with tab_overview:
        st.header("📊 Operations Overview & Dataset Metadata")

        st.markdown(
            """
            Welcome to the **Food Delivery Analytics System**. This application programmatically evaluates 
            operational fulfillment data to uncover root causes of delivery delays, quantify distance vs. time 
            relationships, and evaluate compound traffic/weather bottlenecks.
            """
        )

        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            st.subheader("📋 Active Dataset Metrics")
            if is_empty:
                # No records match — show N/A for all averages
                summary_table = pd.DataFrame({
                    "Metric": [
                        "Total Delivery Records",
                        "Min Delivery Time",
                        "Max Delivery Time",
                        "Average Delivery Time",
                        "Average Delivery Distance",
                        "Average Fleet Speed",
                        "Average Delivery Partner Rating",
                        "Average Delivery Partner Age"
                    ],
                    "Value": ["0", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"]
                })
            else:
                summary_table = pd.DataFrame({
                    "Metric": [
                        "Total Delivery Records",
                        "Min Delivery Time",
                        "Max Delivery Time",
                        "Average Delivery Time",
                        "Average Delivery Distance",
                        "Average Fleet Speed",
                        "Average Delivery Partner Rating",
                        "Average Delivery Partner Age"
                    ],
                    "Value": [
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
            st.table(summary_table)

        with row1_col2:
            st.subheader("📈 Key Statistical Distribution")
            if is_empty:
                st.info("No data available for the selected filter combination.")
            elif "Time_taken (min)" in filtered_df.columns and "distance_km" in filtered_df.columns:
                desc = filtered_df[["Time_taken (min)", "distance_km"]].describe().T
                desc.rename(columns={
                    "count": "Records", "mean": "Mean", "std": "Std Dev",
                    "min": "Min", "25%": "25th %", "50%": "Median",
                    "75%": "75th %", "max": "Max"
                }, inplace=True)
                st.dataframe(desc.round(2), use_container_width=True)

    # ------------------------------------------------------------------------------
    # TAB 2: ADVANCED DATA CLEANING PIPELINE (TRANSPARENT AUDIT)
    # ------------------------------------------------------------------------------
    with tab_cleaning:
        st.header("🧹 Data Cleaning Pipeline & Audit Report")

        st.markdown(
            """
            This section provides full transparency into the programmatic data cleaning and transformation pipeline.
            The pipeline converts messy raw inputs into a structured, analysis-ready format while strictly preserving
            data integrity (e.g. valid `NaN`s in driver ratings or age are never artificially converted to zero).
            """
        )

        # 1. Audit Summary Metric Cards
        aud_c1, aud_c2, aud_c3, aud_c4 = st.columns(4)
        with aud_c1:
            st.metric("Raw Dataset Size", f"{audit_report['raw_rows']:,} rows × {audit_report['raw_cols']} cols")
        with aud_c2:
            st.metric("Cleaned Valid Dataset", f"{audit_report['cleaned_rows']:,} rows × {audit_report['cleaned_cols']} cols")
        with aud_c3:
            st.metric("Duplicates Handled", f"{audit_report['duplicates_removed']} duplicates")
        with aud_c4:
            st.metric("Missing Values Handled", f"{audit_report['total_missing_before']:,} → {audit_report['total_missing_after']:,}")

        st.markdown("---")

        # 2. Executive Before vs. After Dataset Health Comparison Table
        st.subheader("📋 Dataset Health: Before vs. After Cleaning")

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

        st.markdown("---")

        # 3. Interactive Before & After Data Previews
        st.subheader("👀 Data Preview: Raw (Before) vs. Cleaned (After)")

        preview_tab_raw, preview_tab_clean, preview_tab_side = st.tabs([
            "📄 Raw Dataset Sample (First 10 Rows)",
            "✨ Cleaned Dataset Sample (First 10 Rows)",
            "🔄 Side-by-Side Comparison"
        ])

        with preview_tab_raw:
            st.caption("Raw data as loaded directly from the original CSV file before any transformations:")
            st.dataframe(raw_df.head(10), use_container_width=True)

        with preview_tab_clean:
            st.caption("Cleaned, standardized, and structured DataFrame ready for analysis:")
            st.dataframe(cleaned_df.head(10), use_container_width=True)

        with preview_tab_side:
            st.caption("Side-by-side view of key operational columns before vs after:")
            side_col1, side_col2 = st.columns(2)
            with side_col1:
                st.markdown("**Raw Columns (Sample)**")
                raw_preview_cols = [c for c in raw_df.columns if any(k in c.lower() for k in ["time", "weather", "traffic", "age", "rating", "lat", "long", "city"])][:5]
                st.dataframe(raw_df[raw_preview_cols].head(8) if raw_preview_cols else raw_df.head(8), use_container_width=True)
            with side_col2:
                st.markdown("**Cleaned Columns (Standardized)**")
                clean_preview_cols = [c for c in ["Time_taken (min)", "Road_traffic_density", "Weather_conditions", "Delivery_person_Age", "distance_km", "City"] if c in cleaned_df.columns]
                st.dataframe(cleaned_df[clean_preview_cols].head(8) if clean_preview_cols else cleaned_df.head(8), use_container_width=True)

        st.markdown("---")

        # 4. Detailed Missing Values Audit & Chart
        st.subheader("📊 Missing Values Audit per Column (Before vs. After)")

        col_audit_data = []
        for col_name, audit_info in audit_report.get("column_audit", {}).items():
            col_audit_data.append({
                "Original Column": col_name,
                "Standardized Column": audit_info["mapped_to"],
                "Raw Dtype": audit_info["raw_dtype"],
                "Cleaned Dtype": audit_info["clean_dtype"],
                "Raw Missing Count": audit_info["raw_missing"],
                "Raw Missing (%)": f"{audit_info['raw_missing_pct']}%",
                "Cleaned Missing Count": str(audit_info["clean_missing"]),
                "Cleaned Missing (%)": f"{audit_info['clean_missing_pct']}%" if isinstance(audit_info["clean_missing"], int) else "N/A",
            })

        if col_audit_data:
            audit_table_df = pd.DataFrame(col_audit_data)
            st.dataframe(audit_table_df, use_container_width=True, hide_index=True)

        # Plotly Missing Values Chart
        missing_before_dict = audit_report["missing_before"]
        missing_after_dict = audit_report["missing_after"]

        missing_chart_data = []
        for col, count in missing_before_dict.items():
            if count > 0 or missing_after_dict.get(col, 0) > 0:
                missing_chart_data.append({"Column": col, "Stage": "Before Cleaning", "Missing Count": count})
                missing_chart_data.append({"Column": col, "Stage": "After Cleaning", "Missing Count": missing_after_dict.get(col, 0)})

        if missing_chart_data:
            missing_df = pd.DataFrame(missing_chart_data)
            if HAS_PLOTLY:
                fig_miss = px.bar(
                    missing_df,
                    x="Missing Count",
                    y="Column",
                    color="Stage",
                    barmode="group",
                    orientation="h",
                    title="Missing Value Count per Column (Raw vs Cleaned)",
                    color_discrete_map={"Before Cleaning": "#ef4444", "After Cleaning": "#10b981"},
                    text="Missing Count"
                )
                fig_miss.update_traces(textposition="outside")
                st.plotly_chart(fig_miss, use_container_width=True)
            else:
                st.dataframe(missing_df, use_container_width=True)

        st.markdown("---")

        # 5. Step-by-Step Data Transformation Pipeline
        st.subheader("🛠️ Step-by-Step Data Transformation Pipeline")

        pipeline_steps = [
            {
                "Stage": "1. Column Header Standardization",
                "Problem Detected": "Inconsistent column naming conventions across dataset versions (e.g. `Time_taken(min)`, `Road_traffic_density`, `Weatherconditions`).",
                "Transformation Method": "Applied dynamic fuzzy dictionary matching to rename raw headers into clean targets (`Time_taken (min)`, `Road_traffic_density`, `Weather_conditions`, `Delivery_person_Age`, `Delivery_person_Ratings`, `distance_km`, `City`, `Vehicle_condition`).",
                "Impact / Result": f"{len(audit_report['column_mapping'])} columns standardized cleanly."
            },
            {
                "Stage": "2. String Normalization & Noise Removal",
                "Problem Detected": "Prefix noise (e.g., `'conditions Sunny'`), whitespace padding, and string literal `'nan'` values.",
                "Transformation Method": "Stripped whitespace, removed `'conditions '` prefix in Weather_conditions, and mapped string `'nan'` / `'NaN'` literals to true `np.nan`.",
                "Impact / Result": "Categorical values grouped cleanly into true unique buckets without duplicates."
            },
            {
                "Stage": "3. Great-Circle Distance Math (Haversine)",
                "Problem Detected": "Raw dataset lacked direct delivery distance in kilometers.",
                "Transformation Method": "Calculated great-circle Haversine distance ($\text{km}$) using restaurant lat/long and delivery location lat/long coordinates.",
                "Impact / Result": f"Created `distance_km` column with mean distance of {fmt(basic_metrics['avg_distance'] if not is_empty else 0)} km."
            },
            {
                "Stage": "4. Regex Numeric Extraction",
                "Problem Detected": "Numeric metrics stored as mixed string formats (e.g. `'(min) 24'`, `'24 min'`).",
                "Transformation Method": "Extracted float digits using regex `r'(\\d+\\.?\\d*)'` and coerced errors safely into numeric dtype.",
                "Impact / Result": "`Time_taken (min)`, `Delivery_person_Age`, and `Delivery_person_Ratings` converted to true numeric dtypes."
            },
            {
                "Stage": "5. Record Integrity Filtering & Duplicate Removal",
                "Problem Detected": "Exact duplicate rows or non-positive delivery duration values.",
                "Transformation Method": "Filtered non-positive duration values and dropped duplicate records using `.drop_duplicates()`.",
                "Impact / Result": f"{audit_report['duplicates_removed']} duplicate records removed; {audit_report['cleaned_rows']:,} valid rows retained."
            }
        ]

        step_df = pd.DataFrame(pipeline_steps)
        st.table(step_df)

        st.markdown("---")

        # 6. Interactive Column-by-Column Inspector
        st.subheader("🔍 Interactive Column Cleaning Inspector")
        st.caption("Select any dataset column to inspect how its values, dtypes, and missing counts changed during cleaning.")

        col_to_inspect = st.selectbox("Select Column to Inspect", options=list(raw_df.columns))

        if col_to_inspect:
            ci1, ci2, ci3 = st.columns(3)
            raw_dtype = str(raw_df[col_to_inspect].dtype)
            raw_nulls = int(raw_df[col_to_inspect].isnull().sum())

            # Find matching cleaned column name
            mapped_name = audit_report["column_mapping"].get(col_to_inspect.strip(), col_to_inspect.strip())
            clean_exists = mapped_name in cleaned_df.columns

            with ci1:
                st.write(f"**Original Column**: `{col_to_inspect}`")
                st.write(f"**Mapped Target**: `{mapped_name}`")
            with ci2:
                st.write(f"**Dtype Before**: `{raw_dtype}`")
                st.write(f"**Missing Before**: `{raw_nulls:,}` ({round(raw_nulls/len(raw_df)*100, 2)}%)")
            with ci3:
                if clean_exists:
                    clean_nulls = int(cleaned_df[mapped_name].isnull().sum())
                    st.write(f"**Dtype After**: `{cleaned_df[mapped_name].dtype}`")
                    st.write(f"**Missing After**: `{clean_nulls:,}` ({round(clean_nulls/len(cleaned_df)*100, 2)}%)")
                else:
                    st.write("**Status**: Column converted / merged")

            st.write("**Sample Raw Values vs Cleaned Values:**")
            samp_col1, samp_col2 = st.columns(2)
            with samp_col1:
                st.write("*Raw Sample Values (First 5)*")
                st.write(raw_df[col_to_inspect].head(5).tolist())
            with samp_col2:
                if clean_exists:
                    st.write("*Cleaned Sample Values (First 5)*")
                    st.write(cleaned_df[mapped_name].head(5).tolist())

        st.markdown("---")

        with st.expander("🔍 View Complete Cleaned Data Types Table"):
            dtypes_df = pd.DataFrame({
                "Column": [str(c) for c in cleaned_df.columns],
                "Data Type": [str(d) for d in cleaned_df.dtypes]
            })
            st.dataframe(dtypes_df, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------------------
    # TAB 3: COMPETITION QUESTIONS (Q1, Q2, Q3)
    # ------------------------------------------------------------------------------
    with tab_comp:
        st.header("🏆 Competition Questions & Programmatic Solutions")

        if is_empty:
            st.info("No data available for the selected filter combination.")
        else:
            # Q1: Traffic Impact
            st.subheader("Q1: Which road traffic condition has the highest average delivery time?")
            q1_res = answer_q1_traffic_impact(filtered_df)

            qc1, qc2 = st.columns([1, 2])
            with qc1:
                st.info(f"**Highest Traffic Delay Condition**: `{q1_res['highest_condition']}`")
                st.success(f"**Average Delivery Time**: `{q1_res['highest_avg_time']} minutes`")
            with qc2:
                st.write("**Average Delivery Time by Traffic Density Table:**")
                t_df = pd.DataFrame([
                    {"Traffic Density": k, "Avg Delivery Time (min)": v}
                    for k, v in q1_res["traffic_table"].items()
                ])
                st.dataframe(t_df, use_container_width=True)

            st.markdown("---")

            # Q2: Distance Impact
            st.subheader("Q2: How does delivery distance affect delivery time?")
            q2_res = answer_q2_distance_impact(filtered_df)

            qd1, qd2 = st.columns([1, 2])
            with qd1:
                st.metric("Distance vs Time Correlation", f"{q2_res['correlation']}")
                st.write(f"**Conclusion**: {q2_res['interpretation']}")
            with qd2:
                st.write("**Average Delivery Time by Distance Bins:**")
                bin_df = pd.DataFrame([
                    {"Distance Range": k, "Avg Delivery Time (min)": v}
                    for k, v in q2_res["distance_bins_table"].items()
                ])
                st.dataframe(bin_df, use_container_width=True)

            st.markdown("---")

            # Q3: Combined Conditions
            st.subheader("Q3: Which combination of weather condition and traffic density has the highest average delivery time?")
            q3_res = answer_q3_combined_conditions(filtered_df)

            qw1, qw2 = st.columns([1, 2])
            with qw1:
                st.warning(f"**Worst Weather**: `{q3_res['worst_weather']}`")
                st.warning(f"**Worst Traffic**: `{q3_res['worst_traffic']}`")
                st.error(f"**Highest Average Delivery Time**: `{q3_res['highest_avg_time']} minutes`")
            with qw2:
                st.write("**Top Weather + Traffic Combinations:**")
                comb_df = pd.DataFrame([
                    {"Weather + Traffic Combination": k, "Avg Delivery Time (min)": v}
                    for k, v in q3_res["combined_table"].items()
                ])
                st.dataframe(comb_df.head(10), use_container_width=True)

    # ------------------------------------------------------------------------------
    # TAB 4: VISUALIZATIONS
    # ------------------------------------------------------------------------------
    with tab_viz:
        st.header("📈 Publication-Grade Operational Data Visualizations")

        if is_empty:
            st.info("No data available to display for the selected filter combination.")
        else:
            q1_res = answer_q1_traffic_impact(filtered_df)
            q2_res = answer_q2_distance_impact(filtered_df)

            viz_engine = st.radio(
                "Select Visualization Engine:",
                options=["✨ Plotly Interactive Charts", "📊 Seaborn / Matplotlib High-DPI Static PNGs"],
                horizontal=True,
                key="viz_engine_radio"
            )
            st.markdown("---")

            if "Plotly" in viz_engine:
                # ==============================================================
                # PLOTLY INTERACTIVE MODE
                # ==============================================================
                chart_col1, chart_col2 = st.columns(2)

                # CHART 1: Traffic Density Bar Chart
                with chart_col1:
                    st.subheader("Chart 1: Average Delivery Time by Traffic Density")
                    if q1_res["traffic_table"]:
                        traffic_df = pd.DataFrame({
                            "Traffic Density": list(q1_res["traffic_table"].keys()),
                            "Avg Delivery Time (min)": list(q1_res["traffic_table"].values()),
                        })
                        
                        # Color palette mapping
                        color_map = {"Jam": "#dc2626", "High": "#ea580c", "Medium": "#f59e0b", "Low": "#16a34a"}
                        
                        fig1 = px.bar(
                            traffic_df,
                            x="Traffic Density",
                            y="Avg Delivery Time (min)",
                            text="Avg Delivery Time (min)",
                            color="Traffic Density",
                            color_discrete_map=color_map,
                            title="<b>Average Delivery Time by Traffic Density</b>",
                            labels={
                                "Traffic Density": "Road Traffic Density",
                                "Avg Delivery Time (min)": "Average Delivery Time (minutes)"
                            },
                            template="plotly_white",
                        )
                        fig1.update_traces(texttemplate="%{text:.2f} min", textposition="outside", cliponaxis=False)
                        fig1.update_layout(
                            xaxis_title="Road Traffic Density",
                            yaxis_title="Average Delivery Time (minutes)",
                            font=dict(size=12),
                            yaxis=dict(range=[0, max(traffic_df["Avg Delivery Time (min)"]) * 1.2]),
                            showlegend=False
                        )
                        st.plotly_chart(fig1, use_container_width=True)
                        st.caption(f"📌 **Key Finding**: `{q1_res['highest_condition']}` traffic experiences the highest delivery time averaging **{q1_res['highest_avg_time']} min**.")
                    else:
                        st.info("No traffic data available for current filters.")

                # CHART 2: Distance vs. Delivery Time Scatter Plot
                with chart_col2:
                    st.subheader("Chart 2: Delivery Distance vs Delivery Time")
                    valid_scatter = filtered_df[(filtered_df["distance_km"] > 0) & (filtered_df["Time_taken (min)"] > 0)].dropna(subset=["distance_km", "Time_taken (min)"])
                    if len(valid_scatter) > 0:
                        sample_scatter = valid_scatter.sample(min(2500, len(valid_scatter)), random_state=42)
                        
                        fig2 = px.scatter(
                            sample_scatter,
                            x="distance_km",
                            y="Time_taken (min)",
                            opacity=0.45,
                            trendline="ols" if (HAS_STATSMODELS and len(sample_scatter) > 10) else None,
                            trendline_color_override="#dc2626",
                            title="<b>Delivery Distance (km) vs. Delivery Time (minutes)</b>",
                            labels={
                                "distance_km": "Delivery Distance (km)",
                                "Time_taken (min)": "Delivery Time (minutes)"
                            },
                            color_discrete_sequence=["#2563eb"],
                            template="plotly_white",
                        )
                        fig2.update_layout(
                            xaxis_title="Delivery Distance (km)",
                            yaxis_title="Delivery Time (minutes)",
                            font=dict(size=12),
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                        st.caption(f"📌 **Key Finding**: Correlation is **{q2_res['correlation']}** ({q2_res['interpretation']}).")
                    else:
                        st.info("No distance vs time data available for current filters.")

                st.markdown("---")
                chart_col3, chart_col4 = st.columns(2)

                # CHART 3: Weather Condition Bar Chart
                with chart_col3:
                    st.subheader("Chart 3: Average Delivery Time by Weather Condition")
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
                                color_discrete_sequence=px.colors.qualitative.Dark24,
                                title="<b>Average Delivery Time by Weather Condition</b>",
                                labels={
                                    "Weather_conditions": "Weather Condition",
                                    "Time_taken (min)": "Average Delivery Time (minutes)"
                                },
                                template="plotly_white",
                            )
                            fig3.update_traces(texttemplate="%{text:.2f} min", textposition="outside", cliponaxis=False)
                            fig3.update_layout(
                                xaxis_title="Weather Condition",
                                yaxis_title="Average Delivery Time (minutes)",
                                font=dict(size=12),
                                yaxis=dict(range=[0, max(weather_avg["Time_taken (min)"]) * 1.2]),
                                showlegend=False
                            )
                            st.plotly_chart(fig3, use_container_width=True)
                            st.caption(f"📌 **Key Finding**: `{weather_avg.iloc[0]['Weather_conditions']}` weather causes the longest delivery duration (**{weather_avg.iloc[0]['Time_taken (min)']} min**).")
                        else:
                            st.info("No weather data available for current filters.")

                # CHART 4: Distance Bins Analysis Bar Chart
                with chart_col4:
                    st.subheader("Chart 4: Average Delivery Time by Distance Bins")
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
                            title="<b>Average Delivery Time by Distance Tier (km)</b>",
                            labels={
                                "Distance Range": "Distance Range Tier (km)",
                                "Avg Delivery Time (min)": "Average Delivery Time (minutes)"
                            },
                            template="plotly_white",
                        )
                        fig4.update_traces(texttemplate="%{text:.2f} min", textposition="outside", cliponaxis=False)
                        fig4.update_layout(
                            xaxis_title="Distance Tier (km)",
                            yaxis_title="Average Delivery Time (minutes)",
                            font=dict(size=12),
                            yaxis=dict(range=[0, max(bin_df["Avg Delivery Time (min)"]) * 1.2]),
                            showlegend=False
                        )
                        st.plotly_chart(fig4, use_container_width=True)
                        st.caption("📌 **Key Finding**: Delivery time increases progressively across distance tiers.")
                    else:
                        st.info("No distance bin data available for current filters.")

            else:
                # ==============================================================
                # SEABORN / MATPLOTLIB HIGH-DPI STATIC MODE
                # ==============================================================
                c_img1, c_img2 = st.columns(2)
                with c_img1:
                    st.subheader("Chart 1: Average Delivery Time by Traffic Density")
                    img1_path = plot_traffic_impact(q1_res["traffic_table"])
                    if img1_path and os.path.exists(img1_path):
                        st.image(img1_path, use_container_width=True)
                    else:
                        st.info("No traffic data available to render chart image.")

                with c_img2:
                    st.subheader("Chart 2: Delivery Distance vs Delivery Time")
                    img2_path = plot_distance_vs_time(filtered_df)
                    if img2_path and os.path.exists(img2_path):
                        st.image(img2_path, use_container_width=True)
                    else:
                        st.info("No distance vs time data available to render chart image.")

                st.markdown("---")
                c_img3, c_img4 = st.columns(2)
                with c_img3:
                    st.subheader("Chart 3: Average Delivery Time by Weather Condition")
                    img3_path = plot_weather_impact(filtered_df)
                    if img3_path and os.path.exists(img3_path):
                        st.image(img3_path, use_container_width=True)
                    else:
                        st.info("No weather data available to render chart image.")

    # ------------------------------------------------------------------------------
    # TAB 5: BUSINESS INSIGHTS
    # ------------------------------------------------------------------------------
    with tab_insights:
        st.header("💡 Data-Driven Business Insights")

        if is_empty:
            st.info("No data available to generate business insights for the selected filter combination.")
        else:
            q1_res = answer_q1_traffic_impact(filtered_df)
            q2_res = answer_q2_distance_impact(filtered_df)
            q3_res = answer_q3_combined_conditions(filtered_df)
            insights = generate_business_insights(basic_metrics, q1_res, q2_res, q3_res)

            if not insights:
                st.info("Insufficient data to formulate business insights for the current filter selection.")
            else:
                for ins in insights:
                    with st.container():
                        st.subheader(f"Insight {ins['insight_num']}: {ins['title']}")
                        c_ins1, c_ins2 = st.columns([1, 1])
                        with c_ins1:
                            st.markdown(f"**Finding**: {ins['finding']}")
                            st.markdown(f"**Evidence**: `{ins['evidence']}`")
                        with c_ins2:
                            st.markdown(f"**Business Meaning**: {ins['business_meaning']}")
                            st.markdown(f"**Recommended Action**: {ins['recommended_action']}")
                        st.markdown("---")

    # ------------------------------------------------------------------------------
    # TAB 6: AI BUSINESS EXPLANATION & INTERACTIVE CHATBOT
    # ------------------------------------------------------------------------------
    with tab_ai:
        st.header("🤖 AI Business Intelligence & Interactive Assistant")

        ai_subtab_summary, ai_subtab_chat = st.tabs([
            "📊 AI Executive Summary",
            "💬 Interactive AI Assistant Chatbot"
        ])

        if is_empty:
            with ai_subtab_summary:
                st.info("No delivery records available to generate an AI explanation for the selected filters.")
            with ai_subtab_chat:
                st.info("No delivery records available for the AI chatbot to answer questions. Please adjust filters.")
        else:
            q1_res = answer_q1_traffic_impact(filtered_df)
            q2_res = answer_q2_distance_impact(filtered_df)
            q3_res = answer_q3_combined_conditions(filtered_df)

            summary_dict = {
                "total_deliveries": basic_metrics["total_deliveries"],
                "avg_delivery_time": basic_metrics["avg_delivery_time"],
                "min_delivery_time": basic_metrics["min_delivery_time"],
                "max_delivery_time": basic_metrics["max_delivery_time"],
                "avg_distance": basic_metrics["avg_distance"],
                "avg_speed_kmh": basic_metrics["avg_speed_kmh"],
                "avg_rating": basic_metrics["avg_rating"],
                "avg_age": basic_metrics["avg_age"],
                "highest_traffic_condition": q1_res["highest_condition"],
                "highest_traffic_avg_time": q1_res["highest_avg_time"],
                "distance_correlation": q2_res["correlation"],
                "worst_weather": q3_res["worst_weather"],
                "worst_traffic": q3_res["worst_traffic"],
                "worst_combined_avg_time": q3_res["highest_avg_time"],
            }

            # SUBTAB 1: AI Executive Summary
            with ai_subtab_summary:
                st.markdown(
                    """
                    This section uses an LLM API to convert pre-calculated statistics 
                    into a concise executive presentation summary.
                    """
                )

                if "ai_summary_output" not in st.session_state:
                    st.session_state["ai_summary_output"] = None

                if st.button("🚀 Generate / Refresh AI Business Summary", key="ai_generate_btn"):
                    with st.spinner("Generating AI executive explanation from calculated statistics..."):
                        st.session_state["ai_summary_output"] = generate_ai_explanation(summary_dict)

                if st.session_state["ai_summary_output"]:
                    st.markdown(st.session_state["ai_summary_output"])
                else:
                    st.info("Click the button above to generate the LLM explanation from calculated metrics.")

            # SUBTAB 2: Interactive AI Assistant Chatbot
            with ai_subtab_chat:
                st.subheader("💬 Ask AI Anything About the Food Delivery Analytics")
                st.caption("Ask questions about traffic delays, weather impact, distance vs speed, driver ratings, or strategic recommendations.")

                # Initialize Chatbot Session State Messages
                if "chatbot_messages" not in st.session_state:
                    st.session_state["chatbot_messages"] = [
                        {
                            "role": "assistant",
                            "content": (
                                "👋 Hello! I am your **Food Delivery AI Analytics Assistant**.\n\n"
                                "I can answer any questions about your delivery dataset, operational delays, weather bottlenecks, "
                                "distance correlation, and strategic fulfillment recommendations!\n\n"
                                "**Try asking:**\n"
                                "- *What road traffic condition causes the worst delays?*\n"
                                "- *How does weather affect delivery performance?*\n"
                                "- *What is the correlation between delivery distance and time?*\n"
                                "- *What recommendations do you have to reduce delivery times?*"
                            )
                        }
                    ]

                # Sample Quick-Question Prompt Buttons
                st.markdown("**Quick Questions:**")
                q_col1, q_col2, q_col3, q_col4 = st.columns(4)
                preset_prompt = None

                with q_col1:
                    if st.button("🚦 Traffic Delays", key="btn_q_traffic"):
                        preset_prompt = "What road traffic condition causes the worst delivery delays?"
                with q_col2:
                    if st.button("🌧️ Weather Impact", key="btn_q_weather"):
                        preset_prompt = "Which weather condition slows down deliveries the most?"
                with q_col3:
                    if st.button("📏 Distance vs Speed", key="btn_q_distance"):
                        preset_prompt = "How does delivery distance correlate with delivery duration and fleet speed?"
                with q_col4:
                    if st.button("💡 Top Recommendations", key="btn_q_recs"):
                        preset_prompt = "What are the top 3 operational recommendations to optimize fulfillment?"

                st.markdown("---")

                # Render Chat Message History
                for message in st.session_state["chatbot_messages"]:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                # Handle User Input (via chat_input or sample buttons)
                user_input = st.chat_input("Ask a question about the food delivery analytics...") or preset_prompt

                if user_input:
                    # Append user message
                    st.session_state["chatbot_messages"].append({"role": "user", "content": user_input})
                    with st.chat_message("user"):
                        st.markdown(user_input)

                    # Generate AI Assistant response
                    with st.chat_message("assistant"):
                        with st.spinner("AI Assistant is analyzing dataset metrics..."):
                            bot_reply = answer_user_question_ai(
                                user_query=user_input,
                                summary_results=summary_dict,
                                conversation_history=st.session_state["chatbot_messages"]
                            )
                            st.markdown(bot_reply)

                    # Append assistant response to history
                    st.session_state["chatbot_messages"].append({"role": "assistant", "content": bot_reply})

else:
    st.warning("Dataset could not be loaded. Please ensure the CSV file is located in the 'data/' directory.")




