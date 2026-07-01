"""
- Page config, CSS loading, sidebar navigation
- Session-state management (dataset, applicant queue, audit ledger)
- File upload / ingestion flow (delegates to data_utils)
- All Streamlit UI rendering for each page
- Audit-record creation and decision submission
"""

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu

import data_utils as du
import model_utils as mu
from constants import (
    PAY_COLS,
    PAY_COL_LABELS,
    BILL_COLS,
    PAY_AMT_COLS,
    RISK_TIER_COLORS,
    classify_risk,
    is_new_applicant_flag,
    is_override,
)

# Shared Plotly styling — avoids re-typing the same margin/background/legend
# settings on every chart in this file.
PLOTLY_CONFIG = {"displayModeBar": False}
BASE_LAYOUT = dict(margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="rgba(0,0,0,0)")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Credit Risk Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# CSS loader
# ---------------------------------------------------------------------------

def load_css(file_name: str) -> None:
    """Injects the contents of *file_name* as a <style> block."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    css_path = os.path.join(current_dir, file_name)

    try:
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"UI Error: Could not find styling file at {css_path}")


load_css("style.css")


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def get_models():
    return mu.load_credit_models()


rf_model, yj_transformer = get_models()


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "dataset" not in st.session_state:
    st.session_state["dataset"] = None
if "applicant_states" not in st.session_state:
    st.session_state["applicant_states"] = {}
if "low_risk_threshold" not in st.session_state:
    st.session_state["low_risk_threshold"] = 40
if "high_risk_threshold" not in st.session_state:
    st.session_state["high_risk_threshold"] = 70


# ---------------------------------------------------------------------------
# Audit / decision helpers
# ---------------------------------------------------------------------------

def submit_decision(
    app_id: str,
    decision: str,
    dynamic_id_name: str,
    current_prob: float,
    current_risk: str,
    raw_data: pd.Series,
    notes: str,
) -> None:
    """
    Builds an immutable audit record for one applicant decision and commits
    it to session state.
    """
    actual_notes = st.session_state.get(f"just_{app_id}", notes)

    factors = mu.get_risk_factors(raw_data, rf_model, yj_transformer)
    top_factors = factors.tail(3)["Feature"].tolist()

    frozen_history = {
        col: int(raw_data.get(col, 0))
        for col in PAY_COLS
        if col in raw_data
    }

    audit_record = {
        "Decision": decision,
        "Score": round(current_prob, 1),
        "Risk_Tier": current_risk,
        "Top_Drivers": top_factors,
        "PiT_History": frozen_history,
        "Justification": actual_notes or "No manual justification provided.",
        "Timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }

    updated_states = st.session_state["applicant_states"].copy()
    updated_states[app_id] = audit_record
    st.session_state["applicant_states"] = updated_states

    icon = "✅" if decision == "Approved" else "🚫"
    st.toast(f"Audit Log Saved: {dynamic_id_name} '{app_id}' {decision}", icon=icon)


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

with st.sidebar:
    logo_col, text_col = st.columns([1, 3], vertical_alignment="center")

    with logo_col:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        logo_path = os.path.join(current_dir, "logo-removedbg.png")

        if os.path.exists(logo_path):
            st.image(logo_path, use_container_width=True)
        else:
            st.markdown("🔹")

    with text_col:
        st.markdown(
            "<div class='brand-container'>"
            "<h2 class='brand-title'>RiskMetrics</h2>"
            "<p class='brand-subtitle'>Decision-Support System</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='spacer-md'></div>", unsafe_allow_html=True)
    st.markdown("<p class='menu-label'>Main Menu</p>", unsafe_allow_html=True)

    page = option_menu(
        menu_title=None,
        options=[
            "Applicant Assessment",
            "Applicant Archive",
            "Batch Analytics",
            "Engine Diagnostics",
        ],
        icons=["person-vcard", "archive", "cpu", "pie-chart"],
        default_index=0,
        styles={
            "container": {
                "padding": "0!important",
                "background-color": "transparent",
                "border": "none",
            },
            "icon": {"color": "var(--text-color)", "font-size": "16px"},
            "nav-link": {
                "font-size": "15px",
                "text-align": "left",
                "margin": "2px 0px",
                "color": "var(--text-color)",
                "border-radius": "8px",
                "--hover-color": "rgba(128, 128, 128, 0.1)",
            },
            "nav-link-selected": {
                "background-color": "#DDA705",
                "color": "white",
                "font-weight": "600",
            },
        },
    )

    st.markdown(
        "<div class='sidebar-footer'>© 2026 Data Mining Project v6.0</div>",
        unsafe_allow_html=True,
    )


# ===========================================================================
# PAGE: Applicant Assessment
# ===========================================================================

if page == "Applicant Assessment":

    # --- 1. Responsive Header ---
    # vertical_alignment="bottom" ensures the button aligns perfectly with the text
    header_col1, header_col2 = st.columns([3, 1], vertical_alignment="bottom")

    with header_col1:
        st.title("Individual Applicant Assessment")

    with header_col2:
        with st.popover("📁 Bulk Upload Application", use_container_width=True):

            # --- Template generator ---
            st.markdown("**1. Download Application Template**")
            st.caption("Use this formatted CSV to ensure your batch upload is accepted.")

            template_cols = (
                ["ID", "IS_NEW_APPLICANT", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
                + PAY_COLS + BILL_COLS + PAY_AMT_COLS
            )

            csv_template = pd.DataFrame(columns=template_cols).to_csv(index=False).encode("utf-8")

            st.download_button(
                label="📥 Download Blank CSV",
                data=csv_template,
                file_name="RiskMetrics_Batch_Template.csv",
                mime="text/csv",
                use_container_width=True,
            )

            st.divider()

            # --- Batch upload ---
            st.markdown("**2. Upload Batch Data**")
            uploaded_file = st.file_uploader(
                "Upload CSV/Excel file",
                type=["csv", "xlsx", "xls"],
                label_visibility="collapsed",
            )

            if uploaded_file is not None:
                is_new_file = (
                    "processed_filename" not in st.session_state
                    or st.session_state["processed_filename"] != uploaded_file.name
                )

                if is_new_file:
                    with st.spinner("Processing dataset..."):
                        try:
                            # 1. Load raw data
                            if uploaded_file.name.endswith(".csv"):
                                df = pd.read_csv(uploaded_file)
                            elif uploaded_file.name.endswith(".xls"):
                                df = pd.read_excel(uploaded_file, engine="xlrd")
                            else:
                                df = pd.read_excel(uploaded_file, engine="openpyxl")

                            # Blank-string cells and mostly-empty rows/columns are treated as missing
                            df = df.replace(r'^\s*$', np.nan, regex=True)
                            df = df.dropna(how="all", axis=1)
                            if not df.empty:
                                min_valid_cols = int(len(df.columns) * 0.5)
                                df = df.dropna(thresh=min_valid_cols, axis=0)

                            # 2. Clean headers, map columns, standardise casing
                            df = du.clean_dataset_headers(df)
                            df = du.map_columns(df)
                            df.columns = df.columns.str.upper()

                            # 3. Repair missing schema columns
                            df, missing_cols = du.repair_dataset_schema(df)
                            if missing_cols:
                                st.warning(
                                    f"⚠️ **Partial Data Detected:** Missing columns "
                                    f"`{', '.join(missing_cols)}` were filled with "
                                    f"conservative baselines."
                                )

                            # 4. Detect or generate the ID column
                            id_col = du.detect_id_column(df)
                            if id_col is None:
                                df.insert(
                                    0, "Generated_ID",
                                    [f"APP-{i:04d}" for i in range(1, len(df) + 1)],
                                )
                                id_col = "Generated_ID"
                                st.toast("No unique ID column found. Auto-generated temporary IDs.")

                            # 5. Auto-detect new applicants
                            if "IS_NEW_APPLICANT" not in df.columns:
                                df["IS_NEW_APPLICANT"] = du.detect_new_applicants(df)
                                st.toast("Auto-detected new applicants based on empty 6-month histories.")

                            # 6. Commit to session state
                            st.session_state["id_column"] = id_col
                            st.session_state["dataset"] = df
                            st.session_state["applicant_states"] = {
                                str(uid): "Pending" for uid in df[id_col].values
                            }
                            st.session_state["processed_filename"] = uploaded_file.name
                            st.toast(f"Successfully loaded {len(df)} applicants!", icon="✅")

                        except Exception as e:
                            st.error(f"Error loading file: {e}")
                else:
                    st.success(f"File '{uploaded_file.name}' is loaded and ready.")


    # --- 2. Macro-Level Progress Tracking (Moved out of column for full-width span) ---
    if (
        st.session_state.get("dataset") is not None
        and st.session_state.get("applicant_states")
    ):
        total_apps = len(st.session_state["dataset"])
        processed_apps = sum(
            1 for state in st.session_state["applicant_states"].values()
            if isinstance(state, dict)
        )
        progress_fraction = (processed_apps / total_apps) if total_apps > 0 else 0.0
        progress_pct = round(progress_fraction * 100, 2)

        st.progress(progress_fraction, text=f"Batch Progress: {processed_apps} / {total_apps} Processed ({progress_pct:.2f}%)")

    # --- 3. Applicant queue ---
    st.subheader("Select Applicant from Queue")

    pending_applicants = []
    if st.session_state["dataset"] is not None:
        pending_applicants = [
            app_id
            for app_id, state in st.session_state["applicant_states"].items()
            if state == "Pending"
        ]

    if st.session_state["dataset"] is not None and not pending_applicants:
        st.success(
            "🎉 All applicants in the current batch have been processed! "
            "Check the Archive tab for the ledger."
        )

    elif pending_applicants:
        applicant_list = [f"{app_id} (Pending)" for app_id in pending_applicants]
        selected_option = st.selectbox(
            "Applicant ID", options=applicant_list, label_visibility="collapsed"
        )

        selected_id = selected_option.replace(" (Pending)", "")

        df = st.session_state["dataset"]
        id_col = st.session_state["id_column"]

        filtered_df = df[df[id_col].astype(str) == str(selected_id)]
        if filtered_df.empty:
            st.warning(f"Data sync delay for ID: {selected_id}. Refreshing queue...")
            st.rerun()
        else:
            app_data = filtered_df.iloc[0]
        
        is_new = is_new_applicant_flag(app_data.get("IS_NEW_APPLICANT", "False"))
        raw_limit = pd.to_numeric(app_data.get("LIMIT_BAL", np.nan), errors="coerce")
        limit_bal_missing = pd.isna(raw_limit) or str(app_data.get("LIMIT_BAL", "")).strip().lower() in {"", "na", "nan", "none", "not applicable", "not provided"}

        if is_new:
            st.warning(
                "⚠️ **Thin File Detected:** This applicant has no prior credit "
                "history. The risk score is based purely on demographics and "
                "initial credit limit."
            )
        elif limit_bal_missing:
            st.warning(
                "⚠️ **Credit limit missing:** Automated scoring is withheld for this applicant. "
                "Manual review is recommended until the limit is obtained."
            )

        left_col, right_col = st.columns([1, 1], gap="large")

        # ---- Left column: Applicant Profile ----
        with left_col:
            st.subheader("Applicant Intelligence")
            
            data_tabs = st.tabs([
                "Profile Summary", 
                "Payment Trends", 
                "Risk Factors Analysis"
            ])

            # --- TAB 1: Profile Summary ---
            with data_tabs[0]:
                with st.container(border=True):
                    st.markdown("<p class='section-title-sm'>Demographics</p>", unsafe_allow_html=True)

                    d_cols = st.columns(4)
                    with d_cols[0]:
                        st.markdown(f"<div class='profile-item'><span class='profile-label'>Sex<br></span><span class='profile-value'>{du.map_sex(app_data.get('SEX', 0))}</span></div>", unsafe_allow_html=True)
                    with d_cols[1]:
                        st.markdown(f"<div class='profile-item'><span class='profile-label'>Age<br></span><span class='profile-value'>{app_data.get('AGE', 'N/A')}</span></div>", unsafe_allow_html=True)
                    with d_cols[2]:
                        st.markdown(f"<div class='profile-item'><span class='profile-label'>Education<br></span><span class='profile-value'>{du.map_education(app_data.get('EDUCATION', 0))}</span></div>", unsafe_allow_html=True)
                    with d_cols[3]:
                        st.markdown(f"<div class='profile-item'><span class='profile-label'>Marital Status<br></span><span class='profile-value'>{du.map_marriage(app_data.get('MARRIAGE', 0))}</span></div>", unsafe_allow_html=True)

                    st.divider()

                    st.markdown("<p class='section-title-sm'>Financial Information</p>", unsafe_allow_html=True)
                    raw_limit = pd.to_numeric(app_data.get("LIMIT_BAL", np.nan), errors="coerce")
                    limit_bal_missing = pd.isna(raw_limit) or str(app_data.get("LIMIT_BAL", "")).strip().lower() in {"", "na", "nan", "none", "not applicable", "not provided"}
                    display_limit = "Unavailable" if limit_bal_missing else f"$ {raw_limit:,.0f}"
                    st.markdown(
                        f"<div class='profile-item'><span class='profile-label'>Credit Limit Balance<br></span>"
                        f"<span class='profile-value-lg'>{display_limit}</span></div>",
                        unsafe_allow_html=True,
                    )

                    st.divider()

                    st.markdown("<p class='section-title-sm'>6-Month Repayment History</p>", unsafe_allow_html=True)
                    badges = [du.get_history_badge(app_data.get(col, 0)) for col in PAY_COLS]

                    cards_html = "".join(
                        f"<div class='history-card'><div class='history-label'>{lbl}</div><span class='history-badge {css}'>{text}</span></div>"
                        for lbl, (css, text) in zip(PAY_COL_LABELS, badges)
                    )
                    st.markdown(f"<div class='history-grid'>{cards_html}</div>", unsafe_allow_html=True)
            
            # --- TAB 2: Payment Trends ---
            with data_tabs[1]:
                with st.container(border=True):
                    st.markdown("<p class='section-title'>Billed vs. Paid Amounts</p>", unsafe_allow_html=True)
                    
                    trend_df = mu.get_payment_trend(app_data)
                    fig_trend = go.Figure()
                    fig_trend.add_trace(go.Scatter(
                        x=trend_df["Month"], y=trend_df["Billed Amount"], name="Billed Amount",
                        line=dict(color="rgba(128, 128, 128, 0.6)", width=3, dash="dot"), mode="lines+markers",
                    ))
                    fig_trend.add_trace(go.Scatter(
                        x=trend_df["Month"], y=trend_df["Paid Amount"], name="Paid Amount",
                        line=dict(color="#3b82f6", width=3), mode="lines+markers",
                    ))
                    fig_trend.update_layout(
                        **BASE_LAYOUT,
                        height=190,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        yaxis=dict(gridcolor="rgba(128, 128, 128, 0.2)", tickprefix="$"),
                        xaxis=dict(showgrid=False),
                    )
                    st.plotly_chart(fig_trend, use_container_width=True, config=PLOTLY_CONFIG)

            # --- TAB 3: Risk Factors Analysis ---
            with data_tabs[2]:
                with st.container(border=True):
                    if is_new:
                        st.info("⚠️ Risk factor analysis is not applicable for Thin File applicants with no credit history.")
                    else:
                        st.markdown("""
                            <div class='shap-legend' style='margin-top:0px;'>
                                <span class='shap-risk'><b>Red bars (+):</b></span> Pushes closer to default.<br>
                                <span class='shap-safe'><b>Green bars (-):</b></span> Pushes further away from default.
                            </div>
                        """, unsafe_allow_html=True)

                        factors_df = mu.get_risk_factors(app_data, rf_model, yj_transformer)
                        colors = ["#ef4444" if v > 0 else "#22c55e" for v in factors_df["Contribution"]]

                        fig_factors = go.Figure(go.Bar(
                            x=factors_df["Contribution"], y=factors_df["Feature"],
                            orientation="h", marker_color=colors,
                            text=[f"+{v:.2f}" if v > 0 else f"{v:.2f}" for v in factors_df["Contribution"]], textposition="auto",
                        ))
                        fig_factors.update_layout(
                            **BASE_LAYOUT,
                            height=180,
                            xaxis=dict(showgrid=True, gridcolor="rgba(128, 128, 128, 0.2)", zeroline=True, zerolinecolor="rgba(128, 128, 128, 0.5)"),
                            yaxis=dict(showgrid=False),
                        )
                        st.plotly_chart(fig_factors, use_container_width=True, config=PLOTLY_CONFIG)

        # ---- Right column: Risk Assessment ----
        with right_col:
            st.subheader("Model Assessment")

            with st.container(border=True):
                if is_new or limit_bal_missing:
                    st.markdown(
                        "<div style='text-align: center; padding: 20px;'>"
                        "<h3 style='color: #f59e0b; font-weight: 600;'>Prediction Unavailable</h3>"
                        "<p style='color: #6b7280; font-size: 0.9rem;'>Insufficient or incomplete profile data for automated risk assessment.</p>"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                    default_prob = None
                    risk_tag = "NOT APPLICABLE"
                    risk_color = "#f59e0b"
                else:
                    try:
                        default_prob = mu.process_and_predict(app_data, rf_model, yj_transformer)
                    except Exception as exc:
                        default_prob = None
                        st.warning(f"Prediction could not be generated for this applicant: {exc}")

                    risk_tag, risk_color = classify_risk(
                        default_prob,
                        st.session_state["low_risk_threshold"],
                        st.session_state["high_risk_threshold"],
                    )

                    if default_prob is None:
                        st.info("Model score unavailable; manual review required.")
                    else:
                        fig = go.Figure(data=[go.Pie(
                            values=[default_prob, 100 - default_prob], hole=0.75,
                            marker_colors=[risk_color, "rgba(128, 128, 128, 0.2)"],
                            textinfo="none", hoverinfo="none", direction="clockwise", sort=False,
                        )])
                        fig.update_layout(
                            showlegend=False, height=110, margin=dict(t=0, b=0, l=0, r=0),
                            annotations=[dict(text=f"{default_prob:.1f}%", x=0.5, y=0.5, font_size=28, showarrow=False)],
                        )
                        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

                st.markdown(
                    f'<div class="risk-tag-container"><span class="risk-tag" style="color:{risk_color}; background-color:{risk_color}20;">{risk_tag}</span></div>',
                    unsafe_allow_html=True,
                )

                # Collapsed Baselines (Space Saver) - only show if not Thin File
                if not is_new:
                    with st.expander("ℹ️ View Score Baselines", expanded=False):
                        low_t = st.session_state["low_risk_threshold"]
                        high_t = st.session_state["high_risk_threshold"]
                        st.markdown(f"""
                            <div class="baseline-container">
                                <div class="baseline-banner b-low"><div class="baseline-left"><div class="baseline-dot bg-green"></div><span class="baseline-title c-green">Low Risk</span></div><span class="baseline-threshold c-green">&lt; {low_t}%</span></div>
                                <div class="baseline-banner b-mod"><div class="baseline-left"><div class="baseline-dot bg-orange"></div><span class="baseline-title c-orange">Moderate Risk</span></div><span class="baseline-threshold c-orange">{low_t}–{high_t}%</span></div>
                                <div class="baseline-banner b-high"><div class="baseline-left"><div class="baseline-dot bg-red"></div><span class="baseline-title c-red">High Risk</span></div><span class="baseline-threshold c-red">&gt; {high_t}%</span></div>
                            </div>
                        """, unsafe_allow_html=True)

            with st.container(border=True):
                st.markdown("<h3 class='decision-title'>Decision Workflow</h3>", unsafe_allow_html=True)
                
                if is_new:
                    st.info("ℹ️ Manual review and decision required for Thin File applicants. Use demographic data and initial credit limit to assess creditworthiness.")
                
                justification_note = st.text_area(
                    "Operator Justification",
                    placeholder="Required for manual overrides...",
                    key=f"just_{selected_id}",
                    height=68
                )

                action_col1, action_col2 = st.columns(2)
                with action_col1:
                    st.markdown("<span class='reject-marker'></span>", unsafe_allow_html=True)
                    st.button(
                        "Reject",
                        use_container_width=True,
                        on_click=submit_decision,
                        args=(selected_id, "Rejected", id_col, default_prob if default_prob is not None else 0, risk_tag, app_data, justification_note),
                    )
                with action_col2:
                    st.markdown("<span class='approve-marker'></span>", unsafe_allow_html=True)
                    st.button(
                        "Approve",
                        use_container_width=True,
                        on_click=submit_decision,
                        args=(selected_id, "Approved", id_col, default_prob if default_prob is not None else 0, risk_tag, app_data, justification_note),
                    )


# ===========================================================================
# PAGE: Applicant Archive
# ===========================================================================

elif page == "Applicant Archive":
    st.title("Applicant Archive")
    st.markdown(
        "<p class='subtext'>Immutable Point-in-Time (PiT) audit ledger of processed applications.</p>",
        unsafe_allow_html=True,
    )

    if st.session_state.get("dataset") is not None:
        processed_data = {
            k: v
            for k, v in st.session_state["applicant_states"].items()
            if isinstance(v, dict)
        }

        if processed_data:
            df = st.session_state["dataset"]
            id_col = st.session_state["id_column"]
            summary_rows = []
            for app_id, audit in processed_data.items():
                overridden = is_override(audit["Decision"], audit["Risk_Tier"])
                raw_row = df[df[id_col].astype(str) == str(app_id)].iloc[0]
                summary_rows.append({
                    "Applicant ID": str(app_id),
                    "Timestamp": audit.get("Timestamp", "—"),
                    "Age": raw_row.get("AGE", "N/A"),
                    "Credit Limit": raw_row.get("LIMIT_BAL", 0),
                    "Model Score": f"{audit['Score']}% ({audit['Risk_Tier']})",
                    "Final Decision": audit["Decision"],
                    "Delta Flag": "⚠️ OVERRIDE" if overridden else "✅ Aligned",
                    "Risk_Tier_Raw": audit["Risk_Tier"],  # hidden column, used only for filtering
                    "Is_Override_Raw": overridden,        # hidden column, used only for filtering
                })

            summary_df = pd.DataFrame(summary_rows)

            toolbar_col1, toolbar_col2 = st.columns([4, 1], vertical_alignment="bottom")

            # 1. Render Filters on the Left
            with toolbar_col1:
                filter_option = st.radio(
                    "Filter Applications:",
                    options=[
                        "All Records", 
                        "Overrides Only", 
                        "High-Risk Approvals", 
                        "Low-Risk Rejections"
                    ],
                    horizontal=True,
                    label_visibility="collapsed"
                )

            # 2. Apply Filters
            if filter_option == "Overrides Only":
                summary_df = summary_df[summary_df["Is_Override_Raw"]]
            elif filter_option == "High-Risk Approvals":
                summary_df = summary_df[(summary_df["Final Decision"] == "Approved") & (summary_df["Risk_Tier_Raw"] == "HIGH RISK")]
            elif filter_option == "Low-Risk Rejections":
                summary_df = summary_df[(summary_df["Final Decision"] == "Rejected") & (summary_df["Risk_Tier_Raw"] == "LOW RISK")]

            display_df = summary_df.drop(columns=["Risk_Tier_Raw", "Is_Override_Raw"])

            # 3. Build Comprehensive Export Data based on filtered results
            export_rows = []
            for app_id in summary_df["Applicant ID"]:
                audit = processed_data[app_id]
                raw_row = df[df[id_col].astype(str) == str(app_id)].iloc[0]
                
                # Reformat timestamp to dd:mm:yyyy hh:mm
                raw_ts = audit.get("Timestamp", "")
                try:
                    dt_obj = datetime.strptime(raw_ts, "%Y-%m-%d %H:%M:%S UTC")
                    fmt_ts = dt_obj.strftime("%d:%m:%Y %H:%M")
                except ValueError:
                    fmt_ts = raw_ts # Fallback if parsing fails

                export_row = {
                    "Applicant ID": app_id,
                    "Final Decision": audit["Decision"],
                    "Decision Timestamp": fmt_ts,
                    "Model Score (%)": audit["Score"],
                    "Risk Tier": audit["Risk_Tier"],
                    "Age": raw_row.get("AGE", "N/A"),
                    "Gender": du.map_sex(raw_row.get("SEX", 0)),
                    "Education": du.map_education(raw_row.get("EDUCATION", 0)),
                    "Marital Status": du.map_marriage(raw_row.get("MARRIAGE", 0)),
                    "Credit Limit": raw_row.get("LIMIT_BAL", 0),
                }
                for col, label in zip(PAY_COLS, PAY_COL_LABELS):
                    export_row[f"History ({label})"] = du.get_history_badge(raw_row.get(col, 0))[1]
                export_row["Operator Justification"] = audit.get(
                    "Justification", "No manual justification provided."
                )
                export_rows.append(export_row)
            
            export_csv = pd.DataFrame(export_rows).to_csv(index=False).encode("utf-8")

            # 4. Render Export Button on the Right
            with toolbar_col2:
                st.download_button(
                    label="📥 Export to CSV",
                    data=export_csv,
                    file_name=f"RiskMetrics_Archive_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            # 5. Render the Data Table below the Toolbar
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Credit Limit": st.column_config.NumberColumn(format="$%d"),
                },
            )

            # ----------------------------------------------------------------
            # Audit Log Inspection
            # ----------------------------------------------------------------
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<h4>Detailed Audit Log Inspector</h4>", unsafe_allow_html=True)
            st.markdown(
                "<p class='subtext' style='margin-top:-8px;'>Select a specific applicant to view their frozen point-in-time record. <b>Click the dropdown and type to search.</b></p>",
                unsafe_allow_html=True,
            )

            processed_ids = list(processed_data.keys())
            
            if "audit_index" not in st.session_state:
                st.session_state.audit_index = 0
                
            if st.session_state.audit_index >= len(processed_ids):
                st.session_state.audit_index = 0

            def go_prev():
                if st.session_state.audit_index > 0:
                    st.session_state.audit_index -= 1
                    st.session_state.audit_selectbox = processed_ids[st.session_state.audit_index]

            def go_next():
                if st.session_state.audit_index < len(processed_ids) - 1:
                    st.session_state.audit_index += 1
                    st.session_state.audit_selectbox = processed_ids[st.session_state.audit_index]

            def sync_selector():
                st.session_state.audit_index = processed_ids.index(st.session_state.audit_selectbox)

            selected_audit_id = st.selectbox(
                "Search or Select Applicant ID:",
                options=processed_ids,
                index=st.session_state.audit_index,
                key="audit_selectbox",
                on_change=sync_selector,
                label_visibility="collapsed"
            )

            if selected_audit_id:
                app_id = selected_audit_id
                audit = processed_data[app_id]
                
                raw_row = df[df[id_col].astype(str) == str(app_id)].iloc[0]
                
                decision      = audit["Decision"]
                score         = audit["Score"]
                risk_tier     = audit["Risk_Tier"]
                timestamp     = audit.get("Timestamp", "—")
                justification = audit["Justification"]
                top_drivers   = audit["Top_Drivers"]
                pit           = audit["PiT_History"]
                
                overridden = is_override(decision, risk_tier)

                if decision == "Approved":
                    accent      = "#22c55e"
                    badge_cls   = "audit-badge-approved"
                else:
                    accent      = "#ef4444"
                    badge_cls   = "audit-badge-rejected"

                tier_color = RISK_TIER_COLORS.get(risk_tier, "#888")

                pit_cols_chronological = ["PAY_6", "PAY_5", "PAY_4", "PAY_3", "PAY_2", "PAY_0"]
                timeline_bars = ""
                
                for col in pit_cols_chronological:
                    val = int(pit.get(col, 0))
                    if val == -2:
                        bar_color = "#9ca3af" 
                        bar_height = "15px"
                    elif val <= 0:
                        bar_color = "#22c55e"
                        bar_height = "20px"
                    else:
                        bar_color = "#ef4444"
                        bar_height = f"{min(20 + (val * 4), 40)}px"
                        
                    timeline_bars += f"<div title='{col}: {val}' style='flex:1; background-color:{bar_color}; height:{bar_height}; border-radius:3px; transition: height 0.2s ease;'></div>"

                timeline_html = f"""
<div style='display:flex; gap:6px; align-items:flex-end; height:45px; padding-bottom:5px;'>
{timeline_bars}
</div>
<div style='display:flex; justify-content:space-between; font-size:0.65rem; font-weight:600; opacity:0.5; text-transform:uppercase;'>
<span>6 Mo Ago</span>
<span>Current</span>
</div>
"""

                driver_pills_html = "".join(
                    f"<div class='audit-driver-pill'>{d}</div>"
                    for d in top_drivers
                )

                driver_container = (
                    f"<div style='display:flex; flex-wrap:wrap; gap:6px; margin-bottom:8px;'>"
                    f"{driver_pills_html}</div>"
                )

                override_banner = (
                    f"<div class='audit-override-banner'>⚠️ Operator Override — "
                    f"Decision does not align with model recommendation</div>"
                    if overridden else ""
                )

                st.markdown(f"""
<div class="audit-card" style="border-left: 4px solid {accent};">
<div class="audit-card-header">
<div class="audit-header-left">
<span class="audit-app-id">{app_id}</span>
<span class="audit-badge {badge_cls}">{decision}</span>
{('<span class="audit-override-pill">Override</span>' if overridden else '')}
</div>
<div class="audit-header-right">
<span class="audit-timestamp">🕒 {timestamp}</span>
</div>
</div>
{override_banner}
<div class="audit-card-body">
<div class="audit-section">
<div class="audit-section-label">MODEL ASSESSMENT</div>
<div class="audit-score-val" style="color:{accent};">{score}%</div>
<div class="audit-tier-pill" style="color:{tier_color}; border-color:{tier_color}40; background:{tier_color}10;">
{risk_tier}
</div>
</div>
<div class="audit-section audit-section-mid" style="padding: 0 15px;">
<div class="audit-section-label">TOP RISK DRIVERS</div>
{driver_container}
<div class="audit-section-label" style="margin-top:10px;">OPERATOR JUSTIFICATION</div>
<div class="audit-justification" style="margin-top:4px;">"{justification}"</div>
</div>
<div class="audit-section">
<div class="audit-section-label">PAYMENT TRAJECTORY</div>
{timeline_html}
</div>
</div>
</div>
""", unsafe_allow_html=True)
                    
                st.markdown("<br>", unsafe_allow_html=True)
                nav_col1, nav_col2, nav_col3 = st.columns([1, 4, 1])
                
                with nav_col1:
                    st.button(
                        "⬅ Previous", 
                        on_click=go_prev, 
                        disabled=(st.session_state.audit_index == 0),
                        use_container_width=True
                    )
                    
                with nav_col3:
                    st.button(
                        "Next ➡", 
                        on_click=go_next, 
                        disabled=(st.session_state.audit_index == len(processed_ids) - 1),
                        use_container_width=True
                    )

                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("🔍 View Raw Application Data"):
                    st.dataframe(raw_row.to_frame().T, use_container_width=True, hide_index=True)
                
        else:
            st.info("No applicants have been processed yet. Decisions made in the Assessment tab will appear here.")
    else:
        st.warning("Please upload a dataset in the Applicant Assessment tab to begin tracking history.")


# ===========================================================================
# PAGE: Batch Analytics
# ===========================================================================

elif page == "Batch Analytics":
    st.title("Batch Analytics")
    st.markdown(
        "<p class='subtext'>Macro-level demographic and financial analysis of the currently uploaded applicant batch.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr/>", unsafe_allow_html=True)

    if st.session_state.get("dataset") is not None:
        df = st.session_state["dataset"]

        total_applicants = len(df)

        avg_age = int(pd.to_numeric(df["AGE"], errors="coerce").mean()) if "AGE" in df.columns else 0
        avg_limit = pd.to_numeric(df["LIMIT_BAL"], errors="coerce").mean() if "LIMIT_BAL" in df.columns else 0
        avg_limit = avg_limit if pd.notna(avg_limit) else 0

        processed_count = sum(
            1 for s in st.session_state["applicant_states"].values()
            if isinstance(s, dict)
        )
        completion_rate = (processed_count / total_applicants * 100) if total_applicants > 0 else 0

        st.markdown(f"""
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-label">Batch Size</div>
                    <div class="kpi-val">{total_applicants}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Avg. Credit Limit</div>
                    <div class="kpi-val">${avg_limit:,.0f}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Avg. Applicant Age</div>
                    <div class="kpi-val">{avg_age}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Batch Completion</div>
                    <div class="kpi-val">{completion_rate:.1f}%</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Demographic charts ---
        st.subheader("Batch Demographics")
        chart_col1, chart_col2 = st.columns(2, gap="large")

        with chart_col1:
            with st.container(border=True):
                st.markdown("<p class='section-title'>Gender Distribution</p>", unsafe_allow_html=True)

                if "SEX" in df.columns:
                    gender_counts = df["SEX"].apply(du.map_sex).value_counts()
                    fig_gender = go.Figure(data=[go.Pie(
                        labels=gender_counts.index,
                        values=gender_counts.values,
                        hole=0.6,
                        marker_colors=["#3b82f6", "rgba(128, 128, 128, 0.4)"],
                    )])
                    fig_gender.update_layout(**BASE_LAYOUT, height=250, showlegend=True)
                    st.plotly_chart(fig_gender, use_container_width=True, config=PLOTLY_CONFIG)
                else:
                    st.info("Gender data not found in this batch.")

        with chart_col2:
            with st.container(border=True):
                st.markdown("<p class='section-title'>Age Distribution</p>", unsafe_allow_html=True)

                if "AGE" in df.columns:
                    fig_age = go.Figure(data=[go.Histogram(
                        x=df["AGE"], nbinsx=15,
                        marker_color="#22c55e", opacity=0.8,
                    )])
                    fig_age.update_layout(
                        **BASE_LAYOUT,
                        height=250,
                        yaxis=dict(showgrid=True, gridcolor="rgba(128, 128, 128, 0.2)", title="Count"),
                        xaxis=dict(showgrid=False, title="Age"),
                    )
                    st.plotly_chart(fig_age, use_container_width=True, config=PLOTLY_CONFIG)
                else:
                    st.info("Age data not found in this batch.")

        # --- Credit Limit vs Education ---
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("<p class='section-title'>Credit Exposure by Education Level</p>", unsafe_allow_html=True)

            if "EDUCATION" in df.columns and "LIMIT_BAL" in df.columns:
                mapped_edu = df["EDUCATION"].apply(du.map_education)

                fig_box = go.Figure()
                for edu_level in mapped_edu.unique():
                    fig_box.add_trace(go.Box(
                        y=df[mapped_edu == edu_level]["LIMIT_BAL"],
                        name=edu_level,
                        boxpoints=False,
                    ))
                fig_box.update_layout(
                    **BASE_LAYOUT,
                    height=350,
                    yaxis=dict(showgrid=True, gridcolor="rgba(128, 128, 128, 0.2)", tickprefix="$"),
                    xaxis=dict(showgrid=False),
                    showlegend=False,
                )
                st.plotly_chart(fig_box, use_container_width=True, config=PLOTLY_CONFIG)
            else:
                st.info("Education or Credit Limit data not found in this batch.")
    else:
        st.warning(
            "Please navigate to the **Applicant Assessment** tab and upload a "
            "batch dataset to view portfolio analytics."
        )


# ===========================================================================
# PAGE: Engine Diagnostics
# ===========================================================================

elif page == "Engine Diagnostics":
    st.title("Engine Diagnostics")
    st.markdown(
        "<p class='subtext'>Transparency and Analysis for the core prediction engine.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr/>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.subheader("Model Architecture")
        st.info("**Pipeline:** Yeo-Johnson Transformer ➔ Random Forest Classifier")

        with st.container(border=True):
            st.markdown("<p class='section-title'>Hyperparameters</p>", unsafe_allow_html=True)
            params = mu.get_model_params(rf_model)

            for key, val in params.items():
                st.markdown(
                    f"<div class='profile-item'>"
                    f"<span class='profile-label'>{key}</span><br>"
                    f"<span class='profile-value'>{val}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    with col2:
        st.subheader("Global Feature Importance")
        with st.container(border=True):
            st.markdown("<p class='section-title'>Top 10 Drivers of Default Risk</p>", unsafe_allow_html=True)
            st.markdown(
                "<p class='history-desc'>This chart displays the most influential variables "
                "the algorithm uses to evaluate risk across the entire applicant population.</p>",
                unsafe_allow_html=True,
            )

            importance_df = mu.get_global_feature_importance(rf_model)

            fig_global = go.Figure(go.Bar(
                x=importance_df["Importance"],
                y=importance_df["Feature"],
                orientation="h",
                marker_color="#3b82f6",
                text=[f"{v:.3f}" for v in importance_df["Importance"]],
                textposition="auto",
            ))
            fig_global.update_layout(
                **BASE_LAYOUT,
                xaxis=dict(
                    showgrid=True,
                    gridcolor="rgba(128, 128, 128, 0.2)",
                    title="Relative Importance Weight",
                ),
                yaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig_global, use_container_width=True, config=PLOTLY_CONFIG)

    # --- Performance metrics ---
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Model Performance (Threat Matrix)")
    st.markdown(
        "<p class='history-desc'>Evaluating the engine's ability to balance False Positives "
        "(lost revenue) against False Negatives (financial loss) on the testing dataset.</p>",
        unsafe_allow_html=True,
    )

    kpis, cm = mu.get_model_metrics()

    st.markdown(f"""
        <div class="kpi-grid">
            <div class="kpi-card"><div class="kpi-label">Accuracy</div><div class="kpi-val">{kpis['Accuracy']}</div></div>
            <div class="kpi-card"><div class="kpi-label">Precision</div><div class="kpi-val">{kpis['Precision']}</div></div>
            <div class="kpi-card"><div class="kpi-label">Recall (Sensitivity)</div><div class="kpi-val">{kpis['Recall']}</div></div>
            <div class="kpi-card"><div class="kpi-label">F1-Score</div><div class="kpi-val">{kpis['F1-Score']}</div></div>
        </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("<p class='section-title'>Confusion Matrix</p>", unsafe_allow_html=True)

        z_data = [cm[1], cm[0]]   # [[FN, TP], [TN, FP]] for heatmap orientation

        fig_cm = go.Figure(data=go.Heatmap(
            z=z_data,
            x=["Predicted: Paid", "Predicted: Default"],
            y=["Actual: Default", "Actual: Paid"],
            colorscale=[[0, "rgba(128, 128, 128, 0.1)"], [1, "#3b82f6"]],
            text=[[f"<b>{v}</b>" for v in row] for row in z_data],
            texttemplate="%{text}",
            textfont={"size": 18, "color": "var(--text-color)"},
            showscale=False,
            hoverinfo="none",
        ))
        fig_cm.update_layout(
            **BASE_LAYOUT,
            height=250,
            xaxis=dict(side="bottom", showgrid=False, tickfont=dict(size=14)),
            yaxis=dict(showgrid=False, tickfont=dict(size=14)),
        )
        st.plotly_chart(fig_cm, use_container_width=True, config=PLOTLY_CONFIG)

    # --- Dynamic Threshold Adjustment ---
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Risk Threshold Configuration")
    st.markdown(
        "<p class='history-desc'>Dynamically adjust the probability thresholds that determine risk tiers. "
        "Lower thresholds = stricter (fewer approvals), Higher thresholds = lenient (more approvals).</p>",
        unsafe_allow_html=True,
    )

    threshold_col1, threshold_col2 = st.columns([1, 1], gap="large")

    with threshold_col1:
        with st.container(border=True):
            st.markdown("<p class='section-title'>Adjust Thresholds</p>", unsafe_allow_html=True)
            
            new_low = st.slider(
                "LOW RISK Upper Bound",
                min_value=0,
                max_value=100,
                value=st.session_state["low_risk_threshold"],
                step=1,
                help="Default: 40%. Applicants below this score are classified as LOW RISK."
            )
            
            new_high = st.slider(
                "MODERATE RISK Upper Bound",
                min_value=0,
                max_value=100,
                value=st.session_state["high_risk_threshold"],
                step=1,
                help="Default: 70%. Applicants below this score are classified as MODERATE RISK. Above = HIGH RISK."
            )
            
            # Validation: ensure thresholds are in proper order
            if new_low >= new_high:
                st.error("⚠️ LOW RISK threshold must be less than MODERATE RISK threshold!")
            else:
                # Update session state
                st.session_state["low_risk_threshold"] = new_low
                st.session_state["high_risk_threshold"] = new_high
                
                st.success(f"✅ Thresholds Updated: LOW < {new_low}% | MODERATE {new_low}-{new_high}% | HIGH > {new_high}%")
                
                # Reset to defaults button
                if st.button("Reset to Default Thresholds (40 / 70)", use_container_width=True):
                    st.session_state["low_risk_threshold"] = 40
                    st.session_state["high_risk_threshold"] = 70
                    st.rerun()

    with threshold_col2:
        with st.container(border=True):
            st.markdown("<p class='section-title'>Threshold Impact</p>", unsafe_allow_html=True)
            
            # Calculate impact if dataset is loaded
            if st.session_state["dataset"] is not None:
                df = st.session_state["dataset"]
                id_col = st.session_state.get("id_column", None)
                
                if id_col and id_col in df.columns:
                    # Generate predictions for all applicants (excluding Thin Files)
                    predictions = []
                    for idx, row in df.iterrows():
                        is_new = is_new_applicant_flag(row.get("IS_NEW_APPLICANT", "False"))
                        if not is_new:  # Skip Thin Files
                            try:
                                pred = mu.process_and_predict(row, rf_model, yj_transformer)
                                predictions.append(pred)
                            except:
                                pass
                    
                    if predictions:
                        predictions = np.array(predictions)
                        low_t = st.session_state["low_risk_threshold"]
                        high_t = st.session_state["high_risk_threshold"]
                        
                        low_count = np.sum(predictions < low_t)
                        mod_count = np.sum((predictions >= low_t) & (predictions <= high_t))
                        high_count = np.sum(predictions > high_t)
                        total = len(predictions)
                        
                        st.markdown(f"""
                            <div class='profile-item'>
                                <span class='profile-label'>LOW RISK<br></span>
                                <span class='profile-value' style='color: #22c55e;'>{low_count} / {total} ({100*low_count/total:.1f}%)</span>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                            <div class='profile-item'>
                                <span class='profile-label'>MODERATE RISK<br></span>
                                <span class='profile-value' style='color: #eab308;'>{mod_count} / {total} ({100*mod_count/total:.1f}%)</span>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                            <div class='profile-item'>
                                <span class='profile-label'>HIGH RISK<br></span>
                                <span class='profile-value' style='color: #ef4444;'>{high_count} / {total} ({100*high_count/total:.1f}%)</span>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("No valid predictions available in dataset.")
                else:
                    st.info("Load a dataset to see threshold impact.")
            else:
                st.info("📊 Load a dataset in 'Applicant Assessment' to see how thresholds affect classification.")