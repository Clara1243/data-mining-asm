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
    factors = mu.get_risk_factors(raw_data, rf_model, yj_transformer)
    top_factors = factors.tail(3)["Feature"].tolist()

    history_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    frozen_history = {
        col: int(raw_data.get(col, 0))
        for col in history_cols
        if col in raw_data
    }

    audit_record = {
        "Decision": decision,
        "Score": round(current_prob, 1),
        "Risk_Tier": current_risk,
        "Top_Drivers": top_factors,
        "PiT_History": frozen_history,
        "Justification": notes or "No manual justification provided.",
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
        "<div class='sidebar-footer'>© 2026 Data Mining Project v5.0</div>",
        unsafe_allow_html=True,
    )


# ===========================================================================
# PAGE: Applicant Assessment
# ===========================================================================

if page == "Applicant Assessment":

    header_col1, header_col2 = st.columns([3, 1])

    with header_col1:
        st.title("Individual Applicant Assessment")

    with header_col2:
        st.write("")

        with st.popover("📁 Bulk Upload Application", use_container_width=True):

            # --- Template generator ---
            st.markdown("**1. Download Application Template**")
            st.caption("Use this formatted CSV to ensure your batch upload is accepted.")

            template_cols = [
                "ID", "IS_NEW_APPLICANT", "LIMIT_BAL", "SEX", "EDUCATION",
                "MARRIAGE", "AGE",
                "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
                "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
                "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
                "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
                "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
            ]

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

        # Export button (only shown when data is available)
        if (
            st.session_state.get("dataset") is not None
            and st.session_state.get("applicant_states")
        ):

            # --- Macro-Level Progress Tracking ---
            if st.session_state["dataset"] is not None:
                total_apps = len(st.session_state["dataset"])
                processed_apps = sum(1 for state in st.session_state["applicant_states"].values() if state != "Pending")
                progress_pct = int((processed_apps / total_apps) * 100) if total_apps > 0 else 0
                
                st.progress(progress_pct, text=f"Batch Progress: {processed_apps} / {total_apps} Processed ({progress_pct}%)")
                st.markdown("<br>", unsafe_allow_html=True)

    # --- Applicant queue ---
    st.markdown("##### Select Applicant from Queue:")

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
        st.markdown("<br>", unsafe_allow_html=True)

        selected_id = selected_option.replace(" (Pending)", "")

        df = st.session_state["dataset"]
        id_col = st.session_state["id_column"]

        filtered_df = df[df[id_col].astype(str) == str(selected_id)]
        if filtered_df.empty:
            st.warning(f"Data sync delay for ID: {selected_id}. Refreshing queue...")
            st.rerun()
        else:
            app_data = filtered_df.iloc[0]

        left_col, right_col = st.columns([1, 1], gap="large")

        # ---- Left column: Applicant Profile ----
        with left_col:
            st.subheader("Applicant Profile Details")

            raw_new_val = app_data.get("IS_NEW_APPLICANT", "False")
            is_new = str(raw_new_val).strip().lower() in ["true", "1", "1.0", "yes", "t"]

            if is_new:
                st.warning(
                    "⚠️ **Thin File Detected:** This applicant has no prior credit "
                    "history. The risk score is based purely on demographics and "
                    "initial credit limit."
                )

            with st.container(border=True):
                st.markdown("<p class='section-title'>Demographics</p>", unsafe_allow_html=True)

                d_row1_col1, d_row1_col2 = st.columns(2)
                with d_row1_col1:
                    st.markdown(
                        f"<div class='profile-item'><span class='profile-label'>Sex<br></span>"
                        f"<span class='profile-value'>{du.map_sex(app_data.get('SEX', 0))}</span></div>",
                        unsafe_allow_html=True,
                    )
                with d_row1_col2:
                    st.markdown(
                        f"<div class='profile-item'><span class='profile-label'>Age<br></span>"
                        f"<span class='profile-value'>{app_data.get('AGE', 'N/A')}</span></div>",
                        unsafe_allow_html=True,
                    )

                d_row2_col1, d_row2_col2 = st.columns(2)
                with d_row2_col1:
                    st.markdown(
                        f"<div class='profile-item'><span class='profile-label'>Education<br></span>"
                        f"<span class='profile-value'>{du.map_education(app_data.get('EDUCATION', 0))}</span></div>",
                        unsafe_allow_html=True,
                    )
                with d_row2_col2:
                    st.markdown(
                        f"<div class='profile-item'><span class='profile-label'>Marital Status<br></span>"
                        f"<span class='profile-value'>{du.map_marriage(app_data.get('MARRIAGE', 0))}</span></div>",
                        unsafe_allow_html=True,
                    )

                st.divider()
                st.markdown("<p class='section-title'>Financial Information</p>", unsafe_allow_html=True)

                raw_limit = pd.to_numeric(app_data.get("LIMIT_BAL", 0), errors="coerce") or 0
                limit_bal = f"$ {raw_limit:,.0f}"
                st.markdown(
                    f"<div class='profile-item'><span class='profile-label'>Credit Limit Balance<br></span>"
                    f"<span class='profile-value-lg'>{limit_bal}</span></div>",
                    unsafe_allow_html=True,
                )

                st.divider()
                st.markdown("<p class='section-title'>6-Month Repayment History</p>", unsafe_allow_html=True)
                st.markdown(
                    "<p class='history-desc'>Historical payment status "
                    "(-2: No consumption, -1: Paid in full, 0: Revolving, 1-9: Months delayed)</p>",
                    unsafe_allow_html=True,
                )

                badges = [
                    du.get_history_badge(app_data.get(col, 0))
                    for col in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
                ]
                labels = ["Current", "2 Mo Ago", "3 Mo Ago", "4 Mo Ago", "5 Mo Ago", "6 Mo Ago"]

                cards_html = "".join(
                    f"<div class='history-card'>"
                    f"<div class='history-label'>{lbl}</div>"
                    f"<span class='history-badge {css}'>{text}</span>"
                    f"</div>"
                    for lbl, (css, text) in zip(labels, badges)
                )
                st.markdown(
                    f"<div class='history-grid'>{cards_html}</div>",
                    unsafe_allow_html=True,
                )

        # ---- Right column: Risk Assessment ----
        with right_col:
            st.subheader("Default Risk Assessment")

            with st.container(border=True):
                default_prob = mu.process_and_predict(app_data, rf_model, yj_transformer)

                if default_prob < 40:
                    risk_tag, risk_color = "LOW RISK", "#22c55e"
                elif default_prob <= 70:
                    risk_tag, risk_color = "MODERATE RISK", "#eab308"
                else:
                    risk_tag, risk_color = "HIGH RISK", "#ef4444"

                fig = go.Figure(data=[go.Pie(
                    values=[default_prob, 100 - default_prob],
                    hole=0.75,
                    marker_colors=[risk_color, "rgba(128, 128, 128, 0.2)"],
                    textinfo="none",
                    hoverinfo="none",
                    direction="clockwise",
                    sort=False,
                )])
                fig.update_layout(
                    showlegend=False,
                    height=180,
                    margin=dict(t=10, b=0, l=0, r=0),
                    annotations=[dict(
                        text=f"{default_prob:.1f}%",
                        x=0.5, y=0.5,
                        font_size=32,
                        showarrow=False,
                    )],
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

                st.markdown(
                    f'<div class="risk-tag-container">'
                    f'<span class="risk-tag" '
                    f'style="color:{risk_color}; background-color:{risk_color}20;">'
                    f"{risk_tag}</span></div>",
                    unsafe_allow_html=True,
                )

                st.divider()
                st.markdown(
                    "<p class='section-title-sm-spaced'>Score Baselines</p>",
                    unsafe_allow_html=True,
                )
                st.markdown("""
                    <div class="baseline-container">
                        <div class="baseline-banner b-low">
                            <div class="baseline-left">
                                <div class="baseline-dot bg-green"></div>
                                <span class="baseline-title c-green">Low Risk
                                    <span class="baseline-desc">(Auto-Approve Eligible)</span>
                                </span>
                            </div>
                            <span class="baseline-threshold c-green">&lt; 40%</span>
                        </div>
                        <div class="baseline-banner b-mod">
                            <div class="baseline-left">
                                <div class="baseline-dot bg-orange"></div>
                                <span class="baseline-title c-orange">Moderate Risk
                                    <span class="baseline-desc">(Manual Review)</span>
                                </span>
                            </div>
                            <span class="baseline-threshold c-orange">40% – 70%</span>
                        </div>
                        <div class="baseline-banner b-high">
                            <div class="baseline-left">
                                <div class="baseline-dot bg-red"></div>
                                <span class="baseline-title c-red">High Risk
                                    <span class="baseline-desc">(Auto-Reject Eligible)</span>
                                </span>
                            </div>
                            <span class="baseline-threshold c-red">&gt; 70%</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            with st.container(border=True):
                st.markdown("<h3 class='decision-title'>Decision Workflow</h3>", unsafe_allow_html=True)
                st.caption("1. Review Algorithm Risk Assessment above.")

                with st.expander("Decision Justification & Overrides", expanded=False):
                    justification_note = st.text_area(
                        "Enter rationale for approval/rejection",
                        placeholder="Required for manual overrides or exception handling...",
                        key=f"just_{selected_id}",
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                action_col1, action_col2 = st.columns(2)
                with action_col1:
                    st.markdown("<span class='reject-marker'></span>", unsafe_allow_html=True)
                    st.button(
                        "Reject Application",
                        use_container_width=True,
                        on_click=submit_decision,
                        args=(selected_id, "Rejected", id_col, default_prob,
                              risk_tag, app_data, justification_note),
                    )
                with action_col2:
                    st.markdown("<span class='approve-marker'></span>", unsafe_allow_html=True)
                    st.button(
                        "Approve Application",
                        use_container_width=True,
                        on_click=submit_decision,
                        args=(selected_id, "Approved", id_col, default_prob,
                              risk_tag, app_data, justification_note),
                    )

        # ---- Deep Dive Analysis ----
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Deep Dive Analysis")

        chart_col1, chart_col2 = st.columns(2, gap="large")

        with chart_col1:
            with st.container(border=True):
                st.markdown("<p class='section-title'>Payment History Trend</p>", unsafe_allow_html=True)
                st.markdown(
                    "<p class='history-desc'>Comparison of billed vs. paid amounts over the last 6 months.</p>",
                    unsafe_allow_html=True,
                )

                trend_df = mu.get_payment_trend(app_data)

                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(
                    x=trend_df["Month"], y=trend_df["Billed Amount"],
                    name="Billed Amount",
                    line=dict(color="rgba(128, 128, 128, 0.6)", width=3, dash="dot"),
                    mode="lines+markers",
                ))
                fig_trend.add_trace(go.Scatter(
                    x=trend_df["Month"], y=trend_df["Paid Amount"],
                    name="Paid Amount",
                    line=dict(color="#3b82f6", width=3),
                    mode="lines+markers",
                ))
                fig_trend.update_layout(
                    margin=dict(l=0, r=0, t=10, b=0),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(gridcolor="rgba(128, 128, 128, 0.2)", tickprefix="$"),
                    xaxis=dict(showgrid=False),
                )
                st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})

        with chart_col2:
            with st.container(border=True):
                st.markdown("<p class='section-title'>Risk Factors Analysis</p>", unsafe_allow_html=True)
                st.markdown("""
                    <div class='shap-legend'>
                        <b>How to read this chart:</b><br>
                        <span class='shap-risk'><b>Red bars (+):</b></span>
                        Factors pushing the applicant <b>closer</b> to default.<br>
                        <span class='shap-safe'><b>Green bars (-):</b></span>
                        Factors pushing the applicant <b>further away</b> from default.<br>
                        <i>The length of the bar represents the strength of the factor's impact.</i>
                    </div>
                """, unsafe_allow_html=True)

                factors_df = mu.get_risk_factors(app_data, rf_model, yj_transformer)
                colors = ["#ef4444" if v > 0 else "#22c55e" for v in factors_df["Contribution"]]

                fig_factors = go.Figure(go.Bar(
                    x=factors_df["Contribution"],
                    y=factors_df["Feature"],
                    orientation="h",
                    marker_color=colors,
                    text=[f"+{v:.2f}" if v > 0 else f"{v:.2f}" for v in factors_df["Contribution"]],
                    textposition="auto",
                ))
                fig_factors.update_layout(
                    margin=dict(l=0, r=0, t=10, b=0),
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(
                        showgrid=True,
                        gridcolor="rgba(128, 128, 128, 0.2)",
                        zeroline=True,
                        zerolinecolor="rgba(128, 128, 128, 0.5)",
                    ),
                    yaxis=dict(showgrid=False),
                )
                st.plotly_chart(fig_factors, use_container_width=True, config={"displayModeBar": False})


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
                is_override = (
                    (audit["Decision"] == "Approved" and audit["Risk_Tier"] == "HIGH RISK")
                    or (audit["Decision"] == "Rejected" and audit["Risk_Tier"] == "LOW RISK")
                )
                raw_row = df[df[id_col].astype(str) == str(app_id)].iloc[0]
                summary_rows.append({
                    "Applicant ID": str(app_id),
                    "Timestamp": audit.get("Timestamp", "—"),
                    "Age": raw_row.get("AGE", "N/A"),
                    "Credit Limit": raw_row.get("LIMIT_BAL", 0),
                    "Model Score": f"{audit['Score']}% ({audit['Risk_Tier']})",
                    "Final Decision": audit["Decision"],
                    "Delta Flag": "⚠️ OVERRIDE" if is_override else "✅ Aligned",
                    "Risk_Tier_Raw": audit["Risk_Tier"], # Hidden column for accurate filtering
                })

            summary_df = pd.DataFrame(summary_rows)

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

            if filter_option == "Overrides Only":
                summary_df = summary_df[summary_df["Delta Flag"] == "OVERRIDE"]
            elif filter_option == "High-Risk Approvals":
                summary_df = summary_df[(summary_df["Final Decision"] == "Approved") & (summary_df["Risk_Tier_Raw"] == "HIGH RISK")]
            elif filter_option == "Low-Risk Rejections":
                summary_df = summary_df[(summary_df["Final Decision"] == "Rejected") & (summary_df["Risk_Tier_Raw"] == "LOW RISK")]

            display_df = summary_df.drop(columns=["Risk_Tier_Raw"])

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
                
                is_override = (
                    (decision == "Approved" and risk_tier == "HIGH RISK") or 
                    (decision == "Rejected" and risk_tier == "LOW RISK")
                )

                if decision == "Approved":
                    accent      = "#22c55e"
                    badge_cls   = "audit-badge-approved"
                else:
                    accent      = "#ef4444"
                    badge_cls   = "audit-badge-rejected"

                tier_color = {"LOW RISK": "#22c55e", "MODERATE RISK": "#eab308", "HIGH RISK": "#ef4444"}.get(risk_tier, "#888")

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
                    f"<div class='audit-driver-pill' style='display:block; margin-bottom:6px; text-align:center;'>{d}</div>"
                    for d in top_drivers
                )

                override_banner = (
                    f"<div class='audit-override-banner'>⚠️ Operator Override — "
                    f"Decision does not align with model recommendation</div>"
                    if is_override else ""
                )

                st.markdown(f"""
<div class="audit-card" style="border-left: 4px solid {accent};">
<div class="audit-card-header">
<div class="audit-header-left">
<span class="audit-app-id">{app_id}</span>
<span class="audit-badge {badge_cls}">{decision}</span>
{('<span class="audit-override-pill">Override</span>' if is_override else '')}
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
<div class="audit-section audit-section-mid">
<div class="audit-section-label">TOP RISK DRIVERS</div>
<div>{driver_pills_html}</div>
<div class="audit-section-label" style="margin-top:16px;">OPERATOR JUSTIFICATION</div>
<div class="audit-justification">"{justification}"</div>
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
            1 for s in st.session_state["applicant_states"].values() if s != "Pending"
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
                    fig_gender.update_layout(
                        height=250, margin=dict(l=0, r=0, t=10, b=0),
                        showlegend=True, plot_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig_gender, use_container_width=True, config={"displayModeBar": False})
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
                        height=250, margin=dict(l=0, r=0, t=10, b=0),
                        plot_bgcolor="rgba(0,0,0,0)",
                        yaxis=dict(showgrid=True, gridcolor="rgba(128, 128, 128, 0.2)", title="Count"),
                        xaxis=dict(showgrid=False, title="Age"),
                    )
                    st.plotly_chart(fig_age, use_container_width=True, config={"displayModeBar": False})
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
                    height=350, margin=dict(l=0, r=0, t=10, b=0),
                    plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(showgrid=True, gridcolor="rgba(128, 128, 128, 0.2)", tickprefix="$"),
                    xaxis=dict(showgrid=False),
                    showlegend=False,
                )
                st.plotly_chart(fig_box, use_container_width=True, config={"displayModeBar": False})
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
        "<p class='subtext'>Transparency and Explainable AI (XAI) for the core prediction engine.</p>",
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
                margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(
                    showgrid=True,
                    gridcolor="rgba(128, 128, 128, 0.2)",
                    title="Relative Importance Weight",
                ),
                yaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig_global, use_container_width=True, config={"displayModeBar": False})

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
            height=250,
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(side="bottom", showgrid=False, tickfont=dict(size=14)),
            yaxis=dict(showgrid=False, tickfont=dict(size=14)),
        )
        st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})