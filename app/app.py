import streamlit as st
import os
import pandas as pd
import plotly.graph_objects as go
import model_utils as utils
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="Credit Risk Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_css(file_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    css_path = os.path.join(current_dir, file_name)
    
    try:
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"UI Error: Could not find styling file at {css_path}")

load_css("style.css")

@st.cache_resource
def get_models():
    return utils.load_credit_models()

rf_model, yj_transformer = get_models()


if 'dataset' not in st.session_state:
    st.session_state['dataset'] = None
if 'applicant_states' not in st.session_state:
    st.session_state['applicant_states'] = {}


def map_sex(val):
    return "Male" if val == 1 else "Female" if val == 2 else "Unknown"


def map_education(val):
    mapping = {
        1: "Graduate School", 2: "University", 3: "High School",
        4: "Others", 5: "Unknown", 6: "Unknown"
    }
    return mapping.get(val, "Unknown")

def clean_dataset_headers(df):
    """Smart detection and cleaning of staggered or duplicate CSV headers."""
    if df.empty:
        return df
        
    def safe_convert_to_numeric(dataframe):
        """Safely forces numbers to numeric types but leaves text columns alone."""
        for col in dataframe.columns:
            try:
                dataframe[col] = pd.to_numeric(dataframe[col])
            except ValueError:
                pass
        return dataframe

    # 1. Check for a Duplicate Header (e.g., Column is 'ID', Row 0 is also 'ID')
    if all(str(c).strip() == str(v).strip() for c, v in zip(df.columns, df.iloc[0])):
        df = df.iloc[1:].reset_index(drop=True)
        return safe_convert_to_numeric(df)
        
    # 2. Check for Staggered Headers (e.g., Columns are 'X1', 'X2', Row 0 is 'LIMIT_BAL')
    first_row_values = df.iloc[0].astype(str).str.strip().str.upper().values
    
    if 'LIMIT_BAL' in first_row_values or 'ID' in first_row_values or 'AGE' in first_row_values:
        # Promote row 0 to be the actual column names
        df.columns = df.iloc[0].astype(str).str.strip() 
        df = df.iloc[1:].reset_index(drop=True)
        
        df = safe_convert_to_numeric(df)
        
    return df

def detect_id_column(df):
    """Smart detection of the primary key / ID column in an uploaded dataset."""
    # 1. Look for obvious standard names (Case Insensitive)
    known_id_names = ['id', 'clientnum', 'client_id', 'customer_id', 'applicant_id', 'ref_no']
    for col in df.columns:
        if str(col).lower() in known_id_names:
            return col
            
    # 2. Look for columns containing 'id' or 'num' that are 100% unique
    for col in df.columns:
        if ('id' in str(col).lower() or 'num' in str(col).lower()) and df[col].nunique() == len(df):
            return col
            
    # 3. Fallback: If the very first column has completely unique values, assume it's the ID
    if df.iloc[:, 0].nunique() == len(df):
        return df.columns[0]
        
    return None # Return None if no valid ID column is found

def map_marriage(val):
    mapping = {1: "Married", 2: "Single", 3: "Others"}
    return mapping.get(val, "Unknown")


def get_history_badge(val):
    try:
        val = int(val)
        if val <= -1:
            return "badge-gray", f"Paid ({val})"
        if val == 0:
            return "badge-navy", "Revolving (0)"
        return "badge-red", f"{val} Month(s) Late"
    except (ValueError, TypeError):
        return "badge-gray", "Unknown"


# --- Sidebar Navigation ---
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
            unsafe_allow_html=True
        )

    st.markdown("<div class='spacer-md'></div>", unsafe_allow_html=True)
    st.markdown("<p class='menu-label'>Main Menu</p>", unsafe_allow_html=True)

    page = option_menu(
        menu_title=None,
        options=["Applicant Assessment", "Applicant Archive", "Batch Analytics", "Engine Diagnostics"],
        icons=["person-vcard", "archive", "cpu", "pie-chart"],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent", "border": "none"},
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
                "font-weight": "600"
            },
        }
    )

    st.markdown(
        "<div class='sidebar-footer'>© 2026 Data Mining Project v4.2</div>",
        unsafe_allow_html=True
    )

# --- Main Content ---
if page == "Applicant Assessment":

    header_col1, header_col2 = st.columns([3, 1])

    with header_col1:
        st.title("Individual Applicant Assessment")

    with header_col2:
        st.write("")

        with st.popover("📁 Bulk Upload Application", use_container_width=True):
            # --- TEMPLATE GENERATOR ---
            st.markdown("**1. Download Application Template**")
            st.caption("Use this formatted CSV to ensure your batch upload is accepted.")
            
            # Define the exact schema your pipeline expects
            template_cols = [
                'ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 
                'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 
                'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 
                'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
            ]
            
            # Create an empty dataframe with these headers and convert to CSV
            template_df = pd.DataFrame(columns=template_cols)
            csv_template = template_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Download Blank CSV",
                data=csv_template,
                file_name="RiskMetrics_Batch_Template.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            st.divider()

            # ------------------------------

            st.markdown("**2. Upload Batch Data**")
            uploaded_file = st.file_uploader(
                "Upload CSV/Excel file",
                type=["csv", "xlsx", "xls"],
                label_visibility="collapsed"
            )

            if uploaded_file is not None:
                is_new_file = (
                    'processed_filename' not in st.session_state or
                    st.session_state['processed_filename'] != uploaded_file.name
                )
                
                if is_new_file:
                    with st.spinner("Processing dataset..."):
                        try:
                            # 1. Load the raw data
                            if uploaded_file.name.endswith('.csv'):
                                df = pd.read_csv(uploaded_file)
                            elif uploaded_file.name.endswith('.xls'):
                                df = pd.read_excel(uploaded_file, engine='xlrd')
                            else:
                                df = pd.read_excel(uploaded_file, engine='openpyxl')

                            df = clean_dataset_headers(df)

                            # 2. Map standard features and standardize casing
                            df = utils.map_columns(df)
                            df.columns = df.columns.str.upper()

                            # 3. Validate dataset schema for required features
                            is_valid, missing_cols = utils.validate_dataset_schema(df)

                            if not is_valid:
                                st.error(f"Upload rejected. The dataset is missing required credit features: {', '.join(missing_cols)}")
                                st.stop() 

                            # 4. Dynamic Primary Key (ID) Detection
                            id_col = detect_id_column(df)
                            
                            if id_col is None:
                                df.insert(0, 'Generated_ID', [f"APP-{i:04d}" for i in range(1, len(df) + 1)])
                                id_col = 'Generated_ID'
                                st.toast("No unique ID column found. Auto-generated temporary IDs.")
                                
                            # 5. Commit to Session State
                            st.session_state['id_column'] = id_col
                            st.session_state['dataset'] = df
                            
                            # Reset applicant states for the new batch
                            st.session_state['applicant_states'] = {str(uid): "Pending" for uid in df[id_col].values}
                            st.session_state['processed_filename'] = uploaded_file.name
                            
                            st.toast(f"Successfully loaded {len(df)} applicants!", icon="✅")

                        except Exception as e:
                            st.error(f"Error loading file: {e}")
                else:
                    st.success(f"File '{uploaded_file.name}' is loaded and ready.")

        if st.session_state.get('dataset') is not None and st.session_state.get('applicant_states'):
            export_data = [
                {"Applicant ID": k, "State": v}
                for k, v in st.session_state['applicant_states'].items()
            ]
            export_df = pd.DataFrame(export_data)
            csv_buffer = export_df.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="📥 Export Results",
                data=csv_buffer,
                file_name="applicant_decisions.csv",
                mime="text/csv",
                use_container_width=True
            )

    st.markdown("##### Select Applicant from Queue:")

    pending_applicants = []
    if st.session_state['dataset'] is not None:
        pending_applicants = [
            app_id for app_id, state in st.session_state['applicant_states'].items()
            if state == "Pending"
        ]

    # --- NEW QUEUE LOGIC ---
    if st.session_state['dataset'] is not None and not pending_applicants:
        st.success("🎉 All applicants in the current batch have been processed! Check the Archive tab for the ledger.")
        
    elif pending_applicants:
        # We removed the "Select an applicant..." placeholder
        applicant_list = [f"{app_id} (Pending)" for app_id in pending_applicants]
        
        selected_option = st.selectbox("Applicant ID", options=applicant_list, label_visibility="collapsed")
        st.markdown("<br>", unsafe_allow_html=True)

        selected_id = selected_option.split(" ")[0]
        df = st.session_state['dataset']
        id_col = st.session_state['id_column']
        app_data = df[df[id_col].astype(str) == str(selected_id)].iloc[0]

        left_col, right_col = st.columns([1, 1], gap="large")

        with left_col:
            st.subheader("Applicant Profile Details")

            with st.container(border=True):
                st.markdown("<p class='section-title'>Demographics</p>", unsafe_allow_html=True)
                d_row1_col1, d_row1_col2 = st.columns(2)

                with d_row1_col1:
                    st.markdown(
                        f"<div class='profile-item'><span class='profile-label'>Sex<br></span>"
                        f"<span class='profile-value'>{map_sex(app_data.get('SEX', 0))}</span></div>",
                        unsafe_allow_html=True
                    )
                with d_row1_col2:
                    st.markdown(
                        f"<div class='profile-item'><span class='profile-label'>Age<br></span>"
                        f"<span class='profile-value'>{app_data.get('AGE', 'N/A')}</span></div>",
                        unsafe_allow_html=True
                    )

                d_row2_col1, d_row2_col2 = st.columns(2)
                with d_row2_col1:
                    st.markdown(
                        f"<div class='profile-item'><span class='profile-label'>Education<br></span>"
                        f"<span class='profile-value'>{map_education(app_data.get('EDUCATION', 0))}</span></div>",
                        unsafe_allow_html=True
                    )
                with d_row2_col2:
                    st.markdown(
                        f"<div class='profile-item'><span class='profile-label'>Marital Status<br></span>"
                        f"<span class='profile-value'>{map_marriage(app_data.get('MARRIAGE', 0))}</span></div>",
                        unsafe_allow_html=True
                    )

                st.divider()

                st.markdown("<p class='section-title'>Financial Information</p>", unsafe_allow_html=True)
                raw_limit = pd.to_numeric(app_data.get('LIMIT_BAL', 0), errors='coerce') or 0
                limit_bal = f"$ {raw_limit:,.0f}"
                st.markdown(
                    f"<div class='profile-item'><span class='profile-label'>Credit Limit Balance<br></span>"
                    f"<span class='profile-value-lg'>{limit_bal}</span></div>",
                    unsafe_allow_html=True
                )

                st.divider()

                st.markdown("<p class='section-title'>6-Month Repayment History</p>", unsafe_allow_html=True)
                st.markdown(
                    "<p class='history-desc'>Historical payment status "
                    "(-2: No consumption, -1: Paid in full, 0: Revolving, 1-9: Months delayed)</p>",
                    unsafe_allow_html=True
                )

                c1, t1 = get_history_badge(app_data.get('PAY_0', 0))
                c2, t2 = get_history_badge(app_data.get('PAY_2', 0))
                c3, t3 = get_history_badge(app_data.get('PAY_3', 0))
                c4, t4 = get_history_badge(app_data.get('PAY_4', 0))
                c5, t5 = get_history_badge(app_data.get('PAY_5', 0))
                c6, t6 = get_history_badge(app_data.get('PAY_6', 0))

                st.markdown(f"""
                    <div class="history-grid">
                        <div class="history-card"><div class="history-label">Current</div><span class="history-badge {c1}">{t1}</span></div>
                        <div class="history-card"><div class="history-label">2 Mo Ago</div><span class="history-badge {c2}">{t2}</span></div>
                        <div class="history-card"><div class="history-label">3 Mo Ago</div><span class="history-badge {c3}">{t3}</span></div>
                        <div class="history-card"><div class="history-label">4 Mo Ago</div><span class="history-badge {c4}">{t4}</span></div>
                        <div class="history-card"><div class="history-label">5 Mo Ago</div><span class="history-badge {c5}">{t5}</span></div>
                        <div class="history-card"><div class="history-label">6 Mo Ago</div><span class="history-badge {c6}">{t6}</span></div>
                    </div>
                """, unsafe_allow_html=True)

        with right_col:
            st.subheader("Default Risk Assessment")

            with st.container(border=True):
                default_prob = utils.process_and_predict(app_data, rf_model, yj_transformer)

                if default_prob < 40:
                    risk_tag, risk_color = "LOW RISK", "#22c55e"
                elif default_prob <= 70:
                    risk_tag, risk_color = "MODERATE RISK", "#eab308"
                else:
                    risk_tag, risk_color = "HIGH RISK", "#ef4444"

                fig = go.Figure(data=[go.Pie(
                    values=[default_prob, 100-default_prob],
                    hole=0.75,
                    marker_colors=[risk_color, "rgba(128, 128, 128, 0.2)"], # <-- Changed here
                    textinfo='none',
                    hoverinfo='none',
                    direction='clockwise',
                    sort=False
                )])
                
                fig.update_layout(
                    showlegend=False,
                    height=180,
                    margin=dict(t=10, b=0, l=0, r=0),
                    annotations=[dict(
                        text=f"{default_prob:.1f}%",
                        x=0.5, y=0.5,
                        font_size=32,
                        showarrow=False
                    )]
                )

                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

                st.markdown(
                    f'<div class="risk-tag-container"><span class="risk-tag" '
                    f'style="color:{risk_color}; background-color:{risk_color}20;">{risk_tag}</span></div>',
                    unsafe_allow_html=True
                )
                st.divider()

                st.markdown(
                    "<p class='section-title-sm' style='margin-bottom: 12px;'>Score Baselines</p>",
                    unsafe_allow_html=True
                )
                st.markdown("""
                    <div class="baseline-container">
                        <div class="baseline-banner b-low">
                            <div class="baseline-left">
                                <div class="baseline-dot bg-green"></div>
                                <span class="baseline-title c-green">Low Risk <span class="baseline-desc">(Auto-Approve Eligible)</span></span>
                            </div>
                            <span class="baseline-threshold c-green">&lt; 40%</span>
                        </div>
                        <div class="baseline-banner b-mod">
                            <div class="baseline-left">
                                <div class="baseline-dot bg-orange"></div>
                                <span class="baseline-title c-orange">Moderate Risk <span class="baseline-desc">(Manual Review)</span></span>
                            </div>
                            <span class="baseline-threshold c-orange">40% - 70%</span>
                        </div>
                        <div class="baseline-banner b-high">
                            <div class="baseline-left">
                                <div class="baseline-dot bg-red"></div>
                                <span class="baseline-title c-red">High Risk <span class="baseline-desc">(Auto-Reject Eligible)</span></span>
                            </div>
                            <span class="baseline-threshold c-red">&gt; 70%</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            with st.container(border=True):
                text_col, action_col = st.columns([1.5, 1], vertical_alignment="center")

                with text_col:
                    st.markdown("<h3 class='decision-title'>Final Decision</h3>", unsafe_allow_html=True)

                def submit_decision(app_id, decision):
                    st.session_state['applicant_states'][app_id] = decision

                action_col1, action_col2 = st.columns(2)

                with action_col1:
                    st.markdown("<div class='reject-marker'></div>", unsafe_allow_html=True)
                    st.button(
                        "Reject Application", 
                        use_container_width=True,
                        on_click=submit_decision,
                        args=(selected_id, "Rejected") 
                    )

                with action_col2:
                    st.markdown("<div class='approve-marker'></div>", unsafe_allow_html=True)
                    st.button(
                        "Approve Application", 
                        use_container_width=True,
                        on_click=submit_decision,
                        args=(selected_id, "Approved")
                    )

        # ==========================================
        # DEEP DIVE ANALYSIS CHARTS
        # ==========================================
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Deep Dive Analysis")

        chart_col1, chart_col2 = st.columns(2, gap="large")

        # Chart 1: Payment History Trend
        with chart_col1:
            with st.container(border=True):
                st.markdown("<p class='section-title'>Payment History Trend</p>", unsafe_allow_html=True)
                st.markdown("<p class='history-desc'>Comparison of billed vs. paid amounts over the last 6 months.</p>", unsafe_allow_html=True)
                
                trend_df = utils.get_payment_trend(app_data)

                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(
                    x=trend_df['Month'], y=trend_df['Billed Amount'],
                    name="Billed Amount",
                    line=dict(color="rgba(128, 128, 128, 0.6)", width=3, dash="dot"), 
                    mode="lines+markers"
                ))
                fig_trend.add_trace(go.Scatter(
                    x=trend_df['Month'], y=trend_df['Paid Amount'],
                    name="Paid Amount",
                    line=dict(color="#3b82f6", width=3),
                    mode="lines+markers"
                ))
                
                fig_trend.update_layout(
                    margin=dict(l=0, r=0, t=10, b=0),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    plot_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(gridcolor='rgba(128, 128, 128, 0.2)', tickprefix="$"),
                    xaxis=dict(showgrid=False)
                )
                st.plotly_chart(fig_trend, use_container_width=True, config={'displayModeBar': False})

        # Chart 2: Risk Factors Analysis
        with chart_col2:
            with st.container(border=True):
                st.markdown("<p class='section-title'>Risk Factors Analysis</p>", unsafe_allow_html=True)
                
                st.markdown("""
                    <div style='background-color: rgba(128, 128, 128, 0.05); padding: 10px; border-radius: 5px; font-size: 0.9rem; margin-bottom: 15px;'>
                        <b>How to read this chart:</b><br>
                        <span style='color: #ef4444;'><b>Red bars (+):</b></span> Factors pushing the applicant <b>closer</b> to default.<br>
                        <span style='color: #22c55e;'><b>Green bars (-):</b></span> Factors pushing the applicant <b>further away</b> from default.<br>
                        <i>The length of the bar represents the strength of the factor's impact on the final score.</i>
                    </div>
                """, unsafe_allow_html=True)

                factors_df = utils.get_risk_factors(app_data, rf_model, yj_transformer)

                colors = ['#ef4444' if val > 0 else '#22c55e' for val in factors_df['Contribution']]

                fig_factors = go.Figure(go.Bar(
                    x=factors_df['Contribution'],
                    y=factors_df['Feature'],
                    orientation='h',
                    marker_color=colors,
                    text=[f"+{v:.2f}" if v > 0 else f"{v:.2f}" for v in factors_df['Contribution']],
                    textposition='auto'
                ))
                
                fig_factors.update_layout(
                    margin=dict(l=0, r=0, t=10, b=0),
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(
                        showgrid=True, 
                        gridcolor='rgba(128, 128, 128, 0.2)', 
                        zeroline=True, 
                        zerolinecolor='rgba(128, 128, 128, 0.5)'
                    ),
                    yaxis=dict(showgrid=False)
                )
                st.plotly_chart(fig_factors, use_container_width=True, config={'displayModeBar': False})

elif page == "Applicant Archive":
    st.title("Applicant Archive")
    st.markdown(
        "<p class='subtext'>Historical ledger of all processed applicant decisions.</p>", 
        unsafe_allow_html=True
    )
    st.markdown("<hr/>", unsafe_allow_html=True)

    if st.session_state['dataset'] is not None:
        # 1. Identify applicants who are Approved or Rejected
        processed_ids = [
            app_id for app_id, status in st.session_state['applicant_states'].items() 
            if status != "Pending"
        ]
        
        if processed_ids:
            # 2. Extract their data from the main dataset
            df = st.session_state['dataset']
            id_col = st.session_state['id_column']
            # Cast both sides to string to guarantee matches regardless of data type
            str_processed_ids = [str(pid) for pid in processed_ids]
            
            archive_df = df[df[id_col].astype(str).isin(str_processed_ids)].copy()
            archive_df['Final Decision'] = archive_df[id_col].astype(str).map(st.session_state['applicant_states'])
            
            # 4. Clean up the dataframe for the UI
            display_cols = ['ID', 'AGE', 'SEX', 'EDUCATION', 'LIMIT_BAL', 'Final Decision']
            display_df = archive_df[display_cols].copy()
            
            display_df['SEX'] = display_df['SEX'].apply(map_sex)
            display_df['EDUCATION'] = display_df['EDUCATION'].apply(map_education)
            
            # 5. Render the sleek data table
            if 'archive_page' not in st.session_state:
                st.session_state['archive_page'] = 1

            rows_per_page = 10
            total_rows = len(display_df)
            total_pages = max(1, (total_rows - 1) // rows_per_page + 1)

            # Ensure current page doesn't exceed total pages (if filtering changes)
            if st.session_state['archive_page'] > total_pages:
                st.session_state['archive_page'] = total_pages

            # 2. Slice the dataframe for the current page
            start_idx = (st.session_state['archive_page'] - 1) * rows_per_page
            end_idx = start_idx + rows_per_page
            paginated_df = display_df.iloc[start_idx:end_idx]

            # 3. Render the table with the sliced data
            st.dataframe(
                paginated_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "ID": st.column_config.TextColumn("Applicant ID"),
                    "AGE": st.column_config.NumberColumn("Age"),
                    "SEX": st.column_config.TextColumn("Gender"),
                    "EDUCATION": st.column_config.TextColumn("Education"),
                    "LIMIT_BAL": st.column_config.NumberColumn("Credit Limit", format="$%d"),
                    "Final Decision": st.column_config.TextColumn("Decision Status")
                }
            )

            # 4. Render Pagination Controls Below the Table
            st.markdown("<div class='spacer-md'></div>", unsafe_allow_html=True)
            prev_col, text_col, next_col = st.columns([1, 2, 1], vertical_alignment="center")
            
            with prev_col:
                if st.button("< Previous", disabled=(st.session_state['archive_page'] == 1), use_container_width=True):
                    st.session_state['archive_page'] -= 1
                    st.rerun()
            with text_col:
                st.markdown(
                    f"<div style='text-align: center; font-size: 0.9rem; opacity: 0.8;'>"
                    f"Page <b>{st.session_state['archive_page']}</b> of <b>{total_pages}</b>"
                    f"</div>", 
                    unsafe_allow_html=True
                )
            with next_col:
                if st.button("Next >", disabled=(st.session_state['archive_page'] == total_pages), use_container_width=True):
                    st.session_state['archive_page'] += 1
                    st.rerun()
        else:
            st.info("No applicants have been processed yet. Decisions made in the Assessment tab will appear here.")
    else:
        st.warning("Please upload a dataset in the Applicant Assessment tab to begin tracking history.")

elif page == "Batch Analytics":
    st.title("Batch Analytics")
    st.markdown(
        "<p class='subtext'>Macro-level demographic and financial analysis of the currently uploaded applicant batch.</p>",
        unsafe_allow_html=True
    )
    st.markdown("<hr/>", unsafe_allow_html=True)

    if st.session_state.get('dataset') is not None:
        df = st.session_state['dataset']
        
        # --- Batch KPIs ---
        total_applicants = len(df)
        
        # Safely force data to numeric, ignoring text or blank spaces
        if 'AGE' in df.columns:
            age_mean = pd.to_numeric(df['AGE'], errors='coerce').mean()
            avg_age = int(age_mean) if pd.notna(age_mean) else 0
        else:
            avg_age = 0
            
        if 'LIMIT_BAL' in df.columns:
            limit_mean = pd.to_numeric(df['LIMIT_BAL'], errors='coerce').mean()
            avg_limit = limit_mean if pd.notna(limit_mean) else 0
        else:
            avg_limit = 0
        
        # Calculate how many have been processed vs pending
        processed_count = sum(1 for status in st.session_state['applicant_states'].values() if status != "Pending")
        completion_rate = (processed_count / total_applicants) * 100 if total_applicants > 0 else 0

        st.markdown(f"""
            <div class="kpi-grid">
                <div class="kpi-card"><div class="kpi-label">Batch Size</div><div class="kpi-val">{total_applicants}</div></div>
                <div class="kpi-card"><div class="kpi-label">Avg. Credit Limit</div><div class="kpi-val">${avg_limit:,.0f}</div></div>
                <div class="kpi-card"><div class="kpi-label">Avg. Applicant Age</div><div class="kpi-val">{avg_age}</div></div>
                <div class="kpi-card"><div class="kpi-label">Batch Completion</div><div class="kpi-val">{completion_rate:.1f}%</div></div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # --- Dynamic Batch Visualizations ---
        st.subheader("Batch Demographics")
        chart_col1, chart_col2 = st.columns(2, gap="large")

        with chart_col1:
            with st.container(border=True):
                st.markdown("<p class='section-title'>Gender Distribution</p>", unsafe_allow_html=True)
                
                # Safely map gender if the column exists
                if 'SEX' in df.columns:
                    gender_counts = df['SEX'].apply(map_sex).value_counts()
                    
                    fig_gender = go.Figure(data=[go.Pie(
                        labels=gender_counts.index,
                        values=gender_counts.values,
                        hole=0.6,
                        marker_colors=["#3b82f6", "rgba(128, 128, 128, 0.4)"]
                    )])
                    
                    fig_gender.update_layout(
                        height=250, margin=dict(l=0, r=0, t=10, b=0),
                        showlegend=True, plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_gender, use_container_width=True, config={'displayModeBar': False})
                else:
                    st.info("Gender data not found in this batch.")

        with chart_col2:
            with st.container(border=True):
                st.markdown("<p class='section-title'>Age Distribution</p>", unsafe_allow_html=True)
                
                if 'AGE' in df.columns:
                    fig_age = go.Figure(data=[go.Histogram(
                        x=df['AGE'],
                        nbinsx=15,
                        marker_color="#22c55e",
                        opacity=0.8
                    )])
                    
                    fig_age.update_layout(
                        height=250, margin=dict(l=0, r=0, t=10, b=0),
                        plot_bgcolor='rgba(0,0,0,0)',
                        yaxis=dict(showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)', title="Count"),
                        xaxis=dict(showgrid=False, title="Age")
                    )
                    st.plotly_chart(fig_age, use_container_width=True, config={'displayModeBar': False})
                else:
                    st.info("Age data not found in this batch.")

        # --- Credit Limit vs Education ---
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("<p class='section-title'>Credit Exposure by Education Level</p>", unsafe_allow_html=True)
            
            if 'EDUCATION' in df.columns and 'LIMIT_BAL' in df.columns:
                mapped_edu = df['EDUCATION'].apply(map_education)
                
                fig_box = go.Figure()
                for edu_level in mapped_edu.unique():
                    fig_box.add_trace(go.Box(
                        y=df[mapped_edu == edu_level]['LIMIT_BAL'], 
                        name=edu_level,
                        boxpoints=False
                    ))
                
                fig_box.update_layout(
                    height=350, margin=dict(l=0, r=0, t=10, b=0),
                    plot_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)', tickprefix="$"),
                    xaxis=dict(showgrid=False),
                    showlegend=False
                )
                st.plotly_chart(fig_box, use_container_width=True, config={'displayModeBar': False})
            else:
                st.info("Education or Credit Limit data not found in this batch.")

    else:
        # Empty state if they haven't uploaded an Excel file yet
        st.warning("Please navigate to the **Applicant Assessment** tab and upload a batch dataset to view portfolio analytics.")

elif page == "Engine Diagnostics":
    st.title("Engine Diagnostics")
    st.markdown(
        "<p class='subtext'>Transparency and Explainable AI (XAI) for the core prediction engine.</p>", 
        unsafe_allow_html=True
    )
    st.markdown("<hr/>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.subheader("Model Architecture")
        st.info("**Pipeline:** Yeo-Johnson Transformer ➔ Random Forest Classifier")
        
        with st.container(border=True):
            st.markdown("<p class='section-title'>Hyperparameters</p>", unsafe_allow_html=True)
            params = utils.get_model_params(rf_model)
            
            for key, val in params.items():
                st.markdown(
                    f"<div class='profile-item'><span class='profile-label'>{key}</span><br>"
                    f"<span class='profile-value' style='font-size: 1rem;'>{val}</span></div>",
                    unsafe_allow_html=True
                )

    with col2:
        st.subheader("Global Feature Importance")
        with st.container(border=True):
            st.markdown("<p class='section-title'>Top 10 Drivers of Default Risk</p>", unsafe_allow_html=True)
            st.markdown(
                "<p class='history-desc'>This chart displays the most influential variables the algorithm uses to evaluate risk across the entire applicant population.</p>", 
                unsafe_allow_html=True
            )
            
            importance_df = utils.get_global_feature_importance(rf_model)
            
            fig_global = go.Figure(go.Bar(
                x=importance_df['Importance'],
                y=importance_df['Feature'],
                orientation='h',
                marker_color='#3b82f6', # Sleek UI Blue
                text=[f"{v:.3f}" for v in importance_df['Importance']],
                textposition='auto'
            ))
            
            fig_global.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    showgrid=True, 
                    gridcolor='rgba(128, 128, 128, 0.2)',
                    title="Relative Importance Weight"
                ),
                yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig_global, use_container_width=True, config={'displayModeBar': False})

    # ==========================================
    # THREAT MATRIX & KPIs
    # ==========================================
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Model Performance (Threat Matrix)")
    st.markdown(
        "<p class='history-desc'>Evaluating the engine's ability to balance False Positives (lost revenue) against False Negatives (financial loss) on the testing dataset.</p>",
        unsafe_allow_html=True
    )

    kpis, cm = utils.get_model_metrics()

    # 1. Render the Custom KPI Cards
    st.markdown(f"""
        <div class="kpi-grid">
            <div class="kpi-card"><div class="kpi-label">Accuracy</div><div class="kpi-val">{kpis['Accuracy']}</div></div>
            <div class="kpi-card"><div class="kpi-label">Precision</div><div class="kpi-val">{kpis['Precision']}</div></div>
            <div class="kpi-card"><div class="kpi-label">Recall (Sensitivity)</div><div class="kpi-val">{kpis['Recall']}</div></div>
            <div class="kpi-card"><div class="kpi-label">F1-Score</div><div class="kpi-val">{kpis['F1-Score']}</div></div>
        </div>
    """, unsafe_allow_html=True)

    # 2. Render the Confusion Matrix Heatmap
    with st.container(border=True):
        st.markdown("<p class='section-title'>Confusion Matrix</p>", unsafe_allow_html=True)

        z_data = [cm[1], cm[0]] 

        fig_cm = go.Figure(data=go.Heatmap(
            z=z_data,
            x=['Predicted: Paid', 'Predicted: Default'],
            y=['Actual: Default', 'Actual: Paid'],
            colorscale=[[0, 'rgba(128, 128, 128, 0.1)'], [1, '#3b82f6']],
            text=[[f"<b>{v}</b>" for v in row] for row in z_data],
            texttemplate="%{text}",
            textfont={"size": 18, "color": "var(--text-color)"},
            showscale=False,
            hoverinfo="none"
        ))

        fig_cm.update_layout(
            height=250,
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(side='bottom', showgrid=False, tickfont=dict(size=14)),
            yaxis=dict(showgrid=False, tickfont=dict(size=14))
        )

        st.plotly_chart(fig_cm, use_container_width=True, config={'displayModeBar': False})
