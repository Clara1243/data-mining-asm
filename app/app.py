import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import model_utils as utils

st.set_page_config(
    page_title="Credit Risk Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_css(file_name):
    try:
        with open(file_name, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass


load_css("style.css")


@st.cache_resource
def get_model():
    return utils.load_credit_model()


model = get_model()

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
        st.image("logo-removedbg.png", use_container_width=True)

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

    page = st.radio(
        "Navigation",
        ["Applicant Assessment", "Historical Context"],
        label_visibility="collapsed"
    )

    st.markdown(
        "<div class='sidebar-footer'>© 2026 Data Mining Project v3.0</div>",
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
            st.markdown("**Upload Batch Data**")
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
                            if uploaded_file.name.endswith('.csv'):
                                df = pd.read_csv(uploaded_file)
                            elif uploaded_file.name.endswith('.xls'):
                                df = pd.read_excel(uploaded_file, engine='xlrd')
                            else:
                                df = pd.read_excel(uploaded_file, engine='openpyxl')

                            df = utils.map_columns(df)
                            df.columns = df.columns.str.upper()

                            if 'ID' not in df.columns:
                                df.insert(0, 'ID', [f"APP-{i:04d}" for i in range(1, len(df)+1)])

                            st.session_state['dataset'] = df

                            for app_id in df['ID']:
                                if app_id not in st.session_state['applicant_states']:
                                    st.session_state['applicant_states'][app_id] = "Pending"

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

    applicant_list = ["Select an applicant..."] + [f"{app_id} (Pending)" for app_id in pending_applicants]
    selected_option = st.selectbox("Applicant ID", options=applicant_list, label_visibility="collapsed")
    st.markdown("<br>", unsafe_allow_html=True)

    if selected_option != "Select an applicant...":

        selected_id = selected_option.split(" ")[0]
        df = st.session_state['dataset']
        app_data = df[df['ID'] == selected_id].iloc[0]

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
                default_prob = utils.process_and_predict(app_data, model)

                if default_prob < 40:
                    risk_tag, risk_color = "LOW RISK", "#22c55e"
                elif default_prob <= 70:
                    risk_tag, risk_color = "MODERATE RISK", "#f59e0b"
                else:
                    risk_tag, risk_color = "HIGH RISK", "#ef4444"

                fig = go.Figure(data=[go.Pie(
                    values=[default_prob, 100-default_prob],
                    hole=0.75,
                    marker_colors=[risk_color, "#f1f5f9"],
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
                        font_color="#1e293b",
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
                    st.markdown(
                        "<p class='decision-desc'>Review the risk score and make an approval determination.</p>",
                        unsafe_allow_html=True
                    )

                with action_col:
                    btn_col1, btn_col2 = st.columns(2)

                    with btn_col1:
                        st.markdown('<div class="reject-marker"></div>', unsafe_allow_html=True)
                        if st.button("Reject", use_container_width=True):
                            st.session_state['applicant_states'][selected_id] = "Rejected"
                            st.rerun()

                    with btn_col2:
                        st.markdown('<div class="approve-marker"></div>', unsafe_allow_html=True)
                        if st.button("Approve", use_container_width=True):
                            st.session_state['applicant_states'][selected_id] = "Approved"
                            st.rerun()

        # ==========================================
        # NEW SECTION: DEEP DIVE ANALYSIS CHARTS
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
                    line=dict(color="#94a3b8", width=3, dash="dot"),
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
                    yaxis=dict(gridcolor='#e2e8f0', tickprefix="$"),
                    xaxis=dict(showgrid=False)
                )
                st.plotly_chart(fig_trend, use_container_width=True, config={'displayModeBar': False})

        # Chart 2: Risk Factors Analysis
        with chart_col2:
            with st.container(border=True):
                st.markdown("<p class='section-title'>Risk Factors Analysis</p>", unsafe_allow_html=True)
                st.markdown("<p class='history-desc'>Top 5 variables driving the current risk score.</p>", unsafe_allow_html=True)

                factors_df = utils.get_risk_factors(app_data, model)

                # Green for lowering risk, Red for increasing risk
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
                    xaxis=dict(showgrid=True, gridcolor='#e2e8f0', zeroline=True, zerolinecolor='#94a3b8'),
                    yaxis=dict(showgrid=False)
                )
                st.plotly_chart(fig_factors, use_container_width=True, config={'displayModeBar': False})

elif page == "Historical Context":
    st.title("Historical Context")
    st.markdown(
        "<p class='subtext'>Macro-level visualization and association rules of historical credit data.</p>",
        unsafe_allow_html=True
    )
    st.info("Interactive visualizations (Plotly) and Association Rules will be implemented here.")