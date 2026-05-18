import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
from pathlib import Path

# --- COMPONENT IMPORTS ---
from components.data_grids import render_boq_governance_grid
from components.charts import plot_cost_distribution_donut, plot_nrf_fan_chart
from utils.api_client import BoQIntelligenceAPI

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="BoQ Intelligence | AI Governance",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- INJECT CUSTOM CSS ---
def load_css(file_name: str):
    css_path = Path(file_name)
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            
load_css("assets/styles.css")

# --- AUTHENTICATION ABSTRACTION (PROPELAUTH) ---
def verify_authentication():
    """
    Placeholder for PropelAuth logic. 
    Returns a mocked token and user context for local development.
    """
    return {
        "access_token": "mock_propelauth_token_123",
        "user_email": "director@pif.gov.sa",
        "role": "Commercial Director"
    }

user_context = verify_authentication()

# --- CACHED MOCK DATA GENERATOR ---
@st.cache_data
def get_mock_boq_data():
    """Generates a realistic BoQ dataframe matching the ML/Rule Engine outputs."""
    data = {
        "Item_Ref": [f"A.{i}" for i in range(1, 21)],
        "Description": [
            "Plain concrete 50mm 15MPa", "Rebar 16mm @150mm", "Flexible ducts", 
            "DMX control system", "GI conduit", "Sundry items", "Plant Schedule",
            "Timber balustrades", "CCTV Camera pole foundation", "Fire rated cables FP200"
        ] * 2,
        "Qty": np.random.randint(10, 5000, 20),
        "UoM": ["m3", "t", "m", "nr", "m", "item", "item", "m", "m3", "m"] * 2,
        "Rate": np.random.uniform(50, 1500, 20).round(2),
        "Predicted_Category": [
            "Concrete - In situ", "Reinforcement - bars", "Duct - Flexible duct", 
            "Lighting control panel", "Cabling - Copper Conduit", "Unclassified", "Dayworks - Plant",
            "Timber balustrades", "Concrete - In situ", "Cabling - Copper Cable"
        ] * 2,
        "Group_Element": [
            "Substructure", "Substructure", "HVAC", "Electrical", "Electrical", 
            "Preliminaries", "Preliminaries", "Finishes", "Substructure", "Electrical"
        ] * 2,
        "Prediction_Source": [
            "ML_Model", "ML_Model", "Rule_Engine", "Rule_Engine", "Rule_Engine",
            "None", "Rule_Engine", "ML_Model", "Rule_Engine", "Rule_Engine"
        ] * 2,
        "Confidence": [0.95, 0.88, 1.0, 1.0, 1.0, 0.0, 1.0, 0.92, 1.0, 1.0] * 2
    }
    df = pd.DataFrame(data)
    df["Total_Amount"] = df["Qty"] * df["Rate"]
    return df

# Load the data once per session
df_boq = get_mock_boq_data()

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("### 🏗️ **BoQ Intelligence**")
    st.markdown(f"*{user_context['role']}*")
    st.divider()
    
    selected_page = option_menu(
        menu_title=None,
        options=[
            "Command Center",
            "BoQ Predictor",
            "BoQ Ingestion", 
            "NRM Governance", 
            "NRF Forecasting", 
            "Fraud & Anomalies"
        ],
        icons=[
            "bar-chart-fill",
            "cpu-fill",
            "cloud-upload-fill", 
            "check-circle-fill", 
            "graph-up-arrow", 
            "shield-fill-exclamation"
        ],
        menu_icon="cast",
        default_index=1,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#F8FAFC", "font-size": "18px"}, 
            "nav-link": {
                "font-size": "15px", 
                "text-align": "left", 
                "margin":"0px", 
                "--hover-color": "#1E3E62"
            },
            "nav-link-selected": {"background-color": "#1E3E62"},
        }
    )

# --- PAGE ROUTING LOGIC ---
def render_command_center():
    st.title("Executive Command Center")
    st.markdown("Portfolio overview, total BoQ value, and macro inflation risks.")
    
    col1, col2, col3, col4 = st.columns(4)
    total_val = f"SAR {df_boq['Total_Amount'].sum() / 1e6:.2f}M"
    col1.metric("Total BoQ Value Analyzed", total_val)
    col2.metric("Items Auto-Classified", "95.0%", "0.8%")
    col3.metric("Anomalies Detected", "2 Items", "-1")
    col4.metric("Blended Inflation Risk", "4.1%", "0.3%")
    
    st.markdown("---")
    st.subheader("Cost Distribution by NRM Element")
    
    # Render Plotly Donut Chart
    fig_donut = plot_cost_distribution_donut(df_boq)
    st.plotly_chart(fig_donut, width="stretch")

def render_boq_predictor():
    st.title("⚡ BoQ Item Predictor")
    st.markdown("Run **single-item AI inference** against the XGBoost classification engine. Fill in the form below and click **Run Prediction**.")

    api = BoQIntelligenceAPI()

    # ── Section helpers ──────────────────────────────────────────────────────
    def _section(label: str, icon: str = ""):
        st.markdown(f"<p style='font-weight:700;font-size:13px;letter-spacing:.08em;"
                    f"text-transform:uppercase;color:#64748B;margin-bottom:4px;margin-top:18px;'>"
                    f"{icon} {label}</p>", unsafe_allow_html=True)

    # ── FORM ─────────────────────────────────────────────────────────────────
    # Selectbox option lists — defined once so index lookups are DRY
    _UOM_OPTS           = ["", "m", "m2", "m3", "t", "nr", "kg", "item", "lm", "sum"]
    _BASE_DATE_OPTS     = ["", "Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024", "Q1 2025", "Q2 2025", "Q3 2025", "Q4 2025"]
    _RATE_SCOPE_OPTS    = ["", "Supply and Install", "Supply Only", "Install Only", "Labour Only"]
    _CLIENT_LOC_OPTS    = ["", "PIF Non-remote", "PIF Remote", "Commercial"]

    with st.form("boq_predict_form", clear_on_submit=False):
        # ── NRM Hierarchy (optional override) ───────────────────────────────
        _section("NRM Hierarchy (Optional — leave blank to let AI predict)", "🏛️")
        h1, h2, h3, h4 = st.columns(4)
        ge            = h1.text_input("GE Code",       value="5")
        group_element = h2.text_input("Group Element", value="Services")
        e             = h3.text_input("Element Code",  value="5.08")
        element       = h4.text_input("Element",       value="Electrical installations")

        # ── Text Features (core signals) ─────────────────────────────────────
        _section("Item Description  ·  primary classification signal", "📝")
        description = st.text_area(
            "Description *",
            value=(
                "ELECTRICAL INSTALLATIONS Approved LV cables; drawn into conduits "
                "(containment measured separately); including termination as required "
                "and all necessary fixings and accessories"
            ),
            height=100,
        )
        item  = st.text_input("Item / Sub-Description", value="general allowance")
        notes = st.text_area("Notes", placeholder="Any additional context…", height=60)

        # ── Numerical Features ───────────────────────────────────────────────
        _section("Rates & Quantities", "💰")
        n1, n2, n3 = st.columns(3)
        qty          = n1.number_input("Quantity (Qty)",         min_value=0.0, value=1.0,           step=1.0, format="%.2f")
        rate_base    = n2.number_input("Rate — Base Date (SAR)", min_value=0.0, value=24505603.57,   step=1.0, format="%.2f")
        rate_q1_2025 = n3.number_input("Rate — Q1 2025 (SAR)",  min_value=0.0, value=24505604.0,    step=1.0, format="%.2f")

        # ── Categorical Features ─────────────────────────────────────────────
        _section("Project & Scope Metadata", "🗂️")
        c1, c2, c3 = st.columns(3)
        uom      = c1.selectbox("Unit of Measure",   _UOM_OPTS,        index=_UOM_OPTS.index("item"))
        location = c2.text_input("Location",         value="Riyadh")
        project  = c3.text_input("Project",          value="NESMA on DG I – Events Car Park")

        c4, c5, c6 = st.columns(3)
        typology  = c4.text_input("Typology",             value="Car Park")
        base_date = c5.selectbox("Base Date (Quarter)",   _BASE_DATE_OPTS,  index=_BASE_DATE_OPTS.index("Q3 2025"))
        package   = c6.text_input("Package",              value="All")

        c7, c8, c9 = st.columns(3)
        rate_scope      = c7.selectbox("Rate Scope",       _RATE_SCOPE_OPTS, index=_RATE_SCOPE_OPTS.index("Supply and Install"))
        client_location = c8.selectbox("Client Location",  _CLIENT_LOC_OPTS, index=_CLIENT_LOC_OPTS.index("PIF Non-remote"))
        basket_of_goods = c9.text_input("Basket of Goods", value="Cabling - Copper Cable")

        st.markdown("")
        submitted = st.form_submit_button(
            "🚀  Run AI Prediction",
            use_container_width=True,
            type="primary",
        )

    # ── HANDLE SUBMISSION ─────────────────────────────────────────────────────
    if submitted:
        if not description.strip():
            st.error("⚠️  Description is required — it is the primary classification signal.")
            return

        def _opt_str(v: str) -> str | None:
            return v if v else None

        def _opt_float(v) -> float | None:
            return float(v) if v is not None else None

        payload = {
            "items": [{
                "ge":             _opt_str(ge),
                "group_element":  _opt_str(group_element),
                "e":              _opt_str(e),
                "element":        _opt_str(element),
                "description":    description.strip(),
                "item":           _opt_str(item),
                "notes":          _opt_str(notes),
                "qty":            _opt_float(qty),
                "rate_base_date": _opt_float(rate_base),
                "rate_q1_2025":   _opt_float(rate_q1_2025),
                "uom":            _opt_str(uom),
                "location":       _opt_str(location),
                "project":        _opt_str(project),
                "typology":       _opt_str(typology),
                "base_date":      _opt_str(base_date),
                "package":        _opt_str(package),
                "rate_scope":     _opt_str(rate_scope),
                "client_location":_opt_str(client_location),
                "basket_of_goods":_opt_str(basket_of_goods),
            }]
        }

        with st.spinner("Calling inference engine…"):
            try:
                result = api.post_json(
                    "/api/v1/predict",
                    payload,
                    access_token=user_context["access_token"],
                )
                pred = result["predictions"][0]

                # ── RESULTS CARD ─────────────────────────────────────────────
                st.markdown("---")
                st.subheader("🎯 Prediction Result")

                confidence = pred.get("confidence", 0.0)
                source     = pred.get("prediction_source", "—")
                conf_pct   = f"{confidence * 100:.1f}%"

                if confidence >= 0.85:
                    conf_color, conf_label, badge_bg = "#166534", "High Confidence", "#DCFCE7"
                elif confidence >= 0.6:
                    conf_color, conf_label, badge_bg = "#92400E", "Medium Confidence", "#FEF3C7"
                else:
                    conf_color, conf_label, badge_bg = "#991B1B", "Low Confidence", "#FEE2E2"

                source_icon = "🤖" if source == "ML_Model" else "📐"

                # Primary result hero card
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #0B192C 0%, #1E3E62 100%);
                    border-radius: 12px; padding: 28px 32px; color: white;
                    box-shadow: 0 4px 24px rgba(11,25,44,0.18); margin-bottom: 18px;">
                    <div style="font-size:11px;letter-spacing:.1em;text-transform:uppercase;
                                opacity:.7;margin-bottom:8px;">Predicted NRM Category</div>
                    <div style="font-size:28px;font-weight:800;margin-bottom:14px;
                                line-height:1.2;">{pred.get("predicted_category", "—")}</div>
                    <div style="display:flex;gap:12px;flex-wrap:wrap;">
                        <span style="background:rgba(255,255,255,.12);border-radius:20px;
                                     padding:4px 14px;font-size:13px;">
                            {source_icon} {source.replace("_", " ")}
                        </span>
                        <span style="background:{badge_bg};color:{conf_color};border-radius:20px;
                                     padding:4px 14px;font-size:13px;font-weight:700;">
                            {conf_label} · {conf_pct}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Confidence gauge bar
                bar_color = conf_color
                st.markdown(f"""
                <div style="margin-bottom:18px;">
                    <div style="display:flex;justify-content:space-between;
                                font-size:12px;color:#64748B;margin-bottom:4px;">
                        <span>Confidence Score</span><span>{conf_pct}</span>
                    </div>
                    <div style="background:#E2E8F0;border-radius:99px;height:8px;overflow:hidden;">
                        <div style="width:{conf_pct};background:{bar_color};
                                    height:100%;border-radius:99px;
                                    transition:width .5s ease;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # NRM Hierarchy breakdown
                st.markdown("**Predicted NRM Hierarchy**")
                h1c, h2c, h3c, h4c = st.columns(4)
                h1c.metric("GE Code",      pred.get("predicted_ge", "—"))
                h2c.metric("Group Element", pred.get("predicted_group_element", "—"))
                h3c.metric("Element Code", pred.get("predicted_e", "—"))
                h4c.metric("Element",      pred.get("predicted_element", "—"))

                # Meta info
                with st.expander("📋  Full Prediction Payload & Metadata"):
                    mc1, mc2 = st.columns(2)
                    mc1.markdown(f"**Model Version:** `{result.get('model_version', '—')}`")
                    mc2.markdown(f"**Model Name:** `{result.get('model_name', '—')}`")
                    st.json(pred)

            except Exception as err:
                st.error(f"❌ Inference request failed: `{err}`")
                st.info("Make sure the FastAPI server is running and `API_BASE_URL` in your `.env` is correct.")

def render_boq_ingestion():
    st.title("BoQ Ingestion & Smart Mapping")
    st.markdown("Upload raw contractor BoQs and map them to the platform schema.")
    uploaded_file = st.file_uploader("Upload BoQ (Excel/CSV)", type=["xlsx", "csv"])
    if uploaded_file:
        if st.button("Run AI Engine"):
            st.success("File securely transmitted to FastAPI Backend for XGBoost Inference.")

def render_nrm_governance():
    st.title("NRM Classification Governance")
    st.markdown("Human-in-the-loop validation. **Green**: ML Confident | **Blue**: Rule Engine | **Red**: Unclassified/Low Confidence")
    
    # Render interactive AgGrid
    updated_df = render_boq_governance_grid(df_boq)
    
    # Detect manual QS overrides
    if not updated_df.equals(df_boq):
        st.warning("You have unsaved classification overrides.")
        if st.button("Save Overrides to DB"):
            st.success("Overrides saved! ML Retraining queue updated.")

def render_nrf_forecasting():
    st.title("Net Rate Forecast (NRF) Modeling")
    st.markdown("Simulate commodity shocks and predict future rates.")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        commodity = st.selectbox("Select Target Commodity", ["Reinforcement Steel", "Ready-Mix Concrete", "Copper Cabling"])
        shock = st.slider("Market Shock Simulation (%)", -20, 50, 0)
    
    with col2:
        # Render Plotly Fan Chart
        fig_fan = plot_nrf_fan_chart(commodity, shock)
        st.plotly_chart(fig_fan, width="stretch")

def render_fraud_detection():
    st.title("Fraud & Anomaly Detection Hub")
    st.markdown("Review outliers, potential misclassifications, and duplicated pricing.")
    
    anomalies = df_boq[df_boq["Predicted_Category"] == "Unclassified"]
    if not anomalies.empty:
        st.error(f"🚨 {len(anomalies)} High-Risk Items requiring immediate Audit.")
        st.dataframe(anomalies[["Item_Ref", "Description", "Rate", "Total_Amount"]], width="stretch")
    else:
        st.success("No critical anomalies detected in the current BoQ.")

# Execute Routing
if selected_page == "Command Center":
    render_command_center()
elif selected_page == "BoQ Predictor":
    render_boq_predictor()
elif selected_page == "BoQ Ingestion":
    render_boq_ingestion()
elif selected_page == "NRM Governance":
    render_nrm_governance()
elif selected_page == "NRF Forecasting":
    render_nrf_forecasting()
elif selected_page == "Fraud & Anomalies":
    render_fraud_detection()
