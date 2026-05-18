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
                
                ge_code = pred.get("predicted_ge", "—")
                group_el = pred.get("predicted_group_element", "—")
                e_code = pred.get("predicted_e", "—")
                element = pred.get("predicted_element", "—")
                
                st.markdown(f"""
                <div class="nrm-flow-container">
                    <div class="nrm-badge-ge">
                        <span class="nrm-label-ge">GE {ge_code}</span>
                        <strong class="nrm-text">{group_el}</strong>
                    </div>
                    <span class="nrm-arrow">&rarr;</span>
                    <div class="nrm-badge-e">
                        <span class="nrm-label-e">E {e_code}</span>
                        <strong class="nrm-text">{element}</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)



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
    """
    Production-grade BoQ batch ingestion page.

    Pipeline:
        1. Upload  → parse Excel/CSV into a DataFrame.
        2. Map     → auto-detect schema columns; let the QS correct mismatches.
        3. Preview → show the first N rows before committing to the API.
        4. Infer   → POST batch to /api/v1/predict, poll with a progress bar.
        5. Results → enrich the original DataFrame with predictions; render
                     the governance AgGrid and offer a CSV/Excel download.
    """

    # ── Schema vocabulary ────────────────────────────────────────────────────
    SCHEMA_COLS: list[str] = [
        "ge", "group_element", "e", "element",
        "description", "item", "notes",
        "qty", "rate_base_date", "rate_q1_2025",
        "uom", "project", "typology", "location",
        "base_date", "package", "rate_scope", "client_location",
        "basket_of_goods",
    ]

    REQUIRED_COLS: set[str] = {"description"}

    # ── Header ───────────────────────────────────────────────────────────────
    st.title("📂 BoQ Ingestion & Batch Inference")
    st.markdown(
        "Upload a raw Bill of Quantities (Excel or CSV), "
        "map its columns to the AI schema, and run **batch XGBoost inference** "
        "across the entire file in a single request."
    )

    # ── Step 1: File Upload ──────────────────────────────────────────────────
    st.markdown("### Step 1 — Upload File")
    uploaded_file = st.file_uploader(
        "Drop your BoQ file here",
        type=["xlsx", "xls", "csv"],
        help="Excel (.xlsx / .xls) or CSV. First row must be the header.",
    )

    if not uploaded_file:
        # Collapsed Premium Insight Expander
        with st.expander("💡 Architectural Insight — Excel (.xlsx) vs. CSV (Why Excel is preferred)", expanded=False):
            st.write(
                "For the most robust AI classifications and rate analytics, uploading native **Excel (.xlsx)** "
                "files is highly recommended over CSV:"
            )
            st.markdown(
                "- **Bypasses Truncation Errors:** Visual overflows like `###` in Excel are fully preserved "
                "as actual numeric floats in underlying XML. The AI reads them flawlessly.\n"
                "- **CSV Flattening Risk:** Exporting to CSV while columns are visually narrow permanently "
                "flattens numbers into literal text strings (`\"########\"`), leading to complete loss of price precision."
            )



        st.info("⬆️  Upload an Excel or CSV file to begin. The AI engine will handle schema mapping automatically.")
        
        return

    # Contextual CSV Alert
    if uploaded_file.name.endswith(".csv"):
        st.warning(
            "⚠️ **CSV Upload Detected:** "
            "Please ensure this CSV wasn't exported while Excel columns were narrow. "
            "If visual `###` truncations were present during export, they are now literal text strings in this file. "
            "Upload the original `.xlsx` for maximum accuracy."
        )


    # ── Parse file ───────────────────────────────────────────────────────────
    @st.cache_data(show_spinner=False)
    def _parse_file(name: str, data: bytes) -> pd.DataFrame:
        import io
        if name.endswith(".csv"):
            return pd.read_csv(io.BytesIO(data))
        return pd.read_excel(io.BytesIO(data))

    with st.spinner("Parsing file…"):
        try:
            raw_df = _parse_file(uploaded_file.name, uploaded_file.getvalue())
        except Exception as exc:
            st.error(f"❌ Could not parse file: `{exc}`")
            return

    total_rows = len(raw_df)
    st.success(f"✅ Loaded **{total_rows:,} rows** × **{len(raw_df.columns)} columns** from `{uploaded_file.name}`")

    # ── Step 2: Column Mapping ───────────────────────────────────────────────
    st.markdown("### Step 2 — Column Mapping")
    st.markdown(
        "The engine will auto-detect obvious matches. "
        "Fix anything that looks wrong before running inference."
    )

    # Auto-detect: lowercase + strip the file's column names and match greedily
    file_cols: list[str] = list(raw_df.columns)
    file_col_norm: dict[str, str] = {
        c.lower().strip().replace(" ", "_").replace("-", "_"): c
        for c in file_cols
    }

    # Alias map: file column names (normalised) that should map to a schema column.
    # Add entries here whenever a new source system uses non-standard names.
    ALIASES: dict[str, str] = {
        # uom
        "unit":              "uom",
        "units":             "uom",
        "uom_code":          "uom",
        "unit_of_measure":   "uom",
        "unit_of_measurement": "uom",
        "measure":           "uom",
        # qty
        "quantity":          "qty",
        "quantity_nr":       "qty",
        "amount":            "qty",
        # description
        "desc":              "description",
        "item_description":  "description",
        "work_description":  "description",
        # rate fields
        "rate":              "rate_base_date",
        "base_rate":         "rate_base_date",
        "current_rate":      "rate_q1_2025",
        # project / context
        "project_name":      "project",
        "job":               "project",
        "area":              "location",
        "region":            "location",
        "sub_element":       "element",
        "nrm_element":       "element",
        "nrm_group":         "group_element",
    }

    def _auto_detect(schema_col: str) -> str:
        """Return the best file column match for a schema column, or '' if none."""
        # 1. Exact normalised match
        if schema_col in file_col_norm:
            return file_col_norm[schema_col]
        # 2. Alias map: check if any file column's normalised name is a known alias
        for norm, orig in file_col_norm.items():
            if ALIASES.get(norm) == schema_col:
                return orig
        # 3. Partial substring match — only for tokens long enough to be unambiguous.
        #    Single-char schema cols like "e" must NOT greedily match "item", "notes", etc.
        #    Direction is intentionally one-way: the schema column name must appear INSIDE
        #    the file column name (e.g. "description" inside "item_description"). The reverse
        #    is NOT checked — that would let a short file column like "rate" match the longer
        #    schema column "rate_scope", which is almost never the correct mapping.
        if len(schema_col) >= 3:
            for norm, orig in file_col_norm.items():
                if schema_col in norm:
                    return orig
        return ""

    NONE_OPT = "— not mapped —"
    col_opts  = [NONE_OPT] + file_cols

    # Show mapping UI in a tight 3-column grid
    mapping: dict[str, str | None] = {}
    cols_per_row = 3
    schema_chunks = [SCHEMA_COLS[i:i+cols_per_row] for i in range(0, len(SCHEMA_COLS), cols_per_row)]

    with st.expander("🗺️  Column mapping (expand to review / edit)", expanded=True):
        for chunk in schema_chunks:
            grid_cols = st.columns(cols_per_row)
            for col_widget, schema_col in zip(grid_cols, chunk):
                default = _auto_detect(schema_col)
                idx = col_opts.index(default) if default in col_opts else 0
                selected = col_widget.selectbox(
                    f"`{schema_col}`" + (" \\*" if schema_col in REQUIRED_COLS else ""),
                    options=col_opts,
                    index=idx,
                    key=f"map_{schema_col}",
                )
                mapping[schema_col] = None if selected == NONE_OPT else selected

    # Validate required columns are mapped
    missing_required = [c for c in REQUIRED_COLS if not mapping.get(c)]
    if missing_required:
        st.error(f"⚠️  Required column(s) not mapped: **{', '.join(missing_required)}**. Please fix the mapping above.")
        return

    # ── Step 3: Preview ──────────────────────────────────────────────────────
    st.markdown("### Step 3 — Data Preview")
    preview_n = st.slider("Rows to preview", min_value=5, max_value=min(50, total_rows), value=10, step=5)

    # Build a human-readable preview using mapped names
    preview_cols = {v: k for k, v in mapping.items() if v}
    preview_df = raw_df.head(preview_n)[list(preview_cols.keys())].rename(columns=preview_cols)
    st.dataframe(preview_df, use_container_width=True)

    # ── Step 4: Run Inference ────────────────────────────────────────────────
    st.markdown("### Step 4 — Run Batch Inference")

    col_run, col_info = st.columns([1, 3])
    with col_info:
        st.markdown(
            f"<span style='color:#64748B;font-size:13px;'>"
            f"Will send **{total_rows:,} items** to the XGBoost engine via "
            f"`POST /api/v1/predict`.</span>",
            unsafe_allow_html=True,
        )
    run_clicked = col_run.button("🚀  Run AI Engine", type="primary", use_container_width=True)

    if not run_clicked and "ingestion_result" not in st.session_state:
        return

    # Build items payload from the mapped DataFrame
    def _build_items(df: pd.DataFrame, col_map: dict[str, str | None]) -> list[dict]:
        """
        Convert the raw DataFrame into a list of schema-conformant dicts.
        NaN / NaT values are coerced to None so JSON serialisation is clean.
        """
        items: list[dict] = []
        for _, row in df.iterrows():
            item: dict = {}
            for schema_col, file_col in col_map.items():
                if file_col is None:
                    item[schema_col] = None
                    continue
                val = row.get(file_col)
                # Coerce pandas NA types to None
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    item[schema_col] = None
                elif hasattr(val, "item"):          # numpy scalar → python primitive
                    item[schema_col] = val.item()
                else:
                    item[schema_col] = val
            items.append(item)
        return items

    if run_clicked:
        api = BoQIntelligenceAPI()
        items_payload = _build_items(raw_df, mapping)

        progress_bar = st.progress(0, text="Sending batch to inference engine…")
        try:
            result = api.predict_batch(
                items=items_payload,
                access_token=user_context["access_token"],
                timeout=180,
            )
            progress_bar.progress(100, text="Inference complete ✅")
            st.session_state["ingestion_result"]  = result
            st.session_state["ingestion_raw_df"]  = raw_df.copy()
            st.session_state["ingestion_filename"] = uploaded_file.name
        except Exception as exc:
            progress_bar.empty()
            st.error(f"❌ Inference failed: `{exc}`")
            st.info("Make sure the FastAPI server is running and `API_BASE_URL` in `.env` is correct.")
            return

    # ── Step 5: Results ──────────────────────────────────────────────────────
    if "ingestion_result" not in st.session_state:
        return

    result     = st.session_state["ingestion_result"]
    source_df  = st.session_state["ingestion_raw_df"]
    predictions: list[dict] = result.get("predictions", [])

    if not predictions:
        st.warning("The API returned an empty predictions list.")
        return

    st.markdown("---")
    st.markdown("### Step 5 — Results")

    # ── KPI strip ────────────────────────────────────────────────────────────
    pred_df = pd.DataFrame(predictions)

    total          = len(pred_df)
    high_conf      = (pred_df["confidence"] >= 0.85).sum()
    low_conf       = (pred_df["confidence"] < 0.6).sum()
    unclassified   = (pred_df["predicted_category"] == "Others").sum()
    ml_pct         = (pred_df["prediction_source"] == "ML_Model").mean() * 100

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Items",        f"{total:,}")
    k2.metric("High Confidence ≥85%", f"{high_conf:,}", f"{high_conf/total*100:.1f}%")
    k3.metric("Low Confidence <60%",  f"{low_conf:,}",  delta_color="inverse" if low_conf else "normal")
    k4.metric("Unclassified",         f"{unclassified:,}", delta_color="inverse" if unclassified else "normal")
    k5.metric("ML Engine Share",      f"{ml_pct:.1f}%")

    st.markdown("---")

    # ── Enrich original DataFrame with predictions ────────────────────────────
    result_df = source_df.copy().reset_index(drop=True)
    result_df["Predicted_Category"]    = pred_df["predicted_category"]
    result_df["Confidence"]            = pred_df["confidence"].round(4)
    result_df["Prediction_Source"]     = pred_df["prediction_source"]
    result_df["Predicted_GE"]          = pred_df["predicted_ge"]
    result_df["Predicted_Group_Element"] = pred_df["predicted_group_element"]
    result_df["Predicted_E"]           = pred_df["predicted_e"]
    result_df["Predicted_Element"]     = pred_df["predicted_element"]

    # Move prediction columns to front for readability
    pred_cols   = ["Predicted_Category", "Confidence", "Prediction_Source",
                   "Predicted_GE", "Predicted_Group_Element", "Predicted_E", "Predicted_Element"]
    other_cols  = [c for c in result_df.columns if c not in pred_cols]
    result_df   = result_df[pred_cols + other_cols]

    # Persist result_df for NRM Governance page
    st.session_state["df_boq_results"] = result_df

    # Render the governance grid
    st.subheader("AI Classification Results")
    st.markdown("**Green** = ML High Confidence · **Blue** = Rule Engine · **Red** = Low Confidence / Unclassified")
    render_boq_governance_grid(result_df)

    # ── Download ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📥 Export Results")

    def _to_csv_bytes(df: pd.DataFrame) -> bytes:
        return df.to_csv(index=False).encode("utf-8-sig")

    def _to_excel_bytes(df: pd.DataFrame) -> bytes:
        import io
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="BoQ Predictions")
        return buf.getvalue()

    dl1, dl2, _ = st.columns([1, 1, 4])
    stem = Path(st.session_state["ingestion_filename"]).stem
    dl1.download_button(
        label="⬇️  Download CSV",
        data=_to_csv_bytes(result_df),
        file_name=f"{stem}_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )
    dl2.download_button(
        label="⬇️  Download Excel",
        data=_to_excel_bytes(result_df),
        file_name=f"{stem}_predictions.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    # Model metadata footer
    with st.expander("🔬 Model metadata"):
        st.json({
            "model_version": result.get("model_version"),
            "model_name":    result.get("model_name"),
            "items_processed": total,
        })


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
