import pandas as pd
import numpy as np
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, JsCode

def render_boq_governance_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renders an enterprise-grade, editable AgGrid for QS Human-in-the-Loop validation.
    
    Features:
    - Editable 'Predicted_Category' for manual QS overrides.
    - Conditional formatting based on 'Prediction_Source' and 'Confidence'.
    - Built-in filtering and pagination for massive BoQs.
    """
    gb = GridOptionsBuilder.from_dataframe(df)
    
    # Configure global grid options
    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=20)
    gb.configure_default_column(filterable=True, sortable=True, resizable=True)
    
    # Make the Predicted Category editable for the QS
    gb.configure_column("Predicted_Category", editable=True, 
                        cellEditor="agSelectCellEditor", 
                        cellEditorParams={"values": df["Predicted_Category"].unique().tolist()})
    
    # Freeze the description column so it stays visible while scrolling horizontally
    gb.configure_column("Description", pinned="left", width=400)
    
    # Inject JavaScript for Conditional Row Styling (The McKinsey Aesthetic)
    # Green = High Confidence ML, Blue = Expert Rule Engine, Light Coral = Low Confidence/Unclassified
    row_style_jscode = JsCode("""
    function(params) {
        if (params.data.Predicted_Category === 'Unclassified' || params.data.Confidence < 0.6) {
            return {
                'backgroundColor': '#FEE2E2', // Light Red
                'color': '#991B1B',
                'fontWeight': 'bold'
            };
        } else if (params.data.Prediction_Source === 'Rule_Engine') {
            return {
                'backgroundColor': '#E0F2FE', // Light Blue
                'color': '#075985'
            };
        } else {
            return {
                'backgroundColor': '#DCFCE7', // Light Green
                'color': '#166534'
            };
        }
    }
    """)
    gb.configure_grid_options(getRowStyle=row_style_jscode)
    
    grid_options = gb.build()

    # Render the grid
    grid_response = AgGrid(
        df,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        fit_columns_on_grid_load=False,
        theme="alpine", # Clean, professional theme
        allow_unsafe_jscode=True # Required for the conditional formatting
    )
    
    # Return the updated dataframe in case the user made manual edits
    return grid_response['data']
