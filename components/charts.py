import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_cost_distribution_donut(df: pd.DataFrame):
    """
    Renders a McKinsey-style donut chart for NRM Level 1 Cost Distribution.
    """
    # Grouping by a high-level element for the visualization
    summary_df = df.groupby("Group_Element")["Total_Amount"].sum().reset_index()
    
    fig = px.pie(
        summary_df, 
        names="Group_Element", 
        values="Total_Amount", 
        hole=0.5,
        color_discrete_sequence=["#0B192C", "#1E3E62", "#478CCF", "#36C2CE", "#E2E8F0"]
    )
    
    fig.update_layout(
        margin=dict(t=20, b=20, l=20, r=20),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif")
    )
    return fig

def plot_nrf_fan_chart(commodity_name: str, shock_percentage: int):
    """
    Renders a Fan Chart showing historical rates and future NRF predictions 
    with confidence intervals, adjusted by a user-simulated market shock.
    """
    # Generate mock time-series data
    dates_hist = pd.date_range(start="2024-01-01", periods=12, freq="ME")
    dates_pred = pd.date_range(start="2025-01-01", periods=12, freq="ME")
    
    hist_rates = np.linspace(100, 120, 12) + np.random.normal(0, 2, 12)
    
    # Apply shock to the baseline prediction
    base_pred = 120 * (1 + (shock_percentage / 100.0))
    pred_rates = np.linspace(base_pred, base_pred * 1.05, 12)
    upper_bound = pred_rates * 1.15
    lower_bound = pred_rates * 0.85
    
    fig = go.Figure()

    # Historical Data
    fig.add_trace(go.Scatter(
        x=dates_hist, y=hist_rates,
        mode='lines+markers', name='Historical Rate',
        line=dict(color='#0B192C', width=3)
    ))

    # Predicted Data (Upper Bound)
    fig.add_trace(go.Scatter(
        x=dates_pred, y=upper_bound,
        mode='lines', line=dict(width=0),
        showlegend=False
    ))

    # Predicted Data (Lower Bound & Fill)
    fig.add_trace(go.Scatter(
        x=dates_pred, y=lower_bound,
        mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor='rgba(71, 140, 207, 0.2)',
        name='80% Confidence Interval'
    ))

    # Predicted Data (Mean)
    fig.add_trace(go.Scatter(
        x=dates_pred, y=pred_rates,
        mode='lines', name='NRF Prediction (Mean)',
        line=dict(color='#478CCF', width=3, dash='dash')
    ))

    fig.update_layout(
        title=f"Net Rate Forecast: {commodity_name} (Simulation: {shock_percentage}% Shock)",
        xaxis_title="Date",
        yaxis_title="Rate (SAR)",
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif")
    )
    return fig
