import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# ======== Load Data from Excel ========
# Sheet1: Normalized values (for SOFI calculation)
df_normalized = pd.read_excel("data1.xlsx", sheet_name="Sheet1")

# Sheet2: Original values (for display)
df_original = pd.read_excel("data1.xlsx", sheet_name="Sheet2")

# Automatically detect indicator columns (exclude 'Year' and 'SOFI' if present)
indicator_cols = [col for col in df_normalized.columns if col not in ["Year", "SOFI"]]

# Set equal weights for all indicators
weights = np.array([1.0 / len(indicator_cols)] * len(indicator_cols))

def compute_sofi(df, weights):
    values = df
    sofi = np.dot(values, weights)
    return sofi

# Calculate SOFI using normalized values
df_normalized["SOFI"] = compute_sofi(df_normalized[indicator_cols], weights)

# ======== Dash App ========
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H3("SOFI Interactive What-If Simulator"),

    html.Label("Adjust Indicator Value (%)"),
    dcc.Slider(-20, 20, 1, value=0, id="adjust-slider"),

    html.Label("Select Indicator"),
    dcc.Dropdown(
        options=[{"label": i, "value": i} for i in indicator_cols],
        value="Income Inequality",
        id="indicator-dropdown"
    ),

    html.Label("Growth Type"),
    dcc.RadioItems(
        options=[
            {"label": "Linear (one-time change)", "value": "linear"},
            {"label": "Exponential (compound growth)", "value": "exponential"}
        ],
        value="linear",
        id="growth-type"
    ),

    dcc.Graph(id="sofi-graph")
])

@app.callback(
    Output("sofi-graph", "figure"),
    Input("adjust-slider", "value"),
    Input("indicator-dropdown", "value"),
    Input("growth-type", "value")
)
def update_graph(change, indicator, growth_type):
    # Work with normalized data for SOFI calculation
    df_norm_adj = df_normalized.copy()
    # Work with original data for display
    df_orig_adj = df_original.copy()

    # Only apply changes to years >= 2025 (future projections)
    future_years = df_norm_adj[df_norm_adj["Year"] >= 2025].index

    if growth_type == "linear":
        # Linear: Apply same percentage change to all future years
        df_norm_adj.loc[future_years, indicator] *= (1 + change / 100)
        df_orig_adj.loc[future_years, indicator] *= (1 + change / 100)
    else:
        # Exponential: Apply compound growth year over year
        for i, idx in enumerate(future_years):
            # Each year compounds on the previous year's value
            df_norm_adj.loc[idx, indicator] *= (1 + change / 100) ** (i + 1)
            df_orig_adj.loc[idx, indicator] *= (1 + change / 100) ** (i + 1)

    # Recalculate SOFI using normalized adjusted values
    df_norm_adj["SOFI"] = compute_sofi(df_norm_adj[indicator_cols], weights)

    # Create subplots: 1 row, 2 columns
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"{indicator} Over Time", "SOFI Index Over Time"),
        horizontal_spacing=0.12
    )

    # LEFT SUBPLOT: Indicator (using ORIGINAL values for display)
    # Baseline indicator (dashed)
    fig.add_trace(go.Scatter(
        x=df_original["Year"], y=df_original[indicator],
        mode="lines", name="Baseline",
        line={"color": "lightcoral", "dash": "dash"}
    ), row=1, col=1)

    # Adjusted indicator (solid)
    fig.add_trace(go.Scatter(
        x=df_orig_adj["Year"], y=df_orig_adj[indicator],
        mode="lines", name="Adjusted",
        line={"color": "red"}
    ), row=1, col=1)

    # RIGHT SUBPLOT: SOFI (calculated from normalized values)
    # Baseline SOFI (dashed)
    fig.add_trace(go.Scatter(
        x=df_normalized["Year"], y=df_normalized["SOFI"],
        mode="lines", name="Baseline",
        line={"color": "lightgreen", "dash": "dash"},
        showlegend=False
    ), row=1, col=2)

    # Adjusted SOFI (solid)
    fig.add_trace(go.Scatter(
        x=df_norm_adj["Year"], y=df_norm_adj["SOFI"],
        mode="lines", name="Adjusted",
        line={"color": "green"},
        showlegend=False
    ), row=1, col=2)

    # Add vertical lines at 2025 for both subplots
    fig.add_vline(x=2025, line_width=2, line_dash="dot", line_color="gray",
                  row=1, col=1)
    fig.add_vline(x=2025, line_width=2, line_dash="dot", line_color="gray",
                  row=1, col=2, annotation_text="2025")

    # Update layout
    fig.update_xaxes(title_text="Year", row=1, col=1)
    fig.update_xaxes(title_text="Year", row=1, col=2)
    fig.update_yaxes(title_text=indicator, row=1, col=1)
    fig.update_yaxes(title_text="SOFI Index", row=1, col=2)

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        height=500
    )
    return fig

if __name__ == "__main__":
    app.run(debug=True)
