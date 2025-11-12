import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# ======== Load Data from Excel ========
df = pd.read_excel("sample1.xlsx")

# Automatically detect indicator columns (exclude 'Year' and 'SOFI' if present)
indicator_cols = [col for col in df.columns if col not in ["Year", "SOFI"]]

# Set equal weights for all indicators
weights = np.array([1.0 / len(indicator_cols)] * len(indicator_cols))

# print(df)

def compute_sofi(df, weights):
    values = df
    sofi = np.dot(values, weights)
    return sofi

df["SOFI"] = compute_sofi(df[indicator_cols], weights)

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
    df_adj = df.copy()

    # Only apply changes to years >= 2025 (future projections)
    future_years = df_adj[df_adj["Year"] >= 2025].index

    if growth_type == "linear":
        # Linear: Apply same percentage change to all future years
        df_adj.loc[future_years, indicator] *= (1 + change / 100)
    else:
        # Exponential: Apply compound growth year over year
        for i, idx in enumerate(future_years):
            # Each year compounds on the previous year's value
            df_adj.loc[idx, indicator] *= (1 + change / 100) ** (i + 1)

    # Recalculate SOFI for all years with adjusted values
    df_adj["SOFI"] = compute_sofi(df_adj[indicator_cols], weights)

    # Create subplots: 1 row, 2 columns
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"{indicator} Over Time", "SOFI Index Over Time"),
        horizontal_spacing=0.12
    )

    # LEFT SUBPLOT: Indicator
    # Baseline indicator (dashed)
    fig.add_trace(go.Scatter(
        x=df["Year"], y=df[indicator],
        mode="lines", name="Baseline",
        line={"color": "lightcoral", "dash": "dash"}
    ), row=1, col=1)

    # Adjusted indicator (solid)
    fig.add_trace(go.Scatter(
        x=df_adj["Year"], y=df_adj[indicator],
        mode="lines", name="Adjusted",
        line={"color": "red"}
    ), row=1, col=1)

    # RIGHT SUBPLOT: SOFI
    # Baseline SOFI (dashed)
    fig.add_trace(go.Scatter(
        x=df["Year"], y=df["SOFI"],
        mode="lines", name="Baseline",
        line={"color": "lightgreen", "dash": "dash"},
        showlegend=False
    ), row=1, col=2)

    # Adjusted SOFI (solid)
    fig.add_trace(go.Scatter(
        x=df_adj["Year"], y=df_adj["SOFI"],
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
