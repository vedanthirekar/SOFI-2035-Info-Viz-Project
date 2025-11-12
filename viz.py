import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
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

    fig = go.Figure()
    
    # Add baseline (original) data as dashed lines
    fig.add_trace(go.Scatter(x=df["Year"], y=df[indicator],
                             mode="lines", name=f"{indicator} (Baseline)", 
                             line={"color": "lightcoral", "dash": "dash"}))
    fig.add_trace(go.Scatter(x=df["Year"], y=df["SOFI"],
                             mode="lines", name="SOFI (Baseline)", 
                             line={"color": "lightgreen", "dash": "dash"}))
    
    # Add adjusted data as solid lines
    fig.add_trace(go.Scatter(x=df_adj["Year"], y=df_adj[indicator],
                             mode="lines", name=f"{indicator} (Adjusted)", 
                             line={"color": "red"}))
    fig.add_trace(go.Scatter(x=df_adj["Year"], y=df_adj["SOFI"],
                             mode="lines", name="SOFI (Adjusted)", 
                             line={"color": "green"}))
    
    # Add vertical line at 2025 to show where changes begin
    fig.add_vline(x=2025, line_width=2, line_dash="dot", line_color="gray",
                  annotation_text="2025 (Change Start)")
    
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Value / Index",
        template="plotly_white",
        hovermode="x unified"
    )
    return fig

if __name__ == "__main__":
    app.run(debug=True)
