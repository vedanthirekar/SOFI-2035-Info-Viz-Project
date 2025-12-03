"""SOFI Interactive What-If Simulator with multi-variable support."""
import dash
from dash import dcc, html, Input, Output, State, ALL
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

def compute_sofi(df, indicator_weights):
    """Calculate SOFI index from indicator values and weights."""
    values = df
    sofi = np.dot(values, indicator_weights)
    return sofi

# Calculate SOFI using normalized values
df_normalized["SOFI"] = compute_sofi(df_normalized[indicator_cols], weights)

# ======== Dash App ========
app = dash.Dash(__name__)

# Create input fields for all indicators
indicator_inputs = []
for indicator in indicator_cols:
    indicator_inputs.append(
        html.Div([
            html.Label(indicator, style={"fontWeight": "bold", "marginRight": "10px", "minWidth": "200px"}),
            dcc.Input(
                id={"type": "indicator-input", "index": indicator},
                type="number",
                value=0,
                step=0.5,
                style={"width": "80px", "marginRight": "5px"}
            ),
            html.Span("%", style={"marginRight": "10px"}),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"})
    )

app.layout = html.Div([
    html.H2("SOFI Interactive What-If Simulator - Multi-Variable"),
    
    html.Div([
        html.H4("Adjust Multiple Indicators (% change for years >= 2025)"),
        html.Div(indicator_inputs, style={"marginBottom": "20px"}),
        
        html.Div([
            html.Button("Reset All", id="reset-button", n_clicks=0, 
                       style={"marginRight": "10px", "padding": "8px 16px"}),
            html.Button("Apply Changes", id="apply-button", n_clicks=0,
                       style={"padding": "8px 16px", "backgroundColor": "#4CAF50", "color": "white", "border": "none"})
        ], style={"marginBottom": "20px"}),
        
        html.Label("Growth Type", style={"fontWeight": "bold"}),
        dcc.RadioItems(
            options=[
                {"label": "Linear (one-time change)", "value": "linear"},
                {"label": "Exponential (compound growth)", "value": "exponential"}
            ],
            value="linear",
            id="growth-type",
            style={"marginBottom": "20px"}
        ),
    ], style={"padding": "20px", "backgroundColor": "#f5f5f5", "borderRadius": "5px", "marginBottom": "20px"}),

    dcc.Graph(id="sofi-graph")
], style={"padding": "20px"})

# Callback to reset all inputs
@app.callback(
    Output({"type": "indicator-input", "index": ALL}, "value"),
    Input("reset-button", "n_clicks"),
    prevent_initial_call=True
)
def reset_inputs(n_clicks):
    """Reset all indicator inputs to 0."""
    return [0] * len(indicator_cols)


# Main callback to update graph
@app.callback(
    Output("sofi-graph", "figure"),
    Input("apply-button", "n_clicks"),
    State({"type": "indicator-input", "index": ALL}, "value"),
    State({"type": "indicator-input", "index": ALL}, "id"),
    State("growth-type", "value")
)
def update_graph(n_clicks, input_values, input_ids, growth_type):
    """Update graph based on multiple indicator adjustments."""
    # Work with normalized data for SOFI calculation
    df_norm_adj = df_normalized.copy()
    # Work with original data for display
    df_orig_adj = df_original.copy()

    # Only apply changes to years >= 2025 (future projections)
    future_years = df_norm_adj[df_norm_adj["Year"] >= 2025].index

    # Apply changes for each indicator
    changes_applied = {}
    for i, input_id in enumerate(input_ids):
        indicator = input_id["index"]
        change = input_values[i] if input_values[i] is not None else 0
        
        if change != 0:
            changes_applied[indicator] = change
            
            if growth_type == "linear":
                # Linear: Apply same percentage change to all future years
                df_norm_adj.loc[future_years, indicator] *= (1 + change / 100)
                df_orig_adj.loc[future_years, indicator] *= (1 + change / 100)
            else:
                # Exponential: Apply compound growth year over year
                for j, idx in enumerate(future_years):
                    # Each year compounds on the previous year's value
                    df_norm_adj.loc[idx, indicator] *= (1 + change / 100) ** (j + 1)
                    df_orig_adj.loc[idx, indicator] *= (1 + change / 100) ** (j + 1)

    # Recalculate SOFI using normalized adjusted values
    df_norm_adj["SOFI"] = compute_sofi(df_norm_adj[indicator_cols], weights)

    # Determine which indicators to show in detail (those with changes)
    indicators_to_show = list(changes_applied.keys()) if changes_applied else [indicator_cols[0]]
    num_indicators = len(indicators_to_show)
    
    # Create subplots: indicators + SOFI
    fig = make_subplots(
        rows=1, cols=num_indicators + 1,
        subplot_titles=[f"{ind} ({changes_applied.get(ind, 0):+.1f}%)" 
                       for ind in indicators_to_show] + ["SOFI Index"],
        horizontal_spacing=0.08
    )

    # Add traces for each modified indicator
    colors = ["red", "blue", "purple", "orange", "brown", "pink"]
    for idx, indicator in enumerate(indicators_to_show):
        col_idx = idx + 1
        color = colors[idx % len(colors)]
        
        # Baseline indicator (dashed)
        fig.add_trace(go.Scatter(
            x=df_original["Year"], y=df_original[indicator],
            mode="lines", name=f"{indicator} (Baseline)",
            line={"color": color, "dash": "dash", "width": 1.5},
            showlegend=(idx == 0)
        ), row=1, col=col_idx)

        # Adjusted indicator (solid)
        fig.add_trace(go.Scatter(
            x=df_orig_adj["Year"], y=df_orig_adj[indicator],
            mode="lines", name=f"{indicator} (Adjusted)",
            line={"color": color, "width": 2},
            showlegend=(idx == 0)
        ), row=1, col=col_idx)
        
        fig.update_yaxes(title_text=indicator, row=1, col=col_idx)
        fig.add_vline(x=2025, line_width=1, line_dash="dot", line_color="gray",
                     row=1, col=col_idx)

    # SOFI subplot (rightmost)
    sofi_col = num_indicators + 1
    
    # Baseline SOFI (dashed)
    fig.add_trace(go.Scatter(
        x=df_normalized["Year"], y=df_normalized["SOFI"],
        mode="lines", name="SOFI (Baseline)",
        line={"color": "lightgreen", "dash": "dash", "width": 2}
    ), row=1, col=sofi_col)

    # Adjusted SOFI (solid)
    fig.add_trace(go.Scatter(
        x=df_norm_adj["Year"], y=df_norm_adj["SOFI"],
        mode="lines", name="SOFI (Adjusted)",
        line={"color": "green", "width": 3}
    ), row=1, col=sofi_col)

    fig.add_vline(x=2025, line_width=1, line_dash="dot", line_color="gray",
                  row=1, col=sofi_col, annotation_text="2025")

    # Update all x-axes
    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="SOFI Index", row=1, col=sofi_col)

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        height=500,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    
    return fig

if __name__ == "__main__":
    app.run(debug=True)
