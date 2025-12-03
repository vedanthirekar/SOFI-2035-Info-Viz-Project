"""SOFI Interactive What-If Simulator with multi-variable support."""
import dash
from dash import dcc, html, Input, Output, State, ALL, MATCH, ctx
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
app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    html.H2("SOFI Interactive What-If Simulator - Multi-Variable"),

    html.Div([
        html.H4("Add Indicators to Adjust"),

        # Dropdown to select indicator
        html.Div([
            dcc.Dropdown(
                id="indicator-selector",
                options=[{"label": ind, "value": ind} for ind in indicator_cols],
                placeholder="Select an indicator to add...",
                style={"width": "400px", "marginRight": "10px"}
            ),
            html.Button("Add Indicator", id="add-indicator-btn", n_clicks=0,
                       style={"padding": "8px 16px"})
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "20px"}),

        # Container for active indicator inputs
        html.Div(id="active-indicators-container", children=[],
                style={"marginBottom": "20px"}),

        # Action buttons
        html.Div([
            html.Button("Clear All", id="clear-all-btn", n_clicks=0,
                       style={"marginRight": "10px", "padding": "8px 16px"}),
            html.Button("Apply Changes", id="apply-button", n_clicks=0,
                       style={"padding": "8px 16px", "backgroundColor": "#4CAF50",
                              "color": "white", "border": "none"})
        ], style={"marginBottom": "20px"}),

        # Growth type selector
        html.Label("Growth Type", style={"fontWeight": "bold"}),
        dcc.RadioItems(
            options=[
                {"label": "One-time change (apply % once to all future years)", "value": "linear"},
                {"label": "Compound growth (% compounds year-over-year)", "value": "exponential"},
                {"label": "Annual rate (% growth/decline per year from 2025)", "value": "annual_rate"}
            ],
            value="linear",
            id="growth-type",
            style={"marginBottom": "20px"}
        ),
    ], style={"padding": "20px", "backgroundColor": "#f5f5f5",
              "borderRadius": "5px", "marginBottom": "20px"}),

    # Store to track active indicators
    dcc.Store(id="active-indicators-store", data=[]),

    dcc.Graph(id="sofi-graph")
], style={"padding": "20px"})

# Callback to add indicator
@app.callback(
    Output("active-indicators-store", "data"),
    Output("indicator-selector", "value"),
    Input("add-indicator-btn", "n_clicks"),
    Input("clear-all-btn", "n_clicks"),
    Input({"type": "remove-indicator-btn", "index": ALL}, "n_clicks"),
    State("indicator-selector", "value"),
    State("active-indicators-store", "data"),
    prevent_initial_call=True
)
def manage_indicators(add_clicks, clear_clicks, remove_clicks, selected, active):
    """Add or remove indicators from the active list."""
    triggered_id = ctx.triggered_id

    if triggered_id == "clear-all-btn":
        return [], None

    if triggered_id == "add-indicator-btn" and selected and selected not in active:
        return active + [selected], None

    if isinstance(triggered_id, dict) and triggered_id.get("type") == "remove-indicator-btn":
        indicator_to_remove = triggered_id["index"]
        return [ind for ind in active if ind != indicator_to_remove], None

    return active, None


# Callback to update the UI for active indicators
@app.callback(
    Output("active-indicators-container", "children"),
    Input("active-indicators-store", "data")
)
def update_active_indicators_ui(active_indicators):
    """Generate UI elements for active indicators."""
    if not active_indicators:
        return html.Div("No indicators added yet. Select one from the dropdown above.",
                       style={"color": "#666", "fontStyle": "italic"})

    indicator_divs = []
    for ind in active_indicators:
        indicator_divs.append(
            html.Div([
                html.Label(ind, style={"fontWeight": "bold", "marginRight": "10px",
                                      "minWidth": "250px"}),
                dcc.Input(
                    id={"type": "indicator-input", "index": ind},
                    type="number",
                    value=0,
                    step=0.5,
                    style={"width": "80px", "marginRight": "5px"}
                ),
                html.Span("%", style={"marginRight": "10px"}),
                html.Button("×", id={"type": "remove-indicator-btn", "index": ind},
                           style={"padding": "2px 8px", "backgroundColor": "#ff4444",
                                  "color": "white", "border": "none",
                                  "borderRadius": "3px", "cursor": "pointer"})
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"})
        )

    return indicator_divs


# Main callback to update graph
@app.callback(
    Output("sofi-graph", "figure"),
    Input("apply-button", "n_clicks"),
    State({"type": "indicator-input", "index": ALL}, "value"),
    State({"type": "indicator-input", "index": ALL}, "id"),
    State("growth-type", "value")
)
def update_graph(apply_clicks, input_values, input_ids, growth_type):
    """Update graph based on multiple indicator adjustments."""
    df_norm_adj = df_normalized.copy()
    df_orig_adj = df_original.copy()

    future_years = df_norm_adj[df_norm_adj["Year"] >= 2025].index

    # Apply changes for each indicator
    changes_applied = {}
    for i, input_id in enumerate(input_ids):
        ind_name = input_id["index"]
        change = input_values[i] if input_values[i] is not None else 0

        # Track all indicators (even with 0 change)
        changes_applied[ind_name] = change

        if change != 0:
            if growth_type == "linear":
                # One-time change: apply same % to all future years
                df_norm_adj.loc[future_years, ind_name] *= (1 + change / 100)
                df_orig_adj.loc[future_years, ind_name] *= (1 + change / 100)
            elif growth_type == "exponential":
                # Compound growth: % compounds year-over-year
                for j, idx in enumerate(future_years):
                    df_norm_adj.loc[idx, ind_name] *= (1 + change / 100) ** (j + 1)
                    df_orig_adj.loc[idx, ind_name] *= (1 + change / 100) ** (j + 1)
            else:  # annual_rate
                # Annual rate: continuous linear growth/decline from 2025
                # Get the 2024 value (last historical point)
                year_2024_idx = df_norm_adj[df_norm_adj["Year"] == 2024].index
                if len(year_2024_idx) > 0:
                    base_value_norm = df_norm_adj.loc[year_2024_idx[0], ind_name]
                    base_value_orig = df_orig_adj.loc[year_2024_idx[0], ind_name]
                else:
                    # If 2024 doesn't exist, use first future year as base
                    base_value_norm = df_norm_adj.loc[future_years[0], ind_name]
                    base_value_orig = df_orig_adj.loc[future_years[0], ind_name]
                
                # Apply linear rate: each year adds the % to the previous year
                for j, idx in enumerate(future_years):
                    years_from_start = j + 1
                    df_norm_adj.loc[idx, ind_name] = base_value_norm * (1 + (change / 100) * years_from_start)
                    df_orig_adj.loc[idx, ind_name] = base_value_orig * (1 + (change / 100) * years_from_start)

    # Recalculate SOFI
    df_norm_adj["SOFI"] = compute_sofi(df_norm_adj[indicator_cols], weights)

    # Show all added indicators (including those with 0 change)
    indicators_to_show = list(changes_applied.keys())
    all_plots = indicators_to_show + ["SOFI"]
    total_plots = len(all_plots)

    # Calculate grid layout: max 4 columns per row
    max_cols = 4
    num_rows = (total_plots + max_cols - 1) // max_cols
    num_cols = min(total_plots, max_cols)

    # Create subplot titles with positions
    subplot_titles = []
    for ind in indicators_to_show:
        subplot_titles.append(f"{ind} ({changes_applied.get(ind, 0):+.1f}%)")
    subplot_titles.append("SOFI Index")

    # Create subplots with multiple rows
    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.15
    )

    # Add traces for each modified indicator
    colors = ["red", "blue", "purple", "orange", "brown", "pink", "teal", "magenta"]
    for idx, ind_name in enumerate(indicators_to_show):
        row_idx = (idx // max_cols) + 1
        col_idx = (idx % max_cols) + 1
        color = colors[idx % len(colors)]

        fig.add_trace(go.Scatter(
            x=df_original["Year"], y=df_original[ind_name],
            mode="lines", name="Baseline",
            line={"color": color, "dash": "dash", "width": 1.5},
            legendgroup="baseline",
            showlegend=(idx == 0)
        ), row=row_idx, col=col_idx)

        fig.add_trace(go.Scatter(
            x=df_orig_adj["Year"], y=df_orig_adj[ind_name],
            mode="lines", name="Adjusted",
            line={"color": color, "width": 2},
            legendgroup="adjusted",
            showlegend=(idx == 0)
        ), row=row_idx, col=col_idx)

        fig.update_yaxes(title_text=ind_name, row=row_idx, col=col_idx)
        fig.add_vline(x=2025, line_width=1, line_dash="dot", line_color="gray",
                     row=row_idx, col=col_idx)

    # SOFI subplot (last position)
    sofi_idx = len(indicators_to_show)
    sofi_row = (sofi_idx // max_cols) + 1
    sofi_col = (sofi_idx % max_cols) + 1

    fig.add_trace(go.Scatter(
        x=df_normalized["Year"], y=df_normalized["SOFI"],
        mode="lines", name="Baseline",
        line={"color": "green", "dash": "dash", "width": 2},
        legendgroup="baseline",
        showlegend=False
    ), row=sofi_row, col=sofi_col)

    fig.add_trace(go.Scatter(
        x=df_norm_adj["Year"], y=df_norm_adj["SOFI"],
        mode="lines", name="Adjusted",
        line={"color": "green", "width": 3},
        legendgroup="adjusted",
        showlegend=False
    ), row=sofi_row, col=sofi_col)

    fig.add_vline(x=2025, line_width=1, line_dash="dot", line_color="gray",
                  row=sofi_row, col=sofi_col, annotation_text="2025")

    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="SOFI Index", row=sofi_row, col=sofi_col)

    # Adjust height based on number of rows
    plot_height = 400 * num_rows

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        height=plot_height,
        showlegend=True,
        legend={"orientation": "h", "yanchor": "top", "y": -0.2,
                "xanchor": "center", "x": 0.5},
        margin={"b": 80}
    )

    return fig

if __name__ == "__main__":
    app.run(debug=True)
