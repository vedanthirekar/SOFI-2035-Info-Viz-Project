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

# Sheet3: Weights for each indicator
df_weights = pd.read_excel("data1.xlsx", sheet_name="Sheet3")

# Automatically detect indicator columns (exclude 'Year' and 'SOFI' if present)
indicator_cols = [col for col in df_normalized.columns if col not in ["Year", "SOFI"]]

# Negative indicators: lower values are better (normalized values are inverted)
# When user increases these in What-If, we need to DECREASE the normalized values
NEGATIVE_INDICATORS = [
    "Income Inequality",
    "Unemployment",
    "Poverty",
    "Population growth",
    "Mortality rate, infant",
    "Undernourishment",
    "CO2 emissions",
    "Energy-Efficiency",
    "Wars",
    "Terrorism Incidents",
    "Refugees"
]

# Load weights from Sheet3
# Drop any rows with NaN values and reset index
df_weights_clean = df_weights.dropna().reset_index(drop=True)

# Debug: print first few rows
print("Sheet3 data:")
print(df_weights_clean.head())
print("\nIndicator columns:", indicator_cols)

# Create weights dictionary from first two columns
if len(df_weights_clean.columns) >= 2:
    weights_dict = dict(zip(df_weights_clean.iloc[:, 0], df_weights_clean.iloc[:, 1]))
else:
    # Fallback to equal weights if Sheet3 structure is unexpected
    print("Warning: Sheet3 doesn't have expected structure, using equal weights")
    weights_dict = {}

# Map weights to indicator columns
weights_raw = np.array([weights_dict.get(col, 1.0 / len(indicator_cols)) for col in indicator_cols])

# Normalize weights so they sum to 1 (as per SOFI formula)
weights_sum = np.sum(weights_raw)
weights = weights_raw / weights_sum if weights_sum > 0 else weights_raw

print("\nRaw weights:", weights_raw)
print("Sum of raw weights:", weights_sum)
print("Normalized weights:", weights)
print("Sum of normalized weights:", np.sum(weights))

def compute_sofi(df, indicator_weights):
    """Calculate SOFI index from indicator values and weights."""
    values = df
    sofi = np.dot(values, indicator_weights)
    return sofi


# Calculate SOFI using normalized values
df_normalized["SOFI"] = compute_sofi(df_normalized[indicator_cols], weights)

# ======== Dash App ========
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Custom CSS styles
COLORS = {
    "primary": "#2C3E50",
    "secondary": "#3498DB",
    "success": "#27AE60",
    "danger": "#E74C3C",
    "warning": "#F39C12",
    "light": "#ECF0F1",
    "dark": "#34495E",
    "white": "#FFFFFF",
    "background": "#F8F9FA",
    "border": "#DEE2E6"
}

app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.H1("SOFI Dashboard", 
                   style={"color": COLORS["white"], "margin": "0", "fontSize": "2.5rem",
                          "fontWeight": "600", "letterSpacing": "0.5px"}),
            html.P("State of the Future Index - Interactive Analysis Platform",
                  style={"color": COLORS["light"], "margin": "10px 0 0 0", 
                         "fontSize": "1.1rem", "fontWeight": "300"})
        ], style={"maxWidth": "1400px", "margin": "0 auto", "padding": "0 20px"})
    ], style={"background": f"linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['dark']} 100%)",
              "padding": "40px 20px", "marginBottom": "0", "boxShadow": "0 2px 10px rgba(0,0,0,0.1)"}),

    # Tabs
    html.Div([
        dcc.Tabs(id="tabs", value="tab-trends", 
                children=[
                    dcc.Tab(label="Historical Trends", value="tab-trends",
                           style={"padding": "12px 24px", "fontWeight": "500"},
                           selected_style={"padding": "12px 24px", "fontWeight": "600",
                                         "borderTop": f"3px solid {COLORS['success']}",
                                         "backgroundColor": COLORS["white"]}),
                    dcc.Tab(label="What-If Simulator", value="tab-whatif",
                           style={"padding": "12px 24px", "fontWeight": "500"},
                           selected_style={"padding": "12px 24px", "fontWeight": "600",
                                         "borderTop": f"3px solid {COLORS['success']}",
                                         "backgroundColor": COLORS["white"]}),
                    dcc.Tab(label="Correlations", value="tab-correlations",
                           style={"padding": "12px 24px", "fontWeight": "500"},
                           selected_style={"padding": "12px 24px", "fontWeight": "600",
                                         "borderTop": f"3px solid {COLORS['success']}",
                                         "backgroundColor": COLORS["white"]}),
                    dcc.Tab(label="Indicator Analysis", value="tab-analysis",
                           style={"padding": "12px 24px", "fontWeight": "500"},
                           selected_style={"padding": "12px 24px", "fontWeight": "600",
                                         "borderTop": f"3px solid {COLORS['success']}",
                                         "backgroundColor": COLORS["white"]}),
                ],
                style={"maxWidth": "1400px", "margin": "0 auto"}),
    ], style={"backgroundColor": COLORS["white"], "boxShadow": "0 2px 4px rgba(0,0,0,0.05)"}),

    # Content
    html.Div(id="tab-content", style={"maxWidth": "1400px", "margin": "0 auto", "padding": "30px 20px"})
], style={"backgroundColor": COLORS["background"], "minHeight": "100vh", "fontFamily": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"})


# Callback to render tab content
@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value")
)
def render_tab_content(active_tab):
    """Render content based on selected tab."""
    if active_tab == "tab-whatif":
        return html.Div([
            # Control Panel Card
            html.Div([
                html.Div([
                    # Left side - Indicator selection
                    html.Div([
                        html.H4("Add Indicators to Adjust", 
                               style={"color": COLORS["primary"], "marginBottom": "15px", 
                                      "fontSize": "1.1rem", "fontWeight": "600"}),

                        # Dropdown to select indicator
                        html.Div([
                            dcc.Dropdown(
                                id="indicator-selector",
                                options=[{"label": ind, "value": ind} for ind in indicator_cols],
                                placeholder="Select an indicator to add...",
                                style={"width": "400px", "marginRight": "10px"}
                            ),
                            html.Button("Add Indicator", id="add-indicator-btn", n_clicks=0,
                                       style={"padding": "10px 20px", "backgroundColor": COLORS["secondary"],
                                              "color": COLORS["white"], "border": "none", 
                                              "borderRadius": "5px", "cursor": "pointer",
                                              "fontWeight": "500", "transition": "all 0.3s"})
                        ], style={"display": "flex", "alignItems": "center", "marginBottom": "20px"}),

                        # Container for active indicator inputs
                        html.Div(id="active-indicators-container", children=[],
                                style={"marginBottom": "20px"}),
                    ], style={"flex": "1", "marginRight": "40px"}),

                    # Right side - Growth type and buttons
                    html.Div([
                        html.Div([
                            html.Label("Growth Type", 
                                      style={"fontWeight": "600", "marginRight": "8px", 
                                             "color": COLORS["primary"], "fontSize": "1rem"}),
                            html.Span("ⓘ",
                                     title="One-time change: Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
                                           "Sed do eiusmod tempor incididunt ut labore.\n\n"
                                           "Compound growth: Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
                                           "Ut enim ad minim veniam, quis nostrud exercitation.\n\n"
                                           "Annual rate: Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
                                           "Duis aute irure dolor in reprehenderit in voluptate.",
                                     style={"cursor": "help", "color": COLORS["secondary"],
                                            "fontSize": "16px", "fontWeight": "bold",
                                            "marginLeft": "5px"}),
                        ], style={"display": "flex", "alignItems": "center", "marginBottom": "12px"}),
                        dcc.RadioItems(
                            options=[
                                {"label": "One-time change (apply % once to all future years)", "value": "linear"},
                                {"label": "Compound growth (% compounds year-over-year)", "value": "exponential"},
                                {"label": "Annual rate (% growth/decline per year from 2025)", "value": "annual_rate"}
                            ],
                            value="linear",
                            id="growth-type",
                            style={"marginBottom": "30px", "lineHeight": "1.8"}
                        ),

                        # Action buttons
                        html.Div([
                            html.Button("Clear All", id="clear-all-btn", n_clicks=0,
                                       style={"marginRight": "10px", "padding": "10px 20px",
                                              "width": "120px", "backgroundColor": COLORS["white"],
                                              "color": COLORS["danger"], "border": f"2px solid {COLORS['danger']}",
                                              "borderRadius": "5px", "cursor": "pointer",
                                              "fontWeight": "500", "transition": "all 0.3s"}),
                            html.Button("Apply Changes", id="apply-button", n_clicks=0,
                                       style={"padding": "10px 20px", "backgroundColor": COLORS["success"],
                                              "color": COLORS["white"], "border": "none", 
                                              "width": "140px", "borderRadius": "5px",
                                              "cursor": "pointer", "fontWeight": "500",
                                              "transition": "all 0.3s", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"})
                        ]),
                    ], style={"minWidth": "320px"}),
                ], style={"display": "flex"}),
            ], style={"padding": "25px", "backgroundColor": COLORS["white"],
                      "borderRadius": "8px", "marginBottom": "25px",
                      "boxShadow": "0 2px 8px rgba(0,0,0,0.08)", "border": f"1px solid {COLORS['border']}"}),

            # Store to track active indicators
            dcc.Store(id="active-indicators-store", data=[]),

            # Graph Card
            html.Div([
                dcc.Graph(id="sofi-graph", config={"displayModeBar": True, "displaylogo": False})
            ], style={"backgroundColor": COLORS["white"], "borderRadius": "8px",
                     "padding": "20px", "boxShadow": "0 2px 8px rgba(0,0,0,0.08)",
                     "border": f"1px solid {COLORS['border']}"})
        ])

    elif active_tab == "tab-trends":
        # SOFI over time chart
        fig_sofi = go.Figure()
        fig_sofi.add_trace(go.Scatter(
            x=df_normalized["Year"],
            y=df_normalized["SOFI"],
            mode="lines+markers",
            name="SOFI",
            line={"color": COLORS["success"], "width": 3},
            marker={"size": 6}
        ))
        fig_sofi.add_vline(x=2024, line_dash="dot", line_color=COLORS["danger"],
                          annotation_text="Projection Start")
        fig_sofi.update_layout(
            title={"text": "SOFI Index Over Time", "font": {"size": 18, "color": COLORS["primary"]}},
            xaxis_title="Year",
            yaxis_title="SOFI Index",
            template="plotly_white",
            height=400,
            margin={"t": 60}
        )

        # Year-over-year changes
        sofi_yoy = df_normalized["SOFI"].pct_change() * 100
        fig_yoy = go.Figure()
        fig_yoy.add_trace(go.Bar(
            x=df_normalized["Year"][1:],
            y=sofi_yoy[1:],
            marker_color=[COLORS["success"] if x > 0 else COLORS["danger"] for x in sofi_yoy[1:]],
            name="YoY Change"
        ))
        fig_yoy.update_layout(
            title={"text": "Year-over-Year SOFI Change (%)", "font": {"size": 18, "color": COLORS["primary"]}},
            xaxis_title="Year",
            yaxis_title="% Change",
            template="plotly_white",
            height=400,
            margin={"t": 60}
        )

        # Multi-indicator trends
        return html.Div([
            # SOFI Over Time Card
            html.Div([
                dcc.Graph(figure=fig_sofi, config={"displayModeBar": True, "displaylogo": False})
            ], style={"backgroundColor": COLORS["white"], "borderRadius": "8px",
                     "padding": "20px", "marginBottom": "25px",
                     "boxShadow": "0 2px 8px rgba(0,0,0,0.08)",
                     "border": f"1px solid {COLORS['border']}"}),
            
            # YoY Changes Card
            html.Div([
                dcc.Graph(figure=fig_yoy, config={"displayModeBar": True, "displaylogo": False})
            ], style={"backgroundColor": COLORS["white"], "borderRadius": "8px",
                     "padding": "20px", "marginBottom": "25px",
                     "boxShadow": "0 2px 8px rgba(0,0,0,0.08)",
                     "border": f"1px solid {COLORS['border']}"}),
            
            # Multi-Indicator Trends Card
            html.Div([
                html.H4("Individual Indicator Trends", 
                       style={"color": COLORS["primary"], "marginBottom": "20px",
                              "fontSize": "1.2rem", "fontWeight": "600"}),
                html.Div([
                    html.Div([
                        html.Label("Select Indicators to Display:", 
                                  style={"fontWeight": "500", "marginBottom": "8px",
                                         "display": "block", "color": COLORS["dark"]}),
                        dcc.Dropdown(
                            id="trends-indicator-selector",
                            options=[{"label": ind, "value": ind} for ind in indicator_cols],
                            value=indicator_cols[:3],
                            multi=True,
                            style={"width": "500px"}
                        ),
                    ], style={"marginRight": "30px"}),
                    html.Div([
                        html.Label("View:", 
                                  style={"fontWeight": "500", "marginBottom": "8px",
                                         "display": "block", "color": COLORS["dark"]}),
                        dcc.RadioItems(
                            id="trends-view-type",
                            options=[
                                {"label": "Normalized (0-1 scale)", "value": "normalized"},
                                {"label": "Original Values", "value": "original"}
                            ],
                            value="normalized",
                            inline=True,
                            style={"lineHeight": "1.8"}
                        ),
                    ]),
                ], style={"display": "flex", "alignItems": "flex-end", "marginBottom": "25px"}),
                dcc.Graph(id="trends-multi-indicator", config={"displayModeBar": True, "displaylogo": False})
            ], style={"backgroundColor": COLORS["white"], "borderRadius": "8px",
                     "padding": "25px", "boxShadow": "0 2px 8px rgba(0,0,0,0.08)",
                     "border": f"1px solid {COLORS['border']}"})
        ])

    elif active_tab == "tab-analysis":
        # Get latest year data
        latest_year = df_original["Year"].max()

        return html.Div([
            html.Div([
                html.H4("Compare Years", 
                       style={"color": COLORS["primary"], "marginBottom": "20px",
                              "fontSize": "1.2rem", "fontWeight": "600"}),
                html.Div([
                    html.Div([
                        html.Label("Select Year 1:", 
                                  style={"fontWeight": "500", "marginBottom": "8px",
                                         "display": "block", "color": COLORS["dark"]}),
                        dcc.Dropdown(
                            id="analysis-year1",
                            options=[{"label": str(int(y)), "value": y} for y in df_original["Year"]],
                            value=df_original["Year"].min(),
                            style={"width": "200px"}
                        ),
                    ], style={"marginRight": "30px"}),
                    html.Div([
                        html.Label("Select Year 2:", 
                                  style={"fontWeight": "500", "marginBottom": "8px",
                                         "display": "block", "color": COLORS["dark"]}),
                        dcc.Dropdown(
                            id="analysis-year2",
                            options=[{"label": str(int(y)), "value": y} for y in df_original["Year"]],
                            value=latest_year,
                            style={"width": "200px"}
                        ),
                    ], style={"marginRight": "30px"}),
                    html.Div([
                        html.Label("View:", 
                                  style={"fontWeight": "500", "marginBottom": "8px",
                                         "display": "block", "color": COLORS["dark"]}),
                        dcc.RadioItems(
                            id="analysis-view-type",
                            options=[
                                {"label": "Normalized", "value": "normalized"},
                                {"label": "Original", "value": "original"}
                            ],
                            value="normalized",
                            inline=True,
                            style={"lineHeight": "1.8"}
                        ),
                    ]),
                ], style={"display": "flex", "alignItems": "flex-end", "marginBottom": "25px"}),
                dcc.Graph(id="analysis-comparison", config={"displayModeBar": True, "displaylogo": False})
            ], style={"backgroundColor": COLORS["white"], "borderRadius": "8px",
                     "padding": "25px", "boxShadow": "0 2px 8px rgba(0,0,0,0.08)",
                     "border": f"1px solid {COLORS['border']}"})
        ])

    elif active_tab == "tab-correlations":
        # Correlation with SOFI
        # Calculate correlation using normalized data
        sofi_corr = df_normalized[indicator_cols + ["SOFI"]].corr()["SOFI"][:-1]
        
        # Invert correlation sign for negative indicators
        # (because they're inverted in Sheet1, but we want to show real-world correlation)
        sofi_corr_adjusted = sofi_corr.copy()
        for ind in NEGATIVE_INDICATORS:
            if ind in sofi_corr_adjusted.index:
                sofi_corr_adjusted[ind] = -sofi_corr_adjusted[ind]
        
        # Sort by correlation value
        sofi_corr_adjusted = sofi_corr_adjusted.sort_values(ascending=False)
        
        fig_sofi_corr = go.Figure(go.Bar(
            y=sofi_corr_adjusted.index,
            x=sofi_corr_adjusted.values,
            orientation="h",
            marker_color=[COLORS["success"] if x > 0 else COLORS["danger"] for x in sofi_corr_adjusted.values]
        ))
        fig_sofi_corr.update_layout(
            title={"text": "Correlation with SOFI Index", "font": {"size": 18, "color": COLORS["primary"]}},
            xaxis_title="Correlation Coefficient",
            yaxis_title="Indicator",
            template="plotly_white",
            height=800,
            margin={"t": 60}
        )

        return html.Div([
            # Scatter Plot Analysis Card
            html.Div([
                html.H4("Scatter Plot Analysis", 
                       style={"color": COLORS["primary"], "marginBottom": "20px",
                              "fontSize": "1.2rem", "fontWeight": "600"}),
                html.Div([
                    html.Div([
                        html.Label("X-Axis Indicator:", 
                                  style={"fontWeight": "500", "marginBottom": "8px",
                                         "display": "block", "color": COLORS["dark"]}),
                        dcc.Dropdown(
                            id="corr-x-indicator",
                            options=[{"label": ind, "value": ind} for ind in indicator_cols],
                            value=indicator_cols[0],
                            style={"width": "300px"}
                        ),
                    ], style={"marginRight": "30px"}),
                    html.Div([
                        html.Label("Y-Axis Indicator:", 
                                  style={"fontWeight": "500", "marginBottom": "8px",
                                         "display": "block", "color": COLORS["dark"]}),
                        dcc.Dropdown(
                            id="corr-y-indicator",
                            options=[{"label": ind, "value": ind} for ind in indicator_cols],
                            value=indicator_cols[1],
                            style={"width": "300px"}
                        ),
                    ]),
                ], style={"display": "flex", "marginBottom": "25px"}),
                dcc.Graph(id="corr-scatter", config={"displayModeBar": True, "displaylogo": False})
            ], style={"backgroundColor": COLORS["white"], "borderRadius": "8px",
                     "padding": "25px", "marginBottom": "25px",
                     "boxShadow": "0 2px 8px rgba(0,0,0,0.08)",
                     "border": f"1px solid {COLORS['border']}"}),
            
            # Correlation Bar Chart Card
            html.Div([
                dcc.Graph(figure=fig_sofi_corr, config={"displayModeBar": True, "displaylogo": False})
            ], style={"backgroundColor": COLORS["white"], "borderRadius": "8px",
                     "padding": "20px", "boxShadow": "0 2px 8px rgba(0,0,0,0.08)",
                     "border": f"1px solid {COLORS['border']}"})
        ])
    
    return html.Div()

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
    Input("active-indicators-store", "data"),
    State({"type": "indicator-input", "index": ALL}, "value"),
    State({"type": "indicator-input", "index": ALL}, "id")
)
def update_active_indicators_ui(active_indicators, current_values, current_ids):
    """Generate UI elements for active indicators."""
    if not active_indicators:
        return html.Div("No indicators added yet. Select one from the dropdown above.",
                       style={"color": "#6C757D", "fontStyle": "italic", "padding": "15px",
                              "backgroundColor": "#F8F9FA", "borderRadius": "5px",
                              "border": "1px dashed #DEE2E6"})

    # Create a map of existing values
    value_map = {}
    if current_ids and current_values:
        for i, input_id in enumerate(current_ids):
            value_map[input_id["index"]] = current_values[i]

    indicator_divs = []
    for ind in active_indicators:
        # Preserve existing value or default to 0
        existing_value = value_map.get(ind, 0)

        indicator_divs.append(
            html.Div([
                html.Label(ind, style={"fontWeight": "500", "marginRight": "10px",
                                      "minWidth": "250px", "color": COLORS["dark"]}),
                dcc.Input(
                    id={"type": "indicator-input", "index": ind},
                    type="number",
                    value=existing_value,
                    step=0.5,
                    style={"width": "80px", "marginRight": "5px", "padding": "6px 10px",
                           "border": f"1px solid {COLORS['border']}", "borderRadius": "4px"}
                ),
                html.Span("%", style={"marginRight": "10px", "color": COLORS["dark"],
                                     "fontWeight": "500"}),
                html.Button("×", id={"type": "remove-indicator-btn", "index": ind},
                           style={"padding": "4px 10px", "backgroundColor": COLORS["danger"],
                                  "color": COLORS["white"], "border": "none",
                                  "borderRadius": "4px", "cursor": "pointer",
                                  "fontWeight": "600", "fontSize": "14px",
                                  "transition": "all 0.3s"})
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "10px",
                     "padding": "8px", "backgroundColor": COLORS["light"],
                     "borderRadius": "5px", "border": f"1px solid {COLORS['border']}"})
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
            # For negative indicators, invert the change for normalized values
            # (user says "increase poverty by 10%" -> normalized value should decrease)
            is_negative = ind_name in NEGATIVE_INDICATORS
            change_norm = -change if is_negative else change
            change_orig = change  # Original values always follow user input
            
            if growth_type == "linear":
                # One-time change: apply same % to all future years
                df_norm_adj.loc[future_years, ind_name] *= (1 + change_norm / 100)
                df_orig_adj.loc[future_years, ind_name] *= (1 + change_orig / 100)
            elif growth_type == "exponential":
                # Compound growth: % compounds year-over-year
                for j, idx in enumerate(future_years):
                    df_norm_adj.loc[idx, ind_name] *= (1 + change_norm / 100) ** (j + 1)
                    df_orig_adj.loc[idx, ind_name] *= (1 + change_orig / 100) ** (j + 1)
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
                    df_norm_adj.loc[idx, ind_name] = base_value_norm * (1 + (change_norm / 100) * years_from_start)
                    df_orig_adj.loc[idx, ind_name] = base_value_orig * (1 + (change_orig / 100) * years_from_start)

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

    # Add traces for each modified indicator (all in red)
    for idx, ind_name in enumerate(indicators_to_show):
        row_idx = (idx // max_cols) + 1
        col_idx = (idx % max_cols) + 1

        fig.add_trace(go.Scatter(
            x=df_original["Year"], y=df_original[ind_name],
            mode="lines", name="Baseline",
            line={"color": "red", "dash": "dash", "width": 1.5},
            legendgroup="baseline",
            showlegend=(idx == 0)
        ), row=row_idx, col=col_idx)

        fig.add_trace(go.Scatter(
            x=df_orig_adj["Year"], y=df_orig_adj[ind_name],
            mode="lines", name="Adjusted",
            line={"color": "red", "width": 2},
            legendgroup="adjusted",
            showlegend=(idx == 0)
        ), row=row_idx, col=col_idx)

        fig.update_yaxes(title_text=ind_name, row=row_idx, col=col_idx)
        fig.add_vline(x=2024, line_width=1, line_dash="dot", line_color="gray",
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

    fig.add_vline(x=2024, line_width=1, line_dash="dot", line_color="gray",
                  row=sofi_row, col=sofi_col, annotation_text="2024")

    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="SOFI Index", row=sofi_row, col=sofi_col)

    # Adjust height based on number of rows
    plot_height = 400 * num_rows

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        height=plot_height,
        showlegend=True,
        legend={"orientation": "h", "yanchor": "top", "y": -0.05,
                "xanchor": "center", "x": 0.5, "bgcolor": "rgba(255,255,255,0.8)",
                "bordercolor": COLORS["border"], "borderwidth": 1},
        margin={"b": 100, "t": 50, "l": 60, "r": 40},
        font={"family": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
              "color": COLORS["dark"]}
    )

    return fig

# Callback for trends multi-indicator chart
@app.callback(
    Output("trends-multi-indicator", "figure"),
    Input("trends-indicator-selector", "value"),
    Input("trends-view-type", "value")
)
def update_trends_multi(selected_indicators, view_type):
    """Update multi-indicator trends chart."""
    if not selected_indicators:
        return go.Figure()
    
    # Choose data source based on view type
    df_source = df_normalized.copy() if view_type == "normalized" else df_original.copy()
    
    # For normalized view, invert negative indicators back to match original direction
    # (Sheet1 has them inverted for SOFI calculation, but for display we want original direction)
    if view_type == "normalized":
        for ind in NEGATIVE_INDICATORS:
            if ind in df_source.columns:
                # Invert: 1 - value (assuming 0-1 normalization)
                df_source[ind] = 1 - df_source[ind]
    
    fig = go.Figure()
    for ind in selected_indicators:
        fig.add_trace(go.Scatter(
            x=df_source["Year"],
            y=df_source[ind],
            mode="lines+markers",
            name=ind,
            line={"width": 2},
            marker={"size": 4}
        ))
    
    fig.add_vline(x=2024, line_dash="dot", line_color="gray",
                 annotation_text="Projection Start")
    
    y_title = "Normalized Value (0-1)" if view_type == "normalized" else "Original Value"
    fig.update_layout(
        title={"text": f"Selected Indicators Over Time ({view_type.title()} Values)",
               "font": {"size": 18, "color": COLORS["primary"]}},
        xaxis_title="Year",
        yaxis_title=y_title,
        template="plotly_white",
        height=500,
        hovermode="x unified",
        margin={"t": 60},
        font={"family": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
              "color": COLORS["dark"]}
    )
    return fig


# Callback for analysis year comparison
@app.callback(
    Output("analysis-comparison", "figure"),
    Input("analysis-year1", "value"),
    Input("analysis-year2", "value"),
    Input("analysis-view-type", "value")
)
def update_analysis_comparison(year1, year2, view_type):
    """Compare indicators between two years."""
    if year1 is None or year2 is None:
        return go.Figure()
    
    # Choose data source based on view type
    df_source = df_normalized.copy() if view_type == "normalized" else df_original.copy()
    
    # For normalized view, invert negative indicators back to match original direction
    if view_type == "normalized":
        for ind in NEGATIVE_INDICATORS:
            if ind in df_source.columns:
                # Invert: 1 - value (assuming 0-1 normalization)
                df_source[ind] = 1 - df_source[ind]
    
    data1 = df_source[df_source["Year"] == year1].iloc[0]
    data2 = df_source[df_source["Year"] == year2].iloc[0]
    
    values1 = [data1[col] for col in indicator_cols]
    values2 = [data2[col] for col in indicator_cols]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=indicator_cols,
        x=values1,
        name=f"Year {int(year1)}",
        orientation="h",
        marker_color="#6A5ACD",  # Green
        width=0.35  # Make bars fatter
    ))
    fig.add_trace(go.Bar(
        y=indicator_cols,
        x=values2,
        name=f"Year {int(year2)}",
        orientation="h",
        marker_color="#bc5090",  # Purple
        width=0.35  # Make bars fatter
    ))
    
    x_title = "Normalized Value (0-1)" if view_type == "normalized" else "Original Value"
    fig.update_layout(
        title={"text": f"Indicator Comparison: {int(year1)} vs {int(year2)} ({view_type.title()} Values)",
               "font": {"size": 18, "color": COLORS["primary"]}},
        xaxis_title=x_title,
        yaxis_title="Indicator",
        template="plotly_white",
        height=800,
        barmode="group",
        bargap=0.5,  # More space between indicator groups
        bargroupgap=0.3,  # Space between bars within a group
        margin={"t": 60},
        font={"family": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
              "color": COLORS["dark"]}
    )
    return fig


# Callback for correlation scatter plot
@app.callback(
    Output("corr-scatter", "figure"),
    Input("corr-x-indicator", "value"),
    Input("corr-y-indicator", "value")
)
def update_corr_scatter(x_ind, y_ind):
    """Update correlation scatter plot."""
    if x_ind is None or y_ind is None:
        return go.Figure()
    
    # Calculate correlation
    corr = df_normalized[x_ind].corr(df_normalized[y_ind])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_original[x_ind],
        y=df_original[y_ind],
        mode="markers",
        marker={"size": 10, "color": df_original["Year"], "colorscale": "Viridis",
                "showscale": True, "colorbar": {"title": "Year"}},
        text=[f"Year: {int(y)}" for y in df_original["Year"]],
        hovertemplate="%{text}<br>" + f"{x_ind}: %{{x}}<br>{y_ind}: %{{y}}<extra></extra>",
        showlegend=False
    ))
    
    # Add trend line
    z = np.polyfit(df_original[x_ind], df_original[y_ind], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df_original[x_ind].min(), df_original[x_ind].max(), 100)
    fig.add_trace(go.Scatter(
        x=x_trend,
        y=p(x_trend),
        mode="lines",
        name="Trend",
        line={"color": "red", "dash": "dash"}
    ))
    
    fig.update_layout(
        title={"text": f"Correlation: {x_ind} vs {y_ind} (r = {corr:.3f})",
               "font": {"size": 18, "color": COLORS["primary"]}},
        xaxis_title=x_ind,
        yaxis_title=y_ind,
        template="plotly_white",
        height=600,
        margin={"t": 60},
        font={"family": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
              "color": COLORS["dark"]},
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01,
                "bgcolor": "rgba(255,255,255,0.8)", "bordercolor": COLORS["border"], 
                "borderwidth": 1}
    )
    return fig


if __name__ == "__main__":
    app.run(debug=True)
