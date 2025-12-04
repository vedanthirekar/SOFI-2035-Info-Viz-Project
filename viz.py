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

app.layout = html.Div([
    html.H1("SOFI Dashboard", style={"textAlign": "center", "marginBottom": "30px"}),

    # Tabs
    dcc.Tabs(id="tabs", value="tab-whatif", children=[
        dcc.Tab(label="What-If Simulator", value="tab-whatif"),
        dcc.Tab(label="Historical Trends", value="tab-trends"),
        dcc.Tab(label="Indicator Analysis", value="tab-analysis"),
        dcc.Tab(label="Correlations", value="tab-correlations"),
    ]),

    html.Div(id="tab-content")
], style={"padding": "20px"})


# Callback to render tab content
@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value")
)
def render_tab_content(active_tab):
    """Render content based on selected tab."""
    if active_tab == "tab-whatif":
        return html.Div([
            html.H2("What-If Simulator - Multi-Variable", style={"marginTop": "20px"}),

            html.Div([
                html.Div([
                    # Left side - Indicator selection
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
                    ], style={"flex": "1", "marginRight": "30px"}),

                    # Right side - Growth type and buttons
                    html.Div([
                        html.Label("Growth Type", style={"fontWeight": "bold", "marginBottom": "10px"}),
                        dcc.RadioItems(
                            options=[
                                {"label": "One-time change (apply % once to all future years)", "value": "linear"},
                                {"label": "Compound growth (% compounds year-over-year)", "value": "exponential"},
                                {"label": "Annual rate (% growth/decline per year from 2025)", "value": "annual_rate"}
                            ],
                            value="linear",
                            id="growth-type",
                            style={"marginBottom": "30px"}
                        ),

                        # Action buttons
                        html.Div([
                            html.Button("Clear All", id="clear-all-btn", n_clicks=0,
                                       style={"marginRight": "10px", "padding": "8px 16px",
                                              "width": "120px"}),
                            html.Button("Apply Changes", id="apply-button", n_clicks=0,
                                       style={"padding": "8px 16px", "backgroundColor": "#4CAF50",
                                              "color": "white", "border": "none", "width": "140px"})
                        ]),
                    ], style={"minWidth": "280px"}),
                ], style={"display": "flex"}),
            ], style={"padding": "20px", "backgroundColor": "#f5f5f5",
                      "borderRadius": "5px", "marginBottom": "20px"}),

            # Store to track active indicators
            dcc.Store(id="active-indicators-store", data=[]),

            dcc.Graph(id="sofi-graph")
        ])

    elif active_tab == "tab-trends":
        # SOFI over time chart
        fig_sofi = go.Figure()
        fig_sofi.add_trace(go.Scatter(
            x=df_normalized["Year"],
            y=df_normalized["SOFI"],
            mode="lines+markers",
            name="SOFI",
            line={"color": "green", "width": 3},
            marker={"size": 6}
        ))
        fig_sofi.add_vline(x=2025, line_dash="dot", line_color="red",
                          annotation_text="Projection Start")
        fig_sofi.update_layout(
            title="SOFI Index Over Time",
            xaxis_title="Year",
            yaxis_title="SOFI Index",
            template="plotly_white",
            height=400
        )

        # Year-over-year changes
        sofi_yoy = df_normalized["SOFI"].pct_change() * 100
        fig_yoy = go.Figure()
        fig_yoy.add_trace(go.Bar(
            x=df_normalized["Year"][1:],
            y=sofi_yoy[1:],
            marker_color=["green" if x > 0 else "red" for x in sofi_yoy[1:]],
            name="YoY Change"
        ))
        fig_yoy.update_layout(
            title="Year-over-Year SOFI Change (%)",
            xaxis_title="Year",
            yaxis_title="% Change",
            template="plotly_white",
            height=400
        )

        # Multi-indicator trends
        return html.Div([
            html.H2("Historical Trends", style={"marginTop": "20px"}),
            
            dcc.Graph(figure=fig_sofi),
            dcc.Graph(figure=fig_yoy),
            
            html.H4("Individual Indicator Trends", style={"marginTop": "30px"}),
            html.Div([
                html.Div([
                    html.Label("Select Indicators to Display:"),
                    dcc.Dropdown(
                        id="trends-indicator-selector",
                        options=[{"label": ind, "value": ind} for ind in indicator_cols],
                        value=indicator_cols[:5],
                        multi=True,
                        style={"width": "500px"}
                    ),
                ], style={"marginRight": "20px"}),
                html.Div([
                    html.Label("View:"),
                    dcc.RadioItems(
                        id="trends-view-type",
                        options=[
                            {"label": "Normalized (0-1 scale)", "value": "normalized"},
                            {"label": "Original Values", "value": "original"}
                        ],
                        value="normalized",
                        inline=True
                    ),
                ]),
            ], style={"display": "flex", "alignItems": "flex-end", "marginBottom": "20px"}),
            dcc.Graph(id="trends-multi-indicator")
        ])

    elif active_tab == "tab-analysis":
        # Get latest year data
        latest_year = df_original["Year"].max()
        latest_data = df_original[df_original["Year"] == latest_year].iloc[0]
        
        # Current indicator values
        indicator_values = [latest_data[col] for col in indicator_cols]
        fig_current = go.Figure(go.Bar(
            y=indicator_cols,
            x=indicator_values,
            orientation="h",
            marker_color="steelblue"
        ))
        fig_current.update_layout(
            title=f"Current Indicator Values ({int(latest_year)})",
            xaxis_title="Value",
            yaxis_title="Indicator",
            template="plotly_white",
            height=800
        )

        # Contribution to SOFI (weighted)
        latest_norm = df_normalized[df_normalized["Year"] == latest_year].iloc[0]
        contributions = [latest_norm[col] * weights[i] for i, col in enumerate(indicator_cols)]
        fig_contrib = go.Figure(go.Bar(
            y=indicator_cols,
            x=contributions,
            orientation="h",
            marker_color="green"
        ))
        fig_contrib.update_layout(
            title=f"Weighted Contribution to SOFI ({int(latest_year)})",
            xaxis_title="Contribution",
            yaxis_title="Indicator",
            template="plotly_white",
            height=800
        )

        # Weights visualization
        fig_weights = go.Figure(go.Bar(
            y=indicator_cols,
            x=weights,
            orientation="h",
            marker_color="orange"
        ))
        fig_weights.update_layout(
            title="Indicator Weights",
            xaxis_title="Weight",
            yaxis_title="Indicator",
            template="plotly_white",
            height=800
        )

        return html.Div([
            html.H2("Indicator Analysis", style={"marginTop": "20px"}),
            
            html.Div([
                html.Div([dcc.Graph(figure=fig_current)], style={"width": "33%"}),
                html.Div([dcc.Graph(figure=fig_contrib)], style={"width": "33%"}),
                html.Div([dcc.Graph(figure=fig_weights)], style={"width": "33%"}),
            ], style={"display": "flex", "gap": "20px"}),
            
            html.H4("Compare Years", style={"marginTop": "30px"}),
            html.Div([
                html.Div([
                    html.Label("Select Year 1:"),
                    dcc.Dropdown(
                        id="analysis-year1",
                        options=[{"label": str(int(y)), "value": y} for y in df_original["Year"]],
                        value=df_original["Year"].min(),
                        style={"width": "200px"}
                    ),
                ], style={"marginRight": "20px"}),
                html.Div([
                    html.Label("Select Year 2:"),
                    dcc.Dropdown(
                        id="analysis-year2",
                        options=[{"label": str(int(y)), "value": y} for y in df_original["Year"]],
                        value=latest_year,
                        style={"width": "200px"}
                    ),
                ], style={"marginRight": "20px"}),
                html.Div([
                    html.Label("View:"),
                    dcc.RadioItems(
                        id="analysis-view-type",
                        options=[
                            {"label": "Normalized", "value": "normalized"},
                            {"label": "Original", "value": "original"}
                        ],
                        value="normalized",
                        inline=True
                    ),
                ]),
            ], style={"display": "flex", "alignItems": "flex-end", "marginBottom": "20px"}),
            dcc.Graph(id="analysis-comparison")
        ])

    elif active_tab == "tab-correlations":
        # Calculate correlation matrix
        corr_matrix = df_normalized[indicator_cols].corr()
        
        # Correlation heatmap
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=indicator_cols,
            y=indicator_cols,
            colorscale="RdBu",
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 8},
            colorbar={"title": "Correlation"}
        ))
        fig_heatmap.update_layout(
            title="Indicator Correlation Matrix",
            template="plotly_white",
            height=900,
            xaxis={"tickangle": 45},
            yaxis={"tickangle": 0}
        )

        # Correlation with SOFI
        sofi_corr = df_normalized[indicator_cols + ["SOFI"]].corr()["SOFI"][:-1].sort_values(ascending=False)
        fig_sofi_corr = go.Figure(go.Bar(
            y=sofi_corr.index,
            x=sofi_corr.values,
            orientation="h",
            marker_color=["green" if x > 0 else "red" for x in sofi_corr.values]
        ))
        fig_sofi_corr.update_layout(
            title="Correlation with SOFI Index",
            xaxis_title="Correlation Coefficient",
            yaxis_title="Indicator",
            template="plotly_white",
            height=800
        )

        return html.Div([
            html.H2("Correlations", style={"marginTop": "20px"}),
            
            dcc.Graph(figure=fig_sofi_corr),
            dcc.Graph(figure=fig_heatmap),
            
            html.H4("Scatter Plot Analysis", style={"marginTop": "30px"}),
            html.Div([
                html.Div([
                    html.Label("X-Axis Indicator:"),
                    dcc.Dropdown(
                        id="corr-x-indicator",
                        options=[{"label": ind, "value": ind} for ind in indicator_cols],
                        value=indicator_cols[0],
                        style={"width": "300px"}
                    ),
                ], style={"marginRight": "20px"}),
                html.Div([
                    html.Label("Y-Axis Indicator:"),
                    dcc.Dropdown(
                        id="corr-y-indicator",
                        options=[{"label": ind, "value": ind} for ind in indicator_cols],
                        value=indicator_cols[1],
                        style={"width": "300px"}
                    ),
                ]),
            ], style={"display": "flex", "marginBottom": "20px"}),
            dcc.Graph(id="corr-scatter")
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
                       style={"color": "#666", "fontStyle": "italic"})

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
                html.Label(ind, style={"fontWeight": "bold", "marginRight": "10px",
                                      "minWidth": "250px"}),
                dcc.Input(
                    id={"type": "indicator-input", "index": ind},
                    type="number",
                    value=existing_value,
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
    df_source = df_normalized if view_type == "normalized" else df_original
    
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
    
    fig.add_vline(x=2025, line_dash="dot", line_color="gray",
                 annotation_text="Projection Start")
    
    y_title = "Normalized Value (0-1)" if view_type == "normalized" else "Original Value"
    fig.update_layout(
        title=f"Selected Indicators Over Time ({view_type.title()} Values)",
        xaxis_title="Year",
        yaxis_title=y_title,
        template="plotly_white",
        height=500,
        hovermode="x unified"
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
    df_source = df_normalized if view_type == "normalized" else df_original
    
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
        marker_color="lightblue"
    ))
    fig.add_trace(go.Bar(
        y=indicator_cols,
        x=values2,
        name=f"Year {int(year2)}",
        orientation="h",
        marker_color="darkblue"
    ))
    
    x_title = "Normalized Value (0-1)" if view_type == "normalized" else "Original Value"
    fig.update_layout(
        title=f"Indicator Comparison: {int(year1)} vs {int(year2)} ({view_type.title()} Values)",
        xaxis_title=x_title,
        yaxis_title="Indicator",
        template="plotly_white",
        height=800,
        barmode="group"
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
        hovertemplate="%{text}<br>" + f"{x_ind}: %{{x}}<br>{y_ind}: %{{y}}<extra></extra>"
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
        title=f"Correlation: {x_ind} vs {y_ind} (r = {corr:.3f})",
        xaxis_title=x_ind,
        yaxis_title=y_ind,
        template="plotly_white",
        height=600
    )
    return fig


if __name__ == "__main__":
    app.run(debug=True)
