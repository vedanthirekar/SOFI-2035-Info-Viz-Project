"""SOFI Interactive What-If Simulator with multi-variable support."""
import dash
from dash import dcc, html, Input, Output, State, ALL, MATCH, ctx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os

# ======== Load Data from Excel ========
# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data1.xlsx")

# Sheet1: Normalized values (for SOFI calculation)
df_normalized = pd.read_excel(DATA_FILE, sheet_name="Sheet1")

# Sheet2: Original values (for display)
df_original = pd.read_excel(DATA_FILE, sheet_name="Sheet2")

# Sheet3: Weights for each indicator
df_weights = pd.read_excel(DATA_FILE, sheet_name="Sheet3")

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

# Indicator categories for organized viewing
INDICATOR_CATEGORIES = {
    "economic": ["GNI per capita, PPP ", "Income Inequality", "Unemployment", "Poverty", 
                 "CPIA ", "FDI", "R&D "],
    "social": ["Population growth", "Life expectancy", "Mortality rate, infant", 
               "Undernourishment", "Health expenditure", "Physicians", "Drinking water",
               "Literacy, adult", "School enrollment", "Gender equality"],
    "environmental": ["Renewable freshwater", "Biocapacity ", "Forest area", "CO2 emissions",
                     "Energy-Efficiency", "Renewable energy"],
    "governance": ["Wars", "Terrorism Incidents", "Refugees", "Freedom Rights"],
    "technology": ["Patents", "Internet Users"]
}

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

# Neutral Earth Tones - Sophisticated color palette
COLORS = {
    "black": "#0a0908",        # Complete absorption - authority, depth
    "jet": "#22333b",          # Intense darkness - drama, mystery
    "linen": "#eae0d5",        # Subtle creamy neutral - sophistication
    "tan": "#c6ac8f",          # Soft sunlit neutral - warmth
    "stone": "#5e503f",        # Solid mid-brown - grounded reliability
    # Semantic mappings
    "primary": "#22333b",      # Jet Black - main elements
    "secondary": "#5e503f",    # Stone Brown - secondary elements
    "accent": "#c6ac8f",       # Tan - highlights
    "background": "#eae0d5",   # Soft Linen - page background
    "card": "#FFFFFF",         # White cards
    "text": "#0a0908",         # Black text
    "text-light": "#5e503f",   # Stone Brown for lighter text
    "border": "#c6ac8f",       # Tan borders
    "white": "#FFFFFF",
    "success": "#5e503f",      # Stone Brown for positive
    "danger": "#8b4513",       # Saddle brown for negative
    "light": "#eae0d5"         # Linen
}

app.layout = html.Div([
    # Elegant Header with Earth Tones
    html.Div([
        html.Div([
            html.Div([
                html.H1("SOFI Dashboard", 
                       style={"color": COLORS["linen"], "margin": "0", "fontSize": "2.8rem",
                              "fontWeight": "600", "letterSpacing": "2px",
                              "fontFamily": "'Georgia', serif"}),
                html.P("State of the Future Index — Interactive Analysis Platform",
                      style={"color": COLORS["tan"], "margin": "15px 0 0 0", 
                             "fontSize": "1.1rem", "fontWeight": "400", "letterSpacing": "1px"}),
                # html.Div([
                #     html.Span("29 Indicators", style={"marginRight": "30px", "borderRight": f"1px solid {COLORS['stone']}", "paddingRight": "30px"}),
                #     html.Span("1990–2035", style={"marginRight": "30px", "borderRight": f"1px solid {COLORS['stone']}", "paddingRight": "30px"}),
                #     html.Span("Predictive Analytics")
                # ], style={"marginTop": "20px", "fontSize": "0.9rem", "color": COLORS["tan"], 
                #          "letterSpacing": "1px", "textTransform": "uppercase"})
            ])
        ], style={"maxWidth": "1400px", "margin": "0 auto", "padding": "0 30px"})
    ], style={"background": COLORS["jet"],
              "padding": "55px 20px", "marginBottom": "0", 
              "boxShadow": "0 4px 20px rgba(0,0,0,0.2)"}),

    # Elegant Tabs with Earth Tones
    html.Div([
        dcc.Tabs(id="tabs", value="tab-trends", 
                children=[
                    dcc.Tab(label="Historical Trends", value="tab-trends",
                           style={"padding": "16px 32px", "fontWeight": "500", "fontSize": "0.95rem",
                                  "border": "none", "borderBottom": f"3px solid transparent",
                                  "backgroundColor": COLORS["linen"], "color": COLORS["stone"],
                                  "letterSpacing": "0.5px"},
                           selected_style={"padding": "16px 32px", "fontWeight": "600", "fontSize": "0.95rem",
                                         "borderBottom": f"3px solid {COLORS['stone']}",
                                         "backgroundColor": COLORS["white"], "color": COLORS["jet"],
                                         "letterSpacing": "0.5px"}),
                    dcc.Tab(label="What-If Simulator", value="tab-whatif",
                           style={"padding": "16px 32px", "fontWeight": "500", "fontSize": "0.95rem",
                                  "border": "none", "borderBottom": f"3px solid transparent",
                                  "backgroundColor": COLORS["linen"], "color": COLORS["stone"],
                                  "letterSpacing": "0.5px"},
                           selected_style={"padding": "16px 32px", "fontWeight": "600", "fontSize": "0.95rem",
                                         "borderBottom": f"3px solid {COLORS['stone']}",
                                         "backgroundColor": COLORS["white"], "color": COLORS["jet"],
                                         "letterSpacing": "0.5px"}),
                    dcc.Tab(label="Correlations", value="tab-correlations",
                           style={"padding": "16px 32px", "fontWeight": "500", "fontSize": "0.95rem",
                                  "border": "none", "borderBottom": f"3px solid transparent",
                                  "backgroundColor": COLORS["linen"], "color": COLORS["stone"],
                                  "letterSpacing": "0.5px"},
                           selected_style={"padding": "16px 32px", "fontWeight": "600", "fontSize": "0.95rem",
                                         "borderBottom": f"3px solid {COLORS['stone']}",
                                         "backgroundColor": COLORS["white"], "color": COLORS["jet"],
                                         "letterSpacing": "0.5px"}),
                    dcc.Tab(label="Indicator Analysis", value="tab-analysis",
                           style={"padding": "16px 32px", "fontWeight": "500", "fontSize": "0.95rem",
                                  "border": "none", "borderBottom": f"3px solid transparent",
                                  "backgroundColor": COLORS["linen"], "color": COLORS["stone"],
                                  "letterSpacing": "0.5px"},
                           selected_style={"padding": "16px 32px", "fontWeight": "600", "fontSize": "0.95rem",
                                         "borderBottom": f"3px solid {COLORS['stone']}",
                                         "backgroundColor": COLORS["white"], "color": COLORS["jet"],
                                         "letterSpacing": "0.5px"}),
                ],
                style={"maxWidth": "1400px", "margin": "0 auto"}),
    ], style={"backgroundColor": COLORS["linen"], "boxShadow": "0 2px 8px rgba(0,0,0,0.08)",
              "borderBottom": f"1px solid {COLORS['tan']}"}),

    # Content area
    html.Div(id="tab-content", style={"maxWidth": "1400px", "margin": "0 auto", "padding": "40px 30px"})
], style={"backgroundColor": COLORS["linen"], "minHeight": "100vh", 
          "fontFamily": "'Georgia', 'Times New Roman', serif"})


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
                               style={"color": "#22333b", "marginBottom": "15px", 
                                      "fontSize": "1.1rem", "fontWeight": "600", "letterSpacing": "0.5px"}),

                        # Dropdown to select indicator
                        html.Div([
                            dcc.Dropdown(
                                id="indicator-selector",
                                options=[{"label": ind, "value": ind} for ind in indicator_cols],
                                placeholder="Select an indicator to add...",
                                style={"width": "400px", "marginRight": "10px"}
                            ),
                            html.Button("Add Indicator", id="add-indicator-btn", n_clicks=0,
                                       style={"padding": "10px 20px", "backgroundColor": "#5e503f",
                                              "color": "#eae0d5", "border": "none", 
                                              "borderRadius": "4px", "cursor": "pointer",
                                              "fontWeight": "500", "letterSpacing": "0.5px"})
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
                                     title="ONE-TIME CHANGE (Shift):\n"
                                           "Shifts the predicted curve by X% for all future years.\n"
                                           "Example: 2024=100, Predicted 2025=105. Apply +10%.\n"
                                           "Result: 2025=115.5, 2026=121, 2027=126.5\n"
                                           "Use for: Policy changes, infrastructure improvements.\n\n"
                                           "COMPOUND GROWTH (Exponential):\n"
                                           "Growth builds on itself year-over-year from 2024 baseline.\n"
                                           "Example: 2024=100. Apply +10% compound.\n"
                                           "Result: 2025=110, 2026=121, 2027=133.1\n"
                                           "Use for: Technology adoption, economic growth.\n\n"
                                           "ANNUAL RATE (Linear):\n"
                                           "Adds X% of 2024 baseline value every year.\n"
                                           "Example: 2024=100. Apply +10% annual.\n"
                                           "Result: 2025=110, 2026=120, 2027=130\n"
                                           "Use for: Steady improvements, gradual policy changes.",
                                     style={"cursor": "help", "color": COLORS["secondary"],
                                            "fontSize": "16px", "fontWeight": "bold",
                                            "marginLeft": "5px"}),
                        ], style={"display": "flex", "alignItems": "center", "marginBottom": "12px"}),
                        dcc.RadioItems(
                            options=[
                                {"label": "One-time change (shift predicted curve by X%)", "value": "linear"},
                                {"label": "Compound growth (exponential from 2024 baseline)", "value": "exponential"},
                                {"label": "Annual rate (linear growth from 2024 baseline)", "value": "annual_rate"}
                            ],
                            value="linear",
                            id="growth-type",
                            style={"marginBottom": "30px", "lineHeight": "1.8"}
                        ),

                        # Action buttons
                        html.Div([
                            html.Button("Clear All", id="clear-all-btn", n_clicks=0,
                                       style={"marginRight": "10px", "padding": "10px 20px",
                                              "width": "120px", "backgroundColor": "#eae0d5",
                                              "color": "#5e503f", "border": "2px solid #c6ac8f",
                                              "borderRadius": "4px", "cursor": "pointer",
                                              "fontWeight": "500", "letterSpacing": "0.5px"}),
                            html.Button("Apply Changes", id="apply-button", n_clicks=0,
                                       style={"padding": "10px 20px", "backgroundColor": "#22333b",
                                              "color": "#eae0d5", "border": "none", 
                                              "width": "140px", "borderRadius": "4px",
                                              "cursor": "pointer", "fontWeight": "500",
                                              "letterSpacing": "0.5px"})
                        ]),
                    ], style={"minWidth": "320px"}),
                ], style={"display": "flex"}),
            ], style={"padding": "30px", "backgroundColor": "#FFFFFF",
                      "borderRadius": "4px", "marginBottom": "25px",
                      "boxShadow": "0 2px 12px rgba(0,0,0,0.06)", "border": "1px solid #c6ac8f"}),

            # Store to track active indicators
            dcc.Store(id="active-indicators-store", data=[]),

            # Graph Card
            html.Div([
                dcc.Graph(id="sofi-graph", config={"displayModeBar": True, "displaylogo": False})
            ], style={"backgroundColor": "#FFFFFF", "borderRadius": "4px",
                     "padding": "20px", "boxShadow": "0 2px 12px rgba(0,0,0,0.06)",
                     "border": "1px solid #c6ac8f"})
        ])

    elif active_tab == "tab-trends":
        # SOFI over time chart
        fig_sofi = go.Figure()
        fig_sofi.add_trace(go.Scatter(
            x=df_normalized["Year"],
            y=df_normalized["SOFI"],
            mode="lines+markers",
            name="SOFI",
            line={"color": "#5e503f", "width": 3},  # Stone Brown
            marker={"size": 6, "color": "#5e503f"}
        ))
        fig_sofi.add_vline(x=2024, line_dash="dot", line_color="#c6ac8f",  # Tan
                          annotation_text="Projection Start")
        fig_sofi.update_layout(
            title={"text": "SOFI Index Over Time", "font": {"size": 18, "color": "#22333b"}},
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
            marker_color=["#5e503f" if x > 0 else "#c6ac8f" for x in sofi_yoy[1:]],  # Stone/Tan
            name="YoY Change"
        ))
        fig_yoy.update_layout(
            title={"text": "Year-over-Year SOFI Change (%)", "font": {"size": 18, "color": "#22333b"}},
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
            ], style={"backgroundColor": "#FFFFFF", "borderRadius": "4px",
                     "padding": "20px", "marginBottom": "25px",
                     "boxShadow": "0 2px 12px rgba(0,0,0,0.06)",
                     "border": "1px solid #c6ac8f"}),
            
            # YoY Changes Card
            html.Div([
                dcc.Graph(figure=fig_yoy, config={"displayModeBar": True, "displaylogo": False})
            ], style={"backgroundColor": "#FFFFFF", "borderRadius": "4px",
                     "padding": "20px", "marginBottom": "25px",
                     "boxShadow": "0 2px 12px rgba(0,0,0,0.06)",
                     "border": "1px solid #c6ac8f"}),
            
            # Multi-Indicator Trends Card
            html.Div([
                html.H4("Individual Indicator Trends", 
                       style={"color": "#22333b", "marginBottom": "20px",
                              "fontSize": "1.2rem", "fontWeight": "600", "letterSpacing": "0.5px"}),
                html.Div([
                    html.Div([
                        html.Label("Select Indicators to Display:", 
                                  style={"fontWeight": "500", "marginBottom": "8px",
                                         "display": "block", "color": "#5e503f"}),
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
                                         "display": "block", "color": "#5e503f"}),
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
            ], style={"backgroundColor": "#FFFFFF", "borderRadius": "4px",
                     "padding": "25px", "boxShadow": "0 2px 12px rgba(0,0,0,0.06)",
                     "border": "1px solid #c6ac8f"})
        ])

    elif active_tab == "tab-analysis":
        # Get latest year data
        latest_year = df_original["Year"].max()

        return html.Div([
            html.Div([
                html.H4("Compare Years", 
                       style={"color": "#22333b", "marginBottom": "20px",
                              "fontSize": "1.2rem", "fontWeight": "600", "letterSpacing": "0.5px"}),
                html.Div([
                    html.Div([
                        html.Label("Select Year 1:", 
                                  style={"fontWeight": "500", "marginBottom": "8px",
                                         "display": "block", "color": "#5e503f"}),
                        dcc.Dropdown(
                            id="analysis-year1",
                            options=[{"label": str(int(y)), "value": y} for y in df_original["Year"]],
                            value=2005,
                            style={"width": "200px"}
                        ),
                    ], style={"marginRight": "30px"}),
                    html.Div([
                        html.Label("Select Year 2:", 
                                  style={"fontWeight": "500", "marginBottom": "8px",
                                         "display": "block", "color": "#5e503f"}),
                        dcc.Dropdown(
                            id="analysis-year2",
                            options=[{"label": str(int(y)), "value": y} for y in df_original["Year"]],
                            value=2025,
                            style={"width": "200px"}
                        ),
                    ], style={"marginRight": "30px"}),
                    html.Div([
                        html.Label("Category:", 
                                  style={"fontWeight": "500", "marginBottom": "8px",
                                         "display": "block", "color": "#5e503f"}),
                        dcc.Dropdown(
                            id="analysis-category",
                            options=[
                                {"label": "All Indicators", "value": "all"},
                                {"label": "Economic", "value": "economic"},
                                {"label": "Social", "value": "social"},
                                {"label": "Environmental", "value": "environmental"},
                                {"label": "Governance & Security", "value": "governance"},
                                {"label": "Technology", "value": "technology"}
                            ],
                            value="economic",
                            style={"width": "250px"}
                        ),
                    ], style={"marginRight": "30px"}),
                    html.Div([
                        html.Label("View:", 
                                  style={"fontWeight": "500", "marginBottom": "8px",
                                         "display": "block", "color": "#5e503f"}),
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
            ], style={"backgroundColor": "#FFFFFF", "borderRadius": "4px",
                     "padding": "25px", "boxShadow": "0 2px 12px rgba(0,0,0,0.06)",
                     "border": "1px solid #c6ac8f"})
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
            marker_color=["#5e503f" if x > 0 else "#c6ac8f" for x in sofi_corr_adjusted.values]  # Stone/Tan
        ))
        fig_sofi_corr.update_layout(
            title={"text": "Correlation with SOFI Index", "font": {"size": 18, "color": "#22333b"}},
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
                       style={"color": "#22333b", "marginBottom": "20px",
                              "fontSize": "1.2rem", "fontWeight": "600", "letterSpacing": "0.5px"}),
                html.Div([
                    html.Div([
                        html.Label("X-Axis Indicator:", 
                                  style={"fontWeight": "500", "marginBottom": "8px",
                                         "display": "block", "color": "#5e503f"}),
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
                                         "display": "block", "color": "#5e503f"}),
                        dcc.Dropdown(
                            id="corr-y-indicator",
                            options=[{"label": ind, "value": ind} for ind in indicator_cols],
                            value=indicator_cols[1],
                            style={"width": "300px"}
                        ),
                    ]),
                ], style={"display": "flex", "marginBottom": "25px"}),
                dcc.Graph(id="corr-scatter", config={"displayModeBar": True, "displaylogo": False})
            ], style={"backgroundColor": "#FFFFFF", "borderRadius": "4px",
                     "padding": "25px", "marginBottom": "25px",
                     "boxShadow": "0 2px 12px rgba(0,0,0,0.06)",
                     "border": "1px solid #c6ac8f"}),
            
            # Correlation Bar Chart Card
            html.Div([
                dcc.Graph(figure=fig_sofi_corr, config={"displayModeBar": True, "displaylogo": False})
            ], style={"backgroundColor": "#FFFFFF", "borderRadius": "4px",
                     "padding": "20px", "boxShadow": "0 2px 12px rgba(0,0,0,0.06)",
                     "border": "1px solid #c6ac8f"})
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
                       style={"color": "#5e503f", "fontStyle": "italic", "padding": "15px",
                              "backgroundColor": "#eae0d5", "borderRadius": "4px",
                              "border": "1px dashed #c6ac8f"})

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
                                      "minWidth": "250px", "color": "#22333b"}),
                dcc.Input(
                    id={"type": "indicator-input", "index": ind},
                    type="number",
                    value=existing_value,
                    step=0.5,
                    style={"width": "80px", "marginRight": "5px", "padding": "6px 10px",
                           "border": "1px solid #c6ac8f", "borderRadius": "4px"}
                ),
                html.Span("%", style={"marginRight": "10px", "color": "#5e503f",
                                     "fontWeight": "500"}),
                html.Button("×", id={"type": "remove-indicator-btn", "index": ind},
                           style={"padding": "4px 10px", "backgroundColor": "#5e503f",
                                  "color": "#eae0d5", "border": "none",
                                  "borderRadius": "4px", "cursor": "pointer",
                                  "fontWeight": "600", "fontSize": "14px",
                                  "transition": "all 0.3s"})
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "10px",
                     "padding": "10px 12px", "backgroundColor": "#eae0d5",
                     "borderRadius": "4px", "border": "1px solid #c6ac8f"})
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
            
            # Get the 2024 value (last historical point) for all growth types
            year_2024_idx = df_norm_adj[df_norm_adj["Year"] == 2024].index
            if len(year_2024_idx) > 0:
                base_value_norm = df_norm_adj.loc[year_2024_idx[0], ind_name]
                base_value_orig = df_orig_adj.loc[year_2024_idx[0], ind_name]
            else:
                # If 2024 doesn't exist, use first future year as base
                base_value_norm = df_norm_adj.loc[future_years[0], ind_name]
                base_value_orig = df_orig_adj.loc[future_years[0], ind_name]
            
            if growth_type == "linear":
                # One-time change: apply same % to all future years
                df_norm_adj.loc[future_years, ind_name] *= (1 + change_norm / 100)
                df_orig_adj.loc[future_years, ind_name] *= (1 + change_orig / 100)
            elif growth_type == "exponential":
                # Compound growth: % compounds year-over-year from 2024 baseline
                for j, idx in enumerate(future_years):
                    years_from_start = j + 1
                    df_norm_adj.loc[idx, ind_name] = base_value_norm * (1 + change_norm / 100) ** years_from_start
                    df_orig_adj.loc[idx, ind_name] = base_value_orig * (1 + change_orig / 100) ** years_from_start
            else:  # annual_rate
                # Annual rate: continuous linear growth/decline from 2024 baseline
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
            line={"color": "#c6ac8f", "dash": "dash", "width": 1.5},  # Tan baseline
            legendgroup="baseline",
            showlegend=(idx == 0)
        ), row=row_idx, col=col_idx)

        fig.add_trace(go.Scatter(
            x=df_orig_adj["Year"], y=df_orig_adj[ind_name],
            mode="lines", name="Adjusted",
            line={"color": "#5e503f", "width": 2},  # Stone Brown adjusted
            legendgroup="adjusted",
            showlegend=(idx == 0)
        ), row=row_idx, col=col_idx)

        fig.update_yaxes(title_text=ind_name, row=row_idx, col=col_idx)
        fig.add_vline(x=2024, line_width=1, line_dash="dot", line_color="#c6ac8f",
                     row=row_idx, col=col_idx)  # Tan

    # SOFI subplot (last position)
    sofi_idx = len(indicators_to_show)
    sofi_row = (sofi_idx // max_cols) + 1
    sofi_col = (sofi_idx % max_cols) + 1

    fig.add_trace(go.Scatter(
        x=df_normalized["Year"], y=df_normalized["SOFI"],
        mode="lines", name="Baseline",
        line={"color": "#c6ac8f", "dash": "dash", "width": 2},  # Tan baseline
        legendgroup="baseline",
        showlegend=False
    ), row=sofi_row, col=sofi_col)

    fig.add_trace(go.Scatter(
        x=df_norm_adj["Year"], y=df_norm_adj["SOFI"],
        mode="lines", name="Adjusted",
        line={"color": "#22333b", "width": 3},  # Jet Black adjusted
        legendgroup="adjusted",
        showlegend=False
    ), row=sofi_row, col=sofi_col)

    fig.add_vline(x=2024, line_width=1, line_dash="dot", line_color="#c6ac8f",
                  row=sofi_row, col=sofi_col, annotation_text="2024")  # Tan

    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="SOFI Index", row=sofi_row, col=sofi_col)

    # Adjust height based on number of rows
    plot_height = 400 * num_rows

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        height=plot_height,
        showlegend=True,
        legend={"orientation": "h", "yanchor": "top", "y": -0.35,
                "xanchor": "center", "x": 0.5, "bgcolor": "rgba(234,224,213,0.9)",
                "bordercolor": "#c6ac8f", "borderwidth": 1},
        margin={"b": 150, "t": 50, "l": 60, "r": 40},
        font={"family": "'Georgia', 'Times New Roman', serif",
              "color": "#5e503f"}
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
    
    # Earth tone color palette for multi-line charts
    EARTH_TONE_COLORS = [
        "#5e503f",  # Stone Brown
        "#22333b",  # Jet Black
        "#c6ac8f",  # Tan
        "#8b7355",  # Warm Brown
        "#6b5b4f",  # Dark Taupe
        "#a08060",  # Camel
        "#4a4238",  # Dark Stone
        "#9c8b7a",  # Warm Gray
        "#7d6b5d",  # Mocha
        "#b5a08e",  # Sand
    ]
    
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
    for i, ind in enumerate(selected_indicators):
        color = EARTH_TONE_COLORS[i % len(EARTH_TONE_COLORS)]
        fig.add_trace(go.Scatter(
            x=df_source["Year"],
            y=df_source[ind],
            mode="lines+markers",
            name=ind,
            line={"width": 2, "color": color},
            marker={"size": 4, "color": color}
        ))
    
    fig.add_vline(x=2024, line_dash="dot", line_color="#c6ac8f",
                 annotation_text="Projection Start")  # Tan
    
    y_title = "Normalized Value (0-1)" if view_type == "normalized" else "Original Value"
    fig.update_layout(
        title={"text": f"Selected Indicators Over Time ({view_type.title()} Values)",
               "font": {"size": 18, "color": "#22333b"}},
        xaxis_title="Year",
        yaxis_title=y_title,
        template="plotly_white",
        height=500,
        hovermode="x unified",
        margin={"t": 60},
        font={"family": "'Georgia', 'Times New Roman', serif",
              "color": "#5e503f"}
    )
    return fig


# Callback for analysis year comparison
@app.callback(
    Output("analysis-comparison", "figure"),
    Input("analysis-year1", "value"),
    Input("analysis-year2", "value"),
    Input("analysis-category", "value"),
    Input("analysis-view-type", "value")
)
def update_analysis_comparison(year1, year2, category, view_type):
    """Compare indicators between two years."""
    if year1 is None or year2 is None:
        return go.Figure()
    
    # Filter indicators by category
    if category == "all":
        selected_indicators = indicator_cols
    else:
        selected_indicators = INDICATOR_CATEGORIES.get(category, indicator_cols)
    
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
    
    # Use only selected indicators
    values1 = [data1[col] for col in selected_indicators]
    values2 = [data2[col] for col in selected_indicators]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=selected_indicators,
        x=values1,
        name=f"Year {int(year1)}",
        orientation="h",
        marker_color="#5e503f",  # Stone Brown
        width=0.35  # Make bars fatter
    ))
    fig.add_trace(go.Bar(
        y=selected_indicators,
        x=values2,
        name=f"Year {int(year2)}",
        orientation="h",
        marker_color="#c6ac8f",  # Tan
        width=0.35  # Make bars fatter
    ))
    
    # Dynamic height based on number of indicators
    chart_height = max(400, len(selected_indicators) * 25)
    
    category_label = category.title() if category != "all" else "All"
    x_title = "Normalized Value (0-1)" if view_type == "normalized" else "Original Value (Log Scale)"
    
    fig.update_layout(
        title={"text": f"{category_label} Indicators: {int(year1)} vs {int(year2)} ({view_type.title()} Values)",
               "font": {"size": 18, "color": "#22333b"}},
        xaxis_title=x_title,
        yaxis_title="Indicator",
        template="plotly_white",
        height=chart_height,
        barmode="group",
        bargap=0.5,  # More space between indicator groups
        bargroupgap=0.3,  # Space between bars within a group
        margin={"t": 60},
        font={"family": "'Georgia', 'Times New Roman', serif",
              "color": "#5e503f"}
    )
    
    # Use logarithmic scale for original values to show small values better
    if view_type == "original":
        fig.update_xaxes(type="log")
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
        line={"color": "#5e503f", "dash": "dash"}  # Stone Brown trend
    ))
    
    fig.update_layout(
        title={"text": f"Correlation: {x_ind} vs {y_ind} (r = {corr:.3f})",
               "font": {"size": 18, "color": "#22333b"}},
        xaxis_title=x_ind,
        yaxis_title=y_ind,
        template="plotly_white",
        height=600,
        margin={"t": 60},
        font={"family": "'Georgia', 'Times New Roman', serif",
              "color": "#5e503f"},
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01,
                "bgcolor": "rgba(234,224,213,0.9)", "bordercolor": "#c6ac8f", 
                "borderwidth": 1}
    )
    return fig


server = app.server

if __name__ == "__main__":
    app.run(debug=True)
