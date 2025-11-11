import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# ======== Sample Data ========
years = np.arange(1990, 2036)
df = pd.DataFrame({
    "Year": years,
    "GNI per capita": np.linspace(10000, 18000, len(years)),
    "Income Inequality": np.linspace(30, 25, len(years)),
    "Unemployment": np.linspace(6, 4, len(years))
})
indicator_cols = ["GNI per capita", "Income Inequality", "Unemployment"]
weights = np.array([0.4, 0.3, 0.3])

def compute_sofi(df, weights):
    norm = (df - df.min()) / (df.max() - df.min())
    sofi = np.dot(norm, weights)
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

    dcc.Graph(id="sofi-graph")
])

@app.callback(
    Output("sofi-graph", "figure"),
    Input("adjust-slider", "value"),
    Input("indicator-dropdown", "value")
)
def update_graph(change, indicator):
    df_adj = df.copy()
    df_adj[indicator] *= (1 + change / 100)
    df_adj["SOFI"] = compute_sofi(df_adj[indicator_cols], weights)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_adj["Year"], y=df_adj[indicator],
                             mode="lines", name=indicator, line=dict(color="red")))
    fig.add_trace(go.Scatter(x=df_adj["Year"], y=df_adj["SOFI"],
                             mode="lines", name="SOFI Index", line=dict(color="green")))
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Value / Index",
        template="plotly_white"
    )
    return fig

if __name__ == "__main__":
    app.run(debug=True)
