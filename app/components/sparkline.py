"""Sparkline Plotly figure — UI only, no analytics."""

import pandas as pd
import plotly.graph_objects as go


def _hex_to_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b}"


def sparkline_figure(values: pd.Series, color: str = "#2DD4C8") -> go.Figure:
    rgb = _hex_to_rgb(color)
    fig = go.Figure(go.Scatter(
        x=list(range(len(values))),
        y=values.tolist(),
        mode="lines",
        line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor=f"rgba({rgb},0.08)",
    ))
    fig.update_layout(
        height=70,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(visible=False, fixedrange=True),
        yaxis=dict(visible=False, fixedrange=True),
    )
    return fig
