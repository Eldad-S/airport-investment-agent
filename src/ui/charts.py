"""Plotly chart builders shared by the dashboard and the chat answers."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.analytics.scoring import COMPONENT_LABELS, COMPONENTS, CONGESTION_SUBWEIGHTS, SUBMETRIC_LABELS, normalize_weights

COLORS = {"demand": "#2563eb", "capacity_pressure": "#dc2626", "operations": "#059669"}
_LAYOUT = dict(margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.2), height=380)


def score_breakdown_bar(scored: pd.DataFrame, weights: dict[str, float], title: str = "Opportunity score breakdown") -> go.Figure:
    """Horizontal stacked bar: weighted points each component contributes to the score."""
    w = normalize_weights(weights)
    df = scored[scored["status"] == "scored"].copy()
    fig = go.Figure()
    labels = [f"{i} ({r['city']})" for i, r in df.iterrows()]
    for c in COMPONENTS:
        present = df[[f"score_{k}" for k in COMPONENTS]].notna()
        wsum = sum(present[f"score_{k}"] * w[k] for k in COMPONENTS)
        pts = df[f"score_{c}"].fillna(0) * w[c] / wsum
        fig.add_bar(y=labels, x=pts, name=COMPONENT_LABELS[c], orientation="h", marker_color=COLORS[c],
                    hovertemplate="%{y}<br>" + COMPONENT_LABELS[c] + ": %{x:.1f} pts<extra></extra>")
    fig.update_layout(barmode="stack", title=title, xaxis_title="Score points (0-100)",
                      yaxis=dict(autorange="reversed"), **{**_LAYOUT, "height": max(300, 28 * len(df) + 120)})
    return fig


def congestion_percentiles(scored: pd.DataFrame, codes: list[str]) -> go.Figure:
    rows = []
    for c in codes:
        if c not in scored.index:
            continue
        for sub in CONGESTION_SUBWEIGHTS:
            rows.append({"Airport": c, "Indicator": SUBMETRIC_LABELS[sub], "Peer percentile": scored.loc[c, f"pct_{sub}"]})
    df = pd.DataFrame(rows)
    if df.empty:
        return go.Figure()
    fig = px.bar(df, x="Indicator", y="Peer percentile", color="Airport", barmode="group",
                 title="Congestion indicators - percentile vs. peer hub airports (100 = most congested)")
    fig.update_layout(yaxis_range=[0, 100], **_LAYOUT)
    return fig


def component_comparison(scored: pd.DataFrame, codes: list[str]) -> go.Figure:
    rows = [{"Airport": c, "Component": COMPONENT_LABELS[k], "Score": scored.loc[c, f"score_{k}"]}
            for c in codes if c in scored.index for k in COMPONENTS]
    df = pd.DataFrame(rows)
    if df.empty:
        return go.Figure()
    fig = px.bar(df, x="Component", y="Score", color="Airport", barmode="group", title="Component scores (0-100)")
    fig.update_layout(yaxis_range=[0, 100], **_LAYOUT)
    return fig


def monthly_departures(trends: dict[str, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    for code, df in trends.items():
        if not df.empty:
            fig.add_scatter(x=df["date"], y=df["operated_departures"], mode="lines+markers", name=code)
    fig.update_layout(title="Monthly operated departures (BTS reporting carriers)", yaxis_title="Departures", **_LAYOUT)
    return fig


def monthly_delay(trends: dict[str, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    for code, df in trends.items():
        if not df.empty:
            fig.add_scatter(x=df["date"], y=df["dep_delay_rate"] * 100, mode="lines+markers", name=code)
    fig.update_layout(title="Monthly departure delay rate (% of flights >= 15 min late)", yaxis_title="%", **_LAYOUT)
    return fig


def opportunity_map(scored: pd.DataFrame) -> go.Figure:
    df = scored[scored["status"] == "scored"].reset_index()
    df = df[df["state"].isin(set(df["state"]) - {"PR", "VI", "GU"})]
    fig = px.scatter_geo(df, lat="lat", lon="lon", color="opportunity_score", size="enplanements",
                         hover_name="iata", hover_data={"name": True, "opportunity_score": ":.0f", "rank": True,
                                                       "confidence": True, "lat": False, "lon": False, "enplanements": ":,.0f"},
                         color_continuous_scale="RdYlGn", range_color=(20, 80), scope="usa", size_max=28,
                         title="Opportunity score by airport (bubble size = passengers)")
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=460, coloraxis_colorbar=dict(title="Score"))
    return fig


def rank_range(summary: pd.DataFrame, top: int = 15) -> go.Figure:
    df = summary.head(top).reset_index().rename(columns={"index": "iata"})
    fig = go.Figure()
    fig.add_scatter(x=df["worst_rank"], y=df["iata"], mode="markers", name="Worst rank", marker_color="#dc2626")
    fig.add_scatter(x=df["best_rank"], y=df["iata"], mode="markers", name="Best rank", marker_color="#059669")
    fig.add_scatter(x=df["base_rank"], y=df["iata"], mode="markers", name="Base rank",
                    marker=dict(color="#111827", symbol="diamond", size=10))
    for _, r in df.iterrows():
        fig.add_shape(type="line", x0=r["best_rank"], x1=r["worst_rank"], y0=r["iata"], y1=r["iata"],
                      line=dict(color="#9ca3af", width=3), layer="below")
    fig.update_layout(title="Rank range across weight scenarios (shorter line = more robust)",
                      xaxis_title="Rank", yaxis=dict(autorange="reversed"), **{**_LAYOUT, "height": 30 * len(df) + 140})
    return fig
