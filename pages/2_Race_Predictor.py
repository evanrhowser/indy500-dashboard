"""
pages/2_Race_Predictor.py
Race Predictor page for the Indy 500 Historical Dashboard.

Shows the historical speed trend, weather-effect analysis, and a
step-by-step prediction breakdown for the 2025 Indianapolis 500.
"""

import datetime
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="Race Predictor — Indy 500", layout="wide")

RACE_YEAR        = 2026
RACE_DATE        = datetime.date(2026, 5, 24)
YEAR_MIN, YEAR_MAX = 1911, 2024
WEATHER_FEATURES = ["temperature", "wind_speed", "humidity", "precipitation"]
TECH_FEATURES    = ["year", "technology_index"]

FEAT_LABELS = {
    "temperature":   "Temperature",
    "wind_speed":    "Wind Speed",
    "humidity":      "Humidity",
    "precipitation": "Precipitation",
}
FEAT_UNITS = {
    "temperature":   "°F",
    "wind_speed":    " mph",
    "humidity":      "%",
    "precipitation": '"',
}

BG        = "#111827"   # chart background
BG_CARD   = "#1a1a2e"   # metric card background
C_BLUE    = "#4FC3F7"
C_RED     = "#FF6B6B"
C_GOLD    = "#FFD700"

# ── Load models ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        with open("model_trend.pkl",   "rb") as f:
            trend_pkg   = pickle.load(f)
        with open("model_weather.pkl", "rb") as f:
            weather_pkg = pickle.load(f)
        return trend_pkg, weather_pkg
    except FileNotFoundError:
        return None, None

trend_pkg, weather_pkg = load_models()

if trend_pkg is None:
    st.error(
        "Model files not found. Run `python prediction_model.py` from the "
        "project directory first, then reload this page."
    )
    st.stop()

trend_model   = trend_pkg["model"]
weather_model = weather_pkg["model"]

# ── Load historical data ───────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv("data/combined_data.csv")

df = load_data()

# ── Fetch race-day weather ─────────────────────────────────────────────────────
@st.cache_data
def fetch_race_weather(date: datetime.date) -> dict:
    """
    Fetch race-day hourly weather from Open-Meteo.
    May 24 2026 is a future date, so the forecast API is tried first;
    if the date is beyond the 16-day window the historical May average
    is used as a fallback.
    Returns noon conditions + daily precipitation total.
    """
    params = {
        "latitude":           39.7684,
        "longitude":          -86.1581,
        "start_date":         date.isoformat(),
        "end_date":           date.isoformat(),
        "hourly":             ("temperature_2m,relative_humidity_2m,"
                               "precipitation,wind_speed_10m"),
        "temperature_unit":   "fahrenheit",
        "wind_speed_unit":    "mph",
        "precipitation_unit": "inch",
        "timezone":           "America/Indiana/Indianapolis",
    }
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params=params, timeout=15,
        )
        resp.raise_for_status()
        hourly = resp.json().get("hourly", {})
        temp   = hourly.get("temperature_2m",       [])
        rh     = hourly.get("relative_humidity_2m", [])
        precip = hourly.get("precipitation",        [])
        wind   = hourly.get("wind_speed_10m",       [])
        noon   = 12
        if temp and noon < len(temp) and temp[noon] is not None:
            return {
                "temperature":   round(temp[noon], 1),
                "wind_speed":    round(wind[noon], 1),
                "humidity":      round(rh[noon],   1),
                "precipitation": round(sum(x for x in precip if x is not None), 2),
                "source":        "Open-Meteo forecast",
            }
    except Exception:
        pass

    # Fallback: May climate normal from historical data (excluding Aug 2020)
    w   = pd.read_csv("data/weather_data.csv").dropna()
    w   = w[w["year"] != 2020]
    avg = w[WEATHER_FEATURES].mean()
    return {col: round(float(avg[col]), 1) for col in WEATHER_FEATURES} | {
        "source": "historical May average (API unavailable)"
    }

weather = fetch_race_weather(RACE_DATE)

# ── Technology trend line ──────────────────────────────────────────────────────
def trend_at(year: int) -> float:
    ti = (year - YEAR_MIN) / (YEAR_MAX - YEAR_MIN)
    return float(trend_model.predict([[year, ti]])[0])

df = df.copy()
df["trend_speed"] = df["year"].apply(trend_at)

# ── Partial weather effects (10th → 90th percentile) ──────────────────────────
@st.cache_data
def compute_partial_effects() -> dict:
    """
    For each weather variable, compute the speed-adjustment change when
    the variable moves from its 10th to 90th historical percentile while
    all other variables are held at their median.
    """
    base = {f: float(df[f].median()) for f in WEATHER_FEATURES}
    out  = {}
    for feat in WEATHER_FEATURES:
        lo_val = float(df[feat].quantile(0.10))
        hi_val = float(df[feat].quantile(0.90))
        X_lo   = [lo_val if f == feat else base[f] for f in WEATHER_FEATURES]
        X_hi   = [hi_val if f == feat else base[f] for f in WEATHER_FEATURES]
        delta  = float(
            weather_model.predict([X_hi])[0] - weather_model.predict([X_lo])[0]
        )
        out[feat] = {"delta": round(delta, 2), "lo": round(lo_val, 1), "hi": round(hi_val, 1)}
    return out

effects = compute_partial_effects()

# ── 2025 prediction ────────────────────────────────────────────────────────────
trend_2025  = trend_at(RACE_YEAR)
weather_adj = float(weather_model.predict([[
    weather["temperature"],
    weather["wind_speed"],
    weather["humidity"],
    weather["precipitation"],
]])[0])
final_pred  = round(trend_2025 + weather_adj, 1)
mae         = weather_pkg["cv_mae"]
adj_sign    = f"+{weather_adj:.1f}" if weather_adj >= 0 else f"{weather_adj:.1f}"

# ════════════════════════════════════════════════════════════════════════════════
# Layout
# ════════════════════════════════════════════════════════════════════════════════
st.title("🏁 Race Predictor")
st.caption(f"110th Indianapolis 500  ·  {RACE_DATE.strftime('%B %d, %Y')}")

st.markdown("---")

# ── Section 1: Historical speeds + trend line ──────────────────────────────────
st.subheader("Historical Winning Speeds & Technology Trend")

fig_hist = go.Figure()

fig_hist.add_trace(go.Scatter(
    x=df["year"], y=df["avg_speed_mph"],
    mode="lines+markers",
    name="Actual winning speed",
    line=dict(color=C_BLUE, width=1.5),
    marker=dict(size=5, color=C_BLUE),
    hovertemplate="<b>%{x}</b><br>Actual: %{y:.1f} mph<extra></extra>",
))
fig_hist.add_trace(go.Scatter(
    x=df["year"], y=df["trend_speed"],
    mode="lines",
    name=f"Technology trend  (+{trend_pkg['mph_per_year']:.2f} mph/yr)",
    line=dict(color=C_RED, width=2.5, dash="dash"),
    hovertemplate="<b>%{x}</b><br>Trend: %{y:.1f} mph<extra></extra>",
))

fig_hist.update_layout(
    plot_bgcolor=BG, paper_bgcolor=BG,
    font=dict(color="white"),
    xaxis=dict(title="Year",                gridcolor="#2a2a3a", color="white",
               showline=True, linecolor="#444"),
    yaxis=dict(title="Average Speed (mph)", gridcolor="#2a2a3a", color="white",
               showline=True, linecolor="#444"),
    legend=dict(bgcolor="rgba(0,0,0,0.45)", font=dict(color="white"),
                bordercolor="#444", borderwidth=1),
    hovermode="x unified",
    height=380,
    margin=dict(t=20, b=40, l=10, r=10),
)
st.plotly_chart(fig_hist, use_container_width=True)
st.caption(
    f"Trend R² = {trend_pkg['r2']:.3f}  ·  "
    f"Trend MAE = ±{trend_pkg['mae']:.1f} mph  ·  "
    f"{len(df)} races  ·  "
    "Dashed line = what technology alone would predict; "
    "gaps above/below = weather and random variation."
)

st.markdown("---")

# ── Section 2: Weather effects ─────────────────────────────────────────────────
st.subheader("Weather Effects on Race Speed")
st.write(
    "Each bar shows how much the predicted race speed changes when a weather "
    "variable moves from its **10th to 90th historical percentile**, with "
    "everything else held constant.  "
    "🔵 Blue = higher values produce *faster* races · "
    "🔴 Red = higher values produce *slower* races."
)

sorted_eff   = sorted(effects.items(), key=lambda kv: abs(kv[1]["delta"]), reverse=True)
feat_labels  = [FEAT_LABELS[k] for k, _ in sorted_eff]
delta_values = [v["delta"] for _, v in sorted_eff]
bar_colors   = [C_BLUE if d >= 0 else C_RED for d in delta_values]
hover_texts  = [
    (
        f"<b>{FEAT_LABELS[feat]}</b><br>"
        f"Low ({df[feat].quantile(0.10):.1f}{FEAT_UNITS[feat]}) "
        f"→ High ({df[feat].quantile(0.90):.1f}{FEAT_UNITS[feat]})<br>"
        f"Speed change: <b>{v['delta']:+.2f} mph</b>"
    )
    for feat, v in sorted_eff
]

fig_eff = go.Figure()
fig_eff.add_trace(go.Bar(
    x=delta_values,
    y=feat_labels,
    orientation="h",
    marker_color=bar_colors,
    hovertext=hover_texts,
    hoverinfo="text",
    text=[f"{d:+.2f} mph" for d in delta_values],
    textposition="outside",
    textfont=dict(color="white", size=12),
))
fig_eff.add_vline(x=0, line=dict(color="#666", width=1.5, dash="dot"))
fig_eff.update_layout(
    plot_bgcolor=BG, paper_bgcolor=BG,
    font=dict(color="white"),
    xaxis=dict(title="Speed adjustment (mph)", gridcolor="#2a2a3a",
               color="white", zeroline=False),
    yaxis=dict(gridcolor="#2a2a3a", color="white", tickfont=dict(size=13)),
    height=240,
    margin=dict(t=10, b=40, l=10, r=80),
    showlegend=False,
)
st.plotly_chart(fig_eff, use_container_width=True)
st.caption(
    f"Weather model CV MAE = ±{mae:.1f} mph  ·  "
    f"Historical residual range = {weather_pkg['residual_range'][0]:+.1f} to "
    f"{weather_pkg['residual_range'][1]:+.1f} mph from trend"
)

st.markdown("---")

# ── Section 3: 2025 race prediction ───────────────────────────────────────────
st.subheader(f"{RACE_YEAR} Race Day Prediction  (110th Indianapolis 500)")

st.markdown(f"**Race day conditions — Indianapolis, {RACE_DATE.strftime('%B %d, %Y')}**")
st.caption(f"Source: {weather['source']}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("🌡️ Temperature",   f"{weather['temperature']}°F")
c2.metric("💨 Wind Speed",    f"{weather['wind_speed']} mph")
c3.metric("💧 Humidity",      f"{weather['humidity']}%")
c4.metric("🌧️ Precipitation", f'{weather["precipitation"]}"')

st.markdown("&nbsp;")
st.markdown("**Predicted winning speed — step by step**")

col_a, col_b, col_c = st.columns(3)

def _card(col, title, value, unit, subtitle, border_color="#333"):
    col.markdown(
        f"""
        <div style="text-align:center; padding:20px 12px; background:{BG_CARD};
                    border-radius:10px; border:1px solid {border_color};">
          <div style="color:#aaa; font-size:0.82em; margin-bottom:6px;">{title}</div>
          <div style="color:{'white' if border_color=='#333' else border_color};
                      font-size:2.2em; font-weight:700; line-height:1.1;">{value}</div>
          <div style="color:#888; font-size:0.8em; margin-top:2px;">{unit}</div>
          <div style="color:#555; font-size:0.75em; margin-top:8px;
                      line-height:1.4;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

_card(
    col_a,
    "① Technology baseline",
    f"{trend_2025:.1f}",
    "mph",
    f"expected speed at 2025<br>technology level",
)

adj_color = C_BLUE if weather_adj >= 0 else C_RED
adj_label = "faster than trend" if weather_adj >= 0 else "slower than trend"
_card(
    col_b,
    "② Weather adjustment",
    adj_sign,
    "mph",
    f"{adj_label}<br>based on race-day conditions",
    border_color=adj_color,
)

_card(
    col_c,
    "🏆 Predicted winning speed",
    f"{final_pred}",
    "mph",
    f"±{mae:.1f} mph margin of error<br>"
    f"range: {final_pred - mae:.1f} – {final_pred + mae:.1f} mph",
    border_color=C_GOLD,
)

st.markdown("&nbsp;")

# ── Disclaimer ─────────────────────────────────────────────────────────────────
st.info(
    "⚠️ **Disclaimer:** This prediction is produced by a statistical model "
    "built for educational and entertainment purposes only. It is based solely "
    "on historical speed trends and race-day weather conditions. It does not "
    "account for driver skill, car setup, pit strategy, caution laps, crashes, "
    "mechanical failures, or any of the other factors that determine real race "
    "outcomes. This is **not** an official prediction.",
    icon=None,
)
