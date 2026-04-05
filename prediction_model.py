"""
prediction_model.py
Two-stage prediction pipeline for Indy 500 winning average speed.

  Stage 1 — Linear regression on year + technology_index
            Captures the long-run technology trend: how much faster cars
            get each decade independent of weather.

  Stage 2 — Random Forest on weather residuals
            Models only the gap between the technology baseline and the
            actual result, isolating the pure weather effect.

  Final prediction = Stage 1 (trend) + Stage 2 (weather adjustment)

Saves model_trend.pkl and model_weather.pkl for use in the dashboard.
"""

import datetime
import pickle

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

LAT            = 39.7684
LON            = -86.1581
RACE_DATE_2026 = datetime.date(2026, 5, 24)
YEAR_MIN, YEAR_MAX = 1911, 2024

TECH_FEATURES    = ["year", "technology_index"]
WEATHER_FEATURES = ["temperature", "wind_speed", "humidity", "precipitation"]
TARGET           = "avg_speed_mph"

W = 58   # print width

# ── 1. Load ────────────────────────────────────────────────────────────────────
df        = pd.read_csv("data/combined_data.csv")
X_tech    = df[TECH_FEATURES].values
X_weather = df[WEATHER_FEATURES].values
y         = df[TARGET].values

# ── 2. Stage 1 — Technology trend (linear regression) ─────────────────────────
trend_model = LinearRegression()
trend_model.fit(X_tech, y)

baseline_pred = trend_model.predict(X_tech)
residuals     = y - baseline_pred          # positive = faster than expected

trend_r2  = trend_model.score(X_tech, y)
trend_mae = float(np.mean(np.abs(residuals)))

# Derive a single "mph gained per year" figure that is independent of how
# the solver split the coefficient weight across the two collinear inputs.
def _trend_at(yr):
    ti = (yr - YEAR_MIN) / (YEAR_MAX - YEAR_MIN)
    return float(trend_model.predict([[yr, ti]])[0])

mph_per_year = _trend_at(2024) - _trend_at(2023)

# ── 3. Stage 2 — Weather effect (Random Forest on residuals) ──────────────────
weather_model = RandomForestRegressor(n_estimators=200, random_state=42)
weather_model.fit(X_weather, residuals)

importances = dict(zip(WEATHER_FEATURES, weather_model.feature_importances_))

# Cross-validate stage 2 on the residuals.
# Note: residuals are computed from the stage-1 fit on ALL data, so CV
# scores are slightly optimistic — acceptable here given the small dataset
# and low variance of a linear trend model.
_rf_cv = RandomForestRegressor(n_estimators=200, random_state=42)
cv_mae_w = -cross_val_score(_rf_cv, X_weather, residuals,
                             cv=5, scoring="neg_mean_absolute_error")
cv_r2_w  =  cross_val_score(_rf_cv, X_weather, residuals,
                             cv=5, scoring="r2")
weather_mae = float(cv_mae_w.mean())
weather_r2  = float(cv_r2_w.mean())

residual_min = float(residuals.min())
residual_max = float(residuals.max())

# ── 4. Combined model accuracy on training data ────────────────────────────────
combined_pred = baseline_pred + weather_model.predict(X_weather)
combined_mae  = float(np.mean(np.abs(y - combined_pred)))
ss_res = np.sum((y - combined_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
combined_r2   = float(1 - ss_res / ss_tot)

# ── 5. Fetch 2026 race-day weather ────────────────────────────────────────────
def _fetch_hourly(date: datetime.date) -> dict | None:
    params = {
        "latitude":           LAT,
        "longitude":          LON,
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
        resp = requests.get("https://api.open-meteo.com/v1/forecast",
                            params=params, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"  [forecast API error: {exc}]")
        return None

    hourly = resp.json().get("hourly", {})
    temp   = hourly.get("temperature_2m",       [])
    rh     = hourly.get("relative_humidity_2m", [])
    precip = hourly.get("precipitation",        [])
    wind   = hourly.get("wind_speed_10m",       [])

    noon = 12
    if not temp or noon >= len(temp) or temp[noon] is None:
        return None

    return {
        "temperature":   round(temp[noon], 1),
        "wind_speed":    round(wind[noon], 1),
        "humidity":      round(rh[noon],   1),
        "precipitation": round(sum(x for x in precip if x is not None), 2),
    }


def _may_climate_normal() -> dict:
    """Mean of all historical May race-day conditions (excluding 2020/August)."""
    w   = pd.read_csv("data/weather_data.csv").dropna()
    w   = w[w["year"] != 2020]
    avg = w[WEATHER_FEATURES].mean()
    return {col: round(float(avg[col]), 1) for col in avg.index}


print("Fetching 2026 race-day weather forecast...")
weather_2026 = _fetch_hourly(RACE_DATE_2026)
if weather_2026 is not None:
    weather_source = "Open-Meteo forecast"
    print("  Live forecast retrieved.\n")
else:
    weather_2026   = _may_climate_normal()
    weather_source = "historical May climate average (date beyond 16-day forecast window)"
    print("  Using historical May climate average.\n")

# ── 6. Predict 2026 ───────────────────────────────────────────────────────────
tech_2026 = round((2026 - YEAR_MIN) / (YEAR_MAX - YEAR_MIN), 4)  # ≈ 1.018

trend_2026      = _trend_at(2026)
weather_adj     = float(weather_model.predict([[
    weather_2026["temperature"],
    weather_2026["wind_speed"],
    weather_2026["humidity"],
    weather_2026["precipitation"],
]])[0])
predicted_speed = round(trend_2026 + weather_adj, 2)

# ── 7. Save ────────────────────────────────────────────────────────────────────
trend_pkg = {
    "model":         trend_model,
    "features":      TECH_FEATURES,
    "target":        TARGET,
    "r2":            round(trend_r2, 4),
    "mae":           round(trend_mae, 2),
    "mph_per_year":  round(mph_per_year, 3),
}
weather_pkg = {
    "model":            weather_model,
    "features":         WEATHER_FEATURES,
    "target":           "residual_from_trend",
    "cv_mae":           round(weather_mae, 2),
    "cv_r2":            round(weather_r2, 4),
    "importances":      {k: round(v, 4) for k, v in importances.items()},
    "residual_range":   (round(residual_min, 2), round(residual_max, 2)),
    "prediction_2026":  predicted_speed,
    "trend_2026":       round(trend_2026, 2),
    "weather_adj_2026": round(weather_adj, 2),
    "weather_2026":     weather_2026,
    "weather_source":   weather_source,
    "race_date_2026":   RACE_DATE_2026,
    "tech_index_2026":  tech_2026,
    "combined_mae":     round(combined_mae, 2),
    "combined_r2":      round(combined_r2, 4),
}

with open("model_trend.pkl",   "wb") as f: pickle.dump(trend_pkg,   f)
with open("model_weather.pkl", "wb") as f: pickle.dump(weather_pkg, f)

# ── Print results ──────────────────────────────────────────────────────────────
SEP  = "═" * W
sep  = "─" * W

print(SEP)
print("  Stage 1 — Technology Trend  (Linear Regression)")
print(SEP)
print(f"  Training rows   : {len(df)}  "
      f"({int(df['year'].min())}–{int(df['year'].max())})")
print(f"  R²              : {trend_r2:.3f}")
print(f"  MAE             : ±{trend_mae:.1f} mph")
print(f"  Trend rate      : +{mph_per_year:.2f} mph gained per year on average")
print()
print(f"  The linear trend explains {trend_r2*100:.1f}% of the variance in winning")
print(f"  speeds, confirming technology is the dominant driver.")

print()
print(SEP)
print("  Stage 2 — Weather Effect  (Random Forest on residuals)")
print(SEP)
print(f"  Residual range  : {residual_min:+.1f} to {residual_max:+.1f} mph")
print(f"  CV R² (residuals): {weather_r2:.3f}")
print(f"  CV MAE          : ±{weather_mae:.1f} mph")
print()
print("  Variable importances (share of weather-effect prediction):")
for feat, imp in sorted(importances.items(), key=lambda kv: -kv[1]):
    bar   = "█" * round(imp * 34)
    pct   = imp * 100
    print(f"    {feat:<14}  {pct:5.1f}%  {bar}")
print()
print(f"  In plain English: after removing the technology trend, weather")
print(f"  conditions shift the winning speed by ±{weather_mae:.1f} mph on average,")
print(f"  with a historical swing of {residual_min:+.1f} to {residual_max:+.1f} mph.")

print()
print(SEP)
print("  Combined model accuracy (trend + weather, on training data)")
print(SEP)
print(f"  R²  : {combined_r2:.3f}")
print(f"  MAE : ±{combined_mae:.1f} mph")

print()
print(SEP)
print(f"  2026 Race Prediction  —  {RACE_DATE_2026}  (110th Indy 500)")
print(SEP)
print(f"  Weather source  : {weather_source}")
print(f"  Temperature     : {weather_2026['temperature']}°F")
print(f"  Wind speed      : {weather_2026['wind_speed']} mph")
print(f"  Humidity        : {weather_2026['humidity']}%")
print(f"  Precipitation   : {weather_2026['precipitation']}\"")
print()
print(f"  {sep}")
adj_sign = f"+{weather_adj:.1f}" if weather_adj >= 0 else f"{weather_adj:.1f}"
print(f"  Technology trend baseline :  {trend_2026:.1f} mph")
print(f"  Weather adjustment        :  {adj_sign} mph")
print(f"  {sep}")
print(f"  Predicted winning speed   :  {predicted_speed} mph")
print(f"  Margin of error           :  ±{weather_mae:.1f} mph")
print(f"  Plausible range           :  "
      f"{predicted_speed - weather_mae:.1f} – {predicted_speed + weather_mae:.1f} mph")
print(SEP)

print("\nSaved  →  model_trend.pkl   (Stage 1: technology trend)")
print("         model_weather.pkl  (Stage 2: weather effect + 2026 prediction)")
