"""
fetch_weather.py
Fetches historical race-day weather for every Indy 500 in indy500_results.csv
using the free Open-Meteo historical archive API (ERA5, no key required).
Writes results to data/weather_data.csv.

Coverage note: ERA5 data begins in 1940, so races from 1911-1939 will have
no weather data and will be recorded as blank rows.
"""

import datetime
import time

import pandas as pd
import requests

LAT = 39.7684
LON = -86.1581
API_URL = "https://archive-api.open-meteo.com/v1/archive"


def race_date(year: int) -> datetime.date:
    """
    Return the best-estimate calendar date of the Indy 500 for a given year.

    Rules:
      - Pre-1971: traditionally held on May 30; shifted to May 31 when
        May 30 fell on a Sunday.
      - 1971+: held on the Sunday before Memorial Day (last Monday of May).
      - Known exceptions are captured in the overrides dict.
    """
    overrides = {
        2020: datetime.date(2020, 8, 23),  # postponed to August due to COVID-19
    }
    if year in overrides:
        return overrides[year]

    if year < 1971:
        d = datetime.date(year, 5, 30)
        # Shift to May 31 when May 30 falls on a Sunday
        return datetime.date(year, 5, 31) if d.weekday() == 6 else d

    # Find the last Monday of May, then step back one day to get Sunday
    last = datetime.date(year, 5, 31)
    last_monday = last - datetime.timedelta(days=last.weekday())  # weekday 0 = Monday
    return last_monday - datetime.timedelta(days=1)


def fetch_weather(date: datetime.date) -> dict | None:
    """
    Query Open-Meteo archive for hourly data on the given date.

    Returns a dict with race-time conditions (noon reading) and the
    daily precipitation total, or None if the request fails or returns
    no data (e.g. before ERA5 coverage begins in 1940).
    """
    params = {
        "latitude":           LAT,
        "longitude":          LON,
        "start_date":         date.isoformat(),
        "end_date":           date.isoformat(),
        "hourly":             "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
        "temperature_unit":   "fahrenheit",
        "wind_speed_unit":    "mph",
        "precipitation_unit": "inch",
        "timezone":           "America/Indiana/Indianapolis",
    }

    try:
        resp = requests.get(API_URL, params=params, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"    [request error: {exc}]")
        return None

    hourly = resp.json().get("hourly", {})
    temp   = hourly.get("temperature_2m",    [])
    rh     = hourly.get("relative_humidity_2m", [])
    precip = hourly.get("precipitation",     [])
    wind   = hourly.get("wind_speed_10m",    [])

    if not temp:
        return None

    # Use the noon reading (index 12) for point-in-time race conditions.
    # Race starts ~12:00–1:00 PM local time, so noon is a reasonable proxy.
    # Sum all 24 hourly precipitation values for the full-day total.
    noon = 12
    return {
        "temperature":   round(temp[noon],  1) if noon < len(temp)   else None,
        "wind_speed":    round(wind[noon],  1) if noon < len(wind)   else None,
        "humidity":      round(rh[noon],    1) if noon < len(rh)     else None,
        "precipitation": round(sum(x for x in precip if x is not None), 2),
    }


def main():
    results = pd.read_csv("data/indy500_results.csv")
    years   = sorted(results["year"].unique())

    print(f"Fetching weather for {len(years)} races...\n")

    rows = []
    for year in years:
        d = race_date(year)
        print(f"  {year}  ({d})", end="  ", flush=True)

        weather = fetch_weather(d)

        if weather:
            rows.append({"year": year, **weather})
            print(
                f"{weather['temperature']}°F  |  "
                f"wind {weather['wind_speed']} mph  |  "
                f"humidity {weather['humidity']}%  |  "
                f"precip {weather['precipitation']}\""
            )
        else:
            rows.append({
                "year":          year,
                "temperature":   None,
                "wind_speed":    None,
                "humidity":      None,
                "precipitation": None,
            })
            print("no data (outside ERA5 coverage or API error)")

        time.sleep(0.35)  # stay well within the free-tier rate limit

    df = pd.DataFrame(rows, columns=["year", "temperature", "wind_speed",
                                     "humidity", "precipitation"])
    df.to_csv("data/weather_data.csv", index=False)

    covered = df["temperature"].notna().sum()
    print(f"\nDone. {covered}/{len(rows)} races have weather data.")
    print("Results written to data/weather_data.csv")


if __name__ == "__main__":
    main()
