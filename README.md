# Indy 500 Historical Dashboard

An interactive dashboard built with Streamlit to explore and visualize over 100 years of Indianapolis 500 race history.

## Project Structure

```
indy500-dashboard/
├── data/               # Raw and processed datasets (CSV files)
├── pages/              # Streamlit multi-page app pages
├── utils/              # Helper functions for data loading and processing
├── app.py              # Main Streamlit entry point
└── requirements.txt    # Python dependencies
```

## Planned Features

- **Race Results Browser** — Filter and search historical results by year, driver, team, or nationality
- **Winners Over Time** — Timeline of race winners with speed trends and car makes
- **Driver Profiles** — Stats and career summaries for notable drivers
- **Speed & Performance Trends** — How average race speeds have evolved since 1911
- **Starting Grid vs. Finish** — Visualization of how qualifying position correlates with race outcome
- **Prize Money History** — Total purse and winner earnings over the decades
- **Nationality Breakdown** — Geographic origins of drivers and constructors

## Data

Historical race data sourced from public records covering the Indy 500 from 1911 to the present, including:
- Year, winner, car number, team/entrant
- Car make and engine
- Qualifying and race average speeds
- Starting position and laps led
- Prize money

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```
