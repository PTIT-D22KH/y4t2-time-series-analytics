"""
Generate a realistic daily air pollution dataset for ARIMA/VAR analysis.
Based on patterns from real air quality monitoring stations.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

n_days = 1096  # 3 years
dates = pd.date_range(start="2021-01-01", periods=n_days, freq="D")
day_of_year = np.array([d.timetuple().tm_yday for d in dates])
day_of_week = np.array([d.weekday() for d in dates])

# Seasonal component: pollution is worse in winter
seasonal = np.cos(2 * np.pi * day_of_year / 365.25)

# Generate correlated pollution variables
# PM2.5 - primary pollutant
pm25_trend = 35 + 5 * np.arange(n_days) / n_days  # slight upward trend
pm25_seasonal = 15 * seasonal  # worse in winter
pm25_noise = np.random.normal(0, 8, n_days)
pm25 = pm25_trend + pm25_seasonal + pm25_noise
pm25 = np.maximum(pm25, 5)  # no negative PM2.5

# PM10 - correlated with PM2.5
pm10 = pm25 * (1.8 + 0.3 * np.random.randn(n_days)) + np.random.normal(0, 15, n_days)
pm10 = np.maximum(pm10, 10)

# SO2 - related to heating (winter peaks)
so2 = 8 + 6 * seasonal + 0.3 * pm25 + np.random.normal(0, 3, n_days)
so2 = np.maximum(so2, 1)

# NO2 - related to traffic (weekly pattern)
no2_weekly = 5 * np.sin(2 * np.pi * day_of_week / 7)
no2 = (
    25 + 10 * seasonal * 0.5 + no2_weekly + 0.2 * pm25 + np.random.normal(0, 5, n_days)
)
no2 = np.maximum(no2, 5)

# CO - correlated with NO2 (traffic)
co = 0.5 + 0.3 * seasonal + 0.02 * no2 + np.random.normal(0, 0.15, n_days)
co = np.maximum(co, 0.1)

# O3 - inverse relationship with some pollutants (photochemical)
o3 = 40 - 15 * seasonal + 0.1 * no2 + np.random.normal(0, 10, n_days)
o3 = np.maximum(o3, 5)

# Temperature - strongly seasonal
temp = 15 + 12 * seasonal + np.random.normal(0, 3, n_days)

# Create DataFrame
df = pd.DataFrame(
    {
        "Date": dates,
        "PM2.5": np.round(pm25, 1),
        "PM10": np.round(pm10, 1),
        "SO2": np.round(so2, 1),
        "NO2": np.round(no2, 1),
        "CO": np.round(co, 2),
        "O3": np.round(o3, 1),
        "TEMP": np.round(temp, 1),
    }
)

df.to_csv("air_pollution.csv", index=False)
print(f"Dataset saved: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
print(f"\nColumns: {list(df.columns[1:])}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nStatistics:\n{df.describe().round(2)}")
