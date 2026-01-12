import os
import time
import random
import requests
import pandas as pd
from config import (
    NOAA_TOKEN,
    BASE_URL,
    DATASET_ID,
    DATATYPE_ID,
    STATION_ID,
    START_YEAR,
)


# Allow overriding the token with an environment variable for safety
NOAA_TOKEN = os.getenv("NOAA_TOKEN", NOAA_TOKEN)
if NOAA_TOKEN == "":
    raise RuntimeError("NOAA token not provided. Set NOAA_TOKEN in environment or config.py")

HEADERS = {"token": NOAA_TOKEN}


def fetch_year(year: int, max_retries: int = 5) -> list:
    params = {
        "datasetid": DATASET_ID,
        "datatypeid": DATATYPE_ID,
        "stationid": STATION_ID,
        "startdate": f"{year}-01-01",
        "enddate": f"{year}-12-31",
        "limit": 1000,
        "units": "metric",
    }

    attempt = 0
    while attempt < max_retries:
        try:
            resp = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=15)
        except requests.RequestException as exc:
            wait = (2 ** attempt) + random.random()
            print(f"Network error fetching {year} (attempt {attempt+1}/{max_retries}): {exc}. Retrying in {wait:.1f}s")
            time.sleep(wait)
            attempt += 1
            continue

        # retry on server errors or rate limits
        if resp.status_code in (429, 500, 502, 503, 504):
            wait = (2 ** attempt) + random.random()
            print(f"Server returned {resp.status_code} for {year} (attempt {attempt+1}/{max_retries}). Retrying in {wait:.1f}s")
            time.sleep(wait)
            attempt += 1
            continue

        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            print(f"HTTP error fetching {year}: {exc}")
            return []

        try:
            data = resp.json()
        except ValueError:
            print(f"Invalid JSON response for year {year}")
            return []

        return data.get("results", [])

    print(f"Failed to fetch data for year {year} after {max_retries} attempts")
    return []


def main():
    all_data = []

    last_year = pd.Timestamp.today().year
    for year in range(START_YEAR, last_year + 1):
        print(f"Fetching data for year {year}...")
        yearly = fetch_year(year)
        if yearly:
            all_data.extend(yearly)
        else:
            print(f"No data returned for year {year}")

    if not all_data:
        raise RuntimeError("No data fetched from NOAA API. Check token and network connectivity.")

    df = pd.DataFrame(all_data)

    # Normalize fields from NOAA API
    if "date" not in df.columns or "value" not in df.columns:
        raise RuntimeError("Unexpected API response structure; missing 'date' or 'value' fields")

    df["DATE"] = pd.to_datetime(df["date"])
    # NOAA GHCND TMAX values are reported in tenths of degrees — convert to degrees
    df["TMAX"] = df["value"] / 10.0

    df = df[["DATE", "TMAX"]].sort_values("DATE")

    # ensure output directory exists and handle if a file named 'data' exists
    if os.path.exists("data") and not os.path.isdir("data"):
        backup_name = "data.bak"
        print(f"Found a file named 'data' — renaming to '{backup_name}' to create the output directory.")
        os.replace("data", backup_name)

    os.makedirs("data", exist_ok=True)
    out_path = os.path.abspath(os.path.join("data", "daily_temperature.csv"))
    df.to_csv(out_path, index=False)

    print(f"✅ Real-time NOAA data updated: {out_path}")


if __name__ == "__main__":
    main()
