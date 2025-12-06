from datetime import date, timedelta
import requests
import zipfile
import io
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

start_date = date(2020, 9, 14)
base_url = "https://data.binance.vision"

symbol_to_aggs = {symbol: [] for symbol in ["SOLUSDT", "BTCUSDT", "ETHUSDT"]}

for symbol in tqdm(symbol_to_aggs, desc="Downloading by symbol"):

    accumulated_rows = []
    for d in tqdm(pd.date_range(start=start_date, end=date.today()), desc=f"Downloading {symbol} daily data"):
        date_str = d.strftime("%Y-%m-%d")
        url = f"{base_url}/data/futures/um/daily/klines/{symbol}/8h/{symbol}-8h-{date_str}.zip"
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            print(f"Failed to download {url}")
            continue
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_filename = z.namelist()[0]
            with z.open(csv_filename) as f:
                # Binance klines CSV has no headers, specify column names
                df = pd.read_csv(f, header=None, names=[
                    'open_time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'count', 'taker_buy_volume',
                    'taker_buy_quote_volume', 'ignore'
                ])

                df = df.sort_values(by="open_time")
                
                # Ensure we have at least 3 rows (for 8h, 16h, 24h calculations)
                if len(df) < 3:
                    print(f"Not enough data for {symbol} on {date_str}")
                    continue
                
                df["open"] = pd.to_numeric(df["open"], errors="coerce")
                df["close"] = pd.to_numeric(df["close"], errors="coerce")
                
                df = df.dropna(subset=["open", "close"])
                
                # Check again if we still have at least 3 rows after dropping NaNs
                if len(df) < 3:
                    print(f"Not enough data for {symbol} on {date_str} after dropping NaNs")
                    continue
                
                open_price = df["open"]
                close_price = df["close"]

                f_8h_change = close_price.iloc[2] - open_price.iloc[2]
                f_16h_change = close_price.iloc[2] - open_price.iloc[1]
                f_24h_change = close_price.iloc[2] - open_price.iloc[0]
                f_close_price = close_price.iloc[-1]

                volume = pd.to_numeric(df["volume"], errors="coerce")
                if volume.notna().any():
                    f_volume = volume.mean()
                else:
                    f_volume = np.nan
                
                accumulated_rows.append(
                    [symbol, date_str, f_8h_change, f_16h_change, f_24h_change, f_close_price, f_volume]
                )

    os.makedirs("data", exist_ok=True)
    accumulated_df = pd.DataFrame(
        accumulated_rows,
        columns=["symbol", "date", "8h_change", "16h_change", "24h_change", "close_price", "volume"],
    )
    accumulated_df.to_csv(f"data/trading_prices_{symbol}.csv", index=False)

