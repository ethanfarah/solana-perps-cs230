import requests
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pathlib import Path
import zipfile
import io
import gc

start_date = date(2025, 1, 1)
num_days = 100
batch_size = 10
base_url = "https://data.binance.vision"
symbols = ['SOLUSDT', 'BTCUSDT', 'ETHUSDT']

output_dir = "/home/ubuntu/symbol_data"
output_dir.mkdir(parents=True, exist_ok=True)

def add_features(df):
    print(f"  adding features to {len(df):,} rows...", flush=True)
    
    df['log_return'] = np.log(df['price'] / df['price'].shift(1))
    df['direction'] = np.where(df['is_buyer_maker'], -1, 1)
    df['signed_volume'] = df['quantity'] * df['direction']
    print("  [1/7] basic", flush=True)
    
    df['price_ewma_100ms'] = df['price'].ewm(halflife='100ms', times=df['timestamp']).mean()
    df['price_ewma_1s'] = df['price'].ewm(halflife='1s', times=df['timestamp']).mean()
    df['price_ewma_5s'] = df['price'].ewm(halflife='5s', times=df['timestamp']).mean()
    df['price_ewma_30s'] = df['price'].ewm(halflife='30s', times=df['timestamp']).mean()
    df['price_ewma_5m'] = df['price'].ewm(halflife='5min', times=df['timestamp']).mean()
    df['price_ewma_15m'] = df['price'].ewm(halflife='15min', times=df['timestamp']).mean()
    df['price_ewma_60m'] = df['price'].ewm(halflife='60min', times=df['timestamp']).mean()
    print("  [2/7] price_ewma", flush=True)
    
    sq_returns = df['log_return'].pow(2)
    df['realized_vol_100ms'] = sq_returns.ewm(halflife='100ms', times=df['timestamp']).mean().pow(0.5)
    df['realized_vol_1s'] = sq_returns.ewm(halflife='1s', times=df['timestamp']).mean().pow(0.5)
    df['realized_vol_30s'] = sq_returns.ewm(halflife='30s', times=df['timestamp']).mean().pow(0.5)
    df['realized_vol_5m'] = sq_returns.ewm(halflife='5min', times=df['timestamp']).mean().pow(0.5)
    df['realized_vol_15m'] = sq_returns.ewm(halflife='15min', times=df['timestamp']).mean().pow(0.5)
    df['realized_vol_60m'] = sq_returns.ewm(halflife='60min', times=df['timestamp']).mean().pow(0.5)
    print("  [3/7] realized_vol", flush=True)
    
    df['flow_imbalance_100ms'] = df['direction'].ewm(halflife='100ms', times=df['timestamp']).mean()
    df['flow_imbalance_1s'] = df['direction'].ewm(halflife='1s', times=df['timestamp']).mean()
    df['flow_imbalance_5s'] = df['direction'].ewm(halflife='5s', times=df['timestamp']).mean()
    df['flow_imbalance_30s'] = df['direction'].ewm(halflife='30s', times=df['timestamp']).mean()
    df['flow_imbalance_5m'] = df['direction'].ewm(halflife='5min', times=df['timestamp']).mean()
    df['flow_imbalance_15m'] = df['direction'].ewm(halflife='15min', times=df['timestamp']).mean()
    df['flow_imbalance_60m'] = df['direction'].ewm(halflife='60min', times=df['timestamp']).mean()
    print("  [4/7] flow_imbalance", flush=True)
    
    for halflife, suffix in [('100ms', '100ms'), ('1s', '1s'), ('5s', '5s'), ('30s', '30s'), ('5min', '5m'), ('15min', '15m'), ('60min', '60m')]:
        signed_vol_ewma = df['signed_volume'].ewm(halflife=halflife, times=df['timestamp']).mean()
        vol_ewma = df['quantity'].ewm(halflife=halflife, times=df['timestamp']).mean()
        df[f'vw_flow_imbalance_{suffix}'] = signed_vol_ewma / vol_ewma
    print("  [5/7] vw_flow_imbalance", flush=True)
    
    df['momentum_100ms'] = (df['price'] - df['price_ewma_100ms']) / df['price_ewma_100ms']
    df['momentum_1s'] = (df['price'] - df['price_ewma_1s']) / df['price_ewma_1s']
    df['momentum_5s'] = (df['price'] - df['price_ewma_5s']) / df['price_ewma_5s']
    df['momentum_30s'] = (df['price'] - df['price_ewma_30s']) / df['price_ewma_30s']
    df['momentum_5m'] = (df['price'] - df['price_ewma_5m']) / df['price_ewma_5m']
    df['momentum_15m'] = (df['price'] - df['price_ewma_15m']) / df['price_ewma_15m']
    df['momentum_60m'] = (df['price'] - df['price_ewma_60m']) / df['price_ewma_60m']
    print("  [6/7] momentum", flush=True)
    
    for halflife, suffix in [('100ms', '100ms'), ('1s', '1s'), ('5s', '5s'), 
                              ('30s', '30s'), ('5min', '5m'), ('15min', '15m'), ('60min', '60m')]:
        vol_ewma = df['quantity'].ewm(halflife=halflife, times=df['timestamp']).mean()
        df[f'vol_ewma_{suffix}'] = vol_ewma
        df[f'vol_intensity_{suffix}'] = df['quantity'] / vol_ewma
    print("  [7/7] vol_intensity", flush=True)
    
    return df


def download_batch(symbol, batch_start, batch_days):
    dfs = []
    for i in range(batch_days):
        d = batch_start + timedelta(days=i)
        date_str = d.strftime("%Y-%m-%d")
        print(f"{symbol} {date_str} ({i+1}/{batch_days})", flush=True)
        url = f"{base_url}/data/futures/um/daily/aggTrades/{symbol}/{symbol}-aggTrades-{date_str}.zip"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                csv_filename = z.namelist()[0]
                with z.open(csv_filename) as f:
                    df = pd.read_csv(f)
            
            df = df.rename(columns={'transact_time': 'timestamp'})
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['price'] = df['price'].astype(float)
            df['quantity'] = df['quantity'].astype(float)
            dfs.append(df)
        except:
            continue
    
    if not dfs:
        return None
    
    df = pd.concat(dfs, ignore_index=True)
    df = add_features(df)
    return df


for symbol in symbols:
    for batch_idx in range(0, num_days, batch_size):
        file_path = output_dir / f"{symbol}_batch_{batch_idx:03d}.parquet"
        if file_path.exists():
            print(f"Skipping {file_path} (exists)", flush=True)
            continue
        
        batch_start = start_date + timedelta(days=batch_idx)
        batch_days = min(batch_size, num_days - batch_idx)
        
        df = download_batch(symbol, batch_start, batch_days)
        
        if df is not None:
            df.to_parquet(file_path, index=False)
            print(f"Saved {file_path}", flush=True)
            del df
            gc.collect()
