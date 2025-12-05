import pandas as pd
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq
import gc
import warnings
warnings.filterwarnings('ignore')

RESAMPLE_FREQ = '100ms'

output_dir = Path("/home/ubuntu/symbol_data/")
resampled_dir = Path("/home/ubuntu/resampled/")
resampled_dir.mkdir(parents=True, exist_ok=True)


def resample_to_grid(df, freq='100ms'):
    df = df.set_index('timestamp')
    
    price_cols = ['price']
    volume_cols = ['quantity', 'signed_volume']
    feature_cols = [c for c in df.columns if c not in price_cols + volume_cols 
                    and c not in ['agg_trade_id', 'first_trade_id', 'last_trade_id', 
                                  'is_buyer_maker', 'direction', 'log_return']]
    
    agg_dict = {}
    for c in price_cols:
        agg_dict[c] = 'last'
    for c in volume_cols:
        agg_dict[c] = 'sum'
    for c in feature_cols:
        if c in df.columns:
            agg_dict[c] = 'last'
    
    resampled = df.resample(freq).agg(agg_dict)
    resampled[price_cols] = resampled[price_cols].ffill()
    resampled[feature_cols] = resampled[feature_cols].ffill()
    resampled[volume_cols] = resampled[volume_cols].fillna(0)
    resampled['log_return'] = np.log(resampled['price'] / resampled['price'].shift(1))
    
    return resampled.dropna(subset=['price'])


def resample_batch_to_chunks(input_path, symbol, batch_idx, freq='100ms'):
    pf = pq.ParquetFile(input_path)
    num_row_groups = pf.metadata.num_row_groups
    print(f"    {num_row_groups} row groups")
    
    for i in range(num_row_groups):
        out_path = resampled_dir / f"{symbol}_batch_{batch_idx:03d}_rg_{i:03d}.parquet"
        
        if out_path.exists():
            print(f"      Row group {i+1}/{num_row_groups}: skipped (exists)")
            continue
        
        table = pf.read_row_group(i)
        df = table.to_pandas()
        
        resampled = resample_to_grid(df, freq)
        resampled = resampled.astype({col: 'float32' for col in resampled.select_dtypes('float64').columns})
        resampled.to_parquet(out_path)
        
        print(f"      Row group {i+1}/{num_row_groups}: {len(df):,} -> {len(resampled):,}")
        
        del df, table, resampled
        gc.collect()


def resample_all_batches():
    for symbol in ['SOLUSDT', 'BTCUSDT', 'ETHUSDT']:
        files = sorted(output_dir.glob(f"{symbol}_batch_*.parquet"))
        print(f"\nResampling {symbol}: {len(files)} batch files")
        
        for f in files:
            batch_idx = int(f.stem.split('_')[2])
            
            existing = list(resampled_dir.glob(f"{symbol}_batch_{batch_idx:03d}_rg_*.parquet"))
            if existing:
                print(f"  {f.name}: {len(existing)} row group files exist, checking for more...")
            else:
                print(f"  Processing {f.name}...")
            
            resample_batch_to_chunks(f, symbol, batch_idx, RESAMPLE_FREQ)


if __name__ == "__main__":
    print("=" * 60)
    print("Resampling batches to row group files")
    print("=" * 60)
    resample_all_batches()
    print("\nResampling complete.")
