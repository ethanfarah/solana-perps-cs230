import requests
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pathlib import Path
import zipfile
import io
from tqdm import tqdm

start_date = date(2025, 1, 1)
num_days = 100
freq = "100ms"
base_url = "https://data.binance.vision"
symbols = ["SOLUSDT", "BTCUSDT", "ETHUSDT"]

root = Path("/home/ubuntu/hft_data")
trades_dir = root / "trades"
bars_dir = root / "bars"
features_dir = root / "features"
final_path = root / "dataset_20d_100ms.parquet"

for d in [root, trades_dir, bars_dir, features_dir]:
    d.mkdir(parents=True, exist_ok=True)

BASE_FEATURES = [
    "log_return_100ms",
    "momentum_100ms", "momentum_1s", "momentum_5s", "momentum_30s", "momentum_5m", "momentum_15m", "momentum_60m",
    "flow_imbalance_100ms", "flow_imbalance_1s", "flow_imbalance_5s", "flow_imbalance_30s", "flow_imbalance_5m", "flow_imbalance_15m", "flow_imbalance_60m",
    "vw_flow_imbalance_100ms", "vw_flow_imbalance_1s", "vw_flow_imbalance_5s", "vw_flow_imbalance_30s", "vw_flow_imbalance_5m", "vw_flow_imbalance_15m", "vw_flow_imbalance_60m",
    "realized_vol_100ms", "realized_vol_1s", "realized_vol_30s", "realized_vol_5m", "realized_vol_15m", "realized_vol_60m",
    "vol_intensity_100ms", "vol_intensity_1s", "vol_intensity_5s", "vol_intensity_30s", "vol_intensity_5m", "vol_intensity_15m", "vol_intensity_60m",
]

LAG_ROWS = {
    "cur": 0,
    "lag_100ms": 1,
    "lag_500ms": 5,
    "lag_1s": 10,
}


def download_trades(symbol):  # Download agg trades for symbol and normalize schema
    dfs = []
    for i in tqdm(range(num_days), desc=f"Downloading {symbol}", leave=False):
        d = start_date + timedelta(days=i)
        date_str = d.strftime("%Y-%m-%d")
        url = f"{base_url}/data/futures/um/daily/aggTrades/{symbol}/{symbol}-aggTrades-{date_str}.zip"
        r = requests.get(url)  # Pull compressed agg trades for the day
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            df = pd.read_csv(f)  # Load single day trade data
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)  # Combine daily data
    if "transact_time" in df.columns:
        ts_col = "transact_time"
    elif "timestamp" in df.columns:
        ts_col = "timestamp"
    elif "T" in df.columns:
        ts_col = "T"
    else:
        raise ValueError("no timestamp column")
    if "price" in df.columns:
        price_col = "price"
    elif "p" in df.columns:
        price_col = "p"
    else:
        raise ValueError("no price column")
    if "quantity" in df.columns:
        qty_col = "quantity"
    elif "q" in df.columns:
        qty_col = "q"
    else:
        raise ValueError("no quantity column")
    if "is_buyer_maker" in df.columns:
        maker_col = "is_buyer_maker"
    elif "m" in df.columns:
        maker_col = "m"
    else:
        raise ValueError("no is_buyer_maker column")
    out = pd.DataFrame()  # Build cleaned trade frame
    out["timestamp"] = pd.to_datetime(df[ts_col], unit="ms")
    out["price"] = df[price_col].astype("float64")
    out["quantity"] = df[qty_col].astype("float64")
    out["is_buyer_maker"] = df[maker_col].astype(bool)
    out = out.sort_values("timestamp").reset_index(drop=True)
    return out


def build_bars(trades, index):  # Resample trades into aligned bar series
    df = trades.copy()
    df["direction"] = np.where(df["is_buyer_maker"], -1.0, 1.0)  # Trade direction proxy
    df["signed_volume"] = df["quantity"] * df["direction"]  # Signed size per trade
    df = df.set_index("timestamp")
    agg = {
        "price": "last",
        "quantity": "sum",
        "signed_volume": "sum",
    }
    bars = df.resample(freq).agg(agg)  # Aggregate trades per bucket
    bars = bars.reindex(index)  # Align to master index
    bars["price"] = bars["price"].ffill()  # Carry last price forward
    bars["quantity"] = bars["quantity"].fillna(0.0)
    bars["signed_volume"] = bars["signed_volume"].fillna(0.0)
    bars["direction"] = np.sign(bars["signed_volume"])  # Direction of flow
    return bars


def compute_base_features(bars):  # Enrich bars with momentum flow and volatility stats
    df = bars.copy()
    df["log_return_100ms"] = np.log(df["price"] / df["price"].shift(1))  # Instantaneous log return
    sq = df["log_return_100ms"] * df["log_return_100ms"]
    t = df.index
    df["price_ewma_100ms"] = df["price"].ewm(halflife="100ms", times=t).mean()
    df["price_ewma_1s"] = df["price"].ewm(halflife="1s", times=t).mean()
    df["price_ewma_5s"] = df["price"].ewm(halflife="5s", times=t).mean()
    df["price_ewma_30s"] = df["price"].ewm(halflife="30s", times=t).mean()
    df["price_ewma_5m"] = df["price"].ewm(halflife="5min", times=t).mean()
    df["price_ewma_15m"] = df["price"].ewm(halflife="15min", times=t).mean()
    df["price_ewma_60m"] = df["price"].ewm(halflife="60min", times=t).mean()
    df["realized_vol_100ms"] = sq.ewm(halflife="100ms", times=t).mean() ** 0.5
    df["realized_vol_1s"] = sq.ewm(halflife="1s", times=t).mean() ** 0.5
    df["realized_vol_30s"] = sq.ewm(halflife="30s", times=t).mean() ** 0.5
    df["realized_vol_5m"] = sq.ewm(halflife="5min", times=t).mean() ** 0.5
    df["realized_vol_15m"] = sq.ewm(halflife="15min", times=t).mean() ** 0.5
    df["realized_vol_60m"] = sq.ewm(halflife="60min", times=t).mean() ** 0.5
    df["flow_imbalance_100ms"] = df["direction"].ewm(halflife="100ms", times=t).mean()
    df["flow_imbalance_1s"] = df["direction"].ewm(halflife="1s", times=t).mean()
    df["flow_imbalance_5s"] = df["direction"].ewm(halflife="5s", times=t).mean()
    df["flow_imbalance_30s"] = df["direction"].ewm(halflife="30s", times=t).mean()
    df["flow_imbalance_5m"] = df["direction"].ewm(halflife="5min", times=t).mean()
    df["flow_imbalance_15m"] = df["direction"].ewm(halflife="15min", times=t).mean()
    df["flow_imbalance_60m"] = df["direction"].ewm(halflife="60min", times=t).mean()
    eps = 1e-12
    for hl, suf in [("100ms", "100ms"), ("1s", "1s"), ("5s", "5s"), ("30s", "30s"), ("5min", "5m"), ("15min", "15m"), ("60min", "60m")]:
        sv_ewm = df["signed_volume"].ewm(halflife=hl, times=t).mean()  # Flow smoothed at scale
        v_ewm = df["quantity"].ewm(halflife=hl, times=t).mean()  # Volume smoothed at scale
        df[f"vw_flow_imbalance_{suf}"] = sv_ewm / (v_ewm + eps)
    df["momentum_100ms"] = (df["price"] - df["price_ewma_100ms"]) / df["price_ewma_100ms"]
    df["momentum_1s"] = (df["price"] - df["price_ewma_1s"]) / df["price_ewma_1s"]
    df["momentum_5s"] = (df["price"] - df["price_ewma_5s"]) / df["price_ewma_5s"]
    df["momentum_30s"] = (df["price"] - df["price_ewma_30s"]) / df["price_ewma_30s"]
    df["momentum_5m"] = (df["price"] - df["price_ewma_5m"]) / df["price_ewma_5m"]
    df["momentum_15m"] = (df["price"] - df["price_ewma_15m"]) / df["price_ewma_15m"]
    df["momentum_60m"] = (df["price"] - df["price_ewma_60m"]) / df["price_ewma_60m"]
    for hl, suf in [("100ms", "100ms"), ("1s", "1s"), ("5s", "5s"), ("30s", "30s"), ("5min", "5m"), ("15min", "15m"), ("60min", "60m")]:
        v_ewm = df["quantity"].ewm(halflife=hl, times=t).mean()  # Baseline volume per scale
        df[f"vol_ewma_{suf}"] = v_ewm
        df[f"vol_intensity_{suf}"] = df["quantity"] / (v_ewm + eps)  # Volume surprise metric
    return df


def create_lagged(df, asset):  # Build lagged versions of base features by asset
    data = {}
    for feat in BASE_FEATURES:
        if feat not in df.columns:
            continue
        s = df[feat]
        for lag_name, k in LAG_ROWS.items():
            name = f"{asset}_{feat}_{lag_name}"  # Prefix feature with asset and lag
            if k == 0:
                data[name] = s.values  # Current observation
            else:
                data[name] = s.shift(k).values  # Lagged observation
    return pd.DataFrame(data, index=df.index)


def create_targets(sol_df):  # Produce forward return and volatility labels
    n = len(sol_df)
    price = sol_df["price"].values
    fwd_log_return_100ms = np.full(n, np.nan)  # Placeholder for next step return
    fwd_log_return_100ms[:-1] = np.log(price[1:] / price[:-1])
    lr = pd.Series(sol_df["log_return_100ms"].values, index=sol_df.index)
    fwd_vol_30s = lr.rolling(300).std().shift(-300)  # Future realized volatility 30s
    fwd_vol_1m = lr.rolling(600).std().shift(-600)  # Future realized volatility 1m
    out = pd.DataFrame(
        {
            "fwd_log_return_100ms": fwd_log_return_100ms,
            "fwd_vol_30s": fwd_vol_30s.values,
            "fwd_vol_1m": fwd_vol_1m.values,
        },
        index=sol_df.index,
    )
    return out


if __name__ == "__main__":
    trades = {}
    for symbol in tqdm(symbols, desc="Downloading trades"):
        df = download_trades(symbol)
        trades[symbol] = df
        df.to_parquet(trades_dir / f"{symbol}_trades.parquet", index=False)
    start_ts = pd.Timestamp(start_date)
    end_ts = start_ts + timedelta(days=num_days)
    index = pd.date_range(start=start_ts, end=end_ts, freq=freq, inclusive="left")
    bars = {}
    for symbol in tqdm(symbols, desc="Building bars"):
        b = build_bars(trades[symbol], index)
        bars[symbol] = b
        b.to_parquet(bars_dir / f"{symbol}_100ms_bars.parquet")
    feats = {}
    for symbol in tqdm(symbols, desc="Computing features"):
        f = compute_base_features(bars[symbol])
        feats[symbol] = f
        f.to_parquet(features_dir / f"{symbol}_100ms_features.parquet")
    sol = feats["SOLUSDT"]
    btc = feats["BTCUSDT"]
    eth = feats["ETHUSDT"]
    btc_lag = create_lagged(btc, "btc")
    eth_lag = create_lagged(eth, "eth")
    sol_lag = create_lagged(sol, "sol")
    targets = create_targets(sol)
    final = pd.DataFrame(index=index)
    final["price"] = sol["price"].values
    final = final.join(btc_lag).join(eth_lag).join(sol_lag).join(targets)
    final = final.dropna()
    final = final.astype("float32")
    final.to_parquet(final_path)
