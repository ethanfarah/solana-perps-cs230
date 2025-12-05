import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import gc
import warnings
from numba import njit, prange
warnings.filterwarnings('ignore')

resampled_dir = Path("/home/ubuntu/resampled/")

MAX_LAG_ROWS = 3000

RETURN_HORIZONS = {
    '100ms': 1,
    '500ms': 5,
    '1s': 10,
    '5s': 50,
    '10s': 100,
    '30s': 300,
    '1m': 600,
    '5m': 3000,
    '15m': 9000,
}

LAG_PERIODS = np.array([0, 1, 2, 5, 10, 20, 50, 100, 300, 600, 1200, 3000], dtype=np.int64)
LAG_NAMES = ['0', '100ms', '200ms', '500ms', '1s', '2s', '5s', '10s', '30s', '1m', '2m', '5m']

BASE_FEATURES = [
    'log_return',
    'momentum_100ms', 'momentum_1s', 'momentum_5s', 'momentum_30s', 'momentum_5m', 'momentum_15m', 'momentum_60m',
    'flow_imbalance_100ms', 'flow_imbalance_1s', 'flow_imbalance_5s', 'flow_imbalance_30s', 'flow_imbalance_5m', 'flow_imbalance_15m', 'flow_imbalance_60m',
    'vw_flow_imbalance_100ms', 'vw_flow_imbalance_1s', 'vw_flow_imbalance_5s', 'vw_flow_imbalance_30s', 'vw_flow_imbalance_5m', 'vw_flow_imbalance_15m', 'vw_flow_imbalance_60m',
    'realized_vol_100ms', 'realized_vol_1s', 'realized_vol_30s', 'realized_vol_5m', 'realized_vol_15m', 'realized_vol_60m',
    'vol_intensity_100ms', 'vol_intensity_1s', 'vol_intensity_5s', 'vol_intensity_30s', 'vol_intensity_5m', 'vol_intensity_15m', 'vol_intensity_60m',
]


def get_available_features(df):
    return [f for f in BASE_FEATURES if f in df.columns]


@njit(parallel=True)
def apply_lags_numba(base_arr, lags, out):
    n_rows, n_features = base_arr.shape
    n_lags = len(lags)
    
    for f in prange(n_features):
        for l_idx in range(n_lags):
            lag = lags[l_idx]
            col = f * n_lags + l_idx
            if lag == 0:
                for i in range(n_rows):
                    out[i, col] = base_arr[i, f]
            else:
                for i in range(lag, n_rows):
                    out[i, col] = base_arr[i - lag, f]


def warmup_jit():
    dummy_base = np.random.randn(100, 10).astype(np.float64)
    dummy_out = np.full((100, 10 * len(LAG_PERIODS)), np.nan, dtype=np.float64)
    apply_lags_numba(dummy_base, LAG_PERIODS, dummy_out)


CHECKPOINT_PATH = Path("correlation_checkpoint.npz")


class OnlineCorrelation:
    
    def __init__(self, n_features, n_targets, feature_names, target_names):
        self.n_features = n_features
        self.n_targets = n_targets
        self.feature_names = feature_names
        self.target_names = target_names
        
        self.n = np.zeros((n_features, n_targets), dtype=np.float64)
        self.sum_x = np.zeros((n_features, n_targets), dtype=np.float64)
        self.sum_y = np.zeros((n_features, n_targets), dtype=np.float64)
        self.sum_x2 = np.zeros((n_features, n_targets), dtype=np.float64)
        self.sum_y2 = np.zeros((n_features, n_targets), dtype=np.float64)
        self.sum_xy = np.zeros((n_features, n_targets), dtype=np.float64)
        self.processed_files = set()
    
    def save_checkpoint(self, path=CHECKPOINT_PATH):
        np.savez(
            path,
            n=self.n,
            sum_x=self.sum_x,
            sum_y=self.sum_y,
            sum_x2=self.sum_x2,
            sum_y2=self.sum_y2,
            sum_xy=self.sum_xy,
            feature_names=np.array(self.feature_names, dtype=object),
            target_names=np.array(self.target_names, dtype=object),
            processed_files=np.array(list(self.processed_files), dtype=object),
        )
    
    @classmethod
    def load_checkpoint(cls, path=CHECKPOINT_PATH):
        if not path.exists():
            return None
        
        data = np.load(path, allow_pickle=True)
        feature_names = list(data['feature_names'])
        target_names = list(data['target_names'])
        
        obj = cls(len(feature_names), len(target_names), feature_names, target_names)
        obj.n = data['n']
        obj.sum_x = data['sum_x']
        obj.sum_y = data['sum_y']
        obj.sum_x2 = data['sum_x2']
        obj.sum_y2 = data['sum_y2']
        obj.sum_xy = data['sum_xy']
        obj.processed_files = set(data['processed_files'])
        return obj
    
    def update(self, X, Y, batch_size=200000):
        n_rows = X.shape[0]
        
        for start in range(0, n_rows, batch_size):
            end = min(start + batch_size, n_rows)
            X_batch = X[start:end].astype(np.float32)
            Y_batch = Y[start:end].astype(np.float32)
            
            X_valid = (~np.isnan(X_batch)).astype(np.float32)
            Y_valid = (~np.isnan(Y_batch)).astype(np.float32)
            
            X_filled = np.nan_to_num(X_batch, 0)
            Y_filled = np.nan_to_num(Y_batch, 0)
            
            self.n += X_valid.T @ Y_valid
            self.sum_x += X_filled.T @ Y_valid
            self.sum_x2 += (X_filled ** 2).T @ Y_valid
            self.sum_y += X_valid.T @ Y_filled
            self.sum_y2 += X_valid.T @ (Y_filled ** 2)
            self.sum_xy += X_filled.T @ Y_filled
            
            del X_batch, Y_batch, X_valid, Y_valid, X_filled, Y_filled
            gc.collect()
    
    def compute_results(self):
        n = self.n
        mean_x = np.divide(self.sum_x, n, where=n > 0, out=np.zeros_like(self.sum_x))
        mean_y = np.divide(self.sum_y, n, where=n > 0, out=np.zeros_like(self.sum_y))
        
        var_x = np.divide(self.sum_x2, n, where=n > 0, out=np.zeros_like(self.sum_x2)) - mean_x ** 2
        var_y = np.divide(self.sum_y2, n, where=n > 0, out=np.zeros_like(self.sum_y2)) - mean_y ** 2
        cov_xy = np.divide(self.sum_xy, n, where=n > 0, out=np.zeros_like(self.sum_xy)) - mean_x * mean_y
        
        denom = np.sqrt(var_x * var_y)
        corr = np.divide(cov_xy, denom, where=denom > 0, out=np.zeros_like(cov_xy))
        corr = np.clip(corr, -1, 1)
        
        t_stat = np.divide(corr * np.sqrt(n - 2), np.sqrt(1 - corr**2), 
                          where=(np.abs(corr) < 1) & (n > 2), 
                          out=np.zeros_like(corr))
        
        results = []
        for i in range(self.n_features):
            for j in range(self.n_targets):
                if n[i, j] < 100 or var_x[i, j] <= 0 or var_y[i, j] <= 0:
                    continue
                
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat[i, j]), n[i, j] - 2))
                
                results.append({
                    'feature': self.feature_names[i],
                    'target': self.target_names[j],
                    'correlation': corr[i, j],
                    't_stat': t_stat[i, j],
                    'p_value': p_value,
                    'n_obs': int(n[i, j]),
                    'abs_corr': abs(corr[i, j]),
                    'abs_t_stat': abs(t_stat[i, j]),
                })
        
        return pd.DataFrame(results)


def build_file_index(files):
    index = []
    for i, f in enumerate(files):
        df = pd.read_parquet(f, columns=[])
        index.append({
            'file': f,
            'min_time': df.index.min(),
            'max_time': df.index.max(),
        })
        del df
    return index


def find_overlapping_files(file_index, start_time, end_time):
    overlapping = []
    for entry in file_index:
        if entry['max_time'] >= start_time and entry['min_time'] <= end_time:
            overlapping.append(entry['file'])
    return overlapping


def load_time_range(file_index, start_time, end_time):
    files = find_overlapping_files(file_index, start_time, end_time)
    if not files:
        return None
    
    dfs = [pd.read_parquet(f) for f in files]
    result = pd.concat(dfs).sort_index()
    result = result[~result.index.duplicated(keep='last')]
    
    mask = (result.index >= start_time) & (result.index <= end_time)
    return result.loc[mask]


def create_lagged_features_fast(data_dict, features):
    n_rows = len(data_dict['sol'])
    n_lags = len(LAG_PERIODS)
    
    base_arrays = []
    for asset in ['sol', 'btc', 'eth']:
        df = data_dict[asset]
        arr = np.column_stack([
            df[feat].values if feat in df.columns else np.full(n_rows, np.nan)
            for feat in features
        ]).astype(np.float64)
        base_arrays.append(arr)
    
    all_base = np.hstack(base_arrays)
    X = np.full((n_rows, all_base.shape[1] * n_lags), np.nan, dtype=np.float64)
    
    apply_lags_numba(all_base, LAG_PERIODS, X)
    
    return X


def create_targets_fast(sol_df, n_rows):
    price = sol_df['price'].values.astype(np.float64)
    log_ret = sol_df['log_return'].values.astype(np.float64)
    
    target_names = []
    targets = []
    
    for name, periods in RETURN_HORIZONS.items():
        fwd_ret = np.full(n_rows, np.nan)
        if periods < n_rows:
            fwd_ret[:-periods] = (price[periods:] - price[:-periods]) / price[:-periods]
        targets.append(fwd_ret)
        target_names.append(f'fwd_return_{name}')
    
    log_ret_series = pd.Series(log_ret)
    for name, periods in RETURN_HORIZONS.items():
        if periods >= 10:
            fwd_vol = log_ret_series.rolling(periods).std().shift(-periods).values
            targets.append(fwd_vol)
            target_names.append(f'fwd_vol_{name}')
    
    Y = np.column_stack(targets)
    return Y, target_names


def process_correlations(checkpoint_every=5, resume=True):
    files_by_symbol = {}
    file_indices = {}
    for symbol in ['SOLUSDT', 'BTCUSDT', 'ETHUSDT']:
        files = sorted(resampled_dir.glob(f"{symbol}_batch_*_rg_*.parquet"))
        files_by_symbol[symbol] = files
        file_indices[symbol] = build_file_index(files)
    
    sol_sample = pd.read_parquet(files_by_symbol['SOLUSDT'][0])
    features = get_available_features(sol_sample)
    del sol_sample
    gc.collect()
    
    feature_names = []
    for asset in ['sol', 'btc', 'eth']:
        for feat in features:
            for lag_name in LAG_NAMES:
                feature_names.append(f'{asset}_{feat}_lag{lag_name}')
    
    n_features = len(feature_names)
    n_targets = len(RETURN_HORIZONS) + sum(1 for p in RETURN_HORIZONS.values() if p >= 10)
    
    dummy_df = pd.DataFrame({'price': [1.0, 2.0, 3.0], 'log_return': [0.0, 0.1, 0.1]})
    _, target_names = create_targets_fast(dummy_df, 3)
    
    online_corr = None
    if resume:
        online_corr = OnlineCorrelation.load_checkpoint()
    
    if online_corr is None:
        online_corr = OnlineCorrelation(n_features, n_targets, feature_names, target_names)
    
    warmup_jit()
    
    btc_files = files_by_symbol['BTCUSDT']
    prev_tails = {'sol': None, 'btc': None, 'eth': None}
    
    files_to_process = [f for f in btc_files if str(f) not in online_corr.processed_files]
    chunks_since_checkpoint = 0
    
    for idx, btc_file in enumerate(files_to_process):
        btc = pd.read_parquet(btc_file)
        time_min, time_max = btc.index.min(), btc.index.max()
        
        sol = load_time_range(file_indices['SOLUSDT'], time_min, time_max)
        eth = load_time_range(file_indices['ETHUSDT'], time_min, time_max)
        
        if sol is None or eth is None:
            online_corr.processed_files.add(str(btc_file))
            del btc
            gc.collect()
            continue
        
        if prev_tails['btc'] is not None:
            btc = pd.concat([prev_tails['btc'], btc])
            sol = pd.concat([prev_tails['sol'], sol])
            eth = pd.concat([prev_tails['eth'], eth])
        
        btc = btc[~btc.index.duplicated(keep='last')]
        sol = sol[~sol.index.duplicated(keep='last')]
        eth = eth[~eth.index.duplicated(keep='last')]
        
        common_idx = btc.index.intersection(eth.index).intersection(sol.index)
        
        if len(common_idx) < 1000:
            online_corr.processed_files.add(str(btc_file))
            del btc, sol, eth
            gc.collect()
            continue
        
        sol = sol.loc[common_idx]
        btc = btc.loc[common_idx]
        eth = eth.loc[common_idx]
        
        n_rows = len(common_idx)
        
        new_mask = (common_idx >= time_min) & (common_idx <= time_max)
        new_indices = np.where(new_mask)[0]
        
        if len(new_indices) > 0:
            chunk_size = 200000
            n_chunks = (len(new_indices) + chunk_size - 1) // chunk_size
            
            for chunk_idx in range(n_chunks):
                start = chunk_idx * chunk_size
                end = min((chunk_idx + 1) * chunk_size, len(new_indices))
                
                buffer_start = max(0, new_indices[start] - MAX_LAG_ROWS)
                chunk_end = new_indices[end - 1] + 1
                
                sol_chunk = sol.iloc[buffer_start:chunk_end]
                btc_chunk = btc.iloc[buffer_start:chunk_end]
                eth_chunk = eth.iloc[buffer_start:chunk_end]
                
                data_dict = {'sol': sol_chunk, 'btc': btc_chunk, 'eth': eth_chunk}
                X_chunk = create_lagged_features_fast(data_dict, features)
                Y_chunk, _ = create_targets_fast(sol_chunk, len(sol_chunk))
                
                buffer_rows = new_indices[start] - buffer_start
                actual_rows = end - start
                X_new = X_chunk[buffer_rows:buffer_rows + actual_rows].astype(np.float32)
                Y_new = Y_chunk[buffer_rows:buffer_rows + actual_rows].astype(np.float32)
                
                online_corr.update(X_new, Y_new)
                
                del sol_chunk, btc_chunk, eth_chunk, data_dict, X_chunk, Y_chunk, X_new, Y_new
                gc.collect()
        
        online_corr.processed_files.add(str(btc_file))
        chunks_since_checkpoint += 1
        
        if chunks_since_checkpoint >= checkpoint_every:
            online_corr.save_checkpoint()
            chunks_since_checkpoint = 0
        
        prev_tails['btc'] = btc.tail(MAX_LAG_ROWS).copy()
        prev_tails['sol'] = sol.tail(MAX_LAG_ROWS).copy()
        prev_tails['eth'] = eth.tail(MAX_LAG_ROWS).copy()
        
        del btc, sol, eth
        gc.collect()
    
    online_corr.save_checkpoint()
    return online_corr.compute_results()


def extract_lead_lag_results(results_df, source_asset, feature_type='log_return'):
    pattern = f'{source_asset}_{feature_type}_lag'
    subset = results_df[results_df['feature'].str.startswith(pattern)].copy()
    subset['lag'] = subset['feature'].str.extract(r'lag(\w+)$')[0]
    return subset


if __name__ == "__main__":
    results_df = process_correlations()
    
    if results_df is None or len(results_df) == 0:
        print("No results!")
        exit(1)
    
    results_df.to_parquet("correlation_results_raw.parquet", index=False)
    results_df.to_csv("correlation_results_raw.csv", index=False)
    
    print("\nTOP 50 CORRELATIONS:")
    print(results_df.nlargest(50, 'abs_corr')[['feature', 'target', 'correlation', 't_stat']].to_string())
    
    print("\nTOP 50 T-STATS:")
    print(results_df.nlargest(50, 'abs_t_stat')[['feature', 'target', 'correlation', 't_stat']].to_string())
    
    for asset in ['btc', 'eth']:
        lead_lag = extract_lead_lag_results(results_df, asset, 'log_return')
        if len(lead_lag) > 0:
            summary = lead_lag.pivot(index='lag', columns='target', values='correlation')
            print(f"\n{asset.upper()} log_return -> SOL:")
            print(summary.to_string())
            summary.to_csv(f"lead_lag_{asset}_to_sol.csv")