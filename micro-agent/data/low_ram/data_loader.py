import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Generator
import gc


LAG_ROWS = {
    'lag0': 0,
    'lag100ms': 1,
    'lag500ms': 5,
    'lag1s': 10,
}

BASE_FEATURES = [
    'log_return',
    'momentum_100ms', 'momentum_1s', 'momentum_5s', 'momentum_30s', 'momentum_5m', 'momentum_15m', 'momentum_60m',
    'flow_imbalance_100ms', 'flow_imbalance_1s', 'flow_imbalance_5s', 'flow_imbalance_30s', 'flow_imbalance_5m', 'flow_imbalance_15m', 'flow_imbalance_60m',
    'vw_flow_imbalance_100ms', 'vw_flow_imbalance_1s', 'vw_flow_imbalance_5s', 'vw_flow_imbalance_30s', 'vw_flow_imbalance_5m', 'vw_flow_imbalance_15m', 'vw_flow_imbalance_60m',
    'realized_vol_100ms', 'realized_vol_1s', 'realized_vol_30s', 'realized_vol_5m', 'realized_vol_15m', 'realized_vol_60m',
    'vol_intensity_100ms', 'vol_intensity_1s', 'vol_intensity_5s', 'vol_intensity_30s', 'vol_intensity_5m', 'vol_intensity_15m', 'vol_intensity_60m',
]


class DataLoader:
    def __init__(
        self,
        data_dir: Path
    ):
        self.data_dir = Path(data_dir)
        self._build_file_index()

    def _build_file_index(self):
        # Index all parquet files with time ranges for efficient loading
        self.file_index = {}

        for symbol in ['SOLUSDT', 'BTCUSDT', 'ETHUSDT']:
            files = sorted(self.data_dir.glob(f"{symbol}_batch_*_rg_*.parquet"))
            self.file_index[symbol] = []

            # Extract time range from each file without loading full data
            for f in files:
                df = pd.read_parquet(f, columns=[])
                self.file_index[symbol].append({
                    'file': f,
                    'min_time': df.index.min(),
                    'max_time': df.index.max(),
                })
                del df

        self.sol_files = [e['file'] for e in self.file_index['SOLUSDT']]
        self.n_chunks = len(self.sol_files)
    
    def _find_overlapping_files(
        self,
        symbol: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp
    ) -> List[Path]:
        return [
            e['file'] for e in self.file_index[symbol]
            if e['max_time'] >= start_time and e['min_time'] <= end_time
        ]
    
    def _load_time_range(
        self,
        symbol: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp
    ) -> Optional[pd.DataFrame]:
        files = self._find_overlapping_files(symbol, start_time, end_time)
        
        if not files:
            return None
        
        dfs = [pd.read_parquet(f) for f in files]
        result = pd.concat(dfs).sort_index()
        result = result[~result.index.duplicated(keep='last')]
        
        mask = (result.index >= start_time) & (result.index <= end_time)
        return result.loc[mask]
    
    def _get_available_features(
        self,
        df: pd.DataFrame
    ) -> List[str]:
        return [f for f in BASE_FEATURES if f in df.columns]
    
    def _create_lagged_features(
        self,
        df: pd.DataFrame,
        asset: str,
        features: List[str]
    ) -> pd.DataFrame:
        data = {}
        for feat in features:
            for lag_name, lag_rows in LAG_ROWS.items():
                col_name = f'{asset}_{feat}_{lag_name}'
                if lag_rows == 0:
                    data[col_name] = df[feat].values
                else:
                    data[col_name] = df[feat].shift(lag_rows).values
        return pd.DataFrame(data, index=df.index)
    
    def _create_forward_targets(
        self,
        sol: pd.DataFrame
    ) -> pd.DataFrame:
        n_rows = len(sol)
        price = sol['price'].values
        log_return = sol['log_return'].values
        
        fwd_log_return = np.full(n_rows, np.nan)
        fwd_log_return[:-1] = np.log(price[1:] / price[:-1])
        
        log_return_series = pd.Series(log_return, index=sol.index)
        
        return pd.DataFrame({
            'fwd_log_return_100ms': fwd_log_return,
            'fwd_vol_30s': log_return_series.rolling(300).std().shift(-300).values,
            'fwd_vol_1m': log_return_series.rolling(600).std().shift(-600).values,
        }, index=sol.index)
    
    def load_chunk(
        self,
        chunk_idx: int
    ) -> Optional[pd.DataFrame]:
        # Load and align multi-asset data for one time chunk
        if chunk_idx >= self.n_chunks:
            return None

        sol_file = self.sol_files[chunk_idx]
        sol = pd.read_parquet(sol_file)
        time_min, time_max = sol.index.min(), sol.index.max()

        # Load matching time ranges from other assets
        btc = self._load_time_range('BTCUSDT', time_min, time_max)
        eth = self._load_time_range('ETHUSDT', time_min, time_max)

        if btc is None or eth is None:
            del sol
            gc.collect()
            return None

        # Remove duplicate timestamps keeping most recent
        btc = btc[~btc.index.duplicated(keep='last')]
        sol = sol[~sol.index.duplicated(keep='last')]
        eth = eth[~eth.index.duplicated(keep='last')]

        # Find timestamps present in all three assets
        common_idx = btc.index.intersection(sol.index).intersection(eth.index)

        if len(common_idx) < 1000:
            del btc, sol, eth
            gc.collect()
            return None

        # Align all assets to common timestamps
        btc = btc.loc[common_idx]
        sol = sol.loc[common_idx]
        eth = eth.loc[common_idx]

        # Generate lagged features for each asset
        btc_features = self._get_available_features(btc)
        eth_features = self._get_available_features(eth)
        sol_features = self._get_available_features(sol)

        btc_lagged = self._create_lagged_features(btc, 'btc', btc_features)
        eth_lagged = self._create_lagged_features(eth, 'eth', eth_features)
        sol_lagged = self._create_lagged_features(sol, 'sol', sol_features)
        forward_targets = self._create_forward_targets(sol)

        # Combine all features and targets into single dataframe
        out = pd.concat([
            pd.DataFrame({'price': sol['price'].values}, index=common_idx),
            btc_lagged,
            eth_lagged,
            sol_lagged,
            forward_targets,
        ], axis=1)

        # Clean up memory aggressively
        del btc, sol, eth, btc_lagged, eth_lagged, sol_lagged, forward_targets
        gc.collect()

        out = out.dropna()

        if len(out) < 1000:
            return None

        return out
    
    def iter_chunks(
        self,
        start_idx: int = 0,
        end_idx: Optional[int] = None
    ) -> Generator[pd.DataFrame, None, None]:
        if end_idx is None:
            end_idx = self.n_chunks
        
        for idx in range(start_idx, end_idx):
            chunk = self.load_chunk(idx)
            if chunk is not None:
                yield chunk
    
    def get_chunk_info(
        self,
        chunk_idx: int
    ) -> Dict:
        if chunk_idx >= self.n_chunks:
            return {}
        
        entry = self.file_index['SOLUSDT'][chunk_idx]
        return {
            'file': entry['file'],
            'min_time': entry['min_time'],
            'max_time': entry['max_time'],
        }