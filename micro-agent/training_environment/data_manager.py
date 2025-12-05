import pandas as pd
import numpy as np
from pathlib import Path

class DataManager:
    def __init__(self, dataset_path: Path):
        # Load full dataset into memory for fast access
        self.data = pd.read_parquet(dataset_path)
        target_cols = ["price", "fwd_log_return_100ms", "fwd_vol_30s", "fwd_vol_1m"]
        # Separate features from targets and price
        self.feature_cols = [c for c in self.data.columns if c not in target_cols]

        # Convert to numpy for performance
        self.feature_matrix = self.data[self.feature_cols].values.astype('float32')
        self.prices = self.data["price"].values.astype('float64')
        self.fwd_log_return_100ms = self.data["fwd_log_return_100ms"].values.astype('float64')
        self.fwd_vol_30s = self.data["fwd_vol_30s"].values.astype('float64')
        self.fwd_vol_1m = self.data["fwd_vol_1m"].values.astype('float64')

        # Precompute returns for Kelly sizing and rewards
        self.simple_returns = np.zeros_like(self.prices)
        self.simple_returns[:-1] = (self.prices[1:] - self.prices[:-1]) / self.prices[:-1]
        self.simple_returns[-1] = 0.0

        self.n_rows = self.feature_matrix.shape[0]

    def get_split_ranges(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        # Calculate chronological train/val/test split boundaries
        from reinforcement_learning.utils import get_split_indices
        return get_split_indices(self.n_rows, train_ratio, val_ratio, test_ratio)

    def get_slice_view(self, start_row: int, end_row: int):
        # Return array views to avoid copying data
        return {
            'feature_matrix': self.feature_matrix[start_row:end_row],
            'prices': self.prices[start_row:end_row],
            'fwd_log_return_100ms': self.fwd_log_return_100ms[start_row:end_row],
            'fwd_vol_30s': self.fwd_vol_30s[start_row:end_row],
            'fwd_vol_1m': self.fwd_vol_1m[start_row:end_row],
            'simple_returns': self.simple_returns[start_row:end_row],
            'feature_cols': self.feature_cols,
        }
