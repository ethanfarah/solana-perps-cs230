from typing import List, Tuple


def get_feature_keys(columns: List[str]) -> List[str]:
    # Filter out target and price columns to get features
    exclude = {"price", "fwd_log_return_100ms", "fwd_vol_30s", "fwd_vol_1m"}
    return sorted([c for c in columns if c not in exclude])


def get_split_indices(
    n_items: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = None,
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    # Calculate chronological split boundaries for time series data
    if test_ratio is None:
        test_ratio = 1.0 - train_ratio - val_ratio

    total = train_ratio + val_ratio + test_ratio
    if total > 1.0 + 1e-9:
        raise ValueError(
            f"Ratios sum to {total:.3f} > 1.0: train={train_ratio}, val={val_ratio}, test={test_ratio}"
        )

    # Compute split boundaries as integer indices
    train_end = int(n_items * train_ratio)
    val_end = int(n_items * (train_ratio + val_ratio))
    test_end = int(n_items * (train_ratio + val_ratio + test_ratio))

    train_idx = (0, train_end)
    val_idx = (train_end, val_end)
    test_idx = (val_end, test_end)

    return train_idx, val_idx, test_idx
