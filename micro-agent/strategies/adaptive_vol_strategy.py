from typing import List, Dict, Optional
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.ewma_regression import EWMARegression


class AdaptiveVolStrategy:
    def __init__(
        self,
        name: str,
        features: List[str],
        halflife: float = 600.0,
        direction: int = 1,
        target: Optional[str] = None
    ):
        self.name = name
        self.features = features
        self.n_features = len(features)
        self.halflife = halflife
        self.direction = direction
        self.target = target
        # Initialize EWMA regression for online volatility learning
        self.ewma_reg = EWMARegression(self.n_features, halflife) if self.n_features > 0 else None

    def _extract_features(
        self,
        state: Dict
    ) -> np.ndarray:
        # Extract feature values in consistent order
        return np.array([state.get(f, 0.0) for f in self.features], dtype=np.float32)

    def update(
        self,
        state: Dict,
        forward_realized_vol: float
    ):
        # Skip neutral strategies with no directional bias
        if self.direction == 0:
            return
        x = self._extract_features(state)
        if np.isnan(x).any() or np.isnan(forward_realized_vol):
            return

        # Train regression on observed volatility
        self.ewma_reg.update(x, forward_realized_vol)

    def predict(
        self,
        state: Dict
    ) -> float:
        if self.ewma_reg is None:
            return 0.0
        x = self._extract_features(state)
        if np.isnan(x).any():
            return 0.0
        # Apply directional multiplier to prediction
        return self.direction * self.ewma_reg.predict(x)