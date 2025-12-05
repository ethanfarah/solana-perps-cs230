import numpy as np
from typing import Optional


class KellySizer:
    def __init__(
        self,
        leverage_limit: float = 1.0,
        kelly_scale: float = 0.25,
        min_vol: float = 1e-6,
        max_vol: float = 100.0,
        alpha: float = 0.5,
    ):
        # Maximum allowed position size as fraction of equity
        self.leverage_limit = float(leverage_limit)
        # Conservative scaling factor for Kelly criterion
        self.kelly_scale = float(kelly_scale)
        self.min_vol = float(min_vol)
        self.max_vol = float(max_vol)
        # Weight for blending predicted and realized volatility
        self.alpha = float(alpha)

    def _blend_vol(self, pred_vol: float, realized_vol: Optional[float]) -> float:
        # Combine forecast and historical volatility for robustness
        if not np.isfinite(pred_vol):
            pred = 0.0
        else:
            pred = float(pred_vol)
        if realized_vol is None or not np.isfinite(realized_vol):
            return float(np.clip(pred, self.min_vol, self.max_vol))
        real = float(realized_vol)
        pred = float(np.clip(pred, self.min_vol, self.max_vol))
        real = float(np.clip(real, self.min_vol, self.max_vol))
        # Blend variances then take square root
        var = self.alpha * pred * pred + (1.0 - self.alpha) * real * real
        if var <= 0.0:
            return self.min_vol
        return float(var ** 0.5)

    def compute_size(
        self,
        predicted_return_daily: float,
        predicted_vol_daily: float,
        realized_vol_daily: Optional[float],
    ) -> float:
        # Calculate Kelly fraction from return and volatility estimates
        if not np.isfinite(predicted_return_daily):
            return 0.0
        mu = float(predicted_return_daily)
        sigma = self._blend_vol(predicted_vol_daily, realized_vol_daily)
        var = sigma * sigma
        if var <= 1e-12:
            return 0.0
        # Kelly formula: f = mu / sigma^2
        raw = mu / var
        # Apply fractional Kelly for risk control
        scaled = raw * self.kelly_scale
        # Enforce hard leverage limits
        if scaled > self.leverage_limit:
            return self.leverage_limit
        if scaled < -self.leverage_limit:
            return -self.leverage_limit
        return float(scaled)
