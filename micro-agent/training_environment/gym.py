import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.spot_strategy_family import SpotStrategyFamily
from strategies.vol_strategy_family import VolStrategyFamily
from training_environment.action_space import HierarchicalActionSpace
from training_environment.reward import DifferentialSharpeReward
from training_environment.kelly_sizer import KellySizer


@dataclass
class EnvConfig:
    spot_strategy_config: Path
    vol_strategy_config: Path
    step_stride: int = 10
    realized_vol_window: int = 300
    warmup_rows: int = 3000  # 5 minutes at 100ms resolution
    dsr_eta: float = 0.01
    leverage_limit: float = 1.0
    kelly_scale: float = 0.25
    kelly_alpha: float = 0.5
    vol_decision_stride: int = 30


class TradingEnv:
    def __init__(
        self,
        config: EnvConfig,
        data_slice: Dict[str, Any],
        spot_family: Optional[SpotStrategyFamily] = None,
        vol_family: Optional[VolStrategyFamily] = None,
    ):
        self.config = config

        # Use pre-loaded data from DataManager
        self.feature_cols = data_slice['feature_cols']
        self.feature_matrix = data_slice['feature_matrix']
        self.prices = data_slice['prices']
        self.fwd_log_return_100ms = data_slice['fwd_log_return_100ms']
        self.fwd_vol_30s = data_slice['fwd_vol_30s']
        self.fwd_vol_1m = data_slice['fwd_vol_1m']
        self.simple_returns = data_slice['simple_returns']

        self.n_rows = self.feature_matrix.shape[0]

        # Share strategy families across episodes for learning
        if spot_family is not None and vol_family is not None:
            self.spot_family = spot_family
            self.vol_family = vol_family
        else:
            from strategies.load_strategy import load_strategy_config
            spot_cfg = load_strategy_config(config.spot_strategy_config)
            vol_cfg = load_strategy_config(config.vol_strategy_config)
            self.spot_family = SpotStrategyFamily(spot_cfg)
            self.vol_family = VolStrategyFamily(vol_cfg)

        # Build discrete action space from strategy names
        spot_names = list(self.spot_family.strategies.keys())
        vol_names = list(self.vol_family.strategies.keys())
        self.action_space = HierarchicalActionSpace(spot_names, vol_names)

        self.reward_fn = DifferentialSharpeReward(eta=config.dsr_eta)
        self.kelly_sizer = KellySizer(
            leverage_limit=config.leverage_limit,
            kelly_scale=config.kelly_scale,
            alpha=config.kelly_alpha,
        )

        # 864000 timesteps per day at 100ms resolution
        self.daily_vol_scale = np.sqrt(864000.0)
        self.current_vol_strategy = None
        self.current_sigma_daily_pred = None
        self.steps_since_vol_decision = 0

        self.reset()

    def reset(self, skip_warmup: bool = False) -> np.ndarray:
        # Reset episode state to initial values
        self.current_row = 0
        self.step_count = 0
        self.equity = 1.0
        self.position_fraction = 0.0
        self.reward_fn.reset()
        self.current_vol_strategy = None
        self.current_sigma_daily_pred = None
        self.steps_since_vol_decision = 0

        # Prime strategies with initial data before trading
        if not skip_warmup and self.config.warmup_rows > 0:
            warmup_end = min(self.config.warmup_rows, self.n_rows)
            for row in range(warmup_end):
                self._update_spot_strategies(row)
                # Vol strategies update less frequently than spot
                if row % self.config.vol_decision_stride == 0:
                    for vol_name in self.vol_family.strategy_names:
                        self._update_vol_strategy(vol_name, row)
            self.current_row = warmup_end

        return self._get_state(self.current_row)

    def _get_state(self, row: int) -> np.ndarray:
        return self.feature_matrix[row]

    def _row_to_dict(self, row: int) -> Dict[str, float]:
        # Convert feature matrix row to dict for strategy methods
        return {col: float(self.feature_matrix[row, i]) for i, col in enumerate(self.feature_cols)}

    def _realized_vol_daily(self, row: int) -> Optional[float]:
        # Calculate historical volatility from trailing returns window
        if row <= 0:
            return None
        start = max(0, row - self.config.realized_vol_window)
        window = self.simple_returns[start:row]
        if window.size == 0:
            return None
        sigma_100ms = float(window.std())
        # Convert 100ms volatility to daily basis
        return sigma_100ms * self.daily_vol_scale

    def _update_spot_strategies(self, decision_row: int) -> None:
        # Train spot strategies with observed forward return
        target = float(self.fwd_log_return_100ms[decision_row])
        if not np.isfinite(target):
            return
        state_dict = self._row_to_dict(decision_row)
        self.spot_family.update_all(state_dict, target)

    def _update_vol_strategy(self, strategy_name: str, decision_row: int) -> None:
        # Train volatility strategy with observed forward volatility
        target_name = self.vol_family.get_target(strategy_name)
        if target_name == "fwd_vol_30s":
            actual = float(self.fwd_vol_30s[decision_row])
        elif target_name == "fwd_vol_1m":
            actual = float(self.fwd_vol_1m[decision_row])
        else:
            actual = np.nan
        if not np.isfinite(actual):
            return
        state_dict = self._row_to_dict(decision_row)
        self.vol_family.update(strategy_name, state_dict, actual)

    def step(self, spot_action: int, vol_action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        # Execute agent actions and simulate trading over stride
        start_row = self.current_row
        end_row = start_row + self.config.step_stride
        if end_row >= self.n_rows:
            end_row = self.n_rows - 1
            done = True
        else:
            done = False

        # Decode spot strategy selection from action
        spot_name = self.action_space.decode_spot(spot_action).strategy

        # Refresh volatility prediction on stride schedule
        new_vol_decision = False
        if self.current_vol_strategy is None or self.steps_since_vol_decision >= self.config.vol_decision_stride:
            vol_name = self.action_space.decode_vol(vol_action).strategy
            decision_row = start_row
            state_dict = self._row_to_dict(decision_row)
            sigma_daily_pred = float(self.vol_family.predict(vol_name, state_dict))
            self.current_vol_strategy = vol_name
            self.current_sigma_daily_pred = sigma_daily_pred
            self.steps_since_vol_decision = 0
            new_vol_decision = True
        else:
            # Reuse cached volatility prediction
            vol_name = self.current_vol_strategy
            sigma_daily_pred = self.current_sigma_daily_pred
            decision_row = start_row

        # Track portfolio growth across all timesteps in stride
        growth = 1.0
        last_f = 0.0
        last_mu = 0.0
        last_sigma_real = None

        for row in range(start_row, end_row):
            state_dict = self._row_to_dict(row)
            # Get expected return from selected spot strategy
            mu_daily = float(self.spot_family.predict(spot_name, state_dict))
            sigma_daily_real = self._realized_vol_daily(row)
            # Kelly sizing with predicted and realized volatility
            f = float(self.kelly_sizer.compute_size(mu_daily, sigma_daily_pred, sigma_daily_real))
            r = float(self.simple_returns[row])
            # Compound growth with leveraged position
            growth *= 1.0 + f * r
            last_f = f
            last_mu = mu_daily
            last_sigma_real = sigma_daily_real

        # Convert multiplicative growth to additive return
        block_return = growth - 1.0

        # Calculate market return for performance comparison
        market_growth = 1.0
        for row in range(start_row, end_row):
            market_growth *= 1.0 + float(self.simple_returns[row])
        market_return = market_growth - 1.0

        self.equity *= growth
        # Compute Differential Sharpe Ratio reward
        reward = float(self.reward_fn.compute(block_return))

        # Update strategies with observed outcomes
        self._update_spot_strategies(decision_row)
        if new_vol_decision:
            self._update_vol_strategy(vol_name, decision_row)

        self.current_row = end_row
        self.step_count += 1
        self.steps_since_vol_decision += 1

        next_obs = self._get_state(self.current_row)
        info = {
            "equity": self.equity,
            "block_return": block_return,
            "market_return": market_return,
            "position_fraction": last_f,
            "mu_daily": last_mu,
            "sigma_daily_pred": sigma_daily_pred,
            "sigma_daily_real": last_sigma_real,
            "spot_strategy": spot_name,
            "vol_strategy": vol_name,
            "vol_decision": new_vol_decision,
        }
        return next_obs, reward, done, info
