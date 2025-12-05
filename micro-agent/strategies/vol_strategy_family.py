from pathlib import Path
from typing import Dict, Optional
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.adaptive_vol_strategy import AdaptiveVolStrategy
from strategies.load_strategy import load_strategy_config

class VolStrategyFamily:
    def __init__(
        self,
        config_path: Path,
        halflife: float = 600.0
    ):
        # Initialize all volatility prediction strategies from config
        config = load_strategy_config(config_path)
        self.strategies = {
            name: AdaptiveVolStrategy(
                name=name,
                features=cfg["features"],
                halflife=halflife,
                direction=cfg["direction"],
                target=cfg.get("target")
            )
            for name, cfg in config.items()
        }
        self.strategy_names = list(self.strategies.keys())

    def update_all(
        self,
        state: Dict,
        forward_realized_vol: float
    ) -> None:
        # Train all vol strategies with observed volatility
        for strategy in self.strategies.values():
            strategy.update(state, forward_realized_vol)

    def update(
        self,
        name: str,
        state: Dict,
        forward_realized_vol: float
    ) -> None:
        # Train single vol strategy with observed volatility
        self.strategies[name].update(state, forward_realized_vol)

    def predict(
        self,
        name: str,
        state: Dict
    ) -> float:
        # Scale volatility prediction to daily basis
        if self.strategies[name].target == "fwd_vol_30s":
            return self.strategies[name].predict(state) * np.sqrt(2880.0)
        elif self.strategies[name].target == "fwd_vol_1m":
            return self.strategies[name].predict(state) * np.sqrt(1440.0)
        return self.strategies[name].predict(state)

    def get_target(
        self,
        name: str
    ) -> Optional[str]:
        return self.strategies[name].target

    def get_weights(
        self,
        name: str
    ) -> Dict[str, float]:
        # Extract learned feature weights for analysis
        strategy = self.strategies[name]
        if strategy.ewma_reg is None:
            return {}
        return dict(zip(strategy.features, strategy.ewma_reg.weights))

    def reset_all(self):
        # Reset all strategy EWMA states
        for strategy in self.strategies.values():
            if strategy.ewma_reg:
                strategy.ewma_reg.reset()

    def save_all_states(self) -> Dict[str, dict]:
        # Save all strategy states
        return {
            name: strategy.ewma_reg.save_state() if strategy.ewma_reg else None
            for name, strategy in self.strategies.items()
        }

    def restore_all_states(self, saved_states: Dict[str, dict]) -> None:
        # Restore all strategy states
        for name, state in saved_states.items():
            if state and self.strategies[name].ewma_reg:
                self.strategies[name].ewma_reg.restore_state(state)