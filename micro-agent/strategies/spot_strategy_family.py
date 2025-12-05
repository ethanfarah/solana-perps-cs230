from pathlib import Path
from typing import Dict, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.adaptive_spot_strategy import AdaptiveSpotStrategy
from strategies.load_strategy import load_strategy_config

class SpotStrategyFamily:
    def __init__(
        self,
        config_path: Path,
        halflife: float = 600.0
    ):
        # Initialize all spot prediction strategies from config
        config = load_strategy_config(config_path)
        self.strategies = {
            name: AdaptiveSpotStrategy(
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
        forward_log_return: float
    ) -> None:
        # Train all spot strategies with observed return
        for strategy in self.strategies.values():
            strategy.update(state, forward_log_return)

    def predict(
        self,
        name: str,
        state: Dict
    ) -> float:
        pred = self.strategies[name].predict(state)
        # Convert 100ms prediction to daily annualized scale
        if self.strategies[name].target == "fwd_log_return_100ms":
            pred = pred * 864000.0
        return pred

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