from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class SpotAction:
    strategy: str  # Name of spot strategy selected


@dataclass
class VolAction:
    strategy: str  # Name of vol strategy selected


class HierarchicalActionSpace:
    def __init__(
        self,
        spot_strategy_names: List[str],
        vol_strategy_names: List[str]
    ):
        # Map strategy names to discrete action indices
        self.spot_names = spot_strategy_names
        self.vol_names = vol_strategy_names
        self.n_spot_actions = len(spot_strategy_names)
        self.n_vol_actions = len(vol_strategy_names)

    def decode_spot(
        self,
        action: int
    ) -> SpotAction:
        # Convert action index to spot strategy name
        return SpotAction(self.spot_names[action])

    def decode_vol(
        self,
        action: int
    ) -> VolAction:
        # Convert action index to vol strategy name
        return VolAction(self.vol_names[action])

    def encode_spot(
        self,
        strategy: str
    ) -> int:
        # Convert spot strategy name to action index
        return self.spot_names.index(strategy)

    def encode_vol(
        self,
        strategy: str
    ) -> int:
        # Convert vol strategy name to action index
        return self.vol_names.index(strategy)

    def all_spot_actions(
        self
    ) -> List[Tuple[int, SpotAction]]:
        return [(i, self.decode_spot(i)) for i in range(self.n_spot_actions)]

    def all_vol_actions(
        self
    ) -> List[Tuple[int, VolAction]]:
        return [(i, self.decode_vol(i)) for i in range(self.n_vol_actions)]