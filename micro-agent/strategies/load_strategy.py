import json
from pathlib import Path
from typing import Dict

def load_strategy_config(
    path: Path
) -> Dict[str, list]:
    # Load strategy configuration from JSON file
    with open(path) as f:
        return json.load(f)