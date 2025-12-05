import json
from pathlib import Path
import argparse
import numpy as np
import sys
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from training_environment.gym import EnvConfig, TradingEnv
from training_environment.data_manager import DataManager
from strategies.spot_strategy_family import SpotStrategyFamily
from strategies.vol_strategy_family import VolStrategyFamily
from strategies.load_strategy import load_strategy_config
from reinforcement_learning.dqn_agent import DQNAgent
from reinforcement_learning.driver import run_env
from reinforcement_learning.plotting import (
    generate_all_plots,
    plot_training_curves,
    plot_loss_curves,
)


def create_agents(
    state_dim: int,
    n_spot_actions: int,
    n_vol_actions: int,
    hidden_dim: int = 128,
    lr: float = 1e-4,
    gamma: float = 0.99,
    epsilon_decay: int = 50000,
    target_update_freq: int = 1000,
    device: str = None,
):
    # Initialize two independent DQN agents for hierarchical control
    agent_kwargs = {
        "state_dim": state_dim,
        "hidden_dim": hidden_dim,
        "lr": lr,
        "gamma": gamma,
        "epsilon_decay": epsilon_decay,
        "target_update_freq": target_update_freq,
        "device": device,
    }
    spot_agent = DQNAgent(n_actions=n_spot_actions, **agent_kwargs)
    vol_agent = DQNAgent(n_actions=n_vol_actions, **agent_kwargs)
    return spot_agent, vol_agent


def train(
    dataset_path: Path,
    spot_config_path: Path,
    vol_config_path: Path,
    n_epochs: int = 10,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    warmup_rows: int = 3000,
    results_dir: Path = Path("results"),
    hidden_dim: int = 128,
    device: str = None,
    lr: float = 1e-4,
    gamma: float = 0.99,
    epsilon_decay: int = 50000,
    target_update_freq: int = 1000,
    debug: bool = False,
):
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = results_dir / "checkpoints"
    plots_dir = results_dir / "plots"
    checkpoints_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    # Load dataset efficiently with memory-mapped access
    print(f"Loading dataset from {dataset_path}...")
    data_manager = DataManager(dataset_path)
    print(f"Total rows: {data_manager.n_rows:,}")

    # Split dataset chronologically into train/val/test
    train_range, val_range, test_range = data_manager.get_split_ranges(
        train_ratio, val_ratio, test_ratio
    )
    print(f"Train: rows [{train_range[0]:,}, {train_range[1]:,}) = {train_range[1]-train_range[0]:,} rows")
    print(f"Val:   rows [{val_range[0]:,}, {val_range[1]:,}) = {val_range[1]-val_range[0]:,} rows")
    print(f"Test:  rows [{test_range[0]:,}, {test_range[1]:,}) = {test_range[1]-test_range[0]:,} rows")

    env_config = EnvConfig(
        spot_strategy_config=spot_config_path,
        vol_strategy_config=vol_config_path,
        step_stride=10,
        warmup_rows=warmup_rows,
    )

    # Share strategy weights across all environments for learning
    spot_family = SpotStrategyFamily(spot_config_path)
    vol_family = VolStrategyFamily(vol_config_path)

    # Create environments with different data slices but shared strategies
    train_slice = data_manager.get_slice_view(train_range[0], train_range[1])
    train_env = TradingEnv(env_config, train_slice, spot_family, vol_family)

    val_slice = data_manager.get_slice_view(val_range[0], val_range[1])
    val_env = TradingEnv(env_config, val_slice, spot_family, vol_family)

    test_slice = data_manager.get_slice_view(test_range[0], test_range[1])
    test_env = TradingEnv(env_config, test_slice, spot_family, vol_family)

    state_dim = int(train_env.feature_matrix.shape[1])
    n_spot_actions = train_env.action_space.n_spot_actions
    n_vol_actions = train_env.action_space.n_vol_actions
    spot_names = list(train_env.action_space.spot_names)
    vol_names = list(train_env.action_space.vol_names)

    print(f"State dimension: {state_dim}")
    print(f"Spot strategies: {spot_names}")
    print(f"Vol strategies:  {vol_names}")
    print(f"Step stride:     {train_env.config.step_stride}")

    config = {
        "dataset_path": str(dataset_path),
        "n_epochs": n_epochs,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "hidden_dim": hidden_dim,
        "device": str(device) if device else "auto",
        "lr": lr,
        "gamma": gamma,
        "epsilon_decay": epsilon_decay,
        "target_update_freq": target_update_freq,
        "state_dim": state_dim,
        "n_spot_actions": n_spot_actions,
        "n_vol_actions": n_vol_actions,
        "spot_strategies": spot_names,
        "vol_strategies": vol_names,
        "step_stride": train_env.config.step_stride,
        "realized_vol_window": train_env.config.realized_vol_window,
        "leverage_limit": train_env.config.leverage_limit,
        "kelly_scale": train_env.config.kelly_scale,
        "kelly_alpha": train_env.config.kelly_alpha,
    }

    with open(results_dir / "train_config.json", "w") as f:
        json.dump(config, f, indent=2)

    feature_keys = list(train_env.feature_cols)
    with open(results_dir / "feature_keys.json", "w") as f:
        json.dump(feature_keys, f)

    spot_agent, vol_agent = create_agents(
        state_dim=state_dim,
        n_spot_actions=n_spot_actions,
        n_vol_actions=n_vol_actions,
        hidden_dim=hidden_dim,
        device=device,
        lr=lr,
        gamma=gamma,
        epsilon_decay=epsilon_decay,
        target_update_freq=target_update_freq,
    )

    # Prime EWMA strategies once and save checkpoint for epoch resets
    spot_family.reset_all()
    vol_family.reset_all()
    # Train EWMA on initial data without trading
    _ = train_env.reset(skip_warmup=False)
    # Snapshot EWMA state after warmup for deterministic epoch initialization
    warmup_end_states = {
        'spot': spot_family.save_all_states(),
        'vol': vol_family.save_all_states(),
    }
    print(f"Warmup complete: EWMA strategies trained on first {warmup_rows} rows")

    results = {"train": [], "val": []}
    best_val_equity = -np.inf

    epoch_pbar = tqdm(range(n_epochs), desc="Training")
    for epoch in epoch_pbar:
        epoch_pbar.set_description(f"Epoch {epoch + 1}/{n_epochs}")

        # Reset EWMA to warmup checkpoint for reproducibility
        spot_family.restore_all_states(warmup_end_states['spot'])
        vol_family.restore_all_states(warmup_end_states['vol'])

        # Reset episode state and skip warmup computation
        _ = train_env.reset(skip_warmup=True)
        train_env.current_row = warmup_rows
        train_metrics = run_env(
            env=train_env,
            spot_agent=spot_agent,
            vol_agent=vol_agent,
            training=True,
            collect_history=False,
            show_steps=True,
            checkpoint_dir=None,
            epoch=epoch,
            tag="Train",
            debug=debug,
        )

        print(
            f"Train: Equity={train_metrics['avg_equity']:.4f} "
            f"(Std={train_metrics['std_equity']:.4f}), "
            f"SpotLoss={train_metrics['avg_spot_loss']:.6f}, "
            f"VolLoss={train_metrics['avg_vol_loss']:.6f}, "
            f"Epsilon={spot_agent.get_epsilon():.3f}"
        )

        # Checkpoint all model states before validation
        ewma_spot_states = spot_family.save_all_states()
        ewma_vol_states = vol_family.save_all_states()
        dqn_spot_state = spot_agent.save_state()
        dqn_vol_state = vol_agent.save_state()

        # Run validation without training updates
        _ = val_env.reset(skip_warmup=True)
        val_metrics = run_env(
            env=val_env,
            spot_agent=spot_agent,
            vol_agent=vol_agent,
            training=False,
            collect_history=False,
            show_steps=False,
            checkpoint_dir=None,
            epoch=epoch,
            tag="Val",
            debug=debug,
        )

        # Restore training state to prevent validation leakage
        spot_family.restore_all_states(ewma_spot_states)
        vol_family.restore_all_states(ewma_vol_states)
        spot_agent.restore_state(dqn_spot_state)
        vol_agent.restore_state(dqn_vol_state)

        print(
            f"Val:   Equity={val_metrics['avg_equity']:.4f} "
            f"(±{val_metrics['std_equity']:.4f})"
        )

        results["train"].append(
            {
                "epoch": epoch,
                "avg_equity": float(train_metrics["avg_equity"]),
                "std_equity": float(train_metrics["std_equity"]),
                "avg_reward": float(train_metrics["avg_reward"]),
                "avg_spot_loss": float(train_metrics["avg_spot_loss"]),
                "avg_vol_loss": float(train_metrics["avg_vol_loss"]),
                "epsilon": float(spot_agent.get_epsilon()),
                "equities": [float(e) for e in train_metrics["equities"]],
            }
        )

        results["val"].append(
            {
                "epoch": epoch,
                "avg_equity": float(val_metrics["avg_equity"]),
                "std_equity": float(val_metrics["std_equity"]),
                "avg_reward": float(val_metrics["avg_reward"]),
                "equities": [float(e) for e in val_metrics["equities"]],
            }
        )

        # Save best model based on validation performance
        if val_metrics["avg_equity"] > best_val_equity:
            best_val_equity = val_metrics["avg_equity"]
            spot_agent.save(str(checkpoints_dir / "spot_agent_best.pt"))
            vol_agent.save(str(checkpoints_dir / "vol_agent_best.pt"))
            print(f"  *** New best val equity: {best_val_equity:.4f} ***")

        spot_agent.save(str(checkpoints_dir / f"spot_agent_epoch_{epoch}.pt"))
        vol_agent.save(str(checkpoints_dir / f"vol_agent_epoch_{epoch}.pt"))

        with open(results_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        plot_training_curves(results, plots_dir)
        plot_loss_curves(results, plots_dir)

    print("\nFinal Test (using best model)")

    # Load best performing agents from validation
    spot_agent.load(str(checkpoints_dir / "spot_agent_best.pt"))
    vol_agent.load(str(checkpoints_dir / "vol_agent_best.pt"))

    # Checkpoint states before test to prevent leakage
    ewma_spot_states = spot_family.save_all_states()
    ewma_vol_states = vol_family.save_all_states()
    dqn_spot_state = spot_agent.save_state()
    dqn_vol_state = vol_agent.save_state()

    # Evaluate on held-out test set with full metrics
    _ = test_env.reset(skip_warmup=True)
    test_metrics = run_env(
        env=test_env,
        spot_agent=spot_agent,
        vol_agent=vol_agent,
        training=False,
        collect_history=True,
        show_steps=True,
        checkpoint_dir=None,
        epoch=n_epochs,
        tag="Test",
        debug=debug,
    )

    # Restore state to prevent test data leakage
    spot_family.restore_all_states(ewma_spot_states)
    vol_family.restore_all_states(ewma_vol_states)
    spot_agent.restore_state(dqn_spot_state)
    vol_agent.restore_state(dqn_vol_state)

    print(
        f"Test: Equity={test_metrics['avg_equity']:.4f} "
        f"(±{test_metrics['std_equity']:.4f}), "
        f"Min={np.min(test_metrics['equities']):.4f}, "
        f"Max={np.max(test_metrics['equities']):.4f}"
    )

    strategy_counts = {"spot": {}, "vol": {}}
    for s in test_metrics["spot_history"]:
        strategy_counts["spot"][s] = strategy_counts["spot"].get(s, 0) + 1
    for s in test_metrics["vol_history"]:
        strategy_counts["vol"][s] = strategy_counts["vol"].get(s, 0) + 1

    results["test"] = {
        "avg_equity": float(test_metrics["avg_equity"]),
        "std_equity": float(test_metrics["std_equity"]),
        "avg_reward": float(test_metrics["avg_reward"]),
        "min_equity": float(np.min(test_metrics["equities"])),
        "max_equity": float(np.max(test_metrics["equities"])),
        "equities": [float(e) for e in test_metrics["equities"]],
        "strategy_counts": strategy_counts,
    }

    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    generate_all_plots(
        results=results,
        save_dir=plots_dir,
        strategy_counts=strategy_counts,
        strategy_history={
            "spot_history": test_metrics["spot_history"],
            "vol_history": test_metrics["vol_history"],
        },
        pnl_history=test_metrics["pnl_history"],
        position_history=test_metrics["position_history"],
    )

    print(f"\nBest val: {best_val_equity:.4f}, Test: {test_metrics['avg_equity']:.4f}")
    print(f"Results: {results_dir}")

    return spot_agent, vol_agent, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/home/ubuntu/final_submission/data/dataset_0to10d_100ms.parquet"),
    )
    parser.add_argument(
        "--spot_config",
        type=Path,
        default=Path("strategies/spot_strategies.json"),
    )
    parser.add_argument(
        "--vol_config",
        type=Path,
        default=Path("strategies/vol_strategies.json"),
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("results"),
    )
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--warmup_rows", type=int, default=3000)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon_decay", type=int, default=50000)
    parser.add_argument("--target_update_freq", type=int, default=1000)
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    train(
        dataset_path=args.dataset,
        spot_config_path=args.spot_config,
        vol_config_path=args.vol_config,
        n_epochs=args.n_epochs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        warmup_rows=args.warmup_rows,
        results_dir=args.results_dir,
        hidden_dim=args.hidden_dim,
        device=args.device,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_decay=args.epsilon_decay,
        target_update_freq=args.target_update_freq,
        debug=args.debug,
    )
