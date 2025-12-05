import numpy as np
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from training_environment.gym import TradingEnv


def update_spot_agent(
    agent,
    state,
    action,
    reward,
    next_state,
    done,
) -> float:
    agent.store_transition(state, action, reward, next_state, done)
    return agent.update()


def update_vol_agent(
    agent,
    state,
    action,
    reward,
    next_state,
    done,
) -> float:
    agent.store_transition(state, action, reward, next_state, done)
    return agent.update()


def run_env(
    env: TradingEnv,
    spot_agent,
    vol_agent,
    training: bool = True,
    collect_history: bool = False,
    show_steps: bool = False,
    checkpoint_dir: Optional[Path] = None,
    epoch: int = 0,
    tag: str = "",
    debug: bool = False,
) -> Dict:
    # Execute full episode with hierarchical agent control
    state = env.reset()

    if debug:
        print(f"\n{'='*60}")
        print(f"{tag.upper()} - Starting episode")
        print(f"{'='*60}")
        print(f"Initial equity: {env.equity:.6f}")
        print(f"Initial position: {env.position_fraction:.6f}")
        print(f"Environment rows: {env.n_rows}")
        print(f"State shape: {state.shape}")

    episode_reward = 0.0
    spot_losses = []
    vol_losses = []

    equities_history = []
    pnl_history = []
    position_history = []
    spot_history = []
    vol_history = []

    # Track vol agent state for temporal credit assignment
    current_vol_state = None
    current_vol_action = None
    vol_cum_reward = 0.0

    max_steps = (env.n_rows + env.config.step_stride - 1) // env.config.step_stride

    if debug:
        print(f"Max steps: {max_steps}\n")
    step_iter = (
        tqdm(range(max_steps), desc=f"{tag} steps", leave=False, position=2)
        if show_steps
        else range(max_steps)
    )

    for step_idx in step_iter:
        spot_action = spot_agent.select_action(state, training=training)
        vol_action = vol_agent.select_action(state, training=training)

        prev_equity = env.equity
        next_state, reward, done, info = env.step(spot_action, vol_action)

        # Collect equity at each step
        equities_history.append(info["equity"])

        # Debug output every 3000 steps
        if debug and step_idx % 3000 == 0:
            equity_change = info["equity"] - prev_equity
            market_return = info.get('market_return', 0.0)
            block_return = info.get('block_return', 0.0)
            position = info['position_fraction']

            print(f"\nStep {step_idx}:")
            print(f"  Spot action: {spot_action}, Vol action: {vol_action}")
            print(f"  Spot strategy: {info.get('spot_strategy', 'N/A')}")
            print(f"  Vol strategy: {info.get('vol_strategy', 'N/A')}")
            print(f"  Vol decision: {info.get('vol_decision', False)}")
            print(f"  Position: {position:.6f}")
            print(f"  Market return: {market_return:+.8f} (raw market move)")
            print(f"  Portfolio return: {block_return:+.8f} (with position)")
            print(f"  Equity: {prev_equity:.6f} -> {info['equity']:.6f} (Δ={equity_change:+.8f})")
            print(f"  Reward: {reward:.8f}")

            # Verify the math
            expected_portfolio = market_return * position
            if abs(block_return - expected_portfolio) > 0.0001:
                print(f"  WARNING: Expected portfolio return: {expected_portfolio:+.8f} (position * market)")

        if training:
            # Spot agent learns at every step
            spot_loss = update_spot_agent(
                spot_agent, state, spot_action, reward, next_state, done
            )
            spot_losses.append(spot_loss)

            # Vol agent learns on decision stride with cumulative reward
            vol_loss = None
            if info.get("vol_decision", False):
                # Update previous vol decision with accumulated reward
                if current_vol_state is not None and current_vol_action is not None:
                    vol_loss = update_vol_agent(
                        vol_agent,
                        current_vol_state,
                        current_vol_action,
                        vol_cum_reward,
                        state,
                        False,
                    )
                # Start new vol decision tracking
                current_vol_state = state
                current_vol_action = vol_action
                vol_cum_reward = reward
            else:
                # Accumulate reward until next vol decision
                vol_cum_reward += reward

            if vol_loss is not None:
                vol_losses.append(vol_loss)

        if collect_history:
            pnl_history.append(info["equity"] - 1.0)
            position_history.append(info["position_fraction"])
            spot_history.append(info["spot_strategy"])
            vol_history.append(info["vol_strategy"])

        state = next_state
        episode_reward += reward

        if done:
            break

    if show_steps and hasattr(step_iter, "close"):
        step_iter.close()

    # Final vol agent update with terminal flag
    if training and current_vol_state is not None and current_vol_action is not None:
        final_vol_loss = update_vol_agent(
            vol_agent,
            current_vol_state,
            current_vol_action,
            vol_cum_reward,
            state,
            True,
        )
        if final_vol_loss is not None:
            vol_losses.append(final_vol_loss)

    # Compute episode performance metrics
    avg_equity = float(np.mean(equities_history)) if equities_history else 1.0
    std_equity = float(np.std(equities_history)) if equities_history else 0.0
    final_equity = equities_history[-1] if equities_history else 1.0

    if debug:
        print(f"\n{'='*60}")
        print(f"{tag.upper()} - Episode Summary")
        print(f"{'='*60}")
        print(f"Total steps: {len(equities_history)}")
        print(f"Final equity: {final_equity:.6f}")
        print(f"Total return: {(final_equity - 1.0) * 100:.2f}%")
        print(f"Avg equity: {avg_equity:.6f}")
        print(f"Std equity: {std_equity:.6f}")
        print(f"Sharpe (annualized ~100ms steps): {(avg_equity - 1.0) / (std_equity + 1e-8) * np.sqrt(252 * 6.5 * 60 * 60 * 10):.2f}")
        print(f"Total reward: {episode_reward:.6f}")
        print(f"Avg spot loss: {float(np.mean(spot_losses)) if spot_losses else 0.0:.6f}")
        print(f"Avg vol loss: {float(np.mean(vol_losses)) if vol_losses else 0.0:.6f}")

        if equities_history:
            equity_changes = np.diff(equities_history)
            positive_steps = (equity_changes > 0).sum()
            negative_steps = (equity_changes < 0).sum()
            zero_steps = (equity_changes == 0).sum()
            print(f"\nStep outcomes:")
            print(f"  Positive: {positive_steps} ({positive_steps/len(equity_changes)*100:.1f}%)")
            print(f"  Negative: {negative_steps} ({negative_steps/len(equity_changes)*100:.1f}%)")
            print(f"  Zero: {zero_steps} ({zero_steps/len(equity_changes)*100:.1f}%)")
            print(f"  Best step: {equity_changes.max():+.8f}")
            print(f"  Worst step: {equity_changes.min():+.8f}")
        print(f"{'='*60}\n")

    result = {
        "rewards": [episode_reward],
        "equities": equities_history,
        "avg_reward": episode_reward,
        "avg_equity": avg_equity,
        "std_equity": std_equity,
        "avg_spot_loss": float(np.mean(spot_losses)) if spot_losses else 0.0,
        "avg_vol_loss": float(np.mean(vol_losses)) if vol_losses else 0.0,
    }

    if collect_history:
        result["pnl_history"] = pnl_history
        result["position_history"] = position_history
        result["spot_history"] = spot_history
        result["vol_history"] = vol_history

    if training and checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{tag}_" if tag else ""
        spot_agent.save(str(checkpoint_dir / f"{prefix}spot_agent_epoch_{epoch}.pt"))
        vol_agent.save(str(checkpoint_dir / f"{prefix}vol_agent_epoch_{epoch}.pt"))

    return result
