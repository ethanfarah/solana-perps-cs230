import numpy as np
import pandas as pd
import random
from collections import deque
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


class TradingEnv:
    # Environment for trading with discrete inventory changes
    # State: [inventory, close, high, low, volume, macro features] normalized
    # Action: discrete inventory change from -max to +max
    # Reward: price_change * inventory (with optional transform)
    def __init__(
        self,
        features: np.ndarray,
        close_prices: np.ndarray,
        max_inventory: int = 10,
        initial_cash: float = 10000.0,
        reward_transform: str | None = "log",  # options: None, "log", "sqrt"
    ):
        # Store normalized features (T x F) and raw prices (T,)
        self.features = features
        self.close_prices = close_prices
        self.max_inventory = max_inventory
        self.initial_cash = initial_cash
        self.reward_transform = reward_transform

        # Build discrete action space for inventory changes
        self.n_steps = len(close_prices)
        self.action_space = np.arange(-max_inventory, max_inventory + 1, dtype=np.int32)
        self.n_actions = len(self.action_space)

        self.reset()

    def reset(self):
        # Reset episode to initial state
        self.current_step = 0
        self.inventory = 0
        self.cash = self.initial_cash
        return self._get_state()

    def _get_state(self):
        # Build state vector: [normalized_inventory, features]
        inv_norm = self.inventory / self.max_inventory
        feat = self.features[self.current_step]
        state = np.concatenate(([inv_norm], feat), axis=0).astype(np.float32)
        return state

    def step(self, action_index: int):
        # Get current price and decode action
        current_price = float(self.close_prices[self.current_step])
        action_change = int(self.action_space[action_index])

        # Calculate desired inventory with bounds
        desired_inventory = np.clip(self.inventory + action_change, 0, self.max_inventory)
        trade_amount = desired_inventory - self.inventory

        # Check affordability for buy orders
        if trade_amount > 0:
            max_affordable = int(self.cash // current_price)
            trade_amount = min(trade_amount, max_affordable)
            desired_inventory = self.inventory + trade_amount

        # Execute trade and update cash
        self.inventory = int(desired_inventory)
        self.cash -= trade_amount * current_price

        # Advance to next timestep
        self.current_step += 1
        done = self.current_step >= self.n_steps - 1

        # Get next price for reward calculation
        next_price = float(self.close_prices[self.current_step])

        # Reward based on price change and inventory held
        price_change = next_price - current_price
        raw_reward = price_change * self.inventory

        # Apply optional transform to compress extreme rewards
        if self.reward_transform is None:
            reward = raw_reward
        else:
            x = float(raw_reward)
            sign = np.sign(x)
            mag = abs(x)
            if self.reward_transform == "log":
                # Log transform dampens large swings
                reward = sign * np.log1p(mag)
            elif self.reward_transform == "sqrt":
                # Sqrt transform milder than log
                reward = sign * np.sqrt(mag)
            else:
                reward = raw_reward

        # Build next state and info dict
        next_state = self._get_state()
        portfolio_value = self.cash + self.inventory * next_price
        info = {
            "portfolio_value": portfolio_value,
            "inventory": self.inventory,
            "cash": self.cash,
            "price_change": price_change,
            "raw_reward": raw_reward,
        }

        return next_state, float(reward), done, info


class QNetwork(nn.Module):
    # Simple 2-layer MLP for Q-value estimation
    def __init__(self, state_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    # Circular buffer for storing experience tuples
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Add transition to buffer (old ones auto-evicted)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        # Sample random minibatch for training
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


def load_data(csv_path: str, train_ratio: float = 0.95):
    # Load CSV with price and macro features
    df = pd.read_csv(csv_path)

    # Define feature columns (price + volume + macro)
    feature_cols = [
        "close_price", "8h_change", "16h_change", "24h_change", "volume",
        "sp500_1d_change", "sp500_2d_change", "sp500_1w_change",
        "sol_1d_change", "sol_2d_change", "sol_1w_change",
        "eth_1d_change", "eth_2d_change", "eth_1w_change",
        "btc_1d_change", "btc_2d_change", "btc_1w_change"
    ]

    # Convert all features to numeric
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove rows with missing data
    df = df.dropna(subset=feature_cols)

    # Extract features and target prices
    features = df[feature_cols].values.astype(np.float32)
    close_prices = df["close_price"].values.astype(np.float32)

    # Z-score normalization for stable training
    means = features.mean(axis=0)
    stds = features.std(axis=0) + 1e-8
    norm_features = (features - means) / stds

    # Chronological split into train and eval
    split_idx = int(len(norm_features) * train_ratio)

    train_features = norm_features[:split_idx]
    train_prices = close_prices[:split_idx]

    eval_features = norm_features[split_idx:]
    eval_prices = close_prices[split_idx:]

    return (train_features, train_prices), (eval_features, eval_prices), (means, stds)


def train_dqn(
    train_data,
    num_episodes: int = 100,
    buffer_capacity: int = 10_000,
    batch_size: int = 64,
    gamma: float = 0.99,
    lr: float = 1e-3,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_steps: int = 5_000,
    target_update_freq: int = 500,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_features, train_prices = train_data
    # Use logarithmic reward shaping by default; change to "sqrt" or None to experiment
    env = TradingEnv(train_features, train_prices, reward_transform="log")
    state_dim = env._get_state().shape[0]
    n_actions = env.n_actions

    print(f"State dimension: {state_dim}, Actions: {n_actions}")
    print(f"Training on {len(train_prices)} days")

    # Networks
    policy_net = QNetwork(state_dim, n_actions).to(device)
    target_net = QNetwork(state_dim, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(buffer_capacity)

    # Exploration schedule
    def get_epsilon(step):
        return epsilon_end + (epsilon_start - epsilon_end) * np.exp(-step / epsilon_decay_steps)

    total_steps = 0

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0

        done = False
        while not done:
            epsilon = get_epsilon(total_steps)

            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action_index = random.randrange(n_actions)
            else:
                with torch.no_grad():
                    s_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    q_values = policy_net(s_tensor)
                    action_index = int(q_values.argmax(dim=1).item())

            next_state, reward, done, info = env.step(action_index)

            # Store transition
            replay_buffer.push(state, action_index, reward, next_state, done)

            state = next_state
            episode_reward += reward
            total_steps += 1

            # Optimize policy network
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                states = torch.tensor(states, dtype=torch.float32, device=device)
                actions = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
                next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
                dones = torch.tensor(dones.astype(np.float32), dtype=torch.float32, device=device).unsqueeze(1)

                # Q(s,a)
                q_values = policy_net(states).gather(1, actions)

                # Target: r + gamma * max_a' Q_target(s', a') * (1 - done)
                with torch.no_grad():
                    next_q_values = target_net(next_states).max(dim=1, keepdim=True)[0]
                    target = rewards + gamma * next_q_values * (1.0 - dones)

                loss = nn.MSELoss()(q_values, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Periodically update target network
            if total_steps % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes} - Reward: {episode_reward:.2f}, Epsilon: {epsilon:.3f}")

    return policy_net, env


def run_bogo_trader(eval_data, initial_cash: float = 10000.0, max_inventory: int = 10):
    """
    Run a random action baseline (bogo trader) for comparison.
    """
    eval_features, eval_prices = eval_data
    
    env = TradingEnv(
        eval_features,
        eval_prices,
        max_inventory=max_inventory,
        initial_cash=initial_cash,
        reward_transform="log",
    )
    state = env.reset()
    
    done = False
    bogo_pnl_history = []
    
    while not done:
        # Random action from available action space
        action_index = random.randrange(env.n_actions)
        
        next_state, reward, done, info = env.step(action_index)
        state = next_state
        
        # Track P&L at each step
        portfolio_value = info["portfolio_value"]
        pnl = portfolio_value - initial_cash
        bogo_pnl_history.append(pnl)
    
    final_portfolio = info["portfolio_value"]
    final_price = eval_prices[-1]
    
    # Compute final P&L
    pnl = final_portfolio - initial_cash
    pnl_pct = (pnl / initial_cash) * 100
    
    return {
        "final_portfolio": final_portfolio,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "pnl_history": bogo_pnl_history,
    }


def evaluate(policy_net, eval_data, initial_cash: float = 10000.0, max_inventory: int = 10):
    """
    Run the trained policy on the evaluation set and compute total P&L.
    """
    device = next(policy_net.parameters()).device
    eval_features, eval_prices = eval_data

    env = TradingEnv(
        eval_features,
        eval_prices,
        max_inventory=max_inventory,
        initial_cash=initial_cash,
        reward_transform="log",
    )
    state = env.reset()

    done = False
    total_reward = 0.0
    actions_taken = []
    
    # Track P&L over time
    pnl_history = []
    portfolio_values = []
    buy_hold_pnl_history = []
    days = []

    print(f"\n{'='*50}")
    print(f"Evaluating on {len(eval_prices)} days (last 5% of data)")
    print(f"Initial cash: ${initial_cash:.2f}")
    print(f"Starting price: ${eval_prices[0]:.2f}")
    print(f"{'='*50}\n")

    # Run bogo trader (random baseline) for comparison
    print("Running Bogo Trader (random baseline)...")
    bogo_results = run_bogo_trader(eval_data, initial_cash, max_inventory)
    bogo_pnl_history = bogo_results["pnl_history"]
    print(f"Bogo Trader final P&L: ${bogo_results['pnl']:.2f} ({bogo_results['pnl_pct']:+.2f}%)\n")

    buy_hold_units = initial_cash / eval_prices[0]
    day = 0

    while not done:
        with torch.no_grad():
            s_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            q_values = policy_net(s_tensor)
            action_index = int(q_values.argmax(dim=1).item())

        action_change = env.action_space[action_index]
        actions_taken.append(action_change)

        next_state, reward, done, info = env.step(action_index)
        total_reward += reward
        state = next_state
        
        # Track portfolio value and P&L at each step
        portfolio_value = info["portfolio_value"]
        portfolio_values.append(portfolio_value)
        pnl = portfolio_value - initial_cash
        pnl_history.append(pnl)
        
        # Buy and hold P&L for comparison
        current_price = eval_prices[env.current_step - 1] if env.current_step > 0 else eval_prices[0]
        buy_hold_value = buy_hold_units * current_price
        buy_hold_pnl = buy_hold_value - initial_cash
        buy_hold_pnl_history.append(buy_hold_pnl)
        
        days.append(day)
        day += 1

    # Final portfolio value
    final_portfolio = info["portfolio_value"]
    final_price = eval_prices[-1]

    # Compute P&L
    pnl = final_portfolio - initial_cash
    pnl_pct = (pnl / initial_cash) * 100

    # Buy and hold comparison
    buy_hold_units = initial_cash / eval_prices[0]
    buy_hold_value = buy_hold_units * final_price
    buy_hold_pnl = buy_hold_value - initial_cash
    buy_hold_pnl_pct = (buy_hold_pnl / initial_cash) * 100

    print(f"{'='*50}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Final Portfolio Value: ${final_portfolio:.2f}")
    print(f"Final Cash: ${info['cash']:.2f}")
    print(f"Final Inventory: {info['inventory']} units")
    print(f"Final Price: ${final_price:.2f}")
    print(f"")
    print(f"RL Agent P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)")
    print(f"Buy & Hold P&L: ${buy_hold_pnl:.2f} ({buy_hold_pnl_pct:+.2f}%)")
    print(f"Bogo Trader P&L: ${bogo_results['pnl']:.2f} ({bogo_results['pnl_pct']:+.2f}%)")
    print(f"")
    print(f"RL vs Buy & Hold: ${pnl - buy_hold_pnl:.2f} ({pnl_pct - buy_hold_pnl_pct:+.2f}%)")
    print(f"RL vs Bogo: ${pnl - bogo_results['pnl']:.2f} ({pnl_pct - bogo_results['pnl_pct']:+.2f}%)")
    print(f"{'='*50}")

    # Action distribution
    action_counts = {}
    for a in actions_taken:
        action_counts[a] = action_counts.get(a, 0) + 1
    print(f"\nAction distribution:")
    for a in sorted(action_counts.keys()):
        print(f"  Action {a:+d}: {action_counts[a]} times")

    # Plot P&L over time
    plt.figure(figsize=(12, 6))
    plt.plot(days, pnl_history, label='RL Agent P&L', linewidth=2)
    plt.plot(days, buy_hold_pnl_history, label='Buy & Hold P&L', linewidth=2, linestyle='--')
    plt.plot(days, bogo_pnl_history, label='Bogo Trader P&L', linewidth=2, linestyle='-.', alpha=0.7)
    plt.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    plt.xlabel('Day', fontsize=12)
    plt.ylabel('P&L ($)', fontsize=12)
    plt.title('P&L Over Time During Evaluation', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('data/evaluation_pnl_plot.png', dpi=150, bbox_inches='tight')
    print(f"\nP&L plot saved to 'data/evaluation_pnl_plot.png'")
    plt.close()

    return {
        "final_portfolio": final_portfolio,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "buy_hold_pnl": buy_hold_pnl,
        "buy_hold_pnl_pct": buy_hold_pnl_pct,
        "bogo_pnl": bogo_results['pnl'],
        "bogo_pnl_pct": bogo_results['pnl_pct'],
        "total_reward": total_reward,
        "pnl_history": pnl_history,
        "buy_hold_pnl_history": buy_hold_pnl_history,
        "bogo_pnl_history": bogo_pnl_history,
    }


if __name__ == "__main__":
    csv_path = "data/combined_dataset.csv"

    # Load and split data (95% train, 5% eval)
    train_data, eval_data, (means, stds) = load_data(csv_path, train_ratio=0.90)

    print(f"Training set size: {len(train_data[0])} days")
    print(f"Evaluation set size: {len(eval_data[0])} days")
    print()

    # Train the model
    policy_net, train_env = train_dqn(train_data, num_episodes=100)

    # Evaluate on the last 5%
    results = evaluate(policy_net, eval_data)
