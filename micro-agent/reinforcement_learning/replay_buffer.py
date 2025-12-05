import numpy as np

class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        state_dim: int
    ):
        self.capacity = capacity
        self.state_dim = state_dim

        # Preallocate arrays for efficient memory usage
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self.position = 0
        self.size = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: float
    ):
        # Store transition at current position
        idx = self.position

        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done

        # Circular buffer overwrites oldest when full
        self.position = (self.position + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def sample(
        self,
        batch_size: int
    ):
        # Random sampling for decorrelated training batches
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    def __len__(self):
        return self.size
