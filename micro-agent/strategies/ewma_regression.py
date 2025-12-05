import numpy as np

class EWMARegression:
    def __init__(
        self,
        n_features: int,
        halflife: float = 600.0,
        regularization: float = 1e-3
    ):
        self.n_features = n_features
        self.halflife = halflife
        self.regularization = regularization
        # Convert halflife to exponential decay rate
        self.decay = 1 - np.exp(-np.log(2) / halflife)

        # Initialize covariance matrices with regularization
        self.cov_XX = np.eye(n_features) * regularization
        self.cov_Xy = np.zeros(n_features)
        self.weights = np.zeros(n_features)
        self.n_obs = 0

    def update(self, x: np.ndarray, y: float):
        # Skip invalid data to prevent numerical issues
        x = np.asarray(x, dtype=np.float64).ravel()

        if not np.all(np.isfinite(x)) or not np.isfinite(y):
            return

        outer_xx = np.outer(x, x)
        # Use current prediction error for update
        residual = y - self.weights.dot(x)

        # Apply exponential weighting to covariance matrices
        self.cov_XX += self.decay * (outer_xx - self.cov_XX)
        self.cov_Xy += self.decay * (x * residual)

        # Add regularization to prevent singular matrix
        reg_matrix = self.cov_XX + np.eye(self.n_features) * self.regularization

        # Solve for updated regression weights
        try:
            new_weights = np.linalg.solve(reg_matrix, self.cov_Xy)
            if np.all(np.isfinite(new_weights)):
                self.weights = new_weights
        except np.linalg.LinAlgError:
            pass

        self.n_obs += 1

    def predict(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64).ravel()
        # Linear prediction from learned weights
        result = self.weights.dot(x)
        # Safeguard against numerical errors
        if np.isnan(result):
            return 0.0
        if np.isinf(result):
            return 0.0
        return float(result)

    def reset(self):
        # Reset EWMA to initial state
        self.cov_XX = np.eye(self.n_features) * self.regularization
        self.cov_Xy = np.zeros(self.n_features)
        self.weights = np.zeros(self.n_features)
        self.n_obs = 0

    def save_state(self) -> dict:
        # Save complete EWMA state
        return {
            'cov_XX': self.cov_XX.copy(),
            'cov_Xy': self.cov_Xy.copy(),
            'weights': self.weights.copy(),
            'n_obs': self.n_obs,
        }

    def restore_state(self, state: dict) -> None:
        # Restore previously saved EWMA state
        self.cov_XX = state['cov_XX'].copy()
        self.cov_Xy = state['cov_Xy'].copy()
        self.weights = state['weights'].copy()
        self.n_obs = state['n_obs']
