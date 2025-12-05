from dataclasses import dataclass


@dataclass
class DifferentialSharpeReward:
    eta: float = 0.01

    def __post_init__(self):
        # Running estimates of mean and second moment
        self.A = 0.0
        self.B = 0.0

    def compute(self, ret: float) -> float:
        # Calculate deviations from running estimates
        delta_A = ret - self.A
        delta_B = ret * ret - self.B

        # Compute instantaneous variance from moments
        variance = self.B - self.A * self.A
        # Differential Sharpe ratio gradient for online optimization
        if variance > 1e-12:
            dsr = (self.B * delta_A - 0.5 * self.A * delta_B) / (variance ** 1.5)
        else:
            dsr = delta_A

        # Update running statistics with learning rate
        self.A += self.eta * delta_A
        self.B += self.eta * delta_B

        return dsr

    def reset(self):
        self.A = 0.0
        self.B = 0.0
