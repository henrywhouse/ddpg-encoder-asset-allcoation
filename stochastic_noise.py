############################################
###  Noise (for environment exploration) ###
############################################

## Ornstein-Uhlenbeck Process: Used for DDPG Action Noise

import numpy as np

## Ornstein-Uhlenbeck (1930) Process
class OrnsteinUhlenbeck(object):
    """Ornstein-Uhlenbeck process for action noise
    """
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x_0=None):
        """Initialize parameters and noise process"""
        
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x_0 = x_0
        self.reset()


    def __call__(self):
        """Return action noise"""

        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)  # Noise formula
        self.x_prev = x  # Update previous noise value
        return x


    def reset(self):
        """Reset the noise process"""

        self.x_prev = self.x_0 if self.x_0 is not None else np.zeros_like(self.mu)