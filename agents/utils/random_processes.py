'''
/* Copyright (C) 2019 Gerti Tuzi - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MY license, which unfortunately won't be written
 * because no one would care to read it, and in the best of scenarios
 * it will only be used for ego pampering needs.
 */
'''

import numpy as np
from agents.utils.schedule import ConstantSchedule
import copy


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=ConstantSchedule(0.2), dt = 1e-2):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu_init = mu
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        np.random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
        self.mu = self.mu_init * np.ones(self.size)

    def sample(self):
        """Update internal state and return it as a noise sample_set."""
        x = self.state
        dx = self.theta * (self.mu - x)*self.dt + self.sigma()*np.sqrt(self.dt) * np.random.randn(*x.shape)
        self.state = x + dx
        return self.state


class GaussianProcess:
    def __init__(self, size, std = ConstantSchedule(0.2), seed = None):
        self.size = size
        self.std = std
        np.random.seed(seed)

    def reset(self):
        pass

    def sample(self):
        return np.random.randn(*self.size) * self.std()