"""Describe representation for types of assets."""

import math
import numpy as np
from scipy.stats import lognorm


class Stock:
    """Represent Black-Scholes Model Stock (fixed drift and vol)."""

    def __init__(self, s0, mu, sigma):
        """Containerize relavent information."""
        self.s0 = s0
        self.mu = mu
        self.sigma = sigma

    def price_path(self, time):
        """Simulate gbm by following black-scholes stock price dynamics.

        # Requires time is a standard partition of some interval with len > 1
        """
        t_step = time[1] - time[0]
        n = len(time)
        noise = self.sigma * np.random.normal(0, math.sqrt(t_step), n - 1)
        prices = np.zeros(n)
        prices[0] = self.s0
        for i in range(n - 1):
            prices[i + 1] = prices[i] * (1 + self.mu * t_step + noise[i])
        return prices

    def random_final_price(self, T):
        """Generate final price of stock."""
        return self.s0 * np.exp(
            (self.mu - self.sigma ** 2 / 2)
            * T + self.sigma
            * np.random.normal(0, math.sqrt(T))
        )

    def mean_trajectory(self, T):
        """Compute the no noise trajectory of stock."""
        return self.s0 * np.exp(self.mu * T)

    def pdf_curve(self, prices, T):
        """Return theoretical distribution of stock price at t=T."""
        return lognorm.pdf(
            prices,
            self.sigma * math.sqrt(T),
            scale=np.exp(
                np.log(self.s0) + (self.mu - 0.5 * self.sigma ** 2) * T
            )
        )


class MoneyMarket:
    """Represent risk free asset which grows with rate r(t)."""

    def __init__(self, r_func, deterministic=False):
        """Require r_func to be a non-random function of t."""
        self.deterministic = deterministic
        self.r = r_func

    def price_path(self, time):
        """Simulate growth of money market account with any rate structure."""
        t_step = time[1] - time[0]
        n = len(time)
        prices = np.zeros(time)

        prices[0] = 1  # assume money market asset starts at unit price
        for i in range(n - 1):
            # TODO: check if / t_step is necessary (based on short rate data)
            prices[i + 1] = (1 + self.r(time[i]) / t_step) * prices[i]
        return prices

    def mean_trajectory(self, T):
        """Compute trajectory of money market IF rates are non-random."""
        return np.exp(self.r[0] * T)
