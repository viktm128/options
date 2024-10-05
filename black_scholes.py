"""Model stocks,money market, and euro options account under BS."""

import math
from scipy.stats import lognorm
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

TOL = 0.001


class Stock:
    """Represent Black-Scholes Model Stock (fixed drift and vol)."""

    def __init__(self, s0, mu, sigma):
        """Containerize relavent information."""
        self.s0 = s0
        self.mu = mu
        self.sigma = sigma


def deriv(f, x, del_x, prev=1):
    """Numerical approximation for function of one variable."""
    frac = (f(x + del_x) - f(x - del_x)) / (2 * del_x)
    if abs(frac - prev) < TOL:
        return frac
    return deriv(f, x, del_x / 10, frac)


def stock_price_path(S, T, n_step):
    """Simulate GBM by following Black-Scholes stock price dynamics."""
    dt = 1 / n_step
    noise = S.sigma * np.random.normal(0, math.sqrt(T * dt), n_step)
    prices = np.zeros(n_step + 1)
    prices[0] = S.s0
    for i in range(n_step):
        prices[i + 1] = prices[i] * (1 + S.mu * T * dt + noise[i])

    return prices


def final_price(S, T):
    """Generate final price of stock."""
    return S.s0 * np.exp(
        (S.mu - S.sigma ** 2 / 2)
        * T + S.sigma
        * np.random.normal(0, math.sqrt(T))
    )


def plot_paths(num_paths, S, T, n_step=100):
    """Plot dynamics of stock for num_path trials."""
    _, ax = plt.subplots()

    time = np.linspace(0, T, n_step + 1)
    for _ in range(num_paths):
        ax.plot(time, stock_price_path(S, T, n_step))

    ax.plot(time, S.s0 * np.exp(S.mu * time))
    plt.show()


def stock_price_distribution(S, T, n_step, n_sim):
    """Compare distribution of stock price by paths and by closed form."""
    path_distr = np.zeros(n_sim)
    direct_distr = np.zeros(n_sim)
    for i in range(n_sim):
        path_distr[i] = stock_price_path(S, T, n_step)[-1]
        direct_distr[i] = final_price(S, T)

    _, axes = plt.subplots(1, 2)
    axes[0].hist(path_distr, density=True)
    axes[1].hist(direct_distr, density=True)

    prices = np.linspace(0.1, max(max(path_distr), max(direct_distr)), 200)
    axes[0].plot(
        prices,
        lognorm.pdf(
            prices,
            S.sigma * math.sqrt(T),
            scale=np.exp(np.log(S.s0) + (S.mu - 0.5 * S.sigma ** 2) * T)
        )
    )
    axes[1].plot(
        prices,
        lognorm.pdf(
            prices,
            S.sigma * math.sqrt(T),
            scale=np.exp(np.log(S.s0) + (S.mu - 0.5 * S.sigma ** 2) * T)
        )
    )
    plt.show()


if __name__ == "__main__":
    S = Stock(50, 0.06, 0.24)
    T = 3
    n_step = 300
    n_sim = 1000
    # plot_paths(5, S, T, n_step)
    stock_price_distribution(S, T, n_step, n_sim)


# TODO: pull a ticker and automatically generate these graphs
# TODO: need historical mu and vol calculations from historical sample
# TODO: need implied vol calculations from various asset prices
