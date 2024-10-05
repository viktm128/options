"""Model stocks,money market, and euro options account under BS."""

import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from asset import Stock


def plot_paths(num_paths, asset, T, n_step=100):
    """Plot dynamics of asset for num_path trials."""
    _, ax = plt.subplots()

    time = np.linspace(0, T, n_step + 1)
    for _ in range(num_paths):
        ax.plot(time, asset.price_path(time))

    ax.plot(time, asset.mean_trajectory(time))
    plt.show()


def price_distributions(asset, T, n_step=100, n_sim=500):
    """Compare distribution of asset price by paths and by closed form."""
    path_distr = np.zeros(n_sim)
    direct_distr = np.zeros(n_sim)

    time = np.linspace(0, T, n_step + 1)
    for i in range(n_sim):
        path_distr[i] = asset.price_path(time)[-1]
        direct_distr[i] = asset.random_final_price(T)

    _, axes = plt.subplots(1, 2)
    axes[0].hist(path_distr, density=True)
    axes[1].hist(direct_distr, density=True)

    prices = np.linspace(0.1, max(max(path_distr), max(direct_distr)), 200)
    axes[0].plot(
        prices,
        asset.pdf_curve(prices, T)
    )
    axes[1].plot(
        prices,
        asset.pdf_curve(prices, T)
    )
    plt.show()


def main():
    """Driver function."""
    S = Stock(50, 0.06, 0.14)
    T = 10
    n_step = 300
    n_sim = 1000
    # plot_paths(5, S, T, n_step)
    price_distributions(S, T, n_step, n_sim)


if __name__ == "__main__":
    main()

# TODO: pull a ticker and automatically generate these graphs
# TODO: need historical mu and vol calculations from historical sample
# TODO: need implied vol calculations from various asset prices
