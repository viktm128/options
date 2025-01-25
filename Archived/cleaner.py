"""Collect, process, and format data used for model fitting."""

import pandas as pd
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


def build_forward_curve(filename):
    """Read forward futures data and construct smooth forward rate curve."""
    forward_df = pd.read_csv(filename)
    forward_df['Rates'] = (100 - forward_df['Prior Settle']) / 100.00
    cs = CubicSpline(forward_df['Month'], forward_df['Rates'])
    x = np.arange()
    forward_df.plot(kind='scatter', x='Month', y='Rates')
    plt.show()


def read_equity_options_chain_csv(filename):
    """Collect and format 5XSE european options data."""
    df = pd.read_csv(filename)
    df['Last traded'] = df['Last traded'].apply(lambda x: pd.to_datetime(x, dayfirst=True))
    return df


rates_file = "data/sofr_3month_11192024.csv"
build_forward_curve(rates_file)
