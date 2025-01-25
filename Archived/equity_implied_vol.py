"""Imply volatility from a basket of European options on specific equity."""

import math
import datetime
from scipy.stats import norm
import yfinance as yf
import pandas as pd


class EquityVolSmile():
    """Representation of vol surface for a US equity."""
    def __init__(self, ticker, T_max):
        """Perform data collection and represent object."""
        self.ticker = ticker



def options_data(tk):
    """"Scrape options data from yfinace. Credit @TonyLian on Medium."""
    exps = tk.options

    # Get options for each expiration
    options = pd.DataFrame()
    for e in exps:
        opt = tk.option_chain(e)
        opt = pd.DataFrame()._append(opt.calls)._append(opt.puts)
        opt['expirationDate'] = e
        options = options._append(opt, ignore_index=True)

    # Bizarre error in yfinance that gives the wrong expiration date
    # Add 1 day to get the correct expiration date
    options['expirationDate'] = pd.to_datetime(options['expirationDate']) + datetime.timedelta(days = 1)
    options['dte'] = (options['expirationDate'] - datetime.datetime.today()).dt.days / 365
    
    # Boolean column if the option is a CALL
    options['CALL'] = options['contractSymbol'].str[4:].apply(
        lambda x: "C" in x)
    
    options[['bid', 'ask', 'strike']] = options[['bid', 'ask', 'strike']].apply(pd.to_numeric)
    
    # Drop unnecessary and meaningless columns
    options = options.drop(columns = ['contractSize', 'currency', 'change', 'percentChange', 'lastTradeDate', 'lastPrice'])

    return options
