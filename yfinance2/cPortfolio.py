import yfinance2.portfolio as pt
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


class StockPortfolio:
    def __init__(self, name, tickers, per, initial_weights, investment_amount=100000000,
                 gearing=1, rebalance_fee=0.002, expense_ratio=0.008):
        self.name = name
        self.tickers = tickers
        self.per = per
        self.initial_weights = initial_weights
        self.prices = pt.download_data(tickers, per)
        if isinstance(initial_weights, list):
            self.weights = pt.fill_weights(initial_weights, self.prices)
        self.portfolio_amount = pt.stock_amount(self.prices, self.weights, investment_amount)
        self.gearing = gearing
        self.rebalance_fee = rebalance_fee
        self.expense_ratio = expense_ratio
        self.total_return()

    def total_return(self):
        tot_return =  pt.calculate_total_return(self.prices, self.prices.pct_change(), self.weights, self.gearing,
                                                self.rebalance_fee, self.expense_ratio)
        return tot_return

    def port_char(self):
        portfolio_return, portfolio_growth, portfolio_deviation, portfolio_sharpe = pt.portfolio_characteristics(self.total_return())
        return portfolio_return, portfolio_growth, portfolio_deviation, portfolio_sharpe


port1 = StockPortfolio('Den efficiente', ['SXRV.DE', 'DBXG.DE'], '7y', [-1, 2])
print(port1.port_char())
plt.plot(port1.total_return())
plt.show()

tup1 = (3, 2, 5, 'big')
print(tup1)