import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as stats
plt.style.use('dark_background')
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["g", "b", "y", "m"])
import numpy as np
import yfinance2.portfolio as pt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

#------------------SETTINGS---------------------
#yf_tickers =         ['IVV', 'EDV']
#yf_tickers =        ['TQQQ', 'TMF', 'VSCFX', 'IOFIX']
yf_tickers =        ['MGK', 'EDV']
# yf_tickers =        ['IVV', 'TLT']
#yf_tickers =        ['MBB', 'LQD', 'QQQ']
#yf_tickers =        ['EDV', 'QQQ']
#yf_tickers =        ['TQQQ', 'TMF']
#yf_tickers =        ['SXRV.DE', 'DBXG.DE', 'EZA']
#yf_tickers =        ['QQQ', 'TLT', 'EZA']
#bund_2x = '5X61.F'
#yf_tickers =        ['QQQ', 'TLT', 'AFK']
#yf_tickers =        ['SXRV.DE', 'XUTD.DE', 'LGQM.DE']
#yf_tickers =        ['SXRV.DE', bund_2x]


#yf_tickers =         ['DBPG.DE', 'LYMI.F']
#yf_tickers =         ['LYMI.F', 'LYQK.DE']
#yf_tickers =         ['IEAC.AS', 'IBTS.AS']
#yf_tickers =        ['UPRO', 'TQQQ', 'EDV']
#yf_tickers =       ['EQNR.OL', 'TQQQ']
#yf_tickers =       ['GLD', '^GSPC']
#yf_tickers =        ['NOVO-B.CO', 'TRYG.CO', 'NZYM-B.CO', 'BAVA.CO', 'DANSKE.CO', 'LUN.CO', 'RBREW.CO', 'DEMANT.CO',
#                     'AMBU-B.CO', 'ORPHA.CO']
#yf_tickers = ['NPINV.CO', 'CHR.CO', 'COLO-B.CO', 'AMBU-B.CO', 'BAVA.CO', 'NOVO-B.CO', 'TRYG.CO', 'RBREW.CO']
#yf_tickers = ['NPINV.CO', 'COLO-B.CO']


#weights =           [.02, .05, .1, .2, .1, .2, .1, .23]
#weights =           [1.1, .8, -.9]
#weights =           [1.1, .8, -.9]
#weights =           [-.4, -.2, .53, .53, .54]
#weights =           [0.3, 0.2, 0.2, 0.3]
weights =            [.4, .6]
#weights =            [-0.1, 0.1]
#weights = [-.1, 1.1]
#weights =           []


invest_amount = 44000
gearing = 1
per = '12y'
rebalance_fee = 0.0002
expense_ratio = 0.0


weights, yf_tickers = pt.correct_weight_sum(weights, yf_tickers, cash_proxy='ATOIX')
org_ticker_count = len(yf_tickers)
prices = pd.DataFrame(yf.download(yf_tickers, period=per)).loc[:, 'Adj Close']
no_obs = len(prices.index)


# Cleaning data
yf_tickers, prices, no_stocks = pt.clean_prices(prices, yf_tickers)
short_names = pt.clean_columns(prices, yf_tickers)
prices.columns = short_names
prices_pctchange = prices.pct_change()
prices_index = pt.normalize_data_frame(prices)

monthly_price_changes = prices_index.asfreq('BM', method='pad').pct_change()

# Initial weights
weights = pt.fill_weights(weights, prices)
original_weights = pd.DataFrame(weights, copy=True)

# Re-balance amounts
stock_amount, weight_compliant_stock_amount = pt.rebalance_stock_amounts(prices, weights, original_weights, invest_amount)

# Recalculate value of portfolio and weights
stock_value = pt.stock_value(prices, stock_amount)
weights = pt.actual_weights(prices, stock_value)

# Portfolio performance
daily_return = pt.calculate_total_return(prices, prices_pctchange, weights, gearing, rebalance_fee, expense_ratio)
daily_return['Return'] = pt.normalize_data_frame(daily_return['Return'])

# S&P 500 for comparison
comparisons = pd.DataFrame(yf.download(['SPY', '^NDX'], period=per)['Adj Close'])
spy = pt.normalize_data_frame(pt.pad_data(comparisons['SPY']))    # Normalizing to same start date
stoxx = pt.normalize_data_frame(pt.pad_data(comparisons['^NDX']))

# Portfolio characteristics
portfolio_return, portfolio_growth, portfolio_deviation, portfolio_sharpe = pt.portfolio_characteristics(daily_return['Return'])

# Appending index data to data frame
daily_return['SPY'] = spy
daily_return['STOXX'] = stoxx
daily_return['Daily return'] = portfolio_growth
daily_return['Drawdown'] = pt.calculate_drawdown(daily_return['Return'])
daily_return['SPX Drawdown'] = pt.calculate_drawdown(daily_return['SPY'])
daily_return['Drawdown inverse'] = 100-daily_return['Drawdown']
daily_return['SPX drawdown inverse'] = 100-daily_return['SPX Drawdown']

avg_drawdown = daily_return['Drawdown inverse'].median()
spx_drawdown = daily_return['SPX drawdown inverse'].median()

# SPY comparison
spy_return, spy_growth, spy_devation, spy_sharpe = pt.portfolio_characteristics(daily_return['SPY'])

daily_return['Rolling correlation'] = portfolio_growth.rolling(200).corr(spy_growth)
daily_return['Rolling std'] = portfolio_growth.rolling(200).std()


print('y/y Return:', "{:.2f}%".format(float(portfolio_return*100)), 'S&P 500:', "{:.2f}%".format(float(spy_return*100)))
print('Total return', "{:.2f}%".format((float(daily_return['Return'][-1]/100)-1)*100))
print('STD:', "{:.2f}%".format(portfolio_deviation*100), 'S&P 500:', "{:.2f}%".format(spy_devation*100))
print('Sharpe ratio:', "{:.2f}".format(portfolio_sharpe), 'S&P 500:', "{:.2f}".format(spy_sharpe))
print('Portfolio correlation with market:', "{:.2f}".format(portfolio_growth.corr(spy_growth)))
print('Median drawdown:', "{:.2f}%".format(avg_drawdown), 'S&P 500:', "{:.2f}%".format(spx_drawdown))
print('Beta:', "{:.2f}".format((portfolio_growth.cov(spy_growth)/(portfolio_deviation**2))))
print(daily_return['Daily return'].tail()*100)


# Save to Excel
writer = pd.ExcelWriter('csv/stocktable.xlsx', engine='xlsxwriter', date_format="YYYY-MM-DD")
prices.to_excel(writer, sheet_name='prices')
weights.to_excel(writer, sheet_name='weights')
prices_pctchange.to_excel(writer, sheet_name='pct_change')
daily_return.to_excel(writer, sheet_name='Total return')
stock_value.to_excel(writer, sheet_name='Total value')
stock_amount.to_excel(writer, sheet_name='Stock amount')
weight_compliant_stock_amount.to_excel(writer, sheet_name='Compliant amounts')
pd.DataFrame(prices_pctchange*weights, index=prices.index, columns=short_names).to_excel(writer, sheet_name='Return contributions')
writer.save()

plt.subplot(2, 1, 1)
plt.plot(daily_return.iloc[:, 0])
plt.plot(daily_return.iloc[:, 1])
#plt.yscale('log')
plt.legend(['Portfolio', 'S&P 500'])
plt.subplot(2, 3, 4)
plt.plot(daily_return['Drawdown'])
plt.plot(daily_return['SPX Drawdown'], alpha = 0.5)
plt.xlabel('Drawdown')
plt.subplot(2, 3, 5)
plt.xlabel('Return distribution')
n = plt.hist(daily_return['Daily return'], bins=np.arange(-0.04, 0.04, 0.001))
plt.hist(spy_growth, bins=np.arange(-0.04, 0.04, 0.001), alpha = 0.5, color='r')
hist_height = max(n[0])
mu = daily_return['Daily return'].mean()
sigma = portfolio_deviation
x = np.linspace(-0.04, 0.04, 100)
scaling = (hist_height*0.8)/stats.norm.pdf(mu, mu, sigma)
plt.plot(x, stats.norm.pdf(x, mu, sigma)*scaling)
plt.subplot(2, 3, 6)
plt.plot(weights*100)
plt.xlabel("Weights")
plt.figtext(0.03, 0.95, s=short_names, fontsize = 8)
plt.show()
