import yfinance as yf
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import random
import yfinancemodule.portfolio as port
from dateutil.relativedelta import relativedelta
import openpyxl
import matplotlib.pyplot as ply
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# -----------------Settings-------------------
samplestart = dt.date(2014, 5, 11)
inforange = 1    # number of months
rebalancing_cost_bps = 5
fund_value = 10_000_000
original_fund_value = fund_value
dollar_limit = 30_000
pct_limit = dollar_limit/fund_value
#pct_limit = 0.03     # Limit for pct weight in portfolio
fund_type = 'minvar'
market_cap_limit = 0
equal_weight = False
enable_ticker_check = False
# --------------------------------------------

# Retrieving tickers
tickers = port.get_tickers("tickers/tickers.csv", header=1, nrows=500, scramble=True,
                           ownlist=['EXSD.DE'])

# Retrieving stock prices
datamod = port.retrieveprices(tickers[0:10], samplestart, enable_ticker_check, market_cap_limit)

global actualcolumns
actualcolumns = datamod.columns

# Creating rebalancing dates
rebalancingdates = port.create_rebalancing_dates(datamod)

# Calculating returns
diff = datamod.pct_change(axis=0)

# Creating large matrix of weights
weighttotal = port.create_large_weights(datamod, rebalancingdates, rebalancingdates['Rebalancing Dates'], actualcolumns, equal_weight, fund_type, inforange, pct_limit)

# Loop to account for trade costs
if rebalancing_cost_bps != 0:
    for row in rebalancingdates.iterrows():
        diff.loc[row[1][0]] = diff.loc[row[1][0]]-(rebalancing_cost_bps/10000)  #/10000 because in bps

# Loop for calculating returns of portfolio strategy
# Creating large index of rebalancing weights
row_counter = 0
row_rebalancing = inforange

# Removing datamod index
datamod.reset_index(inplace=True)

large_weights = np.zeros((len(datamod.index), len(actualcolumns)))
stocks_amount = np.zeros((len(datamod.index), len(actualcolumns)))

end_loop = (datamod.loc[datamod['Date'] == str(rebalancingdates.iloc[-1, 0])]).index[0]+1

for index in range(0, end_loop):    # Creating large array of weights (weights for every busiday
    if str(datamod.iloc[index, 0]) == str(rebalancingdates.iloc[row_rebalancing, 0]):
        large_weights[row_counter:index, :] = weighttotal.iloc[row_rebalancing-1]
        stocks_amount[row_counter:index, :] = np.floor((weighttotal.iloc[row_rebalancing-1]*fund_value)/datamod.iloc[row_counter, 1:])
        cash = fund_value - (sum(pd.DataFrame(stocks_amount[row_counter, :]*datamod.iloc[row_counter, 1:]).sum(axis=1)))
        fund_value = (sum(pd.DataFrame(stocks_amount[index-1, :]*datamod.iloc[index-1, 1:]).sum(axis=1))) + cash
        row_rebalancing += 1
        row_counter = index

#Adding latest weights and stocks_amount
large_weights[row_counter:, :] = weighttotal.iloc[-1]
large_weights = pd.DataFrame(large_weights, copy=True, columns=actualcolumns)
large_weights = pd.concat([datamod['Date'], large_weights], 1)
pd.DataFrame(large_weights).set_index('Date', drop=True, inplace=True)

stocks_amount[row_counter:, :] = np.floor((weighttotal.iloc[-1]*fund_value)/datamod.iloc[-1, 1:])
stocks_amount = pd.DataFrame(stocks_amount, copy=True, columns=actualcolumns)
stocks_amount = pd.concat([datamod['Date'], stocks_amount], 1)
pd.DataFrame(stocks_amount).set_index('Date', drop=True, inplace=True)

#Calculating fund performance
fund_value = pd.DataFrame(index=datamod['Date'], columns=['Portfolio value'])

fund_start = 0
errorcount = 0
for row in enumerate(stocks_amount.iterrows()):
    amounts = np.array(pd.DataFrame(stocks_amount).iloc[row[0], :])
    if amounts.sum(axis=0) > 0:
        prices = np.array(pd.DataFrame(datamod).iloc[row[0], :])
        temp = np.matmul(amounts, prices[1:])
        if pd.isnull(temp):
            temp = np.nan
        if temp != 1 and fund_start == 0:
            fund_start = row[0]
            first_trade_date = row[0]
        fund_value.iloc[row[0], 0] = temp

#Calculating daily return of fund
large_weights = large_weights.iloc[first_trade_date:]     # Cropping size
stocks_amount = stocks_amount.iloc[first_trade_date:]     # Cropping size
fund_value = fund_value.iloc[first_trade_date-1:]           # Cropping size
weighttotal = weighttotal.iloc[inforange:]         # Cropping size
fund_value.iloc[0, 0] = original_fund_value                 # Initial value of portfolio
fund_value_normalized = fund_value/fund_value.iloc[0]       # Normalizing to same start date
fund_return_diff = pd.DataFrame(fund_value).pct_change()
fund_return_diff.columns = ['Daily returns']
fund_value_normalized.columns = ['Portfolio normalized']
fund_value = pd.DataFrame(fund_value).merge(fund_return_diff, on='Date')
fund_value = pd.DataFrame(fund_value).merge(fund_value_normalized, on='Date')

#Calculating portfolio risk
portfolio_stats = pd.DataFrame(index=['Portfolio', 'SPX same period'], columns=['Std. deviation', 'y/y return', 'Sharpe Ratio'])
stdevation = np.array(pd.DataFrame(fund_return_diff).std())

# y/y return
start_date = dt.datetime.date(fund_value.index[0])
end_date = dt.datetime.date(fund_value.index[-1])
year_part = relativedelta(end_date, start_date).years
month_part = relativedelta(end_date, start_date).months/12
day_part = relativedelta(end_date, start_date).days/364
difference_in_years = year_part + month_part + day_part
latest_fund_value = fund_value.iloc[-1,2]
yreturn = np.array((latest_fund_value ** (1/difference_in_years))-1)

# Pasting to portfolio stat
portfolio_stats.iloc[0, 0] = "{:.2f}%".format(float(stdevation*100))
portfolio_stats.iloc[0, 1] = "{:.2f}%".format(float(yreturn*100))
portfolio_stats.iloc[0, 2] = "{:.2f}".format(float(yreturn)/float(stdevation))   # Sharpe ratio

# S&P 500 for comparison
spy = yf.download('SPY', start_date, dt.date.today())
spy_close = pd.DataFrame(spy['Adj Close']).copy()
spy_close = spy_close/spy_close.iloc[0]     # Normalizing to same start date
latest_spy_value = spy_close.iloc[-1]

#Calculating S&P 500 stats for same period
spy_diff = spy_close.pct_change()
spyreturn = np.array((latest_spy_value ** (1/difference_in_years))-1)
spydeviation = np.array(pd.DataFrame(spy_diff).std())

portfolio_stats.iloc[1, 0] = "{:.2f}%".format(float(spydeviation*100))
portfolio_stats.iloc[1, 1] = "{:.2f}%".format(float(spyreturn*100))
portfolio_stats.iloc[1, 2] = "{:.2f}".format(float(spyreturn)/float(spydeviation))  # Sharpe ratio


print(portfolio_stats.head())
print('no of companies in portfolio:')
print(len(actualcolumns))
print('Average weighting of stocks')
print(weighttotal.mean())


# Save to Excel
writer = pd.ExcelWriter('csv/minvarportfolio.xlsx', engine='xlsxwriter', date_format="YYYY-MM-DD")
datamod.to_excel(writer, sheet_name='prices')
diff.to_excel(writer, sheet_name='returns')
large_weights.to_excel(writer, sheet_name='largeweightsheet')
stocks_amount.to_excel(writer, sheet_name='stocks_amount')
weighttotal.to_excel(writer, sheet_name='minvarweights')
fund_value.to_excel(writer, sheet_name='fund_return')
portfolio_stats.to_excel(writer, sheet_name='portfolio stats')
writer.save()

# Creates two subplots and unpacks the output array immediately
years = mdates.YearLocator()

f, axs = ply.subplots(3, 1)
axs[0].plot(fund_value_normalized['Portfolio normalized'])
axs[0].plot(spy_close[:-1])
axs[0].set_title('Portfolio performance')
axs[1].plot(weighttotal*100)
axs[1].yaxis.set_major_formatter(mtick.PercentFormatter())
axs[1].annotate('Weights',
            xy=(0, 1), xycoords='axes fraction',
            xytext=(10, -10), textcoords='offset pixels',
            horizontalalignment='left',
            verticalalignment='top')
if len(actualcolumns) <= 6: axs[1].legend(actualcolumns)
# axs[2].hist(fund_return_diff['Daily returns'].iloc[1:], bins=50)
axs[2].hist(np.clip(fund_return_diff['Daily returns'].iloc[1:], -0.1, 0.1), bins=50)
ply.show()