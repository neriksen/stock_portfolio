import pandas as pd
import yfinance as yf
import numpy as np
import datetime as dt
import sys
from collections import namedtuple


def download_data(yf_tickers, per, filter_col='Adj Close'):
    prices = pd.DataFrame(yf.download(yf_tickers, period=per)).loc[:, filter_col]
    return prices


def clean_prices(prices, yf_tickers):
    org_ticker_count = len(yf_tickers)
    rows = rows_cols(prices)
    if len(yf_tickers) >= 2:
        for col in prices.iteritems():
            no_empty = len(col[1][pd.isna(col[1])])
            ratio = (no_empty / rows[0]) * 100
            if ratio >= 30:
                yf_tickers.remove(col[0])
                print(col[0], 'was eliminated')
        prices = pd.DataFrame(prices[yf_tickers], copy=True)
        print(org_ticker_count - len(yf_tickers), 'out of', org_ticker_count, 'tickers where eliminated')
        no_stocks = len(prices.columns)
    else:
        no_stocks = 1

    prices = pad_data(prices)

    return yf_tickers, prices, no_stocks


def pad_data(data_frame):
    data_frame.fillna(method='pad', inplace=True)  # Filling empty values with previous value
    data_frame.fillna(method='bfill', inplace=True)  # Filling empty values with previous value
    data_frame.astype(float, copy=True, errors='ignore')
    return data_frame


def clean_columns(prices, yf_tickers):
    short_names = []
    if len(yf_tickers) >= 2:
        for colname in prices.columns:
            try:
                short_names.append(yf.Ticker(colname).info['shortName'])
            except:
                short_names.append(colname)
    else:
        short_names.append(yf.Ticker(prices[0]).info['shortName'])
    return short_names


def fill_weights(weights, prices):   # Intialize weights
    rows, columns = rows_cols(prices)
    try:
        weights = pd.DataFrame(data=np.full((rows, columns), fill_value=weights), index=prices.index,
                               columns=prices.columns)
    except ValueError:
        print('Data for selected tickers not found')
        print('Choose differently or pick shorter time span')
        #sys.exit()
    return weights


def stock_value(prices, stock_amount):
    rows, columns = rows_cols(prices)
    stock_value = pd.DataFrame(np.full((rows, columns), stock_amount * prices), index=prices.index, columns=prices.columns)
    stock_value['Total value'] = pd.DataFrame(stock_value[prices.columns]).sum(axis=1)
    return stock_value


def rows_cols(data_frame):
    rows = len(data_frame.index)
    columns = len(data_frame.columns)
    return rows, columns


def stock_amount(prices, weights, invest_amount):
    rows, columns = rows_cols(prices)
    stock_amount = pd.DataFrame(np.full((rows, columns),
                                        np.floor(np.array(weights.iloc[0, ]) * invest_amount / prices.iloc[0])),
                                index=prices.index, columns=prices.columns)
    return stock_amount


def actual_weights(prices, stock_value):
    rows, columns = rows_cols(prices)
    weights = pd.DataFrame((stock_value[prices.columns]) / np.full((rows, columns), stock_value.iloc[:, -1][:, np.newaxis]), index=prices.index, columns=prices.columns)
    return weights


def correct_weight_sum(weights, yf_tickers, cash_proxy):
    weight_sum = np.round(np.sum(weights), 2)
    try:
        if not weights:
            weights = [1/len(yf_tickers)]*len(yf_tickers)
        else:
            if weight_sum < 1:
                yf_tickers.append(cash_proxy)
                weights.append(np.round(1 - weight_sum, 2))
            elif weight_sum > 1:
                weights /= weight_sum
    except:
        pass
    return weights, yf_tickers


def rebalancing_dates(weights, index):
    check = 0
    try:
        # if dt.datetime.date(weights.index[index]).year > dt.datetime.date(weights.index[index-1]).year:
        #     check = 2
        rebal = dt.datetime.date(weights.index[index]).month in [1, 4, 7, 10]\
            and dt.datetime.date(weights.index[index]).weekday() == 2 \
            and dt.datetime.date(weights.index[index]).day <= 7
        if rebal:
            check = 1
        if dt.datetime.date(weights.index[index]).year > dt.datetime.date(weights.index[index-1]).year:
            check += 1
    except:
        pass
    return check


def weight_compliant_amount(original_weights, stock_value, prices):
    weight_compliant_stock_amount = pd.DataFrame(np.floor((original_weights * stock_value.iloc[:, -1][:, np.newaxis]) / prices))
    return weight_compliant_stock_amount


def calculate_fractional_year(returns):
    assert returns.index.dtype == 'datetime64[ns]'
    start_date = dt.datetime.date(returns.index[0])
    end_date = dt.datetime.date(returns.index[-1])
    difference = end_date - start_date
    difference_in_years = (difference.days + difference.seconds/86400)/365.2425
    return difference_in_years


def normalize_data_frame(data_frame):
    data_frame = pad_data((data_frame/data_frame.iloc[0])*100)
    return data_frame


def portfolio_characteristics(data_frame):
    latest_fund_value = data_frame.iloc[-1]
    portfolio_return = np.array(((latest_fund_value / 100) ** (1 / calculate_fractional_year(data_frame))) - 1)
    portfolio_growth = data_frame.pct_change()
    portfolio_deviation = portfolio_growth.std()
    portfolio_sharpe = portfolio_return / portfolio_deviation
    return portfolio_return, portfolio_growth, portfolio_deviation, portfolio_sharpe


def calculate_total_return(prices, prices_pctchange, weights, gearing, rebalance_fee = 0.01, expense_ratio = 0.008):
    rows = rows_cols(prices)
    daily_return = pd.DataFrame(np.full((rows[0], 1), 100), index=prices.index, columns=['Return'])
    for index, row in enumerate(weights.iterrows()):
        if index == 0:
            pass
        else:
            if rebalancing_dates(weights, index) > 0:
                if rebalancing_dates(weights, index) == 2:  # year change add expense ratio
                    growth_factor = 1 - rebalance_fee - expense_ratio + np.matmul(prices_pctchange.iloc[index, :], weights.iloc[index - 1, :])*gearing
                else:
                    growth_factor = 1 - rebalance_fee + np.matmul(prices_pctchange.iloc[index, :], weights.iloc[index - 1, :])*gearing

                daily_return.iloc[index, 0] = growth_factor * daily_return.iloc[index - 1, 0]
            else:
                growth_factor = 1 + np.matmul(prices_pctchange.iloc[index, :], weights.iloc[index - 1, :])*gearing
                daily_return.iloc[index, 0] = growth_factor * daily_return.iloc[index - 1, 0]
    return daily_return


def rebalance_stock_amounts(prices, weights, original_weights, invest_amount):
    rows = rows_cols(prices)
    amount = stock_amount(prices,  weights, invest_amount)
    value = stock_value(prices, amount)
    weight_compliant_stock_amount = weight_compliant_amount(original_weights, value, prices)

    for index, row in enumerate(amount.iterrows(), start=1):
        if index < rows[0]:
            if rebalancing_dates(amount, index) > 0:
                value = stock_value(prices, amount)
                weight_compliant_stock_amount = weight_compliant_amount(original_weights, value, prices)
                amount.iloc[index, :] = weight_compliant_stock_amount.iloc[index, :]
            else:
                amount.iloc[index, :] = amount.iloc[index - 1, :]
        else:
            pass

    return amount, weight_compliant_stock_amount


def calculate_drawdown(returns):
    rows = len(returns)
    returns = pad_data(returns)
    drawdown = pd.DataFrame(np.full((rows, 3), 100), index=returns.index, columns=['Returns', 'HighValue', 'Drawdown'])
    drawdown['Returns'] = returns
    drawdown['HighValue'] = drawdown.cummax()
    drawdown['Drawdown'] = (drawdown['Returns']/drawdown['HighValue'])*100

    return drawdown['Drawdown']


def save2Excel(sheet_names, dfs):
    writer = pd.ExcelWriter('csv/stocktable.xlsx', engine='xlsxwriter', date_format="YYYY-MM-DD")
    for df in zip(dfs, sheet_names):
        df[0].to_excel(writer, sheet_name=df[1])
    writer.save()

