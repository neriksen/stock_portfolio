import pandas as pd
import random
import datetime as dt
import yfinance as yf
import numpy as np
from dateutil.relativedelta import relativedelta



def get_tickers(filelocation, header, nrows, scramble = True, ownlist = []):
    if len(ownlist) != 0:
        return ownlist
    else:
        tickers = pd.read_csv(filelocation, header=header, squeeze=True, nrows=nrows, index_col=None)
        if scramble == True:
            tickers = sorted(tickers, key=lambda x: random.random())  # Scrambling ticker names
        return tickers


def hasNumbers(inputString):
    ticker_errors = 0
    inputString = inputString.lower()
    if any(char.isdigit() for char in inputString):
       ticker_errors += 1
    if 'fund' in inputString:
       ticker_errors += 1
    if 'trust' in inputString:
        ticker_errors += 1
    if 'income' in inputString:
        ticker_errors += 1
    if ticker_errors > 0:
        return False
    else:
        return True


def retrieveprices(tickers, samplestart, enable_ticker_check, market_cap_limit):
    original_data: object = yf.download(list(tickers), samplestart, dt.date.today(), threads=True)

    data = original_data['Adj Close'].copy()

    # Retrieving data from YF
    tickers = list(original_data['Adj Close'].columns)
    good_tickers = []
    short_names = []
    for index, x in enumerate(tickers):
        company_info: object = yf.Ticker(x).info
        if enable_ticker_check == True:
            if 'shortName' in company_info and 'marketCap' in company_info and 'quoteType' in company_info:
                first_val = original_data['Adj Close'][x][0]
                unique_no = pd.DataFrame(original_data['Adj Close'][x]).nunique(0, dropna=True)
                short_name = company_info['shortName']
                if not np.isnan(first_val) \
                        and hasNumbers(short_name) \
                        and int(unique_no) > 1 \
                        and company_info['marketCap'] > market_cap_limit \
                        and company_info['quoteType'] == 'EQUITY':
                    good_tickers.append(x)
                    short_names.append(short_name)
                    print(short_name)
                else:
                    print('eliminated ' + str(short_name))
        else:
            good_tickers = tickers
            short_names = tickers
    datamod = data[good_tickers].copy()
    datamod.columns = short_names

    print('SUCCES - GENERATING PORTFOLIO')

    datamod.fillna(method='pad', inplace=True)  # Filling empty values with previous value
    datamod.index = pd.to_datetime(datamod.index)

    return datamod


def create_rebalancing_dates(stock_prices):
    global rebalancingdates
    rebalancingdates = []
    temp1 = 0
    total_days = 0

    # Creating rebalancing dates
    for index, row in stock_prices.iterrows():
        current_date = dt.datetime.strptime(str(index), "%Y-%m-%d %H:%M:%S")
        temp2 = dt.datetime.strptime(str(index), "%Y-%m-%d %H:%M:%S").month
        total_days = total_days + 1
        if not temp1 == temp2:
            rebalancingdates.append([(str(current_date), total_days)])
            temp1 = temp2

    rebalancingdates = np.array(rebalancingdates, copy=True).reshape([len(rebalancingdates), 2])
    rebalancingdates = pd.DataFrame(rebalancingdates, copy=True, columns=['Rebalancing Dates', 'Days Elapsed'])
    return rebalancingdates


def create_large_weights(stock_prices, rebalancing_dates, index_for_df, columnnames_for_df, equal_weight,
                         fund_type, inforange, pct_limit):
    weighttotal = pd.DataFrame(index=index_for_df, columns=columnnames_for_df, dtype=object)
    weighttotal.fillna(0, inplace=True)

    # Loop for weights. Calculate keyfigures on new subset
    if equal_weight == False:
        for row in range(0, np.size(rebalancing_dates, 0) - inforange):
            startwindow = rebalancing_dates.iloc[row]['Days Elapsed']
            endwindow = rebalancing_dates.iloc[row + inforange]['Days Elapsed']
            window_df = stock_prices.iloc[int(startwindow):int(endwindow)]
            cov = window_df.cov()

            # Calculating expected returns on window range
            if fund_type == 'minvar':
                ones = np.ones(shape=(len(cov.index), 1))
            else:
                ones = np.array(((window_df.iloc[-1] / window_df.iloc[0]) ** (1 / (inforange / 12))) - 1).reshape(
                    (len(cov.index), 1))  # Efficient portfolio

            zvector = np.linalg.inv(np.array(cov))
            zvector = zvector.dot(ones)
            normzvector = np.transpose(zvector / np.sum(zvector))

            for x in enumerate(np.nditer(normzvector)):  # Added to account for pct limits (see settings)
                if abs(x[1]) < pct_limit:
                    normzvector[0][x[0]] = 0

            linear_weight_scaling = 1 / np.sum(normzvector)  # Added to ensure weights sum to 1
            normzvector = np.multiply(normzvector, linear_weight_scaling)

            if fund_type != 'minvar':
                log_constant = (np.min(normzvector) * (-1)) + 0.01
                normzvector = ((normzvector + log_constant) ** (1 / 4)
                                ) - log_constant  # Added to control crazyness of efficient portfolios
                linear_weight_scaling = 1 / np.sum(normzvector)  # Added to ensure weights sum to 1
                normzvector = np.multiply(normzvector, linear_weight_scaling)
            weighttotal.iloc[row + inforange] = normzvector[0]

    else:

        weighttotal = pd.DataFrame(np.full((len(rebalancingdates), len(columnnames_for_df)), 1 / len(columnnames_for_df)),
                                   index=rebalancingdates['Rebalancing Dates'], columns=columnnames_for_df)

    weighttotal.index = pd.to_datetime(weighttotal.index)  # Converting to datetime

    return weighttotal