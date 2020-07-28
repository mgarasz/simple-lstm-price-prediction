import numpy as np
import pandas as pd
from sklearn import preprocessing
from collections import deque
import random


def classify(current, future, THRESHOLD=1.001):
    if float(future) > float(current)*THRESHOLD:  # if the future price is SUFFICIENTLY higher than the current, that's a buy, or a 1
        return 1
    elif float(future)*THRESHOLD < float(current):  # if the future price is SUFFICIENTLY lower than the current, that's a buy, or a 1
        return 2
    else:  # otherwise... it's a 0!
        return 0

""" INDICATORS """
def moving_average(group, n=7):
    sma = group.rolling(window=n).mean()
    return sma


def moving_average_convergence(group, nslow=26, nfast=12):
    emaslow = pd.ewma(group, span=nslow, min_periods=1)
    emafast = pd.ewma(group, span=nfast, min_periods=1)
    result = pd.DataFrame({'MACD': emafast-emaslow, 'emaSlw': emaslow, 'emaFst': emafast})
    return result



def Bolinger_Bands(stock_price, window_size, num_of_std):
    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std  = stock_price.rolling(window=window_size).std()
    upper_band = rolling_mean + (rolling_std*num_of_std)
    lower_band = rolling_mean - (rolling_std*num_of_std)
    return rolling_std, upper_band, lower_band

    
def preprocess_df(df, SEQ_LEN=30):
    df = df.drop("future", 1)  # don't need this anymore.
    ignore = ['LTC-USD_std', 'LTC-USD_VPT', "target"]

    for col in df.columns:  # go through all of the columns
        if col not in ignore:  # normalize all ... except for the target itself!
            df[col] = df[col].pct_change()  # pct change "normalizes" the different currencies (each crypto coin has vastly diff values, we're really more interested in the other coin's movements)
            df.dropna(inplace=True)  # remove the nas created by pct_change
            df[col] = preprocessing.scale(df[col].values)
            # scale between 0 and 1.
        elif col == "LTC-USD_std":
            df.dropna(inplace=True)  # remove the nas created by pct_change
            df[col] = preprocessing.scale(df[col].values) 

    df.dropna(inplace=True)  # cleanup again... jic.

    sequential_data = []  # this is a list that will CONTAIN the sequences
    prev_days = deque(maxlen=SEQ_LEN)  # These will be our actual sequences. They are made with deque, which keeps the maximum length by popping out older values as new ones come in

    for i in df.values:  # iterate over the values
        prev_days.append([n for n in i[:-1]])  # store all but the target
        if len(prev_days) == SEQ_LEN:  # make sure we have 60 sequences!
            sequential_data.append([np.array(prev_days), i[-1]])  # append those bad boys!


    random.shuffle(sequential_data)  # shuffle for good measure.

    buys = []  # list that will store our buy sequences and targets
    sells = []  # list that will store our sell sequences and targets

    for seq, target in sequential_data:  # iterate over the sequential data
        if target == 0:  # if it's a "not buy"
            sells.append([seq, target])  # append to sells list
        elif target == 1:  # otherwise if the target is a 1...
            buys.append([seq, target])  # it's a buy!

    random.shuffle(buys)  # shuffle the buys
    random.shuffle(sells)  # shuffle the sells!

    lower = min(len(buys), len(sells))  # what's the shorter length?

    buys = buys[:lower]  # make sure both lists are only up to the shortest length.
    sells = sells[:lower]  # make sure both lists are only up to the shortest length.

    sequential_data = buys+sells  # add them together
    random.shuffle(sequential_data)  # another shuffle, so the model doesn't get confused with all 1 class then the other.

    X = []
    y = []

    for seq, target in sequential_data:  # going over our new sequential data
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets/labels (buys vs sell/notbuy)

    return np.array(X), y  # return X and y...and make X a numpy array!

