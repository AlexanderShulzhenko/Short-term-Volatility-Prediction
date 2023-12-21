import pandas as pd
import numpy as np
import os
import datetime as dt
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgbm
import scipy
import scipy.stats as sps
import tensorflow as tf
import gpflow
import pickle
import shap
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, auc, r2_score
from sklearn.feature_selection import RFE, mutual_info_regression
from skfeature.function.similarity_based import fisher_score
from tqdm import tqdm, trange
from copy import deepcopy
from typing import Dict, Tuple

from utils import line_fit


def feature_engineering_candlestick(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Generates features for the Candlestick features module.
    
    Inputs:
    -------
        - df (pd.DataFrame): data frame with candlestick data (o/h/l/c) and target variable merged.
    Outputs:
    -------
        - features (pd.DataFrame): data frame with generated features.
    '''
    
    # Date/Time
    df['week_day'] = df['open_time'].dt.dayofweek
    df['hour'] = df['open_time'].dt.hour

    # Price change
    df['returns_1'] = np.log(df['close'].pct_change(1) + 1)
    df['returns_4'] = np.log(df['close'].pct_change(4) + 1)

    # Garman-Klass Historical Volatility
    c = 2 * np.log(2) - 1
    df['GK_HV_inner'] = 0.5 * (np.log(df['high'] / df['low']))**2 - c * (np.log(df['close'] / df['open']))**2
    df['Garman_Klass_HV_4'] = np.sqrt( df['GK_HV_inner'].rolling(4).mean() )
    
    # Volume
    df['volume_1'] = df['volume'].pct_change(1)
    df['volume_4'] = df['volume'].pct_change(4)
    
    # Candlestick
    df['candle_size'] = abs(df['close'] - df['open'])
    df['mean_candle_size_12'] = df['close'].rolling(12, closed='left').mean()
    df['candle_size_mean_candle_size_12_ratio'] = df['candle_size'] / df['mean_candle_size_12']
    
    candlestick_data = [
        'close_time',
        'week_day',
        'hour',
        'returns_1',
        'returns_4',
        'Garman_Klass_HV_4',
        'volume_1',
        'volume_4',
        'candle_size_mean_candle_size_12_ratio',
        'num_trades',
        'target'
    ]
    
    features = df[candlestick_data]
    
    return features


def feature_engineering_indicators(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Generates features for the Tech. indicators features module.
    
     Inputs:
    -------
        - df (pd.DataFrame): data frame with candlestick data (o/h/l/c) and target variable merged.
    Outputs:
    -------
        - features (pd.DataFrame): data frame with generated features.
    '''
    
    # MA
    df['MA_5'] = df['close'].rolling(5).mean()
    df['MA_40'] = df['close'].rolling(40).mean()
    df['MA_100'] = df['close'].rolling(100).mean()
    
    df['MA_5_under_MA_40'] = np.where(df['MA_5'] < df['MA_40'], 1, 0)
    df['MA_5_under_MA_100'] = np.where(df['MA_5'] < df['MA_100'], 1, 0)
    
    df['MA_5_MA_100_ratio'] = df['MA_5'] / df['MA_40']
    
    # Bollinger Bands
    df['std_10'] = df['close'].rolling(10).std()
    df['bbh'] = df['MA_5'] + 2 * df['std_10']
    df['bbl'] = df['MA_5'] - 2 * df['std_10']
    
    df['bbw%'] = (df['bbh'] - df['bbl']) / df['MA_5']
    
    # RSI
    df['diff'] = df['close'] - df['close'].shift(1)
    df['gains'] = np.where(df['diff'] >= 0, df['diff'], 0)
    df['losses'] = np.where(df['diff'] < 0, -df['diff'], 0)
    df['avg_gains'] = df['gains'].rolling(20).mean()
    df['avg_losses'] = df['losses'].rolling(20).mean()
    df['RS'] = df['avg_gains'] / df['avg_losses']
    df['RSI'] = 100 - 100 / (1 + df['RS'])
    df['smoothed_RSI'] = df['RSI'].rolling(5).mean()
    
    # Stohastic
    df['L_14'] = df['low'].rolling(14).min()
    df['H_14'] = df['high'].rolling(14).max()
    df['stohastic'] = (df['close'] - df['L_14']) / (df['H_14'] - df['L_14'])
    
    # Chaikin Money Flow
    df['mult'] = ((df['close']  -  df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']) 
    df['mf_vol'] = df['mult'] * df['volume']

    df['CMF'] = df['mf_vol'].rolling(20).sum() / df['volume'].rolling(20).sum()
    df['CMF'].fillna(0, inplace=True)
    
    # Rolling Linear Regression
    coefs, r2s = line_fit(df, 'close', 10)
    df['coefs'] = coefs
    df['r2s'] = r2s
    
    indicators = [
        'close_time',
        'MA_5_under_MA_40',
        'MA_5_under_MA_100',
        'bbw%',
        'smoothed_RSI',
        'stohastic',
        'CMF',
        'coefs',
        'r2s',
        'target'
    ]
    return df[indicators]


def feature_engineering_exchange(master_list: pd.DataFrame, agg_trades_collection: list) -> pd.DataFrame:
    '''
    Generates features for the Exchange features module.
    
    Inputs:
    -------
        - master_list (pd.DataFrame): master list;
        - agg_trades_collection (list): list of data frames for each time stamp in master list,
                                        each data frame containing all exchanges 15 minutes before 
                                        the selected time stamp in master list.
    Outputs:
    -------
        - features (pd.DataFrame): data frame with generated features.
    '''
    
    skews = []
    kurtosises = []
    num_buy_trades = []
    num_sell_trades = []
    buy_amounts = []
    sell_amounts = []
    percentile_90 = []
    ba_spread = []
    variations = []
    norm_dists = []
    cauchy_dists = []
    
    # Iterate over all exchange data frames
    for d in tqdm(agg_trades_collection):
        d['returns'] = d['Price'].pct_change()
        d['dolAmount'] = d['Price'] * d['Quantity']

        # Stats
        skews.append(d['returns'].skew())
        kurtosises.append(d['returns'].kurtosis())
        percentile_90.append(d['dolAmount'].quantile(0.9))

        # Buy / Sell split
        buy_ex = d[d['Type']=='BUY']
        sell_ex = d[d['Type']=='SELL']
        num_buy_trades.append(len(buy_ex))
        num_sell_trades.append(len(sell_ex))

        # Bid-Ask spread
        # Get last BUY (price) for each SELL
        bid_ask_spread = pd.merge_asof(d[d['Type']=='SELL'].reset_index(), d[d['Type']=='BUY'].reset_index(), on='index')
        bid_ask_spread['spread'] = bid_ask_spread['Price_y'] - bid_ask_spread['Price_x']
        ba_spread.append(bid_ask_spread['spread'].max())
        variations.append( np.sum(abs(bid_ask_spread['spread'] - bid_ask_spread['spread'].shift(1))) )

        # Kolmogorov-Smirnov test
        prices = d['Price']
        returns = np.log(np.array(prices)[1:]/np.array(prices)[:-1])
        returns = np.sort(returns)
        #construct empirical distribution function
        edf = np.arange(1, len(returns)+1)/len(returns)
        #construct normal distribution function
        mean = np.average(returns)
        std = np.std(returns)
        normal_cdf = sps.norm.cdf(returns, mean, std)
        norm_dists.append(max(abs(edf - normal_cdf)))

        #construct cauchy distribution function
        median = np.quantile(returns, 0.5)
        iq_range = (np.quantile(returns, 0.75) - np.quantile(returns, 0.25))/2
        cauchy_cdf = 1/np.pi*np.arctan((returns - median)/iq_range) + 1/2
        cauchy_dists.append(max(abs(edf - cauchy_cdf)))

    features = deepcopy(master_list)

    features['skews'] = skews
    features['kurtosises'] = kurtosises
    features['percentile_90'] = percentile_90

    features['num_buy_trades'] = num_buy_trades
    features['num_sell_trades'] = num_sell_trades

    features['ba_spread'] = ba_spread
    features['variations'] = variations

    features['norm_dists'] = norm_dists
    features['cauchy_dists'] = cauchy_dists
    
    exchange_data = [
        'close_time',
        'target',
        'skews',
        'kurtosises',
        'percentile_90',
        'num_buy_trades',
        'num_sell_trades',
        'ba_spread',
        'variations',
        'norm_dists',
        'cauchy_dists'
    ]

    features = features[exchange_data]
    return features


def feature_engineering_stochastic(df: pd.DataFrame, inv_cov_dct: Dict) -> pd.DataFrame:
    '''
    Generates features for the Stochastic process features module.
    
    Inputs:
    -------
        - df (pd.DataFrame): data frame with candlestick data (o/h/l/c) and target variable merged.
        - inv_cov_dct (dict): dictionary with kernel types as keys and dictionaries with inversed covariance
                              matrices for each lengthscale as values
    Outputs:
    -------
        - features (pd.DataFrame): data frame with generated features.
    '''
    
    test_res = []

    idx = df[df['target'].notna()].index
    for i in tqdm(idx):
        # get data for last 24 hours and scale it
        raw_path = df['close'][i - 95: i + 1].values
        raw_path_mean = raw_path.mean()
        raw_path_std = raw_path.std()
        price_path = (raw_path - raw_path_mean) / raw_path_std

        # Check for Matern and Ornstein-Uhlenbeck kernels
        res = [df['close_time'][i]]
        for k_type in inv_cov_dct.keys():
            # find p_value of Kolmogorov-Smirnov test for each lengthscale
            p_vals = []
            for ls in range(5, 110, 5):
                standardized_path = inv_cov_dct[k_type][ls] @ price_path
                sorted_path = np.sort(standardized_path.ravel())
                normal_cdf = scipy.stats.norm.cdf(sorted_path, 0, 1)
                edf = np.arange(1, len(sorted_path)+1) / len(sorted_path)

                p_value = np.exp(-max(abs(edf - normal_cdf))**2*len(sorted_path))
                p_vals.append(p_value)

            # Shapiro-Wilk second test on found lengthscale to verify
            lengthscale = (np.argmax(p_vals) + 1) * 5
            standardized_path = inv_cov_dct[k_type][lengthscale] @ price_path
            p_value_shapiro = scipy.stats.shapiro(standardized_path).pvalue

            res += [lengthscale, max(p_vals), p_value_shapiro]

        test_res.append(res)
        
    features = pd.DataFrame(test_res)
    features.columns = ['close_time', 
                        'lengthscale_Matern32', 
                        'p_value_KS_Matern32', 
                        'p_value_SW_Matern32',
                        'lengthscale_OU', 
                        'p_value_KS_OU', 
                        'p_value_SW_OU']

    features = df.merge(features, how='left', on='close_time')
    
    stoch_process = ['close_time',
                     'target',
                     'lengthscale_Matern32', 
                     'p_value_KS_Matern32', 
                     'p_value_SW_Matern32',
                     'lengthscale_OU', 
                     'p_value_KS_OU', 
                     'p_value_SW_OU']
    
    return features[stoch_process]