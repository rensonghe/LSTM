from IPython.core.display import display, HTML

import pandas as pd
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import os
import gc

from joblib import Parallel, delayed

from sklearn import preprocessing, model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import seaborn as sns
import numpy.matlib

#%%
data_2203 = pd.read_csv('ni2202.csv')

col =['datetime','timestamp','last_price','highest','lowest','volume','amount','interest',
      'bid_price1','bid_size1','ask_price1','ask_size1','bid_price2','bid_size2','ask_price2','ask_size2',
      'bid_price3','bid_size3','ask_price3','ask_size3','bid_price4','bid_size4','ask_price4','ask_size4',
      'bid_price5','bid_size5','ask_price5','ask_size5']
data_2203.columns = col

data_2203 = data_2203.drop(['highest','lowest','interest'],axis=1)
# data_2103['last_price_target'] = (data_2103['ask_price1']+data_2103['bid_price1'])/2

#%%
data = data_2203[(data_2203.datetime>='2022-01-01 09:00:00')& (data_2203.datetime<='2022-02-01 08:59:59')]
#%%
data = data.dropna(axis=0,how='any')
#%%
data = data.iloc[1:,:]
#%%
# from sklearn.preprocessing import FunctionTransformer
#
#
# def sin_transformer(period):
#     return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))
#
#
# def cos_transformer(period):
#     return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))
#
# data['sin'] = sin_transformer(60).fit_transform(data)['stime']
#%%
data1 = data.iloc[:,1:].astype(float)
#%%
data1['datetime'] = data['datetime']
#%%
data1 = data1.reset_index(drop=True).reset_index()
# train = data_2103[(data_2103.datetime>='2021-01-04 0:00:00')& (data_2103.datetime<='2021-01-08 23:59:59')]
# test = data_2103[(data_2103.datetime>='2021-01-11 0:00:00')& (data_2103.datetime<='2021-01-11 23:59:59')]


#%%
def calc_wap1(df):
    wap = (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']) / (df['bid_size1'] + df['ask_size1'])
    return wap

# Function to calculate second WAP
def calc_wap2(df):
    wap = (df['bid_price2'] * df['ask_size2'] + df['ask_price2'] * df['bid_size2']) / (df['bid_size2'] + df['ask_size2'])
    return wap

def calc_wap3(df):
    wap = (df['bid_price1'] * df['bid_size1'] + df['ask_price1'] * df['ask_size1']) / (df['bid_size1'] + df['ask_size1'])
    return wap

def calc_wap4(df):
    wap = (df['bid_price2'] * df['bid_size2'] + df['ask_price2'] * df['ask_size2']) / (df['bid_size2'] + df['ask_size2'])
    return wap

def calc_wap5(df):
    rolling = 15
    df['size'] = df['volume'] - df['volume'].shift(1)
    df['pv'] = df['last_price'] * df['size']
    last_price_wap = df['pv'].rolling(rolling).sum() / df['size'].rolling(rolling).sum()
    return last_price_wap

#%%
from scipy.special import gamma

# Calculate the realized volatility
def realized_volatility(series):
    return np.sqrt(np.sum(series ** 2))

def realized_quarticity(series):
    return (np.sum(series**4)*series.shape[0]/3)*100000

def reciprocal_transformation(series):
    return np.sqrt(np.abs(1/series))*100000

def square_root_translation(series):
    return series**(1/2)

# def realized_quadpower_quarticity(series):
#     series = abs(series.rolling(window=4).apply(np.product, raw=True))
#     return ((np.sum(series) * series.shape[0] * (np.pi**2))/4)*1000000

# def realized_tripower_quarticity(series):
#     series = series ** (4/3)
#     series = abs(series).rolling(window=3).apply(np.prod, raw=True)
#     return (series.shape[0]*0.25*((gamma(1/2)**3)/(gamma(7/6)**3))*np.sum(series))*1000000


#%%
# data['wap1'] = calc_wap1(data)
#%%
# data['1'] = np.log(data['wap1'].shift(1)/data['wap1'].shift(2))*100
#%%
# df_feature = data.groupby('datetime').agg({'1' : [realized_quadpower_quarticity,realized_tripower_quarticity]}).reset_index()
#%%

# df_feature.columns = ['_'.join(col) for col in df_feature.columns]
# train['log_return1'] = train.groupby(['datetime'])['wap1'].apply(log_return)
#%%
def book_preprocessor(data):

    df = data

    rolling = 15
    df['volume_size'] = df['volume'] - df['volume'].shift(1)

    # Calculate Wap
    df['wap1'] = calc_wap1(df)
    df['wap2'] = calc_wap2(df)
    df['wap3'] = calc_wap3(df)
    df['wap4'] = calc_wap4(df)
    df['wap5'] = calc_wap5(df)

    df['wap1_shift2'] = df['wap1'].shift(1) - df['wap1'].shift(2)
    df['wap1_shift3'] = df['wap1'].shift(1) - df['wap1'].shift(3)
    df['wap1_shift4'] = df['wap1'].shift(1) - df['wap1'].shift(4)
    df['wap1_shift5'] = df['wap1'].shift(1) - df['wap1'].shift(5)

    df['wap2_shift2'] = df['wap2'].shift(1) - df['wap2'].shift(2)
    df['wap2_shift3'] = df['wap2'].shift(1) - df['wap2'].shift(3)
    df['wap2_shift4'] = df['wap2'].shift(1) - df['wap2'].shift(4)
    df['wap2_shift5'] = df['wap2'].shift(1) - df['wap2'].shift(5)

    df['wap3_shift2'] = df['wap3'].shift(1) - df['wap3'].shift(2)
    df['wap3_shift3'] = df['wap3'].shift(1) - df['wap3'].shift(3)
    df['wap3_shift4'] = df['wap3'].shift(1) - df['wap3'].shift(4)
    df['wap3_shift5'] = df['wap3'].shift(1) - df['wap3'].shift(5)

    df['wap4_shift2'] = df['wap4'].shift(1) - df['wap4'].shift(2)
    df['wap4_shift3'] = df['wap4'].shift(1) - df['wap4'].shift(3)
    df['wap4_shift4'] = df['wap4'].shift(1) - df['wap4'].shift(4)
    df['wap4_shift5'] = df['wap4'].shift(1) - df['wap4'].shift(5)

    df['wap5_shift2'] = df['wap5'].shift(1) - df['wap5'].shift(2)
    df['wap5_shift3'] = df['wap5'].shift(1) - df['wap5'].shift(3)
    df['wap5_shift4'] = df['wap5'].shift(1) - df['wap5'].shift(4)
    df['wap5_shift5'] = df['wap5'].shift(1) - df['wap5'].shift(5)

    df['last_price_shift2'] = df['last_price'].shift(1) - df['last_price'].shift(2)
    df['last_price_shift3'] = df['last_price'].shift(1) - df['last_price'].shift(3)
    df['last_price_shift4'] = df['last_price'].shift(1) - df['last_price'].shift(4)
    df['last_price_shift5'] = df['last_price'].shift(1) - df['last_price'].shift(5)
    df['last_price_shift6'] = df['last_price'].shift(1) - df['last_price'].shift(6)
    df['last_price_shift7'] = df['last_price'].shift(1) - df['last_price'].shift(7)
    df['last_price_shift8'] = df['last_price'].shift(1) - df['last_price'].shift(8)
    df['last_price_shift9'] = df['last_price'].shift(1) - df['last_price'].shift(9)


    df['mid_price'] = np.where(df.volume_size > 0, (df.amount - df.amount.shift(1)) / df.volume_size, df.last_price)
    df['mid_price1'] = (df['ask_price1']+df['bid_price1'])/2
    df['mid_price2'] = (df['ask_price2']+df['bid_price2'])/2
    df['mid_price3'] = (df['ask_price3'] + df['bid_price3']) / 2

    df['press_buy1'] = ((df['mid_price1']/(df['mid_price1']-df['bid_price1']))/((df['mid_price1']/(df['mid_price1']-df['bid_price1']))+(df['mid_price2']/(df['mid_price2']-df['bid_price2']))+(df['mid_price3']/(df['mid_price3']-df['bid_price3']))))*(df['bid_size1']+df['bid_size2']+df['bid_size3'])
    df['press_sell1'] = ((df['mid_price1']/(df['ask_price1']-df['mid_price1']))/((df['mid_price1']/(df['ask_price1']-df['mid_price1']))+(df['mid_price2']/(df['ask_price2']-df['mid_price2']))+(df['mid_price3']/(df['ask_price3']-df['mid_price3']))))*(df['ask_size1']+df['ask_size2']+df['ask_size3'])

    df['order_BS1'] = np.log(df['press_buy1'])-np.log(df['press_sell1'])

    df['HR1'] = ((df['bid_price1']-df['bid_price1'].shift(1))-(df['ask_price1']-df['ask_price1'].shift(1)))/((df['bid_price1']-df['bid_price1'].shift(1))+(df['ask_price1']-df['ask_price1'].shift(1)))

    df['pre_vtA'] = np.where(df.ask_price1==df.ask_price1.shift(1),df.ask_size1-df.ask_size1.shift(1),0)
    df['vtA'] = np.where(df.ask_price1>df.ask_price1.shift(1),df.ask_size1,df.pre_vtA)
    df['pre_vtB'] = np.where(df.bid_price1==df.bid_price1.shift(1),df.bid_size1-df.bid_size1.shift(1),0)
    df['vtB'] = np.where(df.bid_price1>df.bid_price1.shift(1),df.bid_size1,df.pre_vtB)

    df['Oiab'] = df['vtB']-df['vtA']


    df['bid_size1_2_minus'] = df['bid_size1']-df['bid_size2']
    df['ask_size1_2_minus'] = df['ask_size1']-df['ask_size2']

    df['bid_ask_size1_minus'] = df['bid_size1']-df['ask_size1']
    df['bid_ask_size1_plus'] = df['bid_size1']+df['ask_size1']

    df['bid_ask_size2_minus'] = df['bid_size2']-df['ask_size2']
    df['bid_ask_size2_plus'] = df['bid_size2']+df['ask_size2']

    df['bid_size1_shift'] = df['bid_size1']-df['bid_size1'].shift()
    df['bid_size2_shift'] = df['bid_size2']-df['bid_size2'].shift()
    df['ask_size1_shift'] = df['ask_size1']-df['ask_size1'].shift()
    df['ask_size2_shift'] = df['ask_size2']-df['ask_size2'].shift()

    df['bid_ask_size1_spread'] = df['bid_ask_size1_minus']/df['bid_ask_size1_plus']
    df['bid_ask_size2_spread'] = df['bid_ask_size2_minus'] / df['bid_ask_size2_plus']
    df['bid_amount_abs1'] = abs((df['amount']-df['amount'].shift(1))/df['volume_size']-df['bid_price1'])-abs((df['amount']-df['amount'].shift(1))/df['volume_size']-df['ask_price1'])

    # Calculate log returns
    df['log_return1'] = np.log(df['wap1'].shift(1)/df['wap1'].shift(2))*100
    df['log_return2'] = np.log(df['wap2'].shift(1)/df['wap2'].shift(2))*100
    df['log_return3'] = np.log(df['wap3'].shift(1)/df['wap3'].shift(2))*100
    df['log_return4'] = np.log(df['wap4'].shift(1)/df['wap4'].shift(2))*100

    df['log_return_wap1_shift3'] = np.log(df['wap1'].shift(1)/df['wap1'].shift(3))*100
    df['log_return_wap1_shift4'] = np.log(df['wap1'].shift(1)/df['wap1'].shift(4))*100
    df['log_return_wap1_shift5'] = np.log(df['wap1'].shift(1) / df['wap1'].shift(5))*100
    df['log_return_wap1_shift6'] = np.log(df['wap1'].shift(1) / df['wap1'].shift(6))*100
    df['log_return_wap1_shift7'] = np.log(df['wap1'].shift(1) / df['wap1'].shift(7))*100
    df['log_return_wap1_shift8'] = np.log(df['wap1'].shift(1) / df['wap1'].shift(8))*100
    df['log_return_wap1_shift9'] = np.log(df['wap1'].shift(1) / df['wap1'].shift(9))*100


    df['log_return_last_price_shift2'] = np.log(df['last_price'].shift(1)/df['last_price'].shift(2))*100
    df['log_return_last_price_shift3'] = np.log(df['last_price'].shift(1)/df['last_price'].shift(3))*100
    df['log_return_last_price_shift4'] = np.log(df['last_price'].shift(1)/df['last_price'].shift(4))*100
    df['log_return_last_price_shift5'] = np.log(df['last_price'].shift(1)/df['last_price'].shift(5))*100
    df['log_return_last_price_shift6'] = np.log(df['last_price'].shift(1)/df['last_price'].shift(6))*100
    df['log_return_last_price_shift7'] = np.log(df['last_price'].shift(1)/df['last_price'].shift(7))*100
    df['log_return_last_price_shift8'] = np.log(df['last_price'].shift(1)/df['last_price'].shift(8))*100
    df['log_return_last_price_shift9'] = np.log(df['last_price'].shift(1)/df['last_price'].shift(9))*100



    # Calculate wap balance
    df['wap_balance1'] = abs(df['wap1'] - df['wap2'])



    # Calculate spread
    df['price_spread1'] = (df['ask_price1'] - df['bid_price1']) / ((df['ask_price1'] + df['bid_price1']) / 2)
    df['price_spread2'] = (df['ask_price2'] - df['bid_price2']) / ((df['ask_price2'] + df['bid_price2']) / 2)
    df['total_volume1'] = (df['ask_size1'] + df['ask_size2']) + (df['bid_size1'] + df['bid_size2'])
    df['volume_imbalance1'] = abs((df['ask_size1'] + df['ask_size2']) - (df['bid_size1'] + df['bid_size2']))
    df['size_weight'] = ((df['bid_size1']+df['ask_size1'])*0.7)-((df['bid_size2']+df['ask_size2'])*0.3)

    # Dict for aggregations
    create_feature_dict = {
        'wap1_shift2': [np.mean, reciprocal_transformation, square_root_translation],
        'wap1_shift3': [np.mean, reciprocal_transformation, square_root_translation],
        'wap1_shift4': [np.mean, reciprocal_transformation, square_root_translation],
        'wap1_shift5': [np.mean, reciprocal_transformation, square_root_translation],
        'wap2_shift2': [np.mean, reciprocal_transformation, square_root_translation],
        'wap2_shift3': [np.mean, reciprocal_transformation, square_root_translation],
        'wap2_shift4': [np.mean, reciprocal_transformation, square_root_translation],
        'wap2_shift5': [np.mean, reciprocal_transformation, square_root_translation],
        'wap3_shift2': [np.mean, reciprocal_transformation, square_root_translation],
        'wap3_shift3': [np.mean, reciprocal_transformation, square_root_translation],
        'wap3_shift4': [np.mean, reciprocal_transformation, square_root_translation],
        'wap3_shift5': [np.mean, reciprocal_transformation, square_root_translation],
        'wap4_shift2': [np.mean, reciprocal_transformation, square_root_translation],
        'wap4_shift3': [np.mean, reciprocal_transformation, square_root_translation],
        'wap4_shift4': [np.mean, reciprocal_transformation, square_root_translation],
        'wap4_shift5': [np.mean, reciprocal_transformation, square_root_translation],
        'wap5_shift2': [np.mean, reciprocal_transformation, square_root_translation],
        'wap5_shift3': [np.mean, reciprocal_transformation, square_root_translation],
        'wap5_shift4': [np.mean, reciprocal_transformation, square_root_translation],
        'wap5_shift5': [np.mean, reciprocal_transformation, square_root_translation],
        'last_price_shift2': [np.mean, reciprocal_transformation, square_root_translation],
        'last_price_shift3': [np.mean, reciprocal_transformation, square_root_translation],
        'last_price_shift4': [np.mean, reciprocal_transformation, square_root_translation],
        'last_price_shift5': [np.mean, reciprocal_transformation, square_root_translation],
        'last_price_shift6': [np.mean, reciprocal_transformation, square_root_translation],
        'last_price_shift7': [np.mean, reciprocal_transformation, square_root_translation],
        'last_price_shift8': [np.mean, reciprocal_transformation, square_root_translation],
        'last_price_shift9': [np.mean, reciprocal_transformation, square_root_translation],
        'log_return_last_price_shift2': [np.mean, realized_volatility, realized_quarticity, reciprocal_transformation, square_root_translation],
        'log_return_last_price_shift3': [np.mean, realized_volatility, realized_quarticity, reciprocal_transformation, square_root_translation],
        'log_return_last_price_shift4': [np.mean, realized_volatility, realized_quarticity, reciprocal_transformation, square_root_translation],
        'log_return_last_price_shift5': [np.mean, realized_volatility, realized_quarticity, reciprocal_transformation,square_root_translation],
        'log_return_last_price_shift6': [np.mean, realized_volatility, realized_quarticity, reciprocal_transformation,square_root_translation],
        'log_return_last_price_shift7': [np.mean, realized_volatility, realized_quarticity, reciprocal_transformation,square_root_translation],
        'log_return_last_price_shift8': [np.mean, realized_volatility, realized_quarticity, reciprocal_transformation,square_root_translation],
        'log_return_last_price_shift9': [np.mean, realized_volatility, realized_quarticity, reciprocal_transformation,square_root_translation],
        'wap1': [np.mean, reciprocal_transformation, square_root_translation],
        'wap2': [np.mean, reciprocal_transformation, square_root_translation],
        'wap3': [np.mean, reciprocal_transformation, square_root_translation],
        'wap4': [np.mean, reciprocal_transformation, square_root_translation],
        'wap5': [np.mean, reciprocal_transformation, square_root_translation],
        'mid_price': [np.mean, reciprocal_transformation, square_root_translation],
        'mid_price1': [np.mean, reciprocal_transformation, square_root_translation],
        'mid_price2': [np.mean, reciprocal_transformation, square_root_translation],
        'mid_price3': [np.mean, reciprocal_transformation, square_root_translation],
        'press_buy1': [np.mean, reciprocal_transformation, square_root_translation],
        'press_sell1': [np.mean, reciprocal_transformation, square_root_translation],
        'order_BS1': [np.mean, reciprocal_transformation, square_root_translation],
        'HR1': [np.mean, reciprocal_transformation, square_root_translation],
        'vtA': [np.mean, reciprocal_transformation, square_root_translation],
        'vtB': [np.mean, reciprocal_transformation, square_root_translation],
        'Oiab': [np.mean, reciprocal_transformation, square_root_translation],
        'log_return1': [np.mean, reciprocal_transformation, square_root_translation,realized_volatility, realized_quarticity],
        'log_return2': [np.mean, reciprocal_transformation, square_root_translation, realized_volatility, realized_quarticity],
        'log_return3': [np.mean, reciprocal_transformation, square_root_translation, realized_volatility, realized_quarticity],
        'log_return4': [np.mean, reciprocal_transformation, square_root_translation, realized_volatility, realized_quarticity],
        'log_return_wap1_shift3': [np.mean, reciprocal_transformation, square_root_translation, realized_volatility, realized_quarticity],
        'log_return_wap1_shift4': [np.mean, reciprocal_transformation, square_root_translation, realized_volatility, realized_quarticity],
        'log_return_wap1_shift5': [np.mean, reciprocal_transformation, square_root_translation, realized_volatility,realized_quarticity],
        'log_return_wap1_shift6': [np.mean, reciprocal_transformation, square_root_translation, realized_volatility,realized_quarticity],
        'log_return_wap1_shift7': [np.mean, reciprocal_transformation, square_root_translation, realized_volatility,realized_quarticity],
        'log_return_wap1_shift8': [np.mean, reciprocal_transformation, square_root_translation, realized_volatility,realized_quarticity],
        'log_return_wap1_shift9': [np.mean, reciprocal_transformation, square_root_translation, realized_volatility,realized_quarticity],
        'wap_balance1': [np.mean,reciprocal_transformation, square_root_translation],
        'total_volume1': [np.mean,reciprocal_transformation, square_root_translation],
        'volume_imbalance1': [np.mean,reciprocal_transformation, square_root_translation],
        'size_weight': [np.mean,reciprocal_transformation, square_root_translation],
        'bid_size1_2_minus': [np.mean, reciprocal_transformation, square_root_translation],
        'ask_size1_2_minus': [np.mean, reciprocal_transformation, square_root_translation],
        'bid_ask_size1_minus': [np.mean, reciprocal_transformation, square_root_translation],
        'bid_ask_size1_plus': [np.mean, reciprocal_transformation, square_root_translation],
        'bid_ask_size2_minus': [np.mean, reciprocal_transformation, square_root_translation],
        'bid_ask_size2_plus': [np.mean, reciprocal_transformation, square_root_translation],
        'bid_size1_shift': [np.mean, reciprocal_transformation, square_root_translation],
        'bid_size2_shift': [np.mean, reciprocal_transformation, square_root_translation],
        'ask_size1_shift': [np.mean, reciprocal_transformation, square_root_translation],
        'ask_size2_shift': [np.mean, reciprocal_transformation, square_root_translation],
        'bid_ask_size1_spread': [np.mean, reciprocal_transformation, square_root_translation],
        'bid_ask_size2_spread': [np.mean, reciprocal_transformation, square_root_translation],
        'bid_amount_abs1': [np.mean, reciprocal_transformation, square_root_translation],
    }
    # create_feature_dict_time = {
    #     'log_return_last_price_shift2': [np.mean, realized_volatility, realized_quarticity],
    #     'log_return_last_price_shift3': [np.mean, realized_volatility, realized_quarticity],
    #     'log_return_last_price_shift4': [np.mean, realized_volatility, realized_quarticity],
    #     'log_return1': [np.mean, realized_volatility, realized_quarticity],
    #     'log_return2': [np.mean, realized_volatility, realized_quarticity],
    #     'log_return3': [np.mean, realized_volatility, realized_quarticity],
    #     'log_return4': [np.mean, realized_volatility, realized_quarticity],
    #     'log_return_wap1_shift3': [np.mean, realized_volatility, realized_quarticity],
    #     'log_return_wap1_shift4': [np.mean, realized_volatility, realized_quarticity]
    # }

    # Function to get group stats for different windows (seconds in bucket)
    def get_stats_window(fe_dict, index, add_suffix=False):
        # Group by the window
        df_feature = df[df['index'] >= index].groupby(['datetime']).agg(fe_dict).reset_index()
        # Rename columns joining suffix
        df_feature.columns = ['_'.join(col) for col in df_feature.columns]
        # Add a suffix to differentiate windows
        if add_suffix:
            df_feature = df_feature.add_suffix('_' + str(index))
        return df_feature
    # def get_stats_window(fe_dict, add_suffix=False):
    #     # Group by the window
    #     df_feature = df.groupby(['datetime']).agg(fe_dict).reset_index()
    #     # Rename columns joining suffix
    #     df_feature.columns = ['_'.join(col) for col in df_feature.columns]
    #     # Add a suffix to differentiate windows
    #     # if add_suffix:
    #     #     df_feature = df_feature.add_suffix('_' + str(seconds_in_bucket))
    #     return df_feature

    # Get the stats for different windows
    df_feature = get_stats_window(create_feature_dict, index=0, add_suffix=False)
    # df_feature_500 = get_stats_window(create_feature_dict_time, index=25, add_suffix=True)
    # df_feature_400 = get_stats_window(create_feature_dict_time, index=20, add_suffix=True)
    # df_feature_300 = get_stats_window(create_feature_dict_time, index=15, add_suffix=True)
    # df_feature_200 = get_stats_window(create_feature_dict_time, index=10, add_suffix=True)
    # df_feature_100 = get_stats_window(create_feature_dict_time, index=5, add_suffix=True)
    #
    # # Merge all
    # df_feature = df_feature.merge(df_feature_500, how='left', left_on='datetime_', right_on='datetime__25')
    # df_feature = df_feature.merge(df_feature_400, how='left', left_on='datetime_', right_on='datetime__20')
    # df_feature = df_feature.merge(df_feature_300, how='left', left_on='datetime_', right_on='datetime__15')
    # df_feature = df_feature.merge(df_feature_200, how='left', left_on='datetime_', right_on='datetime__10')
    # df_feature = df_feature.merge(df_feature_100, how='left', left_on='datetime_', right_on='datetime__5')
    print(df_feature)
    # # Drop unnecesary time_ids
    # df_feature.drop(['datetime__25', 'datetime__20', 'datetime__15', 'datetime__10', 'datetime__5'], axis=1,
    #                 inplace=True)

    # Create row_id so we can merge
    # stock_id = file_path.split('=')[1]
    # df_feature['row_id'] = df_feature['time_id_'].apply(lambda x: f'{stock_id}-{x}')
    # df_feature.drop(['datetime_'], axis=1, inplace=True)
    return df_feature

#%%
def trade_preprocessor(data):
    df = data
    # df['log_return'] = np.log(df['last_price']).shift()
    df['size'] = df['volume'] - df['volume'].shift(1)
    df['amount'] = df['last_price'] * df['size']

    ask_quantity_1 = df['ask_size1']
    ask_quantity_2 = df['ask_size2']
    ask_quantity_3 = df['ask_size3']
    bid_quantity_1 = df['bid_size1']
    bid_quantity_2 = df['bid_size2']
    bid_quantity_3 = df['bid_size3']

    def weight_pecentage(w1, w2, w3, ask_quantity_1, ask_quantity_2, ask_quantity_3, bid_quantity_1, bid_quantity_2, bid_quantity_3):
        Weight_Ask = (w1 * ask_quantity_1 + w2 * ask_quantity_2 + w3 * ask_quantity_3)
        Weight_Bid = (w1 * bid_quantity_1 + w2 * bid_quantity_2 + w3 * bid_quantity_3)
        W_AB = Weight_Ask / Weight_Bid
        W_A_B = (Weight_Ask - Weight_Bid) / (Weight_Ask + Weight_Bid)

        return W_AB, W_A_B
        # Weight Depth
    w1, w2, w3 = [100.0, 0.0, 0.0]
    df['W_AB_100'], df['W_A_B_100'] = weight_pecentage(w1, w2, w3, ask_quantity_1, ask_quantity_2, ask_quantity_3, bid_quantity_1, bid_quantity_2, bid_quantity_3)
    w1, w2, w3 = [0.0, 100.0, 0.0]
    df['W_AB_010'], df['W_A_B_010'] = weight_pecentage(w1, w2, w3, ask_quantity_1, ask_quantity_2, ask_quantity_3, \
                                           bid_quantity_1, bid_quantity_2, bid_quantity_3)
    w1, w2, w3 = [0.0, 0.0, 100.0]
    df['W_AB_001'], df['W_A_B_001'] = weight_pecentage(w1, w2, w3, ask_quantity_1, ask_quantity_2, ask_quantity_3, \
                                           bid_quantity_1, bid_quantity_2, bid_quantity_3)
    w1, w2, w3 = [90.0, 10.0, 0.0]
    df['W_AB_910'], df['W_A_B_910'] = weight_pecentage(w1, w2, w3, ask_quantity_1, ask_quantity_2, ask_quantity_3, \
                                           bid_quantity_1, bid_quantity_2, bid_quantity_3)
    w1, w2, w3 = [80.0, 20.0, 0.0]
    df['W_AB_820'], df['W_A_B_820'] = weight_pecentage(w1, w2, w3, ask_quantity_1, ask_quantity_2, ask_quantity_3, \
                                           bid_quantity_1, bid_quantity_2, bid_quantity_3)
    w1, w2, w3 = [70.0, 30.0, 0.0]
    df['W_AB_730'], df['W_A_B_730'] = weight_pecentage(w1, w2, w3, ask_quantity_1, ask_quantity_2, ask_quantity_3, \
                                           bid_quantity_1, bid_quantity_2, bid_quantity_3)
    w1, w2, w3 = [60.0, 40.0, 0.0]
    df['W_AB_640'], df['W_A_B_640'] = weight_pecentage(w1, w2, w3, ask_quantity_1, ask_quantity_2, ask_quantity_3, \
                                           bid_quantity_1, bid_quantity_2, bid_quantity_3)
    w1, w2, w3 = [50.0, 50.0, 0.0]
    df['W_AB_550'], df['W_A_B_550'] = weight_pecentage(w1, w2, w3, ask_quantity_1, ask_quantity_2, ask_quantity_3, \
                                           bid_quantity_1, bid_quantity_2, bid_quantity_3)
    w1, w2, w3 = [70.0, 20.0, 10.0]
    df['W_AB_721'], df['W_A_B_721'] = weight_pecentage(w1, w2, w3, ask_quantity_1, ask_quantity_2, ask_quantity_3, \
                                           bid_quantity_1, bid_quantity_2, bid_quantity_3)
    w1, w2, w3 = [50.0, 30.0, 20.0]
    df['W_AB_532'], df['W_A_B_532'] = weight_pecentage(w1, w2, w3, ask_quantity_1, ask_quantity_2, ask_quantity_3, \
                                           bid_quantity_1, bid_quantity_2, bid_quantity_3)
    w1, w2, w3 = [1.0, 1.0, 1.0]
    df['W_AB_111'], df['W_A_B_111'] = weight_pecentage(w1, w2, w3, ask_quantity_1, ask_quantity_2, ask_quantity_3, \
                                           bid_quantity_1, bid_quantity_2, bid_quantity_3)
    w1, w2, w3 = [10.0, 90.0, 1.0]
    df['W_AB_190'], df['W_A_B_190'] = weight_pecentage(w1, w2, w3, ask_quantity_1, ask_quantity_2, ask_quantity_3, \
                                           bid_quantity_1, bid_quantity_2, bid_quantity_3)
    w1, w2, w3 = [20.0, 80.0, 0.0]
    df['W_AB_280'], df['W_A_B_280'] = weight_pecentage(w1, w2, w3, ask_quantity_1, ask_quantity_2, ask_quantity_3, \
                                           bid_quantity_1, bid_quantity_2, bid_quantity_3)
    w1, w2, w3 = [30.0, 70.0, 0.0]
    df['W_AB_370'], df['W_A_B_370'] = weight_pecentage(w1, w2, w3, ask_quantity_1, ask_quantity_2, ask_quantity_3, \
                                           bid_quantity_1, bid_quantity_2, bid_quantity_3)
    w1, w2, w3 = [40.0, 60.0, 0.0]
    df['W_AB_460'], df['W_A_B_460'] = weight_pecentage(w1, w2, w3, ask_quantity_1, ask_quantity_2, ask_quantity_3, \
                                           bid_quantity_1, bid_quantity_2, bid_quantity_3)
    w1, w2, w3 = [10.0, 20.0, 70.0]
    df['W_AB_127'], df['W_A_B_127'] = weight_pecentage(w1, w2, w3, ask_quantity_1, ask_quantity_2, ask_quantity_3, \
                                           bid_quantity_1, bid_quantity_2, bid_quantity_3)
    w1, w2, w3 = [20.0, 30.0, 50.0]
    df['W_AB_235'], df['W_A_B_235'] = weight_pecentage(w1, w2, w3, ask_quantity_1, ask_quantity_2, ask_quantity_3, \
                                           bid_quantity_1, bid_quantity_2, bid_quantity_3)

    # Dict for aggregations
    create_feature_dict = {
        # 'log_return': [realized_volatility],
        # 'seconds_in_bucket': [count_unique],
        'size': [np.mean, reciprocal_transformation, square_root_translation],
        'amount': [np.mean, reciprocal_transformation, square_root_translation],
        'W_AB_100': [np.mean, reciprocal_transformation, square_root_translation],
        'W_A_B_100': [np.mean, reciprocal_transformation, square_root_translation],
        'W_AB_010': [np.mean, reciprocal_transformation, square_root_translation],
        'W_A_B_010': [np.mean, reciprocal_transformation, square_root_translation],
        'W_AB_001': [np.mean, reciprocal_transformation, square_root_translation],
        'W_A_B_001': [np.mean, reciprocal_transformation, square_root_translation],
        'W_AB_910': [np.mean, reciprocal_transformation, square_root_translation],
        'W_A_B_910': [np.mean, reciprocal_transformation, square_root_translation],
        'W_AB_820': [np.mean, reciprocal_transformation, square_root_translation],
        'W_A_B_820': [np.mean, reciprocal_transformation, square_root_translation],
        'W_AB_730': [np.mean, reciprocal_transformation, square_root_translation],
        'W_A_B_730': [np.mean, reciprocal_transformation, square_root_translation],
        'W_AB_640': [np.mean, reciprocal_transformation, square_root_translation],
        'W_A_B_640': [np.mean, reciprocal_transformation, square_root_translation],
        'W_AB_550': [np.mean, reciprocal_transformation, square_root_translation],
        'W_A_B_550': [np.mean, reciprocal_transformation, square_root_translation],
        'W_AB_721': [np.mean, reciprocal_transformation, square_root_translation],
        'W_A_B_721': [np.mean, reciprocal_transformation, square_root_translation],
        'W_AB_532': [np.mean, reciprocal_transformation, square_root_translation],
        'W_A_B_532': [np.mean, reciprocal_transformation, square_root_translation],
        'W_AB_111': [np.mean, reciprocal_transformation, square_root_translation],
        'W_A_B_111': [np.mean, reciprocal_transformation, square_root_translation],
        'W_AB_190': [np.mean, reciprocal_transformation, square_root_translation],
        'W_A_B_190': [np.mean, reciprocal_transformation, square_root_translation],
        'W_AB_280': [np.mean, reciprocal_transformation, square_root_translation],
        'W_A_B_280': [np.mean, reciprocal_transformation, square_root_translation],
        'W_AB_370': [np.mean, reciprocal_transformation, square_root_translation],
        'W_A_B_370': [np.mean, reciprocal_transformation, square_root_translation],
        'W_AB_460': [np.mean, reciprocal_transformation, square_root_translation],
        'W_A_B_460': [np.mean, reciprocal_transformation, square_root_translation],
        'W_AB_127': [np.mean, reciprocal_transformation, square_root_translation],
        'W_A_B_127': [np.mean, reciprocal_transformation, square_root_translation],
        'W_AB_235': [np.mean, reciprocal_transformation, square_root_translation],
        'W_A_B_235': [np.mean, reciprocal_transformation, square_root_translation],
    }
    # create_feature_dict_time = {
    #     # 'log_return': [realized_volatility],
    #     # 'seconds_in_bucket': [count_unique],
    #     'size': [np.mean],
    #     'rolling_mean_size': [np.mean],
    #     'rolling_var_size': [np.mean],
    #     'rolling_std_size': [np.mean],
    #     'rolling_sum_size': [np.mean],
    #     'rolling_min_size': [np.mean],
    #     'rolling_max_size': [np.mean],
    #     'rolling_corr_size': [np.mean],
    #     'rolling_kurt_size': [np.mean],
    #     'rolling_skew_size': [np.mean],
    #     'rolling_median_amount': [np.mean],
    #     'rolling_quantile_25_amount': [np.mean],
    #     'rolling_quantile_50_amount': [np.mean],
    #     'rolling_quantile_75_amount': [np.mean],
    #     'ewm_mean_size': [np.mean],
    #     'ewm_std_size': [np.mean],
    #     'ewm_var_size': [np.mean],
    #     'ewm_corr_size': [np.mean],
    #     'ewm_cov_size': [np.mean],
    #     'ewm_mean_amount': [np.mean],
    #     'ewm_std_amount': [np.mean],
    #     'ewm_var_amount': [np.mean],
    #     'ewm_corr_amount': [np.mean],
    #     'ewm_cov_amount': [np.mean]
    #     # 'order_count': [np.sum],
    # }

    # Function to get group stats for different windows (seconds in bucket)
    def get_stats_window(fe_dict, index, add_suffix=False):
        # Group by the window
        df_feature = df[df['index'] >= index].groupby(['datetime']).agg(fe_dict).reset_index()
        # Rename columns joining suffix
        df_feature.columns = ['_'.join(col) for col in df_feature.columns]
        # Add a suffix to differentiate windows
        if add_suffix:
            df_feature = df_feature.add_suffix('_' + str(index))
        return df_feature
    # df = df.dropna(axis=0, how='any')
    # Get the stats for different windows
    df_feature = get_stats_window(create_feature_dict, index=0, add_suffix=False)
    # df_feature_500 = get_stats_window(create_feature_dict_time, index=25, add_suffix=True)
    # df_feature_400 = get_stats_window(create_feature_dict_time, index=20, add_suffix=True)
    # df_feature_300 = get_stats_window(create_feature_dict_time, index=15, add_suffix=True)
    # df_feature_200 = get_stats_window(create_feature_dict_time, index=10, add_suffix=True)
    # df_feature_100 = get_stats_window(create_feature_dict_time, index=5, add_suffix=True)
    # # Merge all
    # df_feature = df_feature.merge(df_feature_500, how='left', left_on='datetime_', right_on='datetime__25')
    # df_feature = df_feature.merge(df_feature_400, how='left', left_on='datetime_', right_on='datetime__20')
    # df_feature = df_feature.merge(df_feature_300, how='left', left_on='datetime_', right_on='datetime__15')
    # df_feature = df_feature.merge(df_feature_200, how='left', left_on='datetime_', right_on='datetime__10')
    # df_feature = df_feature.merge(df_feature_100, how='left', left_on='datetime_', right_on='datetime__5')
    # print(df_feature)
    # # Drop unnecesary time_ids
    # df_feature.drop(['datetime__25', 'datetime__20', 'datetime__15', 'datetime__10', 'datetime__5'], axis=1,
    #                 inplace=True)

    print(df_feature)

    return df_feature

#%%
# def get_time_stock(df):
#     vol_cols = ['log_return1_realized_volatility', 'log_return2_realized_volatility',
#                 'log_return1_realized_volatility_400', 'log_return2_realized_volatility_400',
#                 'log_return1_realized_volatility_300', 'log_return2_realized_volatility_300',
#                 'log_return1_realized_volatility_200', 'log_return2_realized_volatility_200',
#                 'trade_log_return_realized_volatility', 'trade_log_return_realized_volatility_400',
#                 'trade_log_return_realized_volatility_300', 'trade_log_return_realized_volatility_200']
#
#     # Group by the stock id
#     df_stock_id = df.groupby(['stock_id'])[vol_cols].agg(['mean', 'std', 'max', 'min', ]).reset_index()
#     # Rename columns joining suffix
#     df_stock_id.columns = ['_'.join(col) for col in df_stock_id.columns]
#     df_stock_id = df_stock_id.add_suffix('_' + 'stock')
#
#     # Group by the stock id
#     df_time_id = df.groupby(['time_id'])[vol_cols].agg(['mean', 'std', 'max', 'min', ]).reset_index()
#     # Rename columns joining suffix
#     df_time_id.columns = ['_'.join(col) for col in df_time_id.columns]
#     df_time_id = df_time_id.add_suffix('_' + 'time')
#
#     # Merge with original dataframe
#     df = df.merge(df_stock_id, how='left', left_on=['stock_id'], right_on=['stock_id__stock'])
#     df = df.merge(df_time_id, how='left', left_on=['time_id'], right_on=['time_id__time'])
#     df.drop(['stock_id__stock', 'time_id__time'], axis=1, inplace=True)
#     return df
#%%
import time
start = time.time()
train_data = pd.merge(book_preprocessor(data1), trade_preprocessor(data1), on='datetime_',how='left')
end = time.time()
print('Total Time = %s'%(end-start))

#%%
train_data = pd.merge(book_preprocessor(data1), trade_preprocessor(data1), on='datetime_',how='left')
#%%
test_data = train_data.fillna(0)
test_data = test_data.replace(np.inf, 1)
test_data = test_data.replace(-np.inf, -1)
#%%
test_data['last_price'] = data1['last_price']
#%%
test_data['target'] = np.log(test_data['wap1_mean']/test_data['wap1_mean'].shift(1))*100
test_data['target'] = test_data['target'].shift(-1)
test_data = test_data.dropna(axis=0, how='any')
#%%
test_data = test_data.iloc[:,1:]
#%%

from scipy import stats
import re
import warnings

def calcSpearman(data):

    ic_list = []
    data = data.copy()
    # target = data['target']
    for column in list(data.columns[0:387]):

        ic = data[column].rolling(5000).corr(data['target'])
        ic_mean = np.mean(ic)
        print(ic_mean)
        ic_list.append(ic_mean)

        # print(ic_list)

    return ic_list

IC = calcSpearman(test_data)

IC = pd.DataFrame(IC)
columns = pd.DataFrame(test_data.columns)

IC_columns = pd.concat([IC, columns], axis=1)
col = ['value', 'variable']
IC_columns.columns = col

filter_value = 0.05
filter_value2 = -0.05
x_column = IC_columns.variable[IC_columns.value > filter_value]
y_column = IC_columns.variable[IC_columns.value < filter_value2]

x_column = x_column.tolist()
y_column = y_column.tolist()
final_col = x_column+y_column
#%%
final_data = test_data.reindex(columns=final_col)
#%%
final_data['wap1_mean'] = test_data['wap1_mean']
final_data['datetime'] = train_data['datetime_']
#%%
# test_data['time'] = test_data['datetime_'].str.extract('\d(.{0,18})')
# test_data = test_data.drop_duplicates(subset=['time'],keep='first')
# test_data['stime'] = test_data['datetime_'].str.extract(':.*:(.{1,3})')
# test_data['stime'] = test_data['stime'].astype('float64')
# test_data = test_data[test_data['stime']% 5 == 0]
# test_data = test_data.drop(['stime','time',],axis=1)
# test_data = test_data.reset_index(drop=True)
#%%
test_data = train_data.fillna(0)
#%%
test_data = test_data.replace(np.inf, 1)
#%%
test_data = test_data.replace(-np.inf, -1)
#%%
# test_data['stime'] = test_data['datetime_'].str.extract(':.*:(.{1,3})')
# test_data['stime'] = test_data['stime'].astype('float64')
#
# from sklearn.preprocessing import FunctionTransformer
#
#
# def sin_transformer(period):
#     return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))
#
#
# def cos_transformer(period):
#     return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))
#
# test_data['sin'] = sin_transformer(60).fit_transform(test_data['stime'])
# test_data['cos'] = cos_transformer(60).fit_transform(test_data['stime'])
# test_data = test_data.drop(['stime'],axis=1)
#%%
train_data.to_csv('filter_feature_1tick_387var_rolling15.csv')
#%%
final_data.to_csv('test2202_01_04_01_31.csv')

