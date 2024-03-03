##################
### Data Utils ###
##################

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

def get_market_data_split(data, train_prop):
    """Split MarketData object into training and test sets according to a specified proportion"""

    if type(data) == dict:
        split_index = int(len(data[list(data.keys())[0]])*train_prop)
        train_data = {}
        test_data = {}
        for key in data.keys():
            train_data[key] = data[key].iloc[:split_index]
            test_data[key] = data[key].iloc[split_index:]
    elif type(data) == pd.DataFrame or type(data) == pd.Series:
        split_index = int(len(data)*train_prop)
        train_data = data.iloc[:split_index]
        test_data = data.iloc[split_index:]
    elif type(data) == pd.DatetimeIndex:
        split_index = int(len(data)*train_prop)
        train_data = data[:split_index]
        test_data = data[split_index:]
    elif type(data) == np.ndarray:
        split_index = int(len(data)*train_prop)
        train_data = data[:split_index]
        test_data = data[split_index:]
    else:
        raise TypeError('Data must be a dictionary, pandas dataframe, or numpy array')
    return train_data, test_data


def minmax_scale(data):
    """Apply the MinMax scaler to the data and return the scaled data as a pandas dataframe"""

    scaler = MinMaxScaler()
    if type(data)==pd.Series:
        data = data.values.reshape(-1,1)
    data_sc = scaler.fit_transform(data)
    scaled_df = pd.DataFrame(data_sc, columns=data.columns, index=data.index)
    return scaled_df


def std_scale(data):
    """Apply the Standard scaler to the data and return the scaled data as a pandas dataframe"""

    scaler = StandardScaler()
    data_sc = scaler.fit_transform(data)
    scaled_df = pd.DataFrame(data_sc, columns=data.columns, index=data.index)
    return scaled_df


def pca(df, n_components):
    """Return a pandas dataframe of the first n principal components of the input dataframe"""
    
    principal_components=PCA(n_components=n_components)
    principal_components.fit(df)
    pc=principal_components.transform(df)
    pc_df = pd.DataFrame()
    for i in range(n_components):
        pc_df[f'PC {i+1}'] = pc[:,i]
    pc_df.index = df.index
    return pc_df