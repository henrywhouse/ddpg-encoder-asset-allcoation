#########################
### Market Data Class ###
#########################

import numpy as np
import pandas as pd
from benchmark import Benchmark
from data_utils import minmax_scale, std_scale, get_market_data_split

class MarketData(object):
    """Market Data class that creates a data structure to hold candle data and returns, macroeconomic data, and the risk-free rate
    """
    def __init__(self, price_dict, weights, rf, macro_data, scaler=None, flatten=False, train_prop=None):
        """Initialize the MarketData object"""

        self.flatten = flatten
        self.train_prop = train_prop
        self.price_dict = price_dict     
        self.assets = list(price_dict.keys())
        self.dates = self.price_dict[self.assets[0]].index
        self.returns = self.get_returns()
        self.rf = rf
        self.macro_data = macro_data
        self.weights = weights

        self.scaler = scaler
        if self.scaler is not None:
            self.apply_scaler()

        if self.train_prop is not None:
            self.price_dict_train, self.price_dict_test = get_market_data_split(self.price_dict, self.train_prop)
            self.dates_train, self.dates_test = get_market_data_split(self.dates, self.train_prop)
            self.returns_train, self.returns_test = get_market_data_split(self.returns, self.train_prop)
            self.weights_train, self.weights_test = get_market_data_split(self.weights, self.train_prop)
            self.macro_data_train, self.macro_data_test = get_market_data_split(self.macro_data, self.train_prop)
            self.rf_train, self.rf_test = get_market_data_split(self.rf, self.train_prop)

            self.benchmark_train = self.build_benchmark(self.returns_train, self.weights_train)
            self.benchmark_test = self.build_benchmark(self.returns_test, self.weights_test)

            if not self.flatten:     
                self.asset_feature_array_train = self.build_asset_feature_array(self.price_dict_train)
                self.asset_feature_array_test = self.build_asset_feature_array(self.price_dict_test)
            else:
                self.asset_feature_df_train = self.build_asset_feature_df(self.price_dict_train)
                self.asset_feature_df_test = self.build_asset_feature_df(self.price_dict_test)
        else:
            if not self.flatten: 
                self.asset_feature_array = self.build_asset_feature_array(self.price_dict)
            else:
                self.asset_feature_df = self.build_asset_feature_df(self.price_dict)
            self.benchmark = self.build_benchmark(self.returns, self.weights)


    def get_returns(self):
        """Calculate returns for the assets"""

        returns = pd.DataFrame(index=self.dates)
        for asset in self.assets:
            asset_df = self.price_dict[asset]
            asset_returns = asset_df['Adj Close'].pct_change().fillna(0.0)
            returns[asset] = asset_returns
        return returns


    def build_asset_feature_df(self, price_dict):
        """Builds a dataframe of asset features for the assets in the price_dict"""

        df = price_dict[self.assets[0]]
        for asset in self.assets[1:]:
            df = pd.concat([df, price_dict[asset]], axis=1)
        return df


    def build_asset_feature_array(self, price_dict):
        """Builds a 3D array of asset features for the assets in the price_dict"""

        len_dates = len(price_dict[self.assets[0]])
        len_assets = len(self.assets)
        len_features = len(price_dict[self.assets[0]].columns)

        asset_feature_array = np.zeros(shape=(len_dates, len_assets, len_features))
        for j, asset in enumerate(self.assets):
            asset_df = price_dict[asset]
            feature_values = asset_df.values
            asset_feature_array[:, j, :] = feature_values
        return asset_feature_array        


    def build_benchmark(self, returns, weights):
        """Builds a Benchmark object from the returns and weights"""
        
        return Benchmark(returns, weights)
    

    def apply_scaler(self):
        """Applies the specified scaler to the price_dict and macro_data attributes"""
        
        if self.scaler.lower() == 'minmax':
            for key in self.price_dict.keys():
                self.price_dict[key] = minmax_scale(self.price_dict[key])
            self.macro_data = minmax_scale(self.macro_data)
        if self.scaler.lower() == 'standard':
            for key in self.price_dict.keys():
                self.price_dict[key] = std_scale(self.price_dict[key])
            self.macro_data = std_scale(self.macro_data)