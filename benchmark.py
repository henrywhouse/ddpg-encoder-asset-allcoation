#########################
###  Portfolio Utils  ###
#########################

## Utility methods for pportfolio operations and analysis

import numpy as np
import pandas as pd

class Benchmark:
    """Benchmark class that represents a benchmark for a portfolio of securities
    """
    def __init__(self, df, weights):
        """Initialize the benchmark object"""

        self.assets = df.columns
        self.dates = df.index
        self.returns = df.to_numpy()
        self.weights = weights
        
        self.get_period_return()
        self.get_cumulative_return()


    def get_period_return(self):
        """Returns the period return array for the portfolio"""

        self.period_return = np.sum(self.returns * self.weights, axis=1)
    
    def get_cumulative_return(self):
        """Returns the cumulative return array for the portfolio"""
        
        self.cumulative_return = np.cumprod(self.period_return + 1) - 1

