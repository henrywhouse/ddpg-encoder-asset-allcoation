#########################################
###  Asset Allocation Gym Environment ###
#########################################

# State: Stock returns from period lookback to today
# Action: Portfolio allocation for next holding period
# Reward: Sharpe ratio, beta-adjusted sharpe ratio, or alpha

from gym import Env
import numpy as np
from gym.spaces import Box
import pandas as pd
import ddpg_env_render_plotly as render_plotly
import ddpg_env_render_mpl as render_mpl
from rewards import rolling_sharpe_ratio, rolling_market_ratio, rolling_alpha

class AssetAllocationEnv(Env):
    """Reinforcement learning environment to allocate across a selection of assets. Maximizes reward (sharpe ratio or beta-adjusted sharpe ratio).
       Action := allocation of assets
    """

    def __init__(self, market_data, lookback=20, holding_period=5, window_size=500, 
                 rebalance=False, reward_type='sharpe', train=True):
        """Initialize reinforcement learning environment"""

        super(AssetAllocationEnv, self).__init__()

        # Aggregate Data
        self.market_data = market_data
        self.train = train
        if self.train:
            if self.market_data.flatten:
                self.data = self.market_data.asset_feature_df_train
            else:
                self.data = self.market_data.asset_feature_array_train  # Where observations are drawn from
            self.dates = self.market_data.dates_train  # Dates of observations
            self.stock_returns = self.market_data.returns_train
            self.benchmark = self.market_data.benchmark_train  # Type: Benchmark
            self.macro_data = self.market_data.macro_data_train
            self.rf = self.market_data.rf_train.to_numpy().reshape(-1,1)
        else:
            if self.market_data.flatten:
                self.data = self.market_data.asset_feature_df_test
            else:
                self.data = self.market_data.asset_feature_array_test  # Where observations are drawn from
            self.dates = self.market_data.dates_test # Dates of observations
            self.stock_returns = self.market_data.returns_test
            self.benchmark = self.market_data.benchmark_test  # Type: Benchmark
            self.macro_data = self.market_data.macro_data_test
            self.rf = self.market_data.rf_test.to_numpy().reshape(-1,1)
        if self.rf is None:
            self.rf = np.zeros(len(self.stock_returns),1)
        if self.macro_data is not None:
            self.data = pd.concat([self.data, self.macro_data], axis=1)
            self.data.ffill(inplace=True)
        self.data = self.data.to_numpy()
        self.data_shape = self.data.shape  # Shape of the data
        
        # Market Data
        self.asset_names = self.market_data.assets # names of assets
        self.num_assets = len(self.asset_names)  # Number of assets

        # Observation Parameters
        self.lookback = lookback  # Lookback for each observation
        self.holding_period = holding_period  # Holding period for each observation
        self.forward_returns = np.zeros((self.holding_period,self.num_assets))  # Forward returns (set at each step)
        self.benchmark_forward_returns = np.zeros((self.holding_period,self.num_assets))  # Forward returns (set at each step)
        self.forward_rf = None  # Forward-looking risk-free rate (set at each step)
        self.window_size = window_size  # Episode size
        self.window_dates = None  # Dates for each window
        self.MAX_START = len(self.stock_returns) - (self.window_size + self.holding_period)  # Last possible starting point to avoid out of bounds
        self.beg_ind = 0  # Randomly assigned in reset()
        self.num_steps = (self.window_size - self.lookback) // self.holding_period - 1 # Number of steps in each window (episode)
        self.current_step = 0 # Current step in the window (episode)
        self.rebalance = rebalance
        self.reward_type = reward_type

        # Spaces (Observation & Action)
        self.window_data = np.zeros((self.window_size,self.num_assets))  # Window of observations
        self.obs_shape = (self.lookback, self.data_shape[1]) # Shape of each observation
        self.observation_space = Box(low=np.float32(-100), high=np.float32(np.inf),  # Accounts for returns and prices 
                                            shape=self.obs_shape,  ## Shape of input data
                                            dtype=np.float32)
        self.action_space = Box(low=0, high=1,  # Minimum: No weight; Maximum: Only hold one asset
                                       shape=(self.num_assets,),  # One action per asset
                                       dtype=np.float32)
        
        # Data for plotting & analysis
        self.actions = np.zeros((self.num_steps, self.num_assets))
        self.rewards = np.zeros((self.num_steps,))
        self.portfolio_individual_returns = np.zeros((self.num_steps * self.holding_period + 1, self.num_assets))  # 2-D array  
        self.portfolio_returns = np.zeros((self.num_steps * self.holding_period + 1, 1))  # 2-D array  

        self.benchmark_actions = np.zeros((self.num_steps,self.num_assets))
        self.benchmark_rewards = np.zeros((self.num_steps,))
        self.benchmark_individual_returns = np.zeros((self.num_steps * self.holding_period + 1, self.num_assets))   # 2-D array  
        self.benchmark_returns = np.zeros((self.num_steps * self.holding_period + 1, 1))   # 2-D array  
        self.excess_returns = np.zeros((self.num_steps * self.holding_period + 1,))


    # STEP #
    def step(self, action):
        """Step function that advances the state of the environment and returns the result of the action"""

        done = self._is_done()
        self.forward_returns = self._get_forward_returns(self.stock_returns)
        self.forward_rf = self._get_forward_returns(self.rf)
        if self.benchmark is not None:
            self.benchmark_forward_returns = self._get_forward_returns(self.benchmark.returns)
            bm_weights = np.array([self.benchmark.weights[self.current_ind]])
            bm_reward = self._get_reward(self.benchmark_forward_returns, bm_weights, bm_weights)
            if not done:
                self._update_benchmark_arrays(bm_weights, bm_reward)
        reward = self._get_reward(self.forward_returns, action, bm_weights)
        if not reward < 0 and not reward > 0:
            raise ValueError('Reward must be positive or negative')
        if not done:
            self._update_data_arrays(action, reward)
        self.current_ind += self.holding_period
        self.current_step += 1
        info = {}
        state_ = self._get_observation()
        self._update_excess_returns()
        return state_, reward, done, info
    

    # RESET #
    def reset(self, seed=None, options=None):
        """Reset the environment to the initial (randomly-selected) state"""

        self._reset_attrs(seed)
        observation = self._get_observation()
        self.forward_returns = self._get_forward_returns(self.data)
        info = {}
        return observation, info


    # RENDER #
    def render(self, engine='plotly', plot_benchmark=False, plot_actions=False, plot_relative=False, plot_returns=False, plot_individual=False,
               plot_contribution=False):
        """Render the environment for analysis and visualization"""

        self.plot_benchmark = plot_benchmark
        portfolio_cumulative_returns = self._get_cumulative_return(self.portfolio_returns)
        benchmark_cumulative_returns = self._get_cumulative_return(self.benchmark_returns)
        portfolio_cumulative_individual_returns = self._get_cumulative_return(self.portfolio_individual_returns)
        benchmark_cumulative_individual_returns = self._get_cumulative_return(self.benchmark_individual_returns)

        if engine.lower()=='plotly':
            render_plotly._plot_reward(self.action_dates, self.rewards, self.benchmark_rewards, self.reward_type, self.holding_period, self.plot_benchmark)
            if plot_actions:
                render_plotly._plot_actions(self.action_dates, self.actions, self.benchmark_actions, self.asset_names, plot_relative)
            if plot_returns:
                render_plotly._plot_returns(self.window_dates, portfolio_cumulative_returns, benchmark_cumulative_returns, 
                                portfolio_cumulative_individual_returns, benchmark_cumulative_individual_returns, 
                                self.asset_names, self.num_assets, self.plot_benchmark, plot_individual=plot_individual,
                                plot_contribution=plot_contribution)
        elif engine=='matplotlib':
            render_mpl._plot_reward(self.action_dates, self.rewards, self.benchmark_rewards, self.reward_type, self.holding_period, self.plot_benchmark)
            if plot_actions:
                render_mpl._plot_actions(self.action_dates, self.actions, self.benchmark_actions, self.asset_names, plot_relative)
            if plot_returns:
                render_mpl._plot_returns(self.window_dates, portfolio_cumulative_returns, benchmark_cumulative_returns, 
                                portfolio_cumulative_individual_returns, benchmark_cumulative_individual_returns, 
                                self.asset_names, self.num_assets, self.plot_benchmark, plot_individual=plot_individual,
                                plot_contribution=plot_contribution)
        else:
            raise ValueError('Render engine must be either "plotly" or "matplotlib"')
                               

    ## Private utils for calculations ##
    def _get_observation(self):
        """Returns the next observation from the data"""

        return np.float32(self.data[self.current_ind-self.lookback+1:self.current_ind+1])
    

    def _get_forward_returns(self, data):
        """Returns the forward returns from the data"""

        return np.float32(data[self.current_ind+1:self.current_ind+self.holding_period+1])
       

    def _get_cumulative_return(self, returns):
        """Returns the cumulative returns from the period returns"""

        return np.cumprod(1 + returns, axis=0) - 1


    def _get_portfolio_returns(self, action, observation):
        """Returns the portfolio period returns from the individual period returns"""
        
        if self.rebalance:
            portfolio_returns = observation @ action.T
        else:
            drift_weights = self._get_drift_weights(action)
            portfolio_returns = np.sum(np.multiply(observation, drift_weights), axis=1, keepdims=True)
        return portfolio_returns
    

    def _get_drift_weights(self, action):
        """Returns the drift weights (no daily rebalancing) from the individual period returns"""
        
        returns = 1 + self.forward_returns
        returns[0,:] *= action[0]
        drift_weights = np.cumprod(returns, axis=0)
        row_sums = drift_weights.sum(axis=1, keepdims=True)
        drift_weights = drift_weights / row_sums
        return drift_weights
    
    
    def _get_reward(self, reward_returns, action, bm_weights):
        """Returns the reward from the state and actions taken"""

        forward_reward_portfolio_returns = self._get_portfolio_returns(action, reward_returns)
        if self.reward_type == 'sharpe':
            return rolling_sharpe_ratio(returns=forward_reward_portfolio_returns, rf=self.forward_rf)
        if self.reward_type == 'market_sharpe':
            forward_benchmark_returns = self._get_portfolio_returns(bm_weights, self.benchmark_forward_returns)
            return rolling_market_ratio(returns=forward_reward_portfolio_returns, rf=self.forward_rf, benchmark=forward_benchmark_returns)
        if self.reward_type == 'alpha':
            forward_benchmark_returns = self._get_portfolio_returns(bm_weights, self.benchmark_forward_returns)
            return rolling_alpha(returns=forward_reward_portfolio_returns, rf=self.forward_rf, benchmark=forward_benchmark_returns)
    

    def _update_excess_returns(self):
        """Updates the excess returns"""

        portfolio_cumulative_returns = self._get_cumulative_return(self.portfolio_returns)
        benchmark_cumulative_returns = self._get_cumulative_return(self.benchmark_returns)
        self.excess_returns = np.squeeze(portfolio_cumulative_returns - benchmark_cumulative_returns)


    def _is_done(self):
        """Returns whether the environment is done"""
        
        return self.current_ind + self.holding_period >= self.terminal_ind  # Compare next observation to terminal observation


    def _set_dates(self):
        """Returns the dates for the window, action, and holding periods"""

        self.window_dates = self.dates[self.beg_ind+self.lookback:self.terminal_ind-self.holding_period+1] 
        self.action_dates = self.dates[self.beg_ind+self.lookback:self.terminal_ind-self.holding_period:self.holding_period]
        self.holding_dates = self.dates[self.beg_ind+self.lookback+self.holding_period:self.terminal_ind:self.holding_period]


    def _reset_indexes(self, seed):
        """Reset the indexes for the environment"""

        if seed is not None:
            np.random.seed(seed)
        self.beg_ind = np.random.randint(self.lookback, self.MAX_START)
        self.current_ind = self.beg_ind + self.lookback  # Determines (current) step for the first action
        self.terminal_ind = self.beg_ind + self.window_size  # Determines terminal step
        self.current_step = 0


    def _update_data_arrays(self, action, reward):
        """Update the portfolio data arrays with the action, reward, and resulting returns for the current step"""

        self.actions[self.current_step] = action
        self.rewards[self.current_step] = reward
        portfolio_ind = self.holding_period*self.current_step  # Index of the end of next holding period
        portfolio_individual_returns = np.multiply(self.forward_returns, action)
        self.portfolio_individual_returns[portfolio_ind+1:portfolio_ind+self.holding_period+1] \
            = portfolio_individual_returns
        self.portfolio_returns[portfolio_ind+1:portfolio_ind+self.holding_period+1] \
            = self._get_portfolio_returns(action, self.forward_returns)


    def _update_benchmark_arrays(self, bm_action, bm_reward):
        """Update the benchmark arrays with the action, reward, and resulting returns for the current step"""

        self.benchmark_actions[self.current_step] = bm_action
        self.benchmark_rewards[self.current_step] = bm_reward
        portfolio_ind = self.holding_period*self.current_step  # Index of the end of next holding period
        benchmark_individual_returns = np.multiply(self.forward_returns, bm_action)
        self.benchmark_individual_returns[portfolio_ind+1:portfolio_ind+self.holding_period+1] \
            = benchmark_individual_returns
        self.benchmark_returns[portfolio_ind+1:portfolio_ind+self.holding_period+1] \
            = self._get_portfolio_returns(bm_action, self.benchmark_forward_returns)


    def _reset_attrs(self, seed):
        """Reset the attributes for the environment, selecting a random starting point"""

        self._reset_indexes(seed)
        self.actions = np.zeros((self.num_steps, self.num_assets))
        self.rewards = np.zeros((self.num_steps,)) 
        self._set_dates()


