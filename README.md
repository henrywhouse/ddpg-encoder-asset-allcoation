# LSTM Iterative Price Prediction

## Implement the DDPG algorithm with a customized environment and a customized actor network with gated encoder layers; trained on historical stock prices and macroeconomic indicators to construct efficient portfolios by dynamically allocating across assets.

### Summary of the Algorithm Implementation

##### Introduction
This repository is a data science project that implements the Deep Deterministic Policy Gradient (DDPG) algorithm on a customized stock trading market environment. This implements functionality for customizing the actor and critic networks, adjusting the (Ornstein-Uhlenbeck) noise, adding/pre-processing features through the implementation of a customized MarketData class, implements robust methods for training/testing of the algorithm, multi-GPU distributed training, and algorithm visualization in plotly and matplotlib. 

The data used in this demonstration has three main components: equity price candles, macroeconomic indicators, and the risk-free rate. All data used is from January 3, 2013 to February 1, 2024. The equity price candles are taken for the top 10 stocks (as weighted in the S&P 500). The macroeconomic indicators are the measures of 5-year breakeven inflation, 5-year forward inflation expectation, the federal funds rate, the 10- minus 2-year U.S. treasury spread and the CBOE Volatility Index (VIX). The risk-free rate is measured as the 3-month U.S. treasury bill.

##### Pre-Processing
To pre-process the data, we begin by passing our raw DataFrames into our MarketData class, which structures our data such that we can easily input it into our reinforcement learning environment. The MarketData class calculates returns of the assets and stores them separately. It then applies a MinMaxScaler() scaler to each of the matrices of stock prices and macroeconomic indicators. This is particularly important when including volume data as that is typically many orders of magnitude greater than our price or macroeconomic data, and it can adversely affect the predictive power of the networks. 

##### Reinforcement Learning Environment (AssetAllocationEnv)
The Asset Allocation Gym Environment provides a platform for developing and evaluating reinforcement learning (RL) algorithms in the context of portfolio optimization. The environment encapsulates the dynamics of allocating funds across a set of assets over a specified time horizon. This allocation decision is driven by historical price, volatility, and returns data as well as macroeconomic data. The environment creates a setting where the DDPG agent can maximize a chosen reward metric such as Sharpe ratio or alpha.

One of the primary components of the environment is its state representation, which comprises historical equity and macroeconomic data spanning a configurable lookback period. This historical context empowers RL agents (DDPG in this case) to make informed decisions about future asset allocations as a (human) portfolio manager would. Actions in this environment correspond to the allocation of funds among the available assets for the subsequent holding period. By iteratively interacting with the environment through actions, the RL agent learns optimal allocation strategies that seeks to beat the benchmark at the specified reward metric.

The environment's reward mechanism serves as the guiding force for agent learning. Depending on the specified reward type, agents receive feedback on the quality of their allocation decisions. For instance, a Sharpe ratio-based reward incentivizes agents to achieve higher risk-adjusted returns, promoting prudent risk management in portfolio allocation strategies. Additionally, the environment supports alternative reward metrics such as beta-adjusted Sharpe ratio and alpha, providing flexibility for exploring several investment objectives and strategies.

One episode in the environment consists of iterative walking forward in time one step at a time for the duration of the window size.The agent looks back to analyze patterns historical data and make an asset allocation decision for the holding period. The environment allows the portfolio to be rebalanced each day in the holding period or to allow it to drift after making the first allocation decision. The agent then receives the reward, optimizes the networks, and steps forward to the end of the holding period to perform another iteration until it reaches the terminal state (end of the window).

To facilitate analysis and visualization, the environment offers rendering capabilities using either Plotly or Matplotlib engines. This functionality enables insights into agent behavior, portfolio performance, and the impact of different reward metrics on allocation strategies. It helps to visualize asset allocation, returns, contribution and attribution, and reward metrics. 

##### Actor & Critic Networks
For the neural networks, we use two unique architectures for the (main/target) actor and (main/target) critic networks. The actor network consists of 2 custom (gated) encoder layers followed 3 dense layers, the last dense layer being the output layer with the number of neurons equal to the number of assets to allocate between. A softmax is applied to the output layer to compute the allocation for the next holding period. These architectures are used for the target networks.

The custom encoder layers are built by flattening the data, applying layer normalization, followed by a multi-headed attention layer to capture spatial dependencies. Next comes a dropout layer and another layer normalization. After that is a gated recurrent unit (GRU) layer, aiding in capturing the intertemporal dependencies in the data. The last layer is another dropout layer.

The critic networks are built by placing three dense layers in a row; the output of the network in the Q-value of the state-action pair.

##### The Agent (DDPGAgent)
The DDPG agent acts as an interface between the environment, the replay buffer, the stochastic noise, the neural networks, and the learning functionality. It instantiates the networks, optimizers, and the learner class. The Agent also takes as a parameter the GPU distribution strategy (if multiple GPUs are available) and facilitates distributed training of the networks. This class also implements loading and saving model parameters from disk.

Per the DDPG paper, this agent calls the main actor network to retrieve the action (allocations) and applies the stochastic noise (Ornstein-Uhlenbeck) to encourage exploration. It takes a batch of state, action, reward tensors sampled randomly from the replay buffer to update the critic network parameters and then subsequently updates the target network parameters over time at a slower pace for training stability; after which the actor networks are updated using the new Q-values. 

### Our Dataset and Results 

##### The Dataset
The features used in the sample notebook are three-fold: stock candle data, macroeconomic variables, and the risk-free rate. The candles come from Yahoo Finance, and the macroeconomic indicators/risk-free rate from FRED. The data collected spans January 3, 2013 to February 1, 2024. The candle data is collected from the top ten cap-weighted holdings in the S&P 500 as of February 10, 2024: MSFT, AAPL, NVDA, AMZN, META, GOOGL, GOOG, BRK-B, LLY, and AVGO. To compute the benchmark, we offer two options: the cap-weighted and equal-weighted benchmark; this notebook uses the equal-weighted to minimize single-stock conentration in the portfolio. All of this data together goes into the MarketData class to facilitate seamless interaction with the custom environment. The MarketData class also implements train/test splitting of the data, dividing the data into two disjoint sets of data, dividing a a specific point in time to prevent data leakage, allowing the model to be evaluated fairly and accurately during the testing stage.

##### Instantiation
The notebook creates the middle layers of the neural networks, specified above. It then instantiates the GPU distribution strategy (mirrored) and the DDPG agent. Next, it instantiates the RLInterface, a class built to streamline the process of building the training and testing loops, visualization of the learning curve, and storing relevant data and statistics about the algorithm's performance.

##### Training Loop
The algorithm is trained for 150 episodes. From the learning curve, we can see that on average the performance of the agent improves over time, with alpha increasing from -0.0027% to 0.0011% indicating that the algorithm is learning from the data and making better asset allocation decisions as training progresses. 

We then display a one-episode sample of the algorithm. The plots show it outperforms the benchmark, generating ~0.09% alpha. We can also see how the allocation evolves over the course of the episode as the algorithm makes dynamic allocation decisions. The next plot shows the relative overweights and underweights and how they increase over time. This episode, the algorithm selected to overweight MSFT, NVDA, and BRK-B. It also selected to strategically underweight AVGO, the worst performing stock over the period. 

The largest contributor (70% of returns) was NVDA, adding ~10% to strategy returns versus the benchmark. The largest detractor from returns was the underweight to LLY (contributing 8%), detracting from overall strategy returns by ~1.1%.

For the whole episode, we can see that the algorithm outperformed the benchmark by 8.7%, albeit by taking more risk than the benchmark (beta=1.12). This results in an alpha of -4.83% (below the long-term average of 0.0011%).

Over the course of all the iterations, we can see that the algorithm generates on average 0.068% of excess return over the benchmark, and when adjusted for risk, the alpha is -0.12%. (This is different from the previously references 0.11% because the -0.12% does not take into account the risk-free rate in alpha calculations). The agent tends to take about 8.8% more risk than the benchmark, and generates on average 113.5% return per episode. 

(Caveat: survivorship bias of the top 10 stocks highly influences the absolute return of the algorithm, making relative performance a better metric for algorithm performance.)

##### Testing Loop
The algorithm is tested on 50 iterations. We can see that in the sample episode, the algorithm fails to generate any alpha; however, it generates excess returns versus the benchmark of 11.4%.

On average, we see that the testing samples do not produce superior results to the benchmark, generating both negative alpha and excess return. 

##### Future Improvements
To improve the algorithm, there are a few steps that can be taken. We can further optimize the hyperparameters of the model, including the learning rates, and the hyperparameters of the neural networks, including activation layers and overall architecture.

Additionally, choosing different or more macroeconomic indicators could improve test performance, given the size of the actor and critic networks are >1M parameters each and should be able to support a larger dataset without overfitting. 


### References
DDPG Paper: Lillicrap et. al, "Continuous control with deep reinforcement learning"
Link: https://arxiv.org/abs/1509.02971

This code borrows from the Youtube video "Everything You Need to Know About Deep Deterministic Policy Gradients (DDPG) | Tensorflow 2 Tutorial" from Machine Learning with Phil
Link: https://www.youtube.com/watch?v=4jh32CvwKYw