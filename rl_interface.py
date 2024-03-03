########################################
### Reinforcement Learning Interface ###
########################################

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import plotly.express as px
import plotly.graph_objects as go

class RLInterface(object):
    """Reinforcement learning interface for handling interactions between the agent and evnironment
    """
    def __init__(self, train_env, test_env, agent):
        """Initialize the interface given the environment and agent"""
        
        self.train_env = train_env
        self.test_env = test_env
        self.agent = agent


    def train_agent(self, num_episodes, verbose=1, seed=None, advanced_stats=False):
        """TRAIN agent using experiences from the environment"""

        self._reset_train_history()
        for episode in tqdm(range(num_episodes), desc='Training progress'):
            state, info = self.train_env.reset(seed=seed)
            done = False
            epi_rewards = []
            step = 0
            while not done:
                action = self.predict_action(state, use_noise=True)
                next_state, reward, done, info = self.train_env.step(action)
                if advanced_stats:
                    self._update_train_step_history(state, action, reward, next_state)
                self.agent.buffer.record(state, action, reward, next_state, done)
                self.agent.learn()
                epi_rewards += [reward]
                state = next_state
                if verbose == 2:
                    print('Episode: ', episode+1, 'Step ',  step+1, 'Score: %.4f' % np.mean(epi_rewards),
                          '10 epi avg.: %.4f' % np.mean(self.scores[-10:]))
                step += 1
            self.train_avg_epi_scores.append(np.mean(epi_rewards))
            self.train_reward_history.append(epi_rewards)
            if verbose > 0: 
                print('Episode: {}/{}, Score: {:.4f}, 10 epi avg.: {:.4f}, All-time avg.: {:.4f}'.format(
                      episode + 1, num_episodes, np.mean(epi_rewards), np.mean(self.train_avg_epi_scores),
                      np.mean(self.train_avg_epi_scores)))
            if advanced_stats:
                self._update_train_advanced_stats()
        history = self._compile_train_history(advanced_stats)
        return history
    

    def test_agent(self, num_episodes, verbose=1, seed=None, use_noise=True, advanced_stats=False):
        """TEST agent using experiences from the environment"""

        # Initialize History
        self._reset_test_history()
        for episode in tqdm(range(num_episodes), desc='Testing progress'):
            state, info = self.test_env.reset(seed=seed)
            done = False
            epi_rewards = []
            step = 0
            while not done:
                action = self.predict_action(state, use_noise=use_noise)
                next_state, reward, done, info = self.test_env.step(action)
                # Update Event History
                self._update_test_step_history(state, action, reward, next_state)
                epi_rewards += [reward]
                state = next_state
                if verbose == 2:
                    print('Episode: ', episode+1, 'Step ',  step+1, 'Score: %.4f' % np.mean(epi_rewards),
                          '10 epi avg.: %.4f' % np.mean(self.train_avg_epi_scores[-10:]))
                step += 1
            self.test_avg_epi_scores.append(np.mean(epi_rewards))
            self.test_reward_history.append(epi_rewards)
            if verbose > 0: 
                print('Episode: {}/{}, Score: {:.4f}, 10 epi avg.: {:.4f}, All-time avg.: {:.4f}'.format(
                      episode + 1, num_episodes, np.mean(epi_rewards), np.mean(self.test_avg_epi_scores), 
                      np.mean(self.test_avg_epi_scores)))
            # Update Scores & Statistics
            if advanced_stats:
                self._update_test_advanced_stats()
            history = self._compile_test_history(advanced_stats=advanced_stats)
        return history
    

    def predict_action(self, state, use_noise=True):
        """Make inference on actor neural network and return action"""

        return self.agent.get_action(state, use_noise=use_noise)
    

    def plot_learning(self, scores, rolling_period=10):
        """Plot the learning curve of RL algorithm using Plotly"""

        rolling_avg = pd.Series(scores).rolling(window=rolling_period).mean()
        cumulative_avg = pd.Series(scores).expanding().mean()
        std_dev = pd.Series(scores).rolling(window=rolling_period).std()
        upper_bound = rolling_avg + std_dev
        lower_bound = rolling_avg - std_dev

        df = pd.DataFrame({'Episode': range(len(scores)),
                        'Episode Reward': scores,
                        f'{rolling_period}-Episode Rolling Avg.': rolling_avg,
                        f'Cumulative Rolling Avg.': cumulative_avg,
                        'Upper Bound': upper_bound,
                        'Lower Bound': lower_bound})
        
        fig = px.line(df, x='Episode', y=['Episode Reward', f'{rolling_period}-Episode Rolling Avg.', 'Cumulative Rolling Avg.'],
                    labels={'Episode Reward': 'Episode Reward', 'value': 'Value'},
                    title='Learning Curve')
        
        fig.add_trace(go.Scatter(x=df['Episode'].tolist() + df['Episode'].tolist()[::-1],
                                y=df['Upper Bound'].tolist() + df['Lower Bound'].tolist()[::-1],
                                fill='toself',
                                fillcolor='rgba(0, 0, 255, 0.2)',
                                line=dict(color='rgba(255, 255, 255, 0)'),
                                name='Rolling Avg ± Std. Dev.',
                                showlegend=True))
        fig.show()


    def save_models(self):
        """Save the agent's neural network parameters to disk"""

        self.agent.save_models()

    def load_models(self):
        """Load the agent's neural network parameters from disk"""

        self.agent.load_models()


    ## Private Methods
    def _reset_train_history(self):
        """Reset history for training episodes"""

        self.train_avg_epi_scores = []
        self.train_state_history = []
        self.train_next_state_history = []
        self.train_action_history = []
        self.train_reward_history = []
        self.train_return_history = []
        self.train_excess_returns_history = []
        self.train_volatility_history = []
        self.train_relative_volatility_history = []
        self.train_beta_history = []
        self.train_correlation_history = []
        self.train_alpha_history = []


    def _reset_test_history(self):
        """Reset history for test episodes"""

        self.test_avg_epi_scores = []
        self.test_state_history = []
        self.test_next_state_history = []
        self.test_action_history = []
        self.test_reward_history = []
        self.test_return_history = []
        self.test_excess_returns_history = []
        self.test_volatility_history = []
        self.test_relative_volatility_history = []
        self.test_beta_history = []
        self.test_correlation_history = []
        self.test_alpha_history = []


    def _update_train_step_history(self, state, action, reward, next_state):
        """Update step history with new experience"""

        self.train_state_history += [state]
        self.train_action_history += [action]
        self.train_reward_history += [reward]
        self.train_next_state_history += [next_state]


    def _update_test_step_history(self, state, action, reward, next_state):
        """Update step history with new experience"""

        self.test_state_history += [state]
        self.test_action_history += [action]
        self.test_reward_history += [reward]
        self.test_next_state_history += [next_state]


    def _update_train_advanced_stats(self):
        """Update performance metrics for training episodes"""

        cumulative_portfolio_returns = np.cumprod(1 + self.train_env.portfolio_returns, axis=0)
        cumulative_benchmark_returns = np.cumprod(1 + self.train_env.benchmark_returns, axis=0)
        self.train_return_history += [cumulative_portfolio_returns[-1]]
        self.train_excess_returns_history += [cumulative_portfolio_returns[-1] - cumulative_benchmark_returns[-1]]
        self.train_volatility_history += [np.std(self.train_env.portfolio_returns)*np.sqrt(252)]
        self.train_relative_volatility_history += [np.std(self.train_env.portfolio_returns) / np.std(self.train_env.benchmark_returns)]
        beta = np.cov(np.squeeze(self.train_env.portfolio_returns), 
                        np.squeeze(self.train_env.benchmark_returns))[0,1] / np.var(self.train_env.benchmark_returns)
        self.train_beta_history += [beta]
        correlation = np.corrcoef(np.squeeze(self.train_env.portfolio_returns), np.squeeze(self.train_env.benchmark_returns))[0,1]
        self.train_correlation_history += [correlation]
        self.train_alpha_history += [np.mean(self.train_env.portfolio_returns) - beta*np.mean(self.train_env.benchmark_returns)]
    

    def _update_test_advanced_stats(self):
        """Update performance metrics for test episodes"""

        cumulative_portfolio_returns = np.cumprod(1 + self.train_env.portfolio_returns, axis=0)
        cumulative_benchmark_returns = np.cumprod(1 + self.train_env.benchmark_returns, axis=0)
        self.test_return_history += [cumulative_portfolio_returns[-1]]
        self.test_excess_returns_history += [cumulative_portfolio_returns[-1] - cumulative_benchmark_returns[-1]]
        self.test_volatility_history += [np.std(self.test_env.portfolio_returns)*np.sqrt(252)]
        self.test_relative_volatility_history += [np.std(self.test_env.portfolio_returns) / np.std(self.test_env.benchmark_returns)]
        beta = np.cov(np.squeeze(self.test_env.portfolio_returns), 
                        np.squeeze(self.test_env.benchmark_returns))[0,1] / np.var(self.test_env.benchmark_returns)
        self.test_beta_history += [beta]
        correlation = np.corrcoef(np.squeeze(self.test_env.portfolio_returns), np.squeeze(self.test_env.benchmark_returns))[0,1]
        self.test_correlation_history += [correlation]
        self.test_alpha_history += [np.mean(self.test_env.portfolio_returns) - beta*np.mean(self.test_env.benchmark_returns)*252]


    def _compile_train_history(self, advanced_stats):
        """Compile train history into dictionary for easy access and storage"""

        history = {}
        history['avg_epi_score_history'] = self.train_avg_epi_scores
        if advanced_stats:
            history['state_history'] = self.train_state_history
            history['next_state_history'] = self.train_next_state_history
            history['action_history'] = self.train_action_history
            history['reward_history'] = self.train_reward_history
            history['returns_history'] = self.train_return_history
            history['excess_returns_history'] = self.train_excess_returns_history
            history['volatility_history'] = self.train_volatility_history
            history['relative_volatility_history'] = self.train_relative_volatility_history
            history['beta_history'] = self.train_beta_history
            history['correlation_history'] = self.train_correlation_history
            history['alpha_history'] = self.train_alpha_history
        return history
    

    def _compile_test_history(self, advanced_stats):
        """Compile test history into dictionary for easy access and storage"""

        history = {}
        history['avg_epi_score_history'] = self.test_avg_epi_scores
        if advanced_stats:
            history['state_history'] = self.test_state_history
            history['next_state_history'] = self.test_next_state_history
            history['action_history'] = self.test_action_history
            history['reward_history'] = self.test_reward_history
            history['returns_history'] = self.test_return_history
            history['excess_returns_history'] = self.test_excess_returns_history
            history['volatility_history'] = self.test_volatility_history
            history['relative_volatility_history'] = self.test_relative_volatility_history
            history['beta_history'] = self.test_beta_history
            history['correlation_history'] = self.test_correlation_history
            history['alpha_history'] = self.test_alpha_history
        return history
    

    def print_advanced_stats(self, history):
        """Print advanced statistics from training history"""

        print('Avg. Reward: {:.3f}'.format(np.mean(history['avg_epi_score_history'])))
        print('Avg. Volatility (Agent, Ann.): {:.3f}%'.format(np.mean(history['volatility_history'])*100))
        print('Avg. Risk Utilization (% of benchmark): {:.3f}%'.format(np.mean(history['relative_volatility_history'])*100))
        print('Avg. Beta: {:.3f}'.format(np.mean(history['beta_history'])))
        print('Avg. Correlation: {:.3f}'.format(np.mean(history['correlation_history'])))
        print('Avg. Return (Epi.): {:.3f}%'.format(np.mean(history['returns_history'])*100))
        print('Avg. Excess Return (Epi.): {:.3f}%'.format(np.mean(history['excess_returns_history'])*100))
        print('Avg. Alpha (Ann.): {:.3f}%'.format(np.mean(history['alpha_history'])*100))