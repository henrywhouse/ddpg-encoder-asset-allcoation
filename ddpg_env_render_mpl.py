######################################
### DDPG Allocation Render Methods ###
######################################

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

def _plot_reward(action_dates, rewards, benchmark_rewards, reward_type, holding_period, plot_benchmark=False):
    """Plot a line graph of rolling sharpe ratios for holding period compared to benchmark"""

    if reward_type == 'sharpe':
        fig_title = f'{holding_period}-Day Rolling Sharpe Ratio'
        y_label = 'Rolling Sharpe Ratio'
    elif reward_type == 'market_sharpe':
        fig_title = f'{holding_period}-Day Rolling Beta-Adjusted Sharpe Ratio'
        y_label = 'Beta-Adjusted Sharpe Ratio'
    elif reward_type == 'alpha':
        fig_title = f'{holding_period}-Day Rolling Alpha'
        y_label = 'Alpha'
    else:
        fig_title = 'Agent Reward'
        y_label = 'Reward'

    plt.figure(figsize=(20, 5))
    plt.plot(action_dates, rewards, label='Agent')
    agent_average = np.mean(rewards)
    plt.axhline(y=agent_average, linestyle='--', color='gray', label='Agent Average')

    if plot_benchmark:
        plt.plot(action_dates, benchmark_rewards, label='Benchmark')
        benchmark_average = np.mean(benchmark_rewards)
        plt.axhline(y=benchmark_average, linestyle='--', color='black', label='Benchmark Average')

    plt.title(fig_title)
    plt.xlabel('Date')
    plt.ylabel(y_label)
    plt.legend()
    plt.show()

import matplotlib.pyplot as plt

def _plot_actions(action_dates, actions, benchmark_actions, asset_names, plot_relative=False):
    """Plot a line graph of asset allocations over the episode period"""

    plt.figure(figsize=(20, 5))
    cumulative_actions = actions.cumsum(axis=1)
    
    handles = []  # List to store legend handles for areas
    
    for i in range(actions.shape[1]):
        plt.plot(action_dates, cumulative_actions[:, i], label=None)  
        
    for i in range(actions.shape[1]):
        if i == 0:
            handle = plt.fill_between(action_dates, cumulative_actions[:, i], color='blue', alpha=0.3)
        else:
            handle = plt.fill_between(action_dates, cumulative_actions[:, i-1], cumulative_actions[:, i], color=f'C{i}', alpha=0.3)
        handles.append(handle)  # Append legend handle for area
        
    plt.title('Asset Allocation Over Time')
    plt.xlabel('Date')
    plt.ylabel('Allocation')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_y_axis))
    
    # Create legend with custom handles and labels for areas
    plt.legend(handles, asset_names, loc='upper left')
    
    plt.show()

    if plot_relative:
        # Plot relative weight of strategy to benchmark
        num_assets = len(asset_names)
        rows = num_assets // 2 + num_assets % 2
        fig, axs = plt.subplots(rows, 2, figsize=(15, 3 * rows), sharex=True)

        for i in range(num_assets):
            row = i // 2
            col = i % 2
            relative_weight = actions[:, i] - benchmark_actions[:, i]

            # Find the indices where the relative weight changes sign
            sign_changes = np.where(np.diff(np.sign(relative_weight)) != 0)[0]

            # Iterate through the sign change points and plot segments with different colors
            start_idx = 0
            for change_idx in sign_changes:
                color = 'green' if relative_weight[change_idx] >= 0 else 'red'
                axs[row, col].plot(action_dates[start_idx:change_idx+2], relative_weight[start_idx:change_idx+2], color=color)
                start_idx = change_idx + 1

            # Plot the remaining segment
            color = 'green' if relative_weight[-1] >= 0 else 'red'
            axs[row, col].plot(action_dates[start_idx:], relative_weight[start_idx:], color=color)

            # Fill area between lines
            axs[row, col].fill_between(action_dates, 0, relative_weight, where=(relative_weight >= 0), color='lightgreen', interpolate=True)
            axs[row, col].fill_between(action_dates, 0, relative_weight, where=(relative_weight < 0), color='lightcoral', interpolate=True)

            axs[row, col].set_title(asset_names[i])
            axs[row, col].set_ylabel('Relative Weight')
            axs[row, col].yaxis.set_major_formatter(plt.FuncFormatter(format_y_axis))

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.suptitle('Relative Weight of Portfolio Holdings to Benchmark')
        plt.show()




def _plot_returns(window_dates, portfolio_cumulative_returns, benchmark_cumulative_returns,
                  portfolio_cumulative_individual_returns, benchmark_cumulative_individual_returns,
                  asset_names, num_assets, plot_benchmark=False, plot_individual=False, plot_contribution=False):
    """Plot cumulative returns for the agent strategy compared to the benchmark"""

    plt.figure(figsize=(20, 5))
    plt.plot(window_dates, np.squeeze(portfolio_cumulative_returns), label='Agent')
    if plot_benchmark:
        plt.plot(window_dates, np.squeeze(benchmark_cumulative_returns), label='Benchmark')

    plt.title('Cumulative Portfolio Returns')
    plt.xlabel('Date')
    plt.ylabel('Return (%)')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    plt.legend(loc='upper left')
    plt.show()

    if plot_individual:
        fig, axs = plt.subplots(num_assets // 2 + num_assets % 2, 2, figsize=(15, 3 * (num_assets // 2 + 1)), sharex=True)

        handles = []  # Collect handles for legend
        labels = []   # Collect labels for legend

        for i in range(num_assets):
            row = i // 2
            col = i % 2
            portfolio_line, = axs[row, col].plot(window_dates, portfolio_cumulative_individual_returns[:, i], color='blue')
            benchmark_line, = axs[row, col].plot(window_dates, benchmark_cumulative_individual_returns[:, i], color='red')
            axs[row, col].set_title(asset_names[i])
            axs[row, col].set_ylabel('Returns (%)')
            axs[row, col].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2%}'))

            if i == 0:
                handles.extend([portfolio_line, benchmark_line])
                labels.extend(['Portfolio', 'Benchmark'])
        
        fig.legend(handles, labels, loc='upper left')
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.suptitle('Portfolio Holdings and Benchmarks Contribution to Returns Over Time')
        plt.show()

    if plot_contribution:
        fig, axs = plt.subplots(num_assets // 2 + num_assets % 2, 2, figsize=(15, 3 * (num_assets // 2 + 1)), sharex=True)

        for i in range(num_assets):
            row = i // 2
            col = i % 2
            contribution = portfolio_cumulative_individual_returns[:, i] - benchmark_cumulative_individual_returns[:, i]

            sign_changes = np.where(np.diff(np.sign(contribution)) != 0)[0]

            start_idx = 0
            for change_idx in sign_changes:
                color = 'green' if contribution[change_idx] >= 0 else 'red'
                axs[row, col].plot(window_dates[start_idx:change_idx+2], contribution[start_idx:change_idx+2], color=color)
                start_idx = change_idx + 1

            # Plot the remaining segment
            color = 'green' if contribution[-1] >= 0 else 'red'
            axs[row, col].plot(window_dates[start_idx:], contribution[start_idx:], color=color)

            # Fill area between lines
            axs[row, col].fill_between(window_dates, 0, contribution, where=(contribution >= 0), color='lightgreen', interpolate=True)
            axs[row, col].fill_between(window_dates, 0, contribution, where=(contribution < 0), color='lightcoral', interpolate=True)

            axs[row, col].set_title(asset_names[i])
            axs[row, col].set_ylabel('Excess Return (%)')
            axs[row, col].yaxis.set_major_formatter(plt.FuncFormatter(format_y_axis))

        asset_relative_contribution = portfolio_cumulative_individual_returns - benchmark_cumulative_individual_returns
        rel_con_percentages = [int(np.round(100 * np.abs(asset_relative_contribution[-1, i])/np.sum(np.abs(asset_relative_contribution[-1])), \
                                    decimals=0)) for i in range(num_assets)]
        subplot_titles = [f'{asset_names[i]} - Contribution Proportion: {rel_con_percentages[i]}%' for i in range(num_assets)]
        
        for i, title in enumerate(subplot_titles):
            row = i // 2
            col = i % 2
            axs[row, col].set_title(title)

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.suptitle('Attribution to Returns of Portfolio Holdings vs. Benchmark Holdings')
        plt.show()


def format_y_axis(value, _, decimals=1):
    """Change y-axis to percentage format"""
    if decimals == 0:
        format = f'{value:.0%}'
    elif decimals == 1:
        format = f'{value:.1%}'
    elif decimals == 2:
        format = f'{value:.2%}'
    elif decimals == 3:
        format = f'{value:.3%}'
    return format