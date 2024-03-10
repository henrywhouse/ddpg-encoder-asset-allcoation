######################################
### DDPG Allocation Render Methods ###
######################################

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def _plot_reward(action_dates, rewards, benchmark_rewards, reward_type, holding_period, plot_benchmark=False):
        """Plot a line graph of rolling sharpe ratios for holding period compared to benchmark"""
        
        if reward_type=='sharpe':
           fig_title = f'{holding_period}-Day Rolling Sharpe Ratio'
           y_label = 'Rolling Sharpe Ratio'
        if reward_type=='market_sharpe':
            fig_title = f'{holding_period}-Day Rolling Beta-Adjusted Sharpe Ratio'
            y_label = 'Beta-Adjusted Sharpe Ratio'
        else:
            fig_title = 'Agent Reward'
            y_label = 'Reward'
        traces = []
        ## Agent
        trace = go.Scatter(x=action_dates,  # Assuming self.dates is a list or array of corresponding dates
                           y=rewards,
                           mode='lines',
                           name='Agent')
        traces.append(trace)
        agent_average = np.mean(rewards)  # Plot average of AGENT Sharpe ratio
        ## Benchmark
        if plot_benchmark:
            trace = go.Scatter(x=action_dates,  # Assuming self.dates is a list or array of corresponding dates
                               y=benchmark_rewards,
                               mode='lines',
                               name='Benchmark')
            traces.append(trace)
        ## Agent Average 
        trace = go.Scatter(x=action_dates,
                       y=[agent_average] * len(action_dates),
                       mode='lines',
                       name='Agent Average',
                       line=dict(dash='dash'))
        traces.append(trace)
        ## Benchmark Average
        if plot_benchmark:
            benchmark_average = np.mean(benchmark_rewards)  # Plot average of BENCHMARK Sharpe ratio
            trace = go.Scatter(x=action_dates,
                           y=[benchmark_average] * len(action_dates),
                           mode='lines',
                           name='Benchmark Average',
                           line=dict(dash='dash'))
            traces.append(trace)
        fig = go.Figure(data=traces)
        fig.update_layout(title=fig_title,
                          xaxis_title='Date',
                          yaxis_title=y_label,
                          showlegend=True)
        fig.show()


def _plot_actions(action_dates, actions, benchmark_actions, asset_names, plot_relative=False):
    """Plot a line graph of asset allocations over the episode period"""

    traces = []
    trace = go.Scatter(x=action_dates,
                            y=actions[:,0],
                            name=asset_names[0],
                            stackgroup=1,
                            fill='tozeroy')
    traces.append(trace)
    for i in range(1, actions.shape[1]):
        # accumulated_actions = self.actions[:, :i+1].sum(axis=1)
        trace = go.Scatter(x=action_dates,
                            y=actions[:,i],
                            name=asset_names[i],
                            stackgroup=1,
                            fill='tonexty')
        traces.append(trace)
    fig = go.Figure(data=traces)
    fig.update_layout(title='Asset Allocation Over Time',
                        xaxis_title='Date',
                        yaxis_title='Allocation')
    fig.update_layout(yaxis=dict(tickformat='0%'))
    fig.show()

    if plot_relative:
        # Plot relative weight of strategy to benchmark
        num_assets = len(asset_names)
        rows = num_assets//2
        fig = make_subplots(rows=rows, cols=2, subplot_titles=asset_names, shared_xaxes=True,
                           vertical_spacing=0.05)
        # Iterate through assets and add traces to subplots
        for i in range(num_assets):
            row = i // 2 + 1
            col = i % 2 + 1
            relative_weight = actions[:,i] - benchmark_actions[:,i]
            colors = ['red' if y < 0 else 'green' for y in relative_weight]
            fig.add_trace(go.Scatter(x=action_dates, y=relative_weight, name="attribution", line=dict(color='black')),
                          row=row, col=col)
            fig.add_trace(go.Scatter(x=action_dates, y=relative_weight, mode='markers', marker=dict(color=colors, size=10)),
                          row=row, col=col)
        fig.update_traces(showlegend=False)
        fig.update_layout(title='Strategy Weight Relative to Benchmark', xaxis_title='Date', yaxis_title='Relative Weight', height=200*rows)
        fig.update_layout(showlegend=False)
        _change_yaxis_to_percent(fig, rows, columns=2, decimal_places=2)
        fig.show()


def _plot_returns(window_dates, portfolio_cumulative_returns, benchmark_cumulative_returns, 
                  portfolio_cumulative_individual_returns, benchmark_cumulative_individual_returns,
                  asset_names, num_assets, plot_benchmark=False, plot_individual=False, plot_contribution=False):
    """Plot cumulative returns for the agent strategy compared to the benchmark"""
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=window_dates, y=np.squeeze(portfolio_cumulative_returns), mode='lines', name='Agent'))
    if plot_benchmark:
        fig.add_trace(go.Scatter(x=window_dates, y=np.squeeze(benchmark_cumulative_returns), mode='lines', name='Benchmark'))
    fig.update_layout(title='Cumulative Portfolio Returns', xaxis_title='Date', yaxis_title='Returns (%)')
    fig.update_layout(showlegend=True, yaxis=dict(tickformat='0%'))
    fig.show()

    if plot_individual:
        # Plot individual asset returns
        strategy_legend_labels = []
        benchmark_legend_labels = []
        rows = num_assets // 2
        fig = make_subplots(rows=rows, cols=2, subplot_titles=asset_names, shared_xaxes=True,
                           vertical_spacing=0.05)
        colors = ['blue', 'red']
        # Iterate through assets and add traces to subplots
        for i in range(num_assets):
            row = i // 2 + 1
            col = i % 2 + 1
            fig.add_trace(go.Scatter(x=window_dates, y=portfolio_cumulative_individual_returns[:,i], name='Strategy',
                                     line=dict(color=colors[0])), row=row, col=col)
            strategy_legend_labels.append("Strategy")
            fig.add_trace(go.Scatter(x=window_dates, y=benchmark_cumulative_individual_returns[:,i], name='Benchmark',
                                     line=dict(color=colors[1])), row=row, col=col)
            benchmark_legend_labels.append("Benchmark")
        fig.update_layout(showlegend=False)
        _change_yaxis_to_percent(fig, rows, columns=2, decimal_places=2)
        fig.update_layout(title=' Weighted Cumulative Individual Returns', yaxis_title='Returns (%)', height=200*rows)
        fig.show()

    if plot_contribution:
        # Plot contribution of each asset to portfolio returns
        rows = num_assets // 2
        asset_relative_contribution = portfolio_cumulative_individual_returns - benchmark_cumulative_individual_returns
        rel_con_percentages = [int(np.round(100 * np.abs(asset_relative_contribution[-1, i])/np.sum(np.abs(asset_relative_contribution[-1])), \
                                    decimals=0)) for i in range(num_assets)]
        subplot_titles = [f'{asset_names[i]} - Contribution Proportion: {rel_con_percentages[i]}%' for i in range(num_assets)]
        fig = make_subplots(rows=rows, cols=2, subplot_titles=subplot_titles, shared_xaxes=True,
                           vertical_spacing=0.05)
        # Iterate through assets and add traces to subplots
        for i in range(num_assets):
            row = i // 2 + 1
            col = i % 2 + 1
            contribution = asset_relative_contribution[:,i]
            colors = ['red' if val < 0 else 'green' for val in contribution]
            fig.add_trace(go.Scatter(x=window_dates, y=contribution, name=asset_names[i],
                          line=dict(color='black')), row=row, col=col)
            fig.add_trace(go.Scatter(x=window_dates, y=contribution, mode='markers', name=asset_names[i],
                                     marker=dict(color=colors, size=10)), row=row, col=col)
        fig.update_layout(title='Contribution by Asset', xaxis_title='Date', yaxis_title='Returns (%)', height=225*rows)
        fig.update_layout(showlegend=False)
        _change_yaxis_to_percent(fig, rows, columns=2, decimal_places=1)
        fig.show()


def _change_yaxis_to_percent(fig, rows, columns, decimal_places=0):
    """Change y-axis to percentage format"""
    
    for i in range(rows):
        for j in range(columns):
            tickformat = f'0.{decimal_places}%'
            fig.update_yaxes(tickformat=tickformat, row=i+1, col=j+1)
    