import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

''' Evaluation plot

This plot evaluates the strategy on the backtest data. Three plots are made: (1) a cumulative performance 
plot, (2) a net returns plot and (3) a transactions plot. Plot 1 shows the cumulative profit made on the 
backtest matches. Plot 2 shows the profit per match. Plot 3 shows the bets placed on the matches.

'''

def evaluate(x):
    
    # define plotting data
    flow = np.cumsum(x.winnings - x.stake)
    profit = x.winnings-x.stake
    size = range(len(flow))
    zero = np.zeros(len(flow))

    # define figures
    fig = plt.figure(figsize=(24, 12)) 
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1]) 

    # plot cumulative profit
    ax0 = plt.subplot(gs[0])
    ax0.grid(color='grey', linestyle='-', linewidth=1, alpha=0.5)
    ax0.plot(size, flow, '#4572a7', linewidth = 2)
    ax0.plot(size, zero, '#000000', linewidth = 1)
    ax0.fill_between(size, flow, zero, where=zero >= flow, facecolor='#dbe3ee', interpolate=True)
    ax0.fill_between(size, flow, zero, where=zero <= flow, facecolor='#dbe3ee', interpolate=True)
    ax0.set_title('Cumulative Performance: {}% of €{} invested'.format(
        format(((sum(x.winnings) / sum(x.stake))-1)*100, '.2f'),
        format(sum(x.stake), '.2f')), loc='left', fontsize=15)

    # plot returns
    ax1 = plt.subplot(gs[1])
    ax1.grid(color='grey', linestyle='-', linewidth=1, alpha=0.5)
    ax1.set_ylim((min(profit), max(profit)))
    ax1.bar(size, np.clip(profit, 0, max(profit)), 2, color='#00998f')
    ax1.bar(size, np.clip(profit, min(profit), 0), 2, color='#f66a83')
    ax1.plot(size, zero, color='#000000', linewidth=1)
    ax1.set_title('Returns: {} wins / {} losses'.format(
        format(np.sum(profit > 0)),
        format(np.sum(profit < 0))), loc='left', fontsize=15)

    # plot transactions
    ax2 = plt.subplot(gs[2])
    ax2.grid(color='grey', linestyle='-', linewidth=1, alpha=0.5)
    ax2.bar(size, x.stake, 2, color='#4572a7')
    ax2.set_title('Transactions', loc='left', fontsize=15)
    
    # save figure
    plt.savefig('../output/evaluation.png', bbox_inches='tight')

# load backtest results
results = pd.read_csv('../../data/backtest_results.csv')

# evaluate results
evaluate(results)
