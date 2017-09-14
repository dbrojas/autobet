import numpy as np
import pandas as pd

''' Betting strategy #1:

## STRATEGY ##

This betting strategy compares the confidence of soccer match outcomes from 8 different betting houses 
with a machine learning model. We select the betting house who underestimates the winning team performance
the most. A bet is placed whose size is proportional to the magnitude of the underestimation.

## EXAMPLE ##

Real Madrid vs. Levante
B365 odds: 1.45 / 3.40 / 6.15
BWIN odds: 1.35 / 4.10 / 5.90
ML odds:   1.10 / 3.90 / 7.15

All predictors think Real Madrid is going to win. However, the ML model is more certain than the houses. In
this case, B365 underestimates Real Madrid the most. The algorithm selects this betting house and bets
(1.45/1.1+risk_adj)^2 = $1.32 on Real Madrid (1.45), where risk_adj is a tunable parameter to control the 
amount of risk.
'''

def strategy_1(x, true, preds, risk_adj):
    
    # function to determine magnitude of underestimation
    def _magnitude(odds, preds, risk_adj):
        x = np.array([]).reshape(len(odds), 0)
        for house in set([h[:-1] for h in odds.columns]):
            filter_col = [col for col in odds.columns if col.startswith(house)]
            x = np.concatenate([x, (np.array(odds[filter_col]) / (np.array(preds) + risk_adj))**2], axis=1)
        return pd.DataFrame(x, columns=odds.columns)

    # function to select optimal betting house
    def _selection(x, preds, true):
        idxs = [[np.argmin(p) + (3 * i) for i in range(8)] for p in preds.values]
        action = []
        for i in range(len(x)):
            best = np.max(x.iloc[i, idxs[i]])
            if best > 1:
                action.append((np.argmax(x.iloc[i, idxs[i]]), best, idxs[i][0]))
            else:
                action.append((np.nan, np.nan, np.nan))
        res = pd.concat([pd.DataFrame(action), pd.Series(true)], axis=1)
        res.columns = ['house', 'stake', 'pred', 'true']
        return res
    
    # apply strategy 1
    results = _selection(_magnitude(x, preds, risk_adj), preds, true)

    # get the odds for the winning bets
    win_odds = []
    for res in results[results.pred == results.true].reset_index().values:
        win_odds.append(x.loc[res[0], res[1]])
    win_odds = pd.DataFrame({'index': results[results.pred == results.true].reset_index()['index'],
                             'odd': pd.Series(win_odds)})
    
    # merge winning odds and payoff with results
    results = pd.merge(results.reset_index(), win_odds, how='outer', on='index')
    results = results.fillna(0)
    results = pd.concat([results, results.stake * results.odd], axis=1)
    results.columns = ['match', 'house', 'stake', 'pred', 'true', 'odd', 'winnings']
    
    return results

# load backtest data
x_bt = pd.read_csv('../../data/stack_backtest.csv')
y_bt = x_bt.target
x_bt = x_bt.drop(['target'], axis=1)

# run backtest
odds = x_bt.iloc[:, :24]
preds = x_bt.iloc[:,-3:]
results = strategy_1(odds, y_bt, preds, risk_adj=0.9)
results.to_csv('../../data/backtest_results.csv', index=False)
