# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 10:43:39 2024

@author: ken.t
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime
from math import *
import scipy
import statsmodels.api as sm
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from scipy.optimize import minimize
from scipy.integrate import quad

#identify tickers
stocks = ['TM', 'TSLA', 'GM', 'RACE', 'TTM', 'F', 'HMC', 'VWAGY']

#start and end dates
start = datetime.datetime(2019,1,29)
end = datetime.datetime(2020,1,30)

#use yahoo finance api for historical data
panel_data = web.DataReader(stocks, 'yahoo',start,end)
close = panel_data['Close']

#percent change over time
portfolio_returns = close.pct_change()

#empty list
sr = []

for sharpe_ratio in range(1000):
    weights = np.random.rand(10)
    portfolio_weights = weights/sum(weights)
    weighted_p = portfolio_returns * portfolio_weights
    indi_expected_returns = weighted_p.mean()
    expected_return = indi_expected_returns.sum()
    cov_mat_annual = portfolio_returns.cov() * 252
    portfolio_volatility = (np.dot(portfolio_weights.T, np.dot(cov_mat_annual, portfolio_weights)))
    sharpe_ratio = ((expected_return)/(portfolio_volatility)) * sqrt(252)
    sr.append(sharpe_ratio)

w = np.array(sr).reshape(-1,1)
params = {'bandwidth': np.linspace(.01, 1, 100)}
grid = GridSearchCV(KernelDensity(), params)
grid.fit(w)
cross = grid.best_estimator_.bandwidth

plt.style.use('dark_background')
plt.title("Random Portfolio Sharpe Ratio Distribution")
plt.hist(sr,bins = 100)
plt.show()

kde = sm.nonparametric.KDEUnivariate(sr)
kde.fit(bw = cross)
plt.title('PDF')
plt.plot(kde.support, kde.density, label = f 'Cross')
plt.legend()
plt.show()
print('\n', 'Cross Validation Bandwidth:',cross)

num_bins = 100
counts, egd = np.histogram (sr, bins=num_bins, normed=True)
cdf = np.cumsum (counts)

plt.title('CDF')
plt.plot (egd[1:], cdf/cdf[-1])
plt.show()

plt.title('Quantile')
plt.plot ( cdf/cdf[-1], egd[1:])
plt.show()

format_sr =  [round(x,2) for x in sr]

frst = np.percentile(format_sr,1)
print('\n', '1% Percentile:',frst, 'Sharpe Ratio')
     
twfv = np.percentile(format_sr,25)
print('\n', '25% Percentile:',twfv, 'Sharpe Ratio')

fift = np.percentile(format_sr,50)
print('\n', '50% Percentile:',fift, 'Sharpe Ratio')

svfv = np.percentile(format_sr,75)
print('\n', '75% Percentile:',svfv, 'Sharpe Ratio')

nnty = np.percentile(format_sr,99)
print('\n', '99% Percentile:',nnty, 'Sharpe Ratio')