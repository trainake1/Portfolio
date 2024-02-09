## Sharpe Ratio Calculation
Created by William Sharpe in 1994 

Installations

    pip install pandas
    pip install numpy
    pip install matplotlib
    pip install datetime
    pip install scipy

Once the modules are installed, we begin the code

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


Once we have imported the necessary modules, we can begin to construct the desired portfolio

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


Now we can generate the portfolio distribution of weights, and identify the sharpe ratios for each portfolio.

    #for loop for portfolio construction (you may want to construct 10,000 for the sake of smoothing the histogram -for the sake of this instruction, and time, we will only use 1,000

    for sharpe_ratio in range(100):

        #create random weights
        weights = np.random.rand(8)
        portfolio_weights = weights/sum(weights)

        #assign weights
        weighted_p = portfolio_returns * portfolio_weights

        #begin sharpe ratio calculation
        indi_expected_returns = weighted_p.mean()
        expected_return = indi_expected_returns.sum()
        cov_mat_annual = portfolio_returns.cov() * 252
        portfolio_volatility = (np.dot(portfolio_weights.T, np.dot(cov_mat_annual, portfolio_weights)))
        sharpe_ratio = ((expected_return)/(portfolio_volatility)) * sqrt(252)

        #filter ratios into list
        sr.append(sharpe_ratio)



Plot histogram

    plt.style.use('dark_background')
    plt.title("Random Portfolio Sharpe Ratio Distribution")
    plt.hist(sr,bins = 100)
    plt.show()

Output:

![image](https://github.com/trainake1/Portfolio/assets/158123925/88a191d5-a0e9-487e-8add-5a021c312208)


 In order to plot our results we will use cross validation to determine the best fit

    w = np.array(sr).reshape(-1,1)
    params = {'bandwidth': np.linspace(.01, 1, 100)}
    grid = GridSearchCV(KernelDensity(), params)
    grid.fit(w)
    cross = grid.best_estimator_.bandwidth

Print Prodbaility Density Estimator

    kde = sm.nonparametric.KDEUnivariate(sr)
    kde.fit(bw = cross)
    plt.title('PDF')
    plt.plot(kde.support, kde.density, label = f 'Cross')
    plt.legend()
    plt.show()
    print('\n', 'Cross Validation Bandwidth:',cross)

Output:

![image](https://github.com/trainake1/Portfolio/assets/158123925/28ce8d9d-c433-4652-90a6-084120f4fe2a)


Cross Validation Bandwidth: 0.03

Now plot Cumulative Distribution Function

    plt.title('CDF')
    plt.plot (egd[1:], cdf/cdf[-1])
    plt.show()

Output:

![image](https://github.com/trainake1/Portfolio/assets/158123925/dbea7350-02a6-462f-9234-e4cff486d418)




And we can rework the graph  for a better depiction of the sharpe ratio distribution

    plt.title('Quantile')
    plt.plot ( cdf/cdf[-1], egd[1:])
    plt.show()

![image](https://github.com/trainake1/Portfolio/assets/158123925/a1104da2-509c-4314-bae5-8e2dd7c34a22)



Finally, we can calculate the numerical output of the data

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

Output:

1% Percentile: 0.18 Sharpe Ratio

 25% Percentile: 0.37 Sharpe Ratio

 50% Percentile: 0.45 Sharpe Ratio

 75% Percentile: 0.54 Sharpe Ratio

 99% Percentile: 0.74 Sharpe Ratio

We can see that for the selected basket of stocks, that the average portfolio will return somewhere around 0.45. This will differ based on the selected securities in step 1
