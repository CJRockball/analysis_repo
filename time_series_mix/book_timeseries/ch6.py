#%%
# import dependencies

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PythonTsa.datadir import getdtapath
from PythonTsa.plot_acf_pacf import acf_pacf_fig
from PythonTsa.LjungBoxtest import plot_LB_pvalue
from statsmodels.graphics.tsaplots import plot_predict

from PythonTsa.Selecting_arma2 import choose_arma2
from scipy import stats
from scipy.stats import norm

from PythonTsa.ModResidDiag import plot_ResidDiag

import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

dtapath = getdtapath()

#%%

pgret = pd.read_csv(dtapath + 'monthly returns of Procter n Gamble stock n 3 market indexes 1961 to 2016.csv',
                    header=0)

display(pgret)

# %% Set up series

pgret = pgret['RET']
dates = pd.date_range('1960-01', periods=len(pgret), freq='ME')
pgret.index = dates

# Values are too small affects convergence
pgret = 100*pgret
pgret.plot(title='PnG returns')


# %% Plot 

sm.tsa.kpss(pgret, regression='c', nlags='auto')
plot_LB_pvalue(pgret, noestimatedcoef=0, nolags=36)
plt.show()
acf_pacf_fig(pgret**2, lag=48)
plt.show()
plot_LB_pvalue(pgret**2, noestimatedcoef=0, nolags=36)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
hfig = ax.hist(pgret, bins=40, density=True, label='Histogram')
kde = sm.nonparametric.KDEUnivariate(pgret)
kde.fit()

ax.plot(kde.support, kde.density, label='KDE')
smean = np.mean(pgret)
scal = np.std(pgret, ddof=1)
normden = norm.pdf(kde.support, loc=smean, scale=scal)
ax.plot(kde.support, normden, label='Normal density')
ax.legend(loc='best')
plt.show()


# %% 

from arch import arch_model

archmod = arch_model(pgret).fit()
print(archmod.summary())

# %%

archresid = archmod.std_resid
plot_LB_pvalue(archresid, noestimatedcoef=0, nolags=36)
plt.show()
plot_LB_pvalue(archresid**2, noestimatedcoef=0, nolags=36)
plt.show()

garchT = arch_model(pgret, p=1, q=1, dist='StudentsT')
res = garchT.fit()
print(res.summary())

# %%
from statsmodels.graphics.api import qqplot

archresidT = res.std_resid
plot_LB_pvalue(archresidT, noestimatedcoef=0, nolags=36)
plt.show()
plot_LB_pvalue(archresidT, noestimatedcoef=0, nolags=36)
plt.show()

qqplot(archresidT, stats.t, distargs=(9.62), line='q', fit=True)
plt.show()

# %%
