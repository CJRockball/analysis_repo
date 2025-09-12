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
from PythonTsa.ModResidDiag import plot_ResidDiag

import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

dtapath = getdtapath()


#%% ------------------------- Drugs Australia ---------------------------------

h02 = pd.read_csv(dtapath + 'h02July1991June2008.csv', header=0)
display(h02)

# %%
timeindex = pd.date_range(start='1991-07', periods=len(h02), freq='ME')
h02.index = timeindex
h02 = pd.Series(h02['h02'])
    
h02.plot(title='Australia Drugs')
plt.show()

# %% Differentiate data

Dh02 = sm.tsa.statespace.tools.diff(h02, k_diff=0, 
                    k_seasonal_diff=1, seasonal_periods=12)
Dh02.plot()
plt.show()

# %% Check if stationary

acf_pacf_fig(Dh02, both=True, lag=36)
plt.show()
sm.tsa.kpss(Dh02, regression='c', nlags='auto')

# %% Try model

sarima005210 = sm.tsa.SARIMAX(h02, order=(0,0,5), seasonal_order=(2,1,0,12))
sarimaMod005210 = sarima005210.fit(disp=False)
print(sarimaMod005210.summary())

#%%

resid005210 = sarimaMod005210.resid[12:]
plot_ResidDiag(resid005210, noestimatedcoef=7, nolags=48, lag=36)
plt.show()

# %% Term L4 is not significant. Fit again without the term

sarima0054210 = sm.tsa.SARIMAX(h02, order=(0,0,[1,1,1,0,1]),
                                          seasonal_order=(2,1,0,12))
sarimaMod0054210 = sarima0054210.fit(disp=False)
print(sarimaMod0054210.summary())

resid0054210 = sarimaMod0054210.resid[12:]
plot_ResidDiag(resid0054210, noestimatedcoef=6, nolags=48, lag=36)
plt.show()

# %% ---------------------------- Global Temperature --------------------------------------

gtem = pd.read_csv(dtapath + 'GlobalTemperature.txt', header=None, sep='\s+')
gtemts = pd.concat([gtem.loc[:,0], gtem.loc[:,1]], ignore_index='true')
for i in range(2,gtem.shape[1]):
    gtemts = pd.concat([gtemts, gtem.loc[:,i]], ignore_index='true')

timeindex = pd.date_range(start='1856-01', periods=len(gtemts), freq='ME')
gtemts.index = timeindex

gtemts.plot(title='Monthly Temperature')
plt.show()

ygtemts = gtemts.resample(rule='12ME', kind='period').mean()
ytimeindex = pd.date_range('1856', periods=len(ygtemts), freq='YE')
ygtemts.index = ytimeindex

ygtemts.plot(title='Yearly temperature')
plt.show()

# %%

temp = gtemts['1970-01':'2005-12']
COS = np.zeros((len(temp), 6))
SIN = np.zeros((len(temp), 6))
# tim is a time variable
tim = np.zeros((len(temp)))
for i in range(36):
    for j in range(12):
        tim[i*12+j] = 1970.0 + i + j/12



# %% Fit a fourier seriers with ols
import math

_pi = math.pi
for i in range(6):
    COS[:,i] = np.cos(2*_pi*(i+1)*tim)
    SIN[:,i] = np.sin(2*_pi*(i+1)*tim)

# TIME is standardized tim for reducing computation error
TIME = (tim - np.mean(tim)) / np.sqrt(np.var(tim))

Z = np.column_stack((TIME, COS[:,0], SIN[:,0], COS[:,1], SIN[:,1],
                     COS[:,2], SIN[:,2], COS[:,3], SIN[:,3],
                     COS[:,4], SIN[:,4], COS[:,5], SIN[:,5]))
Z = sm.add_constant(Z)

OLSmod = sm.OLS(temp,Z).fit()
print(OLSmod.summary())

# %% Fit a smaller series

X = np.column_stack((TIME, SIN[:,0], SIN[:,1]))
X = sm.add_constant(X)
OLSmodel1 = sm.OLS(temp, X).fit()
print(OLSmodel1.summary())

# %% Check residuals

OLSResid = OLSmodel1.resid
acf_pacf_fig(OLSResid, both=True, lag=36)
plt.show()

sm.tsa.kpss(OLSResid, regression='c', nlags='auto')

# %% Model residuals as AR(2)

ar2 = ARIMA(OLSResid, order=(2,0,0), trend='n').fit()
print(ar2.summary())

# %% Check residuals of AR fit of residuals
from PythonTsa.LjungBoxtest import plot_LB_pvalue

ar2Resid = ar2.resid
acf_pacf_fig(ar2Resid, both=True, lag=36)
plt.show()
plot_LB_pvalue(ar2Resid, noestimatedcoef=2, nolags=30)
plt.show()

# %%

Y = np.column_stack((TIME, SIN[:,0], SIN[:,1]))
regar = ARIMA(temp, order=(2,0,0), exog=Y, trend='c').fit()
print(regar.summary())

#%%

regarResid = regar.resid
acf_pacf_fig(regarResid, both=True, lag=36)
plt.show()
plot_LB_pvalue(regarResid, noestimatedcoef=5, nolags=30)
plt.show()


# %%
