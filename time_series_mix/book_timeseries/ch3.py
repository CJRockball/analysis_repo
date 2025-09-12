#%% ------------------------ North Atlantic Oscillation --------------------------
# import dependencies

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PythonTsa.datadir import getdtapath
from PythonTsa.plot_acf_pacf import acf_pacf_fig
from PythonTsa.LjungBoxtest import plot_LB_pvalue
from statsmodels.graphics.tsaplots import plot_predict

import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

dtapath = getdtapath()

#%% ---------------------- Basic MA process -----------------------
# MA(2) simulation X = ep + 0.6ep(t-1) - 0.3ep(t-2)
# arbitrarilly assume std = 0.5

ep = np.random.normal(0, 0.5, size=300)
df_sim = pd.DataFrame(ep, columns=['ep'])
df_sim['X'] = df_sim['ep'] + 0.6*df_sim['ep'].shift(1) - 0.3*df_sim['ep'].shift(2)

df_sim['X'].plot(title='MA(2) Simulated Data')

#%%
from statsmodels.graphics.tsaplots import plot_acf

acf_pacf_fig(df_sim['X'].dropna(), both=True, lag=20)
plt.show()

plot_acf(df_sim['X'].dropna())
plt.show()

display(df_sim)
#%% --------------- North Atlantic Oscillation --------------------------------

#%% import data

dtapath = getdtapath()
nao = pd.read_csv(dtapath + 'nao.csv', header=0)

display(nao)

# %% Change the inde to months, set the data to a SERIES

timeindex = pd.date_range('1950-01', periods=len(nao), freq='M')
nao.index = timeindex
display(nao)

nao['index'].plot(title='NAO index')
plt.show()

naots = nao['index']
display(naots)

# %% Check ac and kpss and ad

acf_pacf_fig(naots, both=True, lag=48)
plt.show()

print('KPSS test')
sm.tsa.stattools.kpss(naots, regression='c', nlags=50)
# print('ADFULLER test')
# sm.tsa.stattools.adfuller(naots, autolag='AIC')

# %% Fit AR(1) model with constant

arl = ARIMA(naots, order=(1,0,0), trend='c').fit()
print(arl.summary())

#%% Fit AR(1) model without constant

arl_nc = ARIMA(naots, order=(1,0,0), trend='n').fit()
print(arl_nc.summary())

# %% Check residuals

resid1 = arl.resid

acf_pacf_fig(resid1, both=True, lag=48)
plt.show()

plot_LB_pvalue(resid1, noestimatedcoef=1, nolags=50)
plt.show()

# %% option1 plot


fig, ax = plt.subplots()
ax = naots.loc['1950-01':].plot(ax=ax)
plot_predict(arl, start='1960-04', end='2019-12', ax=ax)
plt.show()

# %% option2 plot

pred = arl.get_prediction(start='2010-04', end='2019-04')
predicts = pred.predicted_mean
predconf = pred.conf_int()
predframe = pd.concat([naots, predicts, predconf], axis=1)
predframe.plot()
plt.show()

# %% --------------------- Global Annual Mean Surface Air Temperature Changes from 1880 to 1985 ---------

from PythonTsa.Selecting_arma2 import choose_arma2
from scipy import stats
from PythonTsa.ModResidDiag import plot_ResidDiag

#%% Load data, set up series and plot

tep = pd.read_csv(dtapath + 'Global mean surface air temp changes 1880-1985.csv', header=None)
display(tep)

timeindex = pd.date_range('1880-12', periods=len(tep), freq='YE-DEC')
tep.index = timeindex
display(tep)

tepts = pd.Series(tep[0], name ='tep')
tepts.plot(title='Temp Change')
plt.show()

# %% To remove trend, differentiate

dtepts = tepts.diff(1)
dtepts = dtepts.dropna()
dtepts.name = 'dtep'
dtepts.plot()
plt.show()

sm.tsa.kpss(dtepts, regression='c', nlags='auto')

# %% Choose ARMA model

acf_pacf_fig(dtepts, both=True, lag=20)
plt.show()
choose_arma2(dtepts, max_p=7, max_q=7, ctrl=1.03)

# %% Fit model, plot eval

arma13 = ARIMA(dtepts, order=(1,0,3), trend='c').fit()
print(arma13.summary())

# %% check residuals

resid13 = arma13.resid
stats.normaltest(resid13)

#%%
plot_ResidDiag(resid13, noestimatedcoef=2, nolags=20, lag=25)
plt.show()

# %%

def plot_ts(df, model, t_data_start, t_pred_start, t_pred_end):
    fig, ax = plt.subplots()
    ax = df.loc[t_data_start:].plot(ax=ax)
    plot_predict(model, start=t_pred_start, end=t_pred_end, ax=ax)
    plt.show()

plot_ts(dtepts, arma13, '1880-12', '1960-12', '1990-12')




# %% ---------------------- US T-Bills Interest Rate ---------------------------------------

rat = pd.read_csv(dtapath + 'USbill.csv', header=None)
display(rat)

#%% set up data. Split in train, test

timeindex = pd.date_range('1950-1', periods=len(rat), freq='M')
rat.index = timeindex
rat.rename(columns={0:'time', 1:'bill'}, inplace=True)
display(rat)

y = rat[:456]
y=y['bill']
y.plot(title='T-Bill Interest Rate')   

# %% Data has trend. Take log and differentiate

ly = np.log(y)
dly = ly.diff(1)
dly = dly.dropna()

display(dly)
dly.plot(title='Data logged and differentiated')
plt.show()

# %% Check if stationary with acf, pacf, kpss

acf_pacf_fig(dly, both=True, lag=24)
plt.show()
sm.tsa.stattools.kpss(dly, regression='c', nlags='auto')

# %% Model selection

choose_arma2(dly, max_p=6, max_q=6, ctrl=1.05)

# %%

res = sm.tsa.arma_order_select_ic(dly, max_ar=6, max_ma=7, 
                                 ic = ['aic', 'bic', 'hqic'], trend='n')
print(res.aic_min_order)
print(res.bic_min_order)
print(res.hqic_min_order)

# %% Make model 6,0

arima610 = ARIMA(ly, order=(6,1,0), trend='n').fit()
print(arima610.summary())

# %% check residauls

resid610 = arima610.resid
plot_ResidDiag(resid610, noestimatedcoef=6, nolags=24, lag=24)
plt.show()
stats.normaltest(resid610)

# %%

def plot_ts(df, model, t_data_start, t_pred_start, t_pred_end):
    test = rat['bill'].iloc[456:]
    ltest = np.log(test)
    fig, ax = plt.subplots()
    ax = df.loc[t_data_start:].plot(ax=ax)
    ax = ltest.plot(color='red', label='test data', ax=ax)

    plot_predict(model, start=t_pred_start, end=t_pred_end, ax=ax)
    plt.show()
    
ax = plot_ts(ly, arima610, '1980-01', '1980-01', '1988-06')


# %% Predict data

gpred610 = arima610.get_prediction(start='1980-01', end='1988-06')
frame = gpred610.summary_frame(alpha=0.05)

#%%

frame['mean'].plot()




# %%
