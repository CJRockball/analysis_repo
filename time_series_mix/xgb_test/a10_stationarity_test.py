#%%
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
from statsmodels.regression.linear_model import OLS
from statsmodels.formula.api import ols

dtapath = getdtapath()

from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from xgboost import plot_importance

#%% ------------------------- Drugs Australia ---------------------------------

df = pd.read_csv(dtapath + 'h02July1991June2008.csv', header=0)
df.columns = ['drug']

timeindex = pd.date_range('1991-06', periods=len(df), freq='M')
df.index = timeindex
display(df)
df.plot(title='Australia Drug Sales Monthly')

# %% Check acf

acf_pacf_fig(df, both=True, lag=30)
plt.show()

# %% Calculate trend

def pred_trend(df, col_name):
    df_temp = df.copy(deep=True)
    # # Add trend fit a + bx
    df_temp['xx'] = range(len(df_temp)) 
    df_temp['xx_norm'] = df_temp.xx / df_temp.xx.max()
    df_temp['const'] = 1.0

    df_temp = df_temp.drop(columns=['xx'])
        
    lin_model = OLS.from_formula(f'{col_name} ~ xx_norm + const + np.power(xx_norm,2)', df_temp).fit()
    print(lin_model.summary())  

    lin_pred = lin_model.predict(df_temp)
    df_temp['trend'] = lin_pred
    
    plt.figure()
    plt.plot(df_temp[col_name])
    plt.plot(lin_pred)
    plt.show()
    
    df = pd.concat([df, df_temp.trend.to_frame()], axis=1)
    return df

df2 = pred_trend(df, 'drug')

# %% Remove trend

df2['detrend_drug'] = df2.drug - df2.trend
display(df2)

df2['detrend_drug'].plot()


# %% Remove seasonal by differencing
dfa = df2.copy(deep=True)

dfa['12_shift'] = dfa['detrend_drug'].shift(12)
dfa['diff'] = dfa['detrend_drug'] - dfa['12_shift']

dfa['diff'].plot()
acf_pacf_fig(dfa['diff'].dropna(), both=True, lag=30)
sm.tsa.kpss(dfa['diff'].dropna(), regression='c', nlags='auto')


# %%
from numpy import polyfit

# fit polynomial: x^2*b1 + x*b2 + ... + bn
X = [i%12 for i in range(0, len(df2))]
y = df2['detrend_drug'].values
degree = 4
coef = polyfit(X, y, degree)
print('Coefficients: %s' % coef)
# create curve
curve = list()
for i in range(len(X)):
    value = coef[-1]
    for d in range(degree+1):
        value += X[i]**(degree-d) * coef[d]
    curve.append(value)

curve_series = pd.Series(curve, index=df2.index)
# plot curve over original data
plt.plot(df2['detrend_drug'])
plt.plot(curve_series, color='red', linewidth=3)
plt.show()

# %%

df_p = df2.copy(deep=True)
df_p['poly_fit'] = curve_series.values
df_p['de_seasoned'] = df_p['detrend_drug'] - df_p['poly_fit']
display(df_p)

df_p['de_seasoned'].plot()
acf_pacf_fig(df_p['de_seasoned'], both=True, lag=30)
sm.tsa.kpss(df_p['de_seasoned'], regression='c', nlags='auto')

# %% fit models
from scipy.stats import norm
from scipy.optimize import curve_fit

def model_add(x, a, a1, b1, a2, b2, a3, b3, phi, theta):
    omega = 2*np.pi/12
    y_pred = a + a1*np.sin(omega*x + phi)+ b1*np.cos(omega*x+theta) +\
        a2*np.sin(2*omega*x + phi)+ b2*np.cos(2*omega*x+theta) +\
            a3*np.sin(3*omega*x + phi)+ b3*np.cos(3*omega*x+theta)
    return y_pred

def RSS(y, y_pred):
    return np.sqrt((y-y_pred)**2).sum()

#%%

df_model = df2['detrend_drug'].copy(deep=True)
df_model = df_model.to_frame()

data_length = df_model.shape[0]
xx = np.arange(0,data_length,1)

params, cov = curve_fit(model_add, xdata=xx, ydata=df_model['detrend_drug'], method='lm')

param_list = ['a', 'a1', 'b1', 'a2', 'b2', 'a3', 'b3', 'phi', 'theta']

print('\n Model 1 \n')
std_dev = np.sqrt(np.diag(cov))
for name,p,sd in zip(param_list, params, std_dev):
    print('{0} :  {1:0.3}  CI ~normally [{2:0.2e},{3:0.2e}]'.format(name, p, p-1.96*sd,p+1.96*sd))

df_model['model'] = model_add(xx, *params)

df_model.plot(figsize=(12,4), style=['s','^-'], markersize=4, linewidth=2)

print('Residual Sum of Squares RSS')
print(f"RSS model: {round(RSS(df_model['detrend_drug'], df_model['model']),2)}")

# %%
df_f = df2.copy(deep=True)

df_f['model'] = df_model['model']
df_f['de_seasoned'] = df_f['detrend_drug'] - df_f['model']
df_f['de_seasoned'].plot(figsize=(12,8), title='Residuals')

acf_pacf_fig(df_f['de_seasoned'], both=True, lag=30)
sm.tsa.kpss(df_f['de_seasoned'], regression='c', nlags='auto')

#%% diff 12

df_f['12_shift'] = df_f['de_seasoned'].shift(12)
df_f['diff'] = df_f['de_seasoned'] - df_f['12_shift']

df_f['diff'].plot(figsize=(12,8))
acf_pacf_fig(df_f['diff'].dropna(), both=True, lag=30)
sm.tsa.kpss(df_f['diff'].dropna(), regression='c', nlags='auto')

# %% Choose model
# Result
choose_arma2(df_f['diff'].dropna(), max_p=15, max_q=5, ctrl=1.03)

# %% Fit model 8,8
# For de_seasoned
#ar1 = ARIMA(df_f['de_seasoned'], order=([0,1,1,1,1,0,0,0,0,0,0,1],0,[0,0,0,0,1]), trend='n').fit()
# For diff
ar1 = ARIMA(df_f['diff'].dropna(), order=([0,1,1,0,0,0,0,0,0,0,0,1],0,0), trend='n').fit()
print(ar1.summary())

# %% 8,8 seems like a good model

resid1 = ar1.resid
stats.normaltest(resid1)

plot_ResidDiag(resid1, noestimatedcoef=3, nolags=30, lag=30)
plt.show()

# %%---------------------------------- LOG ---------------------------------------
# %% Calculate trend

def pred_trend(df, col_name):
    df_temp = df.copy(deep=True)
    # # Add trend fit a + bx
    df_temp['xx'] = range(len(df_temp)) 
    df_temp['xx_norm'] = df_temp.xx / df_temp.xx.max()
    df_temp['const'] = 1.0

    df_temp = df_temp.drop(columns=['xx'])
        
    lin_model = OLS.from_formula(f'{col_name} ~ xx_norm + const + np.power(xx_norm,2)', df_temp).fit()
    print(lin_model.summary())  

    lin_pred = lin_model.predict(df_temp)
    df_temp['trend'] = lin_pred
    
    plt.figure()
    plt.plot(df_temp[col_name])
    plt.plot(lin_pred)
    plt.show()
    
    df = pd.concat([df, df_temp.trend.to_frame()], axis=1)
    return df

df_l = df.copy(deep=True)
df_l['log_drug'] = np.log(df_l.drug)

display(df_l)

df_l['log_drug'].plot()

df_l = pred_trend(df_l, 'log_drug')
display(df_l)

df_l['de_trend'] = df_l['log_drug'] - df_l['trend']
df_l['de_trend'].plot()


#%%
def model_add(x, a, a1, b1, a2, b2, a3, b3, phi, theta):
    omega = 2*np.pi/12
    y_pred = a + a1*np.sin(omega*x + phi)+ b1*np.cos(omega*x+theta) +\
        a2*np.sin(2*omega*x + phi)+ b2*np.cos(2*omega*x+theta) +\
            a3*np.sin(3*omega*x + phi)+ b3*np.cos(3*omega*x+theta)
    return y_pred

def RSS(y, y_pred):
    return np.sqrt((y-y_pred)**2).sum()

#%%

df_model = df_l['de_trend'].copy(deep=True)
df_model = df_model.to_frame()

data_length = df_model.shape[0]
xx = np.arange(0,data_length,1)

params, cov = curve_fit(model_add, xdata=xx, ydata=df_model['de_trend'], method='lm')

param_list = ['a', 'a1', 'b1', 'a2', 'b2', 'a3', 'b3', 'phi', 'theta']

print('\n Model 1 \n')
std_dev = np.sqrt(np.diag(cov))
for name,p,sd in zip(param_list, params, std_dev):
    print('{0} :  {1:0.3}  CI ~normally [{2:0.2e},{3:0.2e}]'.format(name, p, p-1.96*sd,p+1.96*sd))

df_model['model'] = model_add(xx, *params)

df_model.plot(figsize=(12,4), style=['s','^-'], markersize=4, linewidth=2)

print('Residual Sum of Squares RSS')
print(f"RSS model: {round(RSS(df_model['de_trend'], df_model['model']),2)}")


# %%
df_f2 = df_l.copy(deep=True)

df_f2['model'] = df_model['model']
df_f2['de_seasoned'] = df_f2['de_trend'] - df_f2['model']
df_f2['de_seasoned'].plot(figsize=(12,8), title='Residuals')

acf_pacf_fig(df_f2['de_seasoned'], both=True, lag=30)
sm.tsa.kpss(df_f2['de_seasoned'], regression='c', nlags='auto')


#%% diff 12

df_f2['12_shift'] = df_f2['de_seasoned'].shift(12)
df_f2['diff'] = df_f2['de_seasoned'] - df_f2['12_shift']

df_f2['diff'].plot(figsize=(12,8))
acf_pacf_fig(df_f2['diff'].dropna(), both=True, lag=30)
sm.tsa.kpss(df_f2['diff'].dropna(), regression='c', nlags='auto')


# %%
