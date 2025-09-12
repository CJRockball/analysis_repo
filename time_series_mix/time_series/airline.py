#%%
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.graphics.api import qqplot
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg, ar_select_order, AutoRegResults

# %%

df = pd.read_csv('airline-passengers.csv')
display(df)

#%%

df['Month'] = pd.to_datetime(df.Month, format='%Y-%m')
df.set_index('Month', inplace=True)
display(df)
print(df.info())
df['Passengers'].plot(figsize=(8,4), color='green', title='Airline Passengers')
plt.show()

# %%

decompose_result = seasonal_decompose(df['Passengers'], model='additive', period=int(12), extrapolate_trend='freq')

trend = decompose_result.trend
sesonal = decompose_result.seasonal
residual = decompose_result.resid

decompose_result.plot()
plt.show()
residual.hist(bins=40, figsize=(8,6))
plt.show()

# %% Test for stationarity. 
# Null hypothesis is that unit root exists, i.e. is stationary
dftest = adfuller(residual, autolag = 'AIC')

print("1. ADF : ",dftest[0])
print("2. P-Value : ", dftest[1])
print("3. Num Of Lags : ", dftest[2])
print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
print("5. Critical Values :")
for key, val in dftest[4].items():
    print("\t",key, ": ", val)



# %% Plot autocorrelation and partial autocorrelation

plot_acf(residual, lags=50)
plt.show()
plot_pacf(residual, lags=50)
plt.show()
# %% Plot residuals to check fit

residuals = residual.copy(deep=True)
residuals.index = pd.DatetimeIndex(residuals.index).to_period('M')

# %% check AC on residuals with AIC

mod = ar_select_order(residuals, maxlag=40, ic='aic', old_names=True)

aic = []
for key, val in mod.aic.items():
    if key != 0:
        aic.append((key[-1], val))

aic.sort()
x,y = [x for x,y in aic],[y for x,y in aic]

plt.scatter(x, y)
plt.plot([0,40],[y[13],y[13]], 'tab:orange') # AR(13) Same as adf test
plt.text(3,y[13]+1, '{0}'.format(round(y[13],3)),color='tab:orange')
#plt.plot([0,40],[y[20],y[20]], 'k--')
#plt.text(3,y[20]-0.004, '{0}'.format(round(y[20],3)))
plt.title("AIC Criterion")
plt.xlabel("Lags in AR Model")
plt.ylabel("AIC")
plt.show()

#%% check fit autoregressive series to residuals

model = AutoReg(residuals, lags=13, old_names=True,trend='n')
model_fit  = model.fit()
coef = model_fit.params
res = model_fit.resid
res.index = res.index.to_timestamp()
print(model_fit.summary())


# %% Plot AR model

fig, axs = plt.subplots(2,2, figsize=(8,6))
fig.suptitle('Residuals after AR(15) model')
axs[0,0].plot(res) #all residuals of the residuals
axs[1,0].plot(res[-200:]) # last 200 residuals (which is all the residuals)
plot_acf(res, lags=20, ax=axs[0,1])
plot_pacf(res, lags=20, ax=axs[1,1])
plt.show()


# %% qq-plot on AR fitted residuals
import numpy as np
from scipy.stats import norm

qqplot(res, marker='x', dist=norm, loc=0, scale=np.std(res), line='45')
plt.show()
# %% Use convolution to denoise data
from scipy import signal

def apply_convolution(x, window):
    conv = np.repeat([0., 1., 0.], window)
    filtered = signal.convolve(x, conv, mode='same') / window
    return filtered

denoised = df.apply(lambda x: apply_convolution(x, 3))
denoised['Passengers'].plot(figsize=(12,6))
plt.ylabel('Temperature (deg C)')
plt.show()
# Data is too short and not noisy enough....

# %% Get the trend

df['Trend'] = df['Passengers'].rolling(window=12).mean()
plt.ylabel('Passengers')
df['Trend'].plot(figsize=(8,4), color='tab:green', title='Rolling mean passengers')
df['Passengers'].plot(figsize=(8,4), color='tab:red', title='Rolling mean passengers')
plt.show()

# %%
# Get seasonal variation by subtracting trend
df['Seasonal_a'] = df['Passengers'] - df['Trend']
df['Seasonal_a'].plot(figsize=(8,4), color='red', title='Seasonal component')
plt.show()
# Not a constant amplitude

# Get multiplicative seasonal component
df['Seasonal_m'] = df['Passengers'] / df['Trend']
df['Seasonal_m'].plot(figsize=(8,4), color='red', title='Seasonal component')
plt.show()
# Looks constant

# %% fit models
from scipy.stats import norm
from scipy.optimize import curve_fit

def model_add(x, a, b,  a1, phi):
    omega = 2*np.pi/12
    y_pred = a + b*x + (a1*x)*np.sin(omega*x + phi)
    return y_pred

def RSS(y, y_pred):
    return np.sqrt((y-y_pred)**2).sum()

#%% Prep data. Get just passenger data and make ordinal index
import datetime as dt

df_model = df['Passengers'].copy(deep=True)
df_model = df_model.to_frame()
#df_model.dropna(inplace=True)
#df_model['Passengers'] = df_model['Seasonal_m']
#display(df_model)


# %%

data_length = df_model.shape[0]
xx = np.arange(0,data_length,1)

params, cov = curve_fit(model_add, xdata=xx, ydata=df_model['Passengers'], method='lm')

param_list = ['a', 'b', 'a1', 'phi']

print('\n Model 1 \n')
std_dev = np.sqrt(np.diag(cov))
for name,p,sd in zip(param_list, params, std_dev):
    print('{0} :  {1:0.3}  CI ~normally [{2:0.2e},{3:0.2e}]'.format(name, p, p-1.96*sd,p+1.96*sd))

df_model['model'] = model_add(xx, *params) #df_model.index-first_ord

# if not isinstance(df_model.index, pd.DatetimeIndex):
#     df_model.index = df_model.index.map(dt.datetime.fromordinal)
    
df_model.plot(figsize=(12,4), style=['s','^-','k--'], markersize=4, linewidth=2)

print('Residual Sum of Squares RSS')
print(f"  RSS model: {round(RSS(df_model['Passengers'], df_model['model']),2)}")

# %%
