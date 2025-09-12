#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy import signal

from statsmodels.graphics.api import qqplot
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg, ar_select_order, AutoRegResults

%load_ext autoreload
%autoreload 2

from fit_fcn import fit_data


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

#%%

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

#%% fit large transformed model

def model_add(x, a, b, a1, a2, a3, b1, b2, b3, phi, theta):
    omega = 2*np.pi/12
    y_pred = a + b*x + a1*np.sin(omega*x + phi) + a2*np.sin(2*omega*x + phi) + a3*np.sin(3*omega*x + phi) +\
        b1*np.sin(omega*x + theta) + b2*np.sin(2*omega*x + theta) + b3*np.sin(3*omega*x + theta)
    return y_pred

param_list = ['a', 'b', 'a1', 'a2', 'a3','b1', 'b2', 'b3', 'phi', 'theta']

df_model = np.log(df['Passengers']).copy(deep=True)
df_model = df_model.to_frame()
df_model.dropna(inplace=True)

params, df_out = fit_data(df_model, model_add, param_list)

#%%
def model_add(x, a, b, a1,a2, b1, b2,  phi, theta):
    omega = 2*np.pi/12
    y_pred = a + b*x + a1*np.sin(omega*x + phi) + b1*np.sin(omega*x + theta) \
        + a2*np.sin(2*omega*x + phi) + b2*np.sin(2*omega*x + theta)
    return y_pred

param_list = ['a', 'b', 'a1','a2', 'b1', 'b2', 'phi', 'theta']

df_model = np.log(df['Passengers']).copy(deep=True)
df_model = df_model.to_frame()
df_model.dropna(inplace=True)

params, df_out = fit_data(df_model, model_add, param_list)

# %% Put model calculation back to main df

# print(params)
# display(df_out)
# display(df)
df['log_Model'] = df_out['model']
df['Model'] = np.exp(df_out['model'])
df['residual'] = df['Passengers'] - df['Model']
display(df)

# %%

df['residual'].plot(figsize=(10,5), title='Residuals from real model')
plt.show()

fig, axs = plt.subplots(2,2, figsize=(12,8))
fig.suptitle('Residuals after de-trending and removing seasonality from the DAT')
axs[0,0].plot(df['residual'])
axs[1,0].plot(df['residual'])
plot_acf(df['residual'], lags=20, ax=axs[0,1])
plot_pacf(df['residual'], lags=20, ax=axs[1,1])
plt.show()

# %%
import scipy.stats as stats

stats.probplot(df['residual'], dist="norm", plot=plt)
plt.title("Probability Plot model")
plt.show()

# %%

mu, std = norm.fit(df['residual'])
z = (df['residual'] - mu)/std
plt.hist(df['residual'], density=True, alpha=0.6, bins=100, label='Temp Error')

xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
data = np.random.randn(100000)

plt.plot(x, p, 'k', linewidth=2, label='Normal Dist')
plt.plot([std*2,std*2],[0,ymax])

print('P(Z > 2): {:0.3}% vs Normal Distibution: {:0.3}% '.format(len(z[z >= 2])/len(z)*100, (1-norm.cdf(2))*100))
print('SKEW    : {:0.3}'.format(stats.skew(z)))
print('KURTOSIS: {:0.3}'.format(stats.kurtosis(z)+3))
plt.ylabel('Density')
plt.xlabel('Temperature Error')
plt.legend()
plt.show()


# %% plot monthly passenger variation over a year

df['month'] = df.index.month

# get variation per month
df_var = df.groupby(['month'])['Passengers'].agg(['mean', 'std'])
display(df_var)

df_var['std'].plot(figsize=(6,6), color='green', title='Passenger std by month')
plt.show()
plt.figure(figsize=(6,6))
plt.scatter(df_var['mean'], df_var['std'])
plt.show()

# %%
