#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.graphics.api import qqplot
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg, ar_select_order, AutoRegResults

#%%

df = pd.read_csv('daily-min-temperatures.txt')
display(df)

# %%

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
display(df)
df['Temp'].plot(figsize=(8,4), color='green', title='Temperature 10year')

# %% standard decompose

decompose_result = seasonal_decompose(df['Temp'], model='additive', period=365, extrapolate_trend='freq')

trend = decompose_result.trend
seasonal = decompose_result.seasonal
residual = decompose_result.resid

decompose_result.plot()
plt.show()
residual.hist(bins=60, figsize=(8,6))
plt.show()

# %% Explore data
# use convolution to denoise data
from scipy import signal

def apply_convolution(x, window):
    conv = np.repeat([0., 1., 0.], window)
    filtered = signal.convolve(x, conv, mode='same') / window
    return filtered

denoised = df.apply(lambda x: apply_convolution(x, 60))
denoised['Temp'].plot(figsize=(12,6))
plt.ylabel('Temperature (deg C)')
plt.show()

#%% Get trend

df['trend'] = df['Temp'].rolling(window=365).mean()
plt.ylabel('Temp')
df['Temp'].plot(figsize=(8,4), color='tab:blue', title='Rolling mean passengers')
df['trend'].plot(figsize=(8,4), color='tab:orange', linewidth=2, title='Rolling mean passengers')
plt.show()

# Trend is flat. Set it to mean
trend = df['Temp'].mean()
df['Ttrend'] = trend
display(df)

# %%

df['seasonal'] = df['Temp'] - df['Ttrend']
df['seasonal'].plot(figsize=(8,6), color='green', title='Temp seasonal')
plt.show()


# %% fit models
from scipy.stats import norm
from scipy.optimize import curve_fit

def model_add(x, a,  a1, phi):
    omega = 2*np.pi/365
    y_pred = a + a1*np.sin(omega*x + phi)
    return y_pred

def RSS(y, y_pred):
    return np.sqrt((y-y_pred)**2).sum()

#%%

df_model = df['Temp'].copy(deep=True)
df_model = df_model.to_frame()

data_length = df_model.shape[0]
xx = np.arange(0,data_length,1)

params, cov = curve_fit(model_add, xdata=xx, ydata=df_model['Temp'], method='lm')

param_list = ['a', 'a1', 'phi']

print('\n Model 1 \n')
std_dev = np.sqrt(np.diag(cov))
for name,p,sd in zip(param_list, params, std_dev):
    print('{0} :  {1:0.3}  CI ~normally [{2:0.2e},{3:0.2e}]'.format(name, p, p-1.96*sd,p+1.96*sd))

df_model['model'] = model_add(xx, *params)

df_model.plot(figsize=(12,4), style=['s','^-','k--'], markersize=4, linewidth=2)

print('Residual Sum of Squares RSS')
print(f"  RSS model: {round(RSS(df_model['Temp'], df_model['model']),2)}")

# %%

df['model'] = df_model['model']
df['residuals'] = df['Temp'] - df['model']
df['residuals'].plot(figsize=(12,8), title='Residuals')

# %% Check autocorrelation in residuals

fig, axs = plt.subplots(2,2, figsize=(12,8))
fig.suptitle('Residuals after de-trending and removing seasonality from the DAT')
axs[0,0].plot(df['residuals'])
axs[1,0].plot(df['residuals'])
plot_acf(df['residuals'], lags=40, ax=axs[0,1])
plot_pacf(df['residuals'], lags=40, ax=axs[1,1])
plt.show()

# %% Check qq plot
import scipy.stats as stats

stats.probplot(df['residuals'], dist="norm", plot=plt)
plt.title("Probability Plot model")
plt.show()

# %%

mu, std = norm.fit(df['residuals'])
z = (df['residuals'] - mu)/std
plt.hist(df['residuals'], density=True, alpha=0.6, bins=100, label='Temp Error')

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

# %%

residuals = df['residuals']
residuals.index = pd.DatetimeIndex(residuals.index).to_period('D')

model = AutoReg(residuals, lags=1, old_names=True,trend='n')
model_fit  = model.fit()
coef = model_fit.params
res = model_fit.resid
# res.index = res.index.to_timestamp()
print(model_fit.summary())

# %%

kappa = 1 - coef[0]
print("Kappa is estimated as: {:0.3}".format(kappa))

# %%
df['day_number'] = df.index.dayofyear
df['month'] = df.index.month

display(df)

#%% The variance doesn't change much. Trying three different methods to calculate/plot it
# groupby, convolution, percentage

def apply_convolution(x, window):
    conv = np.repeat([0., 1., 0.], window)
    filtered = signal.convolve(x, conv, mode='same') / window
    return filtered

# daily mean and std
df_var = df.groupby(['day_number'])['Temp'].agg(['mean', 'std'])
# normalized std
df_var['pct_var'] = df_var['std']/df_var['mean'] * 100
# filtered std
df_std = df_var['std'].copy(deep=True).to_frame()
df_filter = df_std.apply(lambda x: apply_convolution(x, 30))
display(df_filter)


df_var['std'].plot(figsize=(8,6), title='Varying residuals')
#df_var['pct_var'].plot(color='orange', title='%Varying residuals')
df_filter['std'].plot(color='orange', linewidth=4)
plt.show()

# %% Use piecewise constant function to model variance
import datetime as dt

vol_days = df.groupby(['day_number'])['Temp'].agg(['mean', 'std'])
vol_months = df.groupby(['month'])['Temp'].agg(['mean', 'std'])

vol_days['days'] = vol_days.index
def change_month(row):
    date = dt.datetime(2016,1,1) + dt.timedelta(row.days - 1)
    return vol_months.loc[date.month, 'std']

vol_days['vol_month'] = vol_days.apply(change_month, axis=1)

vol_days[['std', 'vol_month']].plot(linewidth=2)
plt.ylabel('Std Dev (deg C)')
plt.xlim(0,366)
plt.show()

# %%

display(vol_days)


# %% Simulate weather

def T_model(x, a, a1, phi):
    omega = 2*np.pi/365.25
    T = a + a1*np.sin(omega*x + phi)
    return T


def dT_model(x, a, a1, phi):
    omega=2*np.pi/365.25
    dT = a1*omega*np.cos(omega*x + phi)
    return dT


def euler_step(row, kappa, M):
    """Function for euler scheme approximation step in
    modified OH dynamics for temperature simulations
    Inputs:
    - dataframe row with columns: T, Tbar, dTbar and vol
    - kappa: rate of mean reversion
    Output:
    - temp: simulated next day temperatures
    """
    if row['Tbar_shift'] != np.nan:
        T_i = row['Tbar']
    else:
        T_i = row['Tbar_shift']
    T_det = T_i + row['dTbar']
    T_mrev = kappa*(row['Tbar'] - T_i)
    sigma = row['vol']*np.random.randn(M)
    return T_det + T_mrev + sigma


def get_vol(row):
    return vol_days.loc[row.day_number,'vol_month']


def monte_carlo_temp(trading_dates, Tbar_params, vol_model, M=1, kappa=0.438):
    """Function for euler scheme approximation step in
    modified OH dynamics for temperature simulations
    Inputs:
    - dataframe row with columns: T, Tbar, dTbar and vol
    - kappa: rate of mean reversion
    Output:
    - temp: simulated next day temperatures
    """
    
    #trading_date = np.arange(trading_dates.shape[0])
    trading_date = trading_dates.map(dt.datetime.toordinal)
    
    # Use Modified Ornstein-Uhlenbeck process with estimated parameters to simulate Tbar DAT
    Tbars = T_model(trading_date, *Tbar_params)

    # Use derivative of modified OH process SDE to calculate change of Tbar
    dTbars = dT_model(trading_date, *Tbar_params)
    
    # Create DataFrame with thi
    mc_temps = pd.DataFrame(data=np.array([Tbars, dTbars]).T,
                            index=trading_dates, columns=['Tbar', 'dTbar'])
    
    # Create columns for day in year
    mc_temps['day_number'] = mc_temps.index.dayofyear
    
    # Volatility 
    mc_temps['vol'] = mc_temps.apply(get_vol, axis=1)
    
    # Shift Tbar one day (lagged Tbar series)
    mc_temps['Tbar_shift'] = mc_temps['Tbar'].shift(1)    

    # Apply Eurler Step Pandas Function
    data = mc_temps.apply(euler_step, args=[kappa, M], axis=1)
    
    # Create final DataFrame of all simulations
    mc_sims = pd.DataFrame(data=[x for x in [y for y in data.values]],
                           index=trading_dates, columns=range(1,M+1))

    return mc_temps, mc_sims


# %% Run simulations

no_sims = 5
trading_dates = pd.date_range(start='1991-01-01', end='1992-06-30', freq='D')
mc_temps, mc_sims = monte_carlo_temp(trading_dates, params, vol_days, no_sims)

#%% Check dataframes

display(mc_temps)
display(mc_sims)

#%% Plot simulation and mean

mc_sims[1].plot(figsize=(12,6), alpha=0.6, marker='*')
mc_temps['Tbar'].plot(color='orange')
plt.show()

#%% Simulate high summer and vinter

no_sims = 10_000
trading_dates_winter = pd.date_range(start='1991-07-01', end='1991-07-01', freq='D')
mc_temps_winter, mc_sims_winter = monte_carlo_temp(trading_dates_winter, params, vol_days, no_sims)

trading_dates_summer = pd.date_range(start='1991-01-01', end='1991-01-01', freq='D')
mc_temps_summer, mc_sims_summer = monte_carlo_temp(trading_dates_summer, params, vol_days, no_sims)

# %% Plot distributions

plt.figure(figsize=(12,6))
plt.title('Winter vs Summer Temperature MC Sims')

Tbar_summer = mc_temps_summer.iloc[-1,:]['Tbar']
Tbar_winter = mc_temps_winter.iloc[-1,:]['Tbar']

plt.hist(mc_sims_summer.iloc[-1,:], bins=20, alpha=0.5, label='Summer', color='orange')
plt.plot([Tbar_summer, Tbar_summer], [0,1600], linewidth=4, label='Summer average', color='orange')

plt.hist(mc_sims_winter.iloc[-1,:], bins=20, alpha=0.5, label='Winter', color='blue')
plt.plot([Tbar_winter, Tbar_winter], [0,1600], linewidth=4, label='Winter average', color='blue')
plt.legend()
plt.show()


