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

dtapath = getdtapath()

from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

#%% ------------------------- Drugs Australia ---------------------------------

df = pd.read_csv(dtapath + 'h02July1991June2008.csv', header=0)
df.columns = ['drug']

timeindex = pd.date_range('1991-06', periods=len(df), freq='M')
df.index = timeindex
#display(df)

#%% --------------------- Female Births ----------------------------------------

dfb = pd.read_csv('data/female_births.txt')
timeindex = pd.date_range('1959-01-01', periods=len(dfb), freq='D')
dfb = dfb.drop(columns=['Date'])
dfb.index = timeindex
dfb.plot(title='Female Births')
display(dfb)

#%%
n_in = 14
test_size = 12

# Transform time series to supervised learning
def s_to_sup(data:pd.DataFrame, n_in:int=1, dropnan:bool=True) -> pd.DataFrame:
    """
    Make dataframe with lagged features.   
    Leaves the original data in col 0.
    Each following col is lagged 1 time.
    data_out: pd.DatFrame, shape: (org rows x n_int+1)
    """

    data_out = data.copy(deep=True)
	# Shifts the data to make it lagged
    for i in range(1, n_in+1):
        data_lag = data.shift(i)
        data_lag.columns = [f'birth_lag{i}']
        data_out = pd.concat([data_out, data_lag], axis=1)

    # Drop nan
    if dropnan:
        data_out.dropna(inplace=True)

    return data_out

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data.iloc[:-n_test, :], data.iloc[-n_test:, :]


# Make lag features
data = s_to_sup(dfb, n_in=n_in, dropnan=True)
# display(data)

# Split train/test
train, test = train_test_split(data, test_size)
# print('train:', train.shape, type(train))
# print('test', test.shape, type(test))

# Get X,y matrix
trainX, trainy = train.iloc[:,1:], train.iloc[:,0]
# print(type(trainX))
# print(trainX.shape)
# print(type(trainy))
# print(trainy.shape)

#%% Dummy model (persistance model)

data_mean = dfb.Births.mean()
mean_model_series = pd.Series([data_mean]*12, index=test.index)

mean_error = mean_absolute_error(test.iloc[:,0], mean_model_series)
print(f"MAE: {mean_error:.3f}")

#%%
# fit model
model = XGBRegressor(objective='reg:squarederror',  eval_metric=None, n_estimators=1000)
model.fit(trainX, trainy)

#%% Do one step prediction and add predicted data to X input

def one_predict(df_test:pd.DataFrame, n_in:int, test_size:int):
    """
    Predicts one step at a time and then adds prediction to the next input vector
    """
    one_step_prediction = []
    testX = df_test.iloc[0,1:].to_frame().T

    # make a one-step prediction
    for _ in range(test_size):
        yhat = model.predict(testX)
        # Save prediction to list
        one_step_prediction.append(yhat[0])
        # Update input data
        new_data = [yhat[0]] + testX.iloc[0,:-1].to_list()
        testX.iloc[0] = new_data

    # Put predictions in a pandas series
    df_y_hat = pd.Series(one_step_prediction, index=df_test.index)
    # estimate prediction error
    error = mean_absolute_error(df_test.iloc[:, 0], df_y_hat)
    return df_y_hat, error

df_y_hat, error = one_predict(test, n_in, test_size)
print(f'MAE: {error:.3f}')

# plot expected vs preducted
plt.plot(test.iloc[:,0], label='Expected')
plt.plot(df_y_hat, label='Predicted')
plt.legend()
plt.xticks(rotation=45, ha='right')
plt.show()

#%% Check residuals

in_sample = model.predict(trainX)
in_sample_series = pd.Series(in_sample, index=trainX.index)

xgb_resid = trainX.iloc[:,0] - in_sample_series

plot_ResidDiag(xgb_resid, noestimatedcoef=6, nolags=24, lag=25)
plt.show()

#%%
plt.figure()
plt.plot(trainX.iloc[:,0], label="Expected")
plt.plot(in_sample_series, alpha=0.4, label='Predicted')
plt.legend()
plt.xticks(rotation=45, ha='right')
plt.show()



#%% ------------------  Statistical data test  ----------------------------
# Remove last 12 for out-of-sample test

dfb2 = dfb.iloc[:-12, :]
dfb_test = dfb.iloc[-12:, :]

display(dfb2)
dfb2.plot(title='Data Series')
plt.show()

# %% Check data
# Check if data set is stationary and inspect acf/pacf

acf_pacf_fig(dfb2, both=True, lag=30)
plt.show()
sm.tsa.kpss(dfb2, regression='c', nlags='auto')

# %% Choose model
# Result
choose_arma2(dfb2, max_p=8, max_q=8, ctrl=1.03)

# %% Fit model 8,8

ar1 = ARIMA(dfb2, order=(8,0,8), trend='c').fit()
print(ar1.summary())

#%% Refine model

ar1mod = ARIMA(dfb2, order=([1,0,1,1,1,0,0,1],0,[1,0,1,0,1,0,0,1]),
               trend='c').fit()
print(ar1mod.summary())
#Checking residuals give a not very good result

# %% 8,8 seems like a good model

resid1 = ar1.resid
stats.normaltest(resid1)


#%% 8,8 seems like a good model

plot_ResidDiag(resid1, noestimatedcoef=2, nolags=24, lag=25)
plt.show()

# %%

arima_forecast = ar1.get_forecast(steps=12)
df_forecast = arima_forecast.summary_frame(alpha=0.05)
print(df_forecast)

# %%

error = mean_absolute_error(dfb_test, df_forecast['mean'])
print(f'MAE: {error:.3f}')

plt.figure()
plt.plot(dfb_test, label='Expected')
plt.plot(df_forecast['mean'], label='Predicted')
plt.legend()
plt.xticks(rotation=45, ha='right')
plt.show()


# %%
