#%%
%load_ext autoreload
%autoreload 2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PythonTsa.datadir import getdtapath
from PythonTsa.plot_acf_pacf import acf_pacf_fig
from PythonTsa.plot_multi_ACF import multi_ACFfig
from PythonTsa.LjungBoxtest import plot_LB_pvalue
from statsmodels.graphics.tsaplots import plot_predict

from PythonTsa.Selecting_arma2 import choose_arma2
from scipy import stats
from PythonTsa.ModResidDiag import plot_ResidDiag

import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from arch import arch_model

from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor



#%% Load Data

df = pd.read_csv('airline-passengers.csv')
timeindex = pd.date_range('1949-01', periods=len(df), freq='ME')
df.index = timeindex
df.drop(columns=['Month'], inplace=True)

display(df)
df.plot(title='Airline Passengers')


# %% Seasonal lag seams reasonable

ldf = np.log(df)
acf_pacf_fig(ldf,both=True, lag=30)

Dldf = ldf.diff(12).dropna()

display(Dldf)
Dldf.plot()
plt.show()

# Make train/test
df_train = Dldf.iloc[:-12,:]
df_test = Dldf.iloc[-12:,:]

#%% ------------------ STATISTICAL DATA ANALYSIS ARIMA/SARIMA -----------------------
# Check stationary
acf_pacf_fig(df_train, both=True, lag=26)
plt.show()
sm.tsa.kpss(df_train, regression='c', nlags='auto')

#%%  Help choose ARIMA order

choose_arma2(df_train, max_p=12, max_q=5, ctrl=1.03)

#%%5 Fit model

arima1 = ARIMA(df_train, order=(10,0,3), trend='c').fit()
print(arima1.summary())

#%% Check residuals

resid1 = arima1.resid
plot_ResidDiag(resid1, noestimatedcoef=3, nolags=25, lag=25)
plt.show()
acf_pacf_fig(resid1**2, both=True, lag=25)

# %% It's very hard to find a working model

ts_mod = sm.tsa.SARIMAX(df, order=([0,1,0,1,0,0,1,1,0,1],1,0), seasonal_order=(1,1,0,12))
ts_mod_fit = ts_mod.fit(disp=False)
print(ts_mod_fit.summary())

resid2 = ts_mod_fit.resid
plot_ResidDiag(resid2, noestimatedcoef=13, nolags=28, lag=28)


# %% Forecast

y_pred = arima1.get_forecast(12)
y_pred_frame = y_pred.summary_frame(alpha=0.05)

#%% Reconstruct data

df_rest = ldf.copy(deep=True)
df_rest['diff'] = Dldf
# Deshift train data
df_rest['dediff'] = df_rest['diff'] + df_rest['Passengers'].shift(12)
# Deshift prediction
df_rest['ldpred'] = y_pred_frame['mean']
df_rest['lpred'] = df_rest['ldpred'] + df_rest['Passengers'].shift(12).iloc[-12:]
# Delog prediction
df_rest['pred'] = np.exp(df_rest['lpred'])
# Delog original data
df_rest['original'] = np.exp(df_rest['Passengers'])

display(df_rest)

#%%

error = mean_absolute_error(df_rest.original[-12:], df_rest.pred[-12:])
print(f'MAE: {error:.3f}')

plt.figure()
plt.plot(df_rest.original, label='Data')
plt.plot(df_rest.pred, label='Prediction')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.show()

plt.figure()
plt.plot(df_rest[-12:].original, label='Data')
plt.plot(df_rest.pred, label='Prediction')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.show()

# %% ------------------------  XGB  --------------------------------
# Make regressed data

df_xgb = df.copy(deep=True)
#df_feat['drug'] = np.log(df_feat['drug'] + 1)
# Make lags
for i in range(12):
    df_xgb[f'drug_lag{i+1}'] = df_xgb.iloc[:,i].shift(1)

#display(df_xgb.iloc[:14,:])

df_xgb = df_xgb.dropna()

xgb_train = df_xgb.iloc[:-12,:]
xgb_trainX = xgb_train.iloc[:,1:]
xgb_trainy = xgb_train.iloc[:,0]
xgb_test = df_xgb.iloc[-12:,:]
xgb_testX = xgb_test.iloc[:-12,1:]
xgb_testy = xgb_test.iloc[-12:,0]

#%%
xgb_model = XGBRegressor(objective='reg:squarederror',  eval_metric=mean_absolute_error, 
                         n_estimators=75)
xgb_model.fit(xgb_trainX, xgb_trainy, eval_set = [(xgb_trainX, xgb_trainy)])

# %% Predictions
N_IN = 12
TEST_SIZE = 12

def one_predict(df_test:pd.DataFrame, n_in:int, test_size:int):
    """
    Predicts one step at a time and then adds prediction to the next input vector
    """
    one_step_prediction = []
    testX = df_test.iloc[0,1:].to_frame().T

    # make a one-step prediction
    for _ in range(test_size):
        yhat = xgb_model.predict(testX)
        # Save prediction to list
        one_step_prediction.append(yhat[0])
        # Update input data
        new_data = [yhat[0]] + testX.iloc[0,:-1].to_list()
        testX.iloc[0] = new_data

    # Put predictions in a pandas series
    df_y_hat = pd.Series(one_step_prediction, index=df_test.index)
    return df_y_hat

df_y_hat = one_predict(xgb_test, N_IN, TEST_SIZE)

# %%

error = mean_absolute_error(xgb_testy, df_y_hat)
print(f'MAE: {error:.3f}')

# plot expected vs preducted
plt.plot(df, label='Expected')
plt.plot(df_y_hat, label='Predicted')
plt.legend()
plt.xticks(rotation=45, ha='right')
plt.show()

# plot expected vs preducted
plt.plot(xgb_testy, label='Expected')
plt.plot(df_y_hat, label='Predicted')
plt.legend()
plt.xticks(rotation=45, ha='right')
plt.show()

# %% Check feature importance
# xgboost import plot_importance

fig, ax = plt.subplots(figsize=(12,6))
plot_importance(xgb_model, max_num_features=20, ax=ax)
plt.show()

#%% ---------------------- XGB2 ------------------------------------
import os
path = "/home/patrick/Python/timeseries/weather"
os.chdir(path)

import xgb_test.xgb_util as xu
from xgboost import plot_importance

#%%
df_xgb2 = df.copy(deep=True)

SEASONAL_DIFF = 12
ORD_DIFF = 0
df_diff = xu.get_diff_df(df_xgb2, ord_diff=ORD_DIFF, seasonal_diff=SEASONAL_DIFF)

N_LAGS = 12
MOVING_WIN = False
df_lags = xu.get_lags(df_diff, N_LAGS, moving_win=MOVING_WIN)
#df_lags = xu.get_features(df_lags)
#display(df_lags)

N_TEST = 12
N_VAL = 0
#trainX, trainy, valX, valy, testX, testy = xu.train_test(df_lags, N_TEST, N_VAL)
trainX, trainy, testX, testy = xu.train_test(df_lags, N_TEST, N_VAL)

# %% Def model
eval_metrics = ['mae']
xgb_model1 = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
                        #   eval_metric=eval_metrics,
                        #   n_estimators=3000, max_depth=5, 
                        #   learning_rate=0.001, colsample_bytree=0.6)
xgb_model1.fit(trainX, trainy)

# %% Set up and train xgb model

eval_set = [(trainX, trainy),(valX, valy)]
eval_metrics = ['mae']

xgb_model1 = XGBRegressor(objective='reg:squarederror', eval_metric=eval_metrics,
                          n_estimators=3000, max_depth=5, 
                          learning_rate=0.001, colsample_bytree=0.6)

xgb_model1.fit(trainX, trainy,
               eval_set=eval_set,
               verbose=False)

#%%

history = xgb_model1.evals_result()
xx = range(0,len(history['validation_0']['mae']))

# Plot training
plt.figure()
plt.plot(xx,history["validation_0"]['mae'], label="Train")
plt.plot(xx,history['validation_1']['mae'], label='Test')
plt.legend()
plt.show()
#%%
# Plot feature importance
fig, ax = plt.subplots(figsize=(12,6))
plot_importance(xgb_model1, max_num_features=20, ax=ax)
plt.show();

#%% One step prediction

y_hat_series = xu.one_step_pred(trainX, testX, N_TEST, N_LAGS, xgb_model1, moving_win=MOVING_WIN)
#isplay(y_hat_series)

# plt.figure()
# plt.plot(testy, label='diff Data')
# plt.plot(y_hat_series, label='Prediction')
# plt.show()

# Restore series from lag
y_hat_df = y_hat_series.to_frame()
y_hat_df.columns = ['pred']
y_hat_df['data'] = testy
y_hat_df['lags'] = df_xgb2.iloc[-2*N_TEST:-N_TEST,0].values

y_hat_df['final_pred'] = y_hat_df['pred'] + y_hat_df['lags']
y_hat_df['recon_data'] = y_hat_df['data'] + y_hat_df['lags']
y_hat_df['org_data'] = df_xgb2.iloc[-N_TEST:,0]
display(y_hat_df)

# Plot Data and predictions
error = mean_absolute_error(df_xgb2.iloc[-N_TEST:,0], y_hat_df['final_pred'])
print(f'MAE: {error:.3f}')

plt.figure()
plt.plot(df_xgb2.iloc[-N_TEST:,0], label='Org Data')
plt.plot(y_hat_df['final_pred'], label='Pred Data')
plt.legend()
plt.show()


# %%




# %%
