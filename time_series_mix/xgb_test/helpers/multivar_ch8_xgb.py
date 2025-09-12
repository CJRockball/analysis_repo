#%%
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
from xgboost import plot_importance

dtapath = getdtapath()


# %% Set up df

uscc = pd.read_csv(dtapath + 'USEconomicChange.csv')
timeindex = pd.date_range('1970-03', periods=len(uscc), freq='QE')
uscc.index = timeindex
uscc.rename(columns={'Consumption':"cons", 'Income':"inc", 'Production':'prod', 'Savings':"sav","Unemployment":"unem"},
            inplace=True)
uscc.drop(columns=['Time'], inplace=True)
display(uscc)

# %% EDA

multi_ACFfig(uscc, nlags=12)
plt.show()


display(uscc.corr())

#%% Check cross lag correlation
N_CROSS_LAGS = 2
cols = ['cons', 'inc','prod', 'sav', 'unem']
df_lags_test = uscc.copy(deep=True)

for col_name in cols:
    cols_dict = {f"{col_name}_lag{l}":df_lags_test[col_name].shift(l) 
                    for l in range(1,N_CROSS_LAGS)}
    df_lags_test = df_lags_test.assign(**cols_dict)
    
df_lags_test.dropna(inplace=True)
#display(df_lags_test)

#%%

corrs = df_lags_test.corr()
corrs = corrs.reindex(sorted(corrs.columns), axis=1)
corrs.sort_index(inplace=True) 
corrs = corrs[::-1]
NO_FACTORS = corrs.shape[0]
# display(corrs)

# x_label_names = corrs.columns
# y_label_names = corrs.index.tolist()

# plt.figure()
# plt.pcolormesh(corrs, edgecolors='k', linewidth=2)
# plt.xticks(np.arange(NO_FACTORS), x_label_names, rotation=45, ha='right')
# plt.yticks(np.arange(NO_FACTORS), y_label_names)
# plt.colorbar()
# ax = plt.gca()
# ax.set_aspect('equal')
# plt.show()

#%%

corrs2 = corrs[::-1].where(np.tril(np.ones(corrs.shape)).astype(bool))
#display(corrs2)
s = corrs2.unstack().dropna()
#display(s)
so = s.sort_values(kind="quicksort")
#display(so)

high_corr = so[(so.values > 0.15) & (so.values < 1) | (so.values < -0.15)]
display(high_corr)

#%%

acf_pacf_fig(uscc['cons'], both=True, lag=40)



# %% Split datta

df_train = uscc.iloc[:-8,:]
trainX = df_train[['inc', 'prod', 'sav','unem']]
trainy = df_train['cons']
df_test = uscc.iloc[-8:,:]
testX = df_test[['inc', 'prod', 'sav','unem']]
testy = df_test['cons']

trainX = sm.add_constant(trainX, prepend=False)
testX = sm.add_constant(testX, prepend=False)

#%% Copy Rob Handyman lesson 9.2

sarimaxmod = SARIMAX(endog=trainy, exog=trainX['inc'], trend='c', order=(1,0,2))
sarimaxfit = sarimaxmod.fit(disp=False)
print(sarimaxfit.summary())

# y = 

#%% fit sarimax model

sarimaxmod = SARIMAX(endog=trainy, exog=trainX, order=(1,0,1))
sarimaxfit = sarimaxmod.fit(disp=False)
print(sarimaxfit.summary())


# %% Evaluate residuals

sarimaxresid = sarimaxfit.resid
acf_pacf_fig(sarimaxresid, both=True, lag=20)
plt.show()
plot_LB_pvalue(sarimaxresid, noestimatedcoef=2, nolags=20)
plt.show()

# %%

sarimaxresid = sarimaxfit.resid
acf_pacf_fig(sarimaxresid**2, both=True, lag=20)
plt.show()
plot_LB_pvalue(sarimaxresid**2, noestimatedcoef=2, nolags=20)
plt.show()

# %% Fit GARCH to residuals

archmod = arch_model(sarimaxresid, mean='Zero').fit(disp='off')
print(archmod.summary())

# %%

archresid = archmod.std_resid
plot_LB_pvalue(archresid, noestimatedcoef=0, nolags=20)
plt.show()
plot_LB_pvalue(archresid**2, noestimatedcoef=0, nolags=20)
plt.show()


# %% Predict, combine

s_pred = sarimaxfit.get_forecast(8, exog=testX)
s_pred_frame = s_pred.summary_frame(alpha=0.05) 
#print(s_pred.summary_frame(alpha=0.05))

simmod = arch_model(None, mean='Zero')
g_pred = simmod.simulate(archmod.params, 8)
#display(g_pred)

tot_pred = s_pred_frame['mean'] + g_pred.data.values

# %%
error = mean_absolute_error(df_test.cons, s_pred_frame['mean'])
print(f'MAE: {error:.3f}')

plt.figure()
plt.plot(df_test.cons, label='data')
plt.plot(s_pred_frame['mean'], label='Mean Pred')
plt.plot(tot_pred, label='GARCH Pred')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.show()

# %% ------------------------- XGB  ------------------------------
# Helper functions

# Set constants
N_LAGS = 1
N_TEST = 8

# Predict function
def get_model(df_trainX: pd.DataFrame, df_trainy:pd.DataFrame) -> pd.DataFrame:
    xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
    xgb_model.fit(trainX, trainy)

    yhat = xgb_model.predict(testX)
    df_yhat = pd.Series(yhat, testy.index)
    return xgb_model, df_yhat

def get_train_test(df:pd.DataFrame, N_TEST:int):
    df_train = df.iloc[:-N_TEST, :]
    trainX = df_train.iloc[:,1:]
    trainy = df_train.iloc[:,0]
    df_test = df.iloc[-N_TEST:, :]
    testX = df_test.iloc[:,1:]
    testy = df_test.iloc[:,0]
    return trainX, trainy, testX, testy
    
def eval_data(yhat:pd.DataFrame, truey:pd.DataFrame,
              df_X:pd.DataFrame, pred_model):
    error = mean_absolute_error(truey, yhat)
    print(f'MAE: {error:.3f}')
    
    plt.figure()
    plt.plot(df_X, label='Data')
    plt.plot(yhat, label='Predicted')
    plt.show()
    
    plt.figure()
    plt.plot(truey, label='Data')
    plt.plot(yhat, label='Predicted')
    plt.show()
    
    fig, ax = plt.subplots(figsize=(12,6))
    plot_importance(pred_model, max_num_features=20, ax=ax)
    plt.show()
    return

#%% Initial basic setup
# Make initial test
# Split data
# Train
# Predict
# Evaluate

xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
xgb_model.fit(trainX, trainy)

yhat = xgb_model.predict(testX)
df_yhat = pd.Series(yhat, testy.index)

error = mean_absolute_error(testy, yhat)
print(f'MAE: {error:.3f}')

plt.figure()
plt.plot(uscc.cons, label='Data')
plt.plot(df_yhat, label='XGB Pred')
plt.legend()
plt.show()

plt.figure()
plt.plot(testy, label='Data')
plt.plot(df_yhat, label='XGB Pred')
plt.legend()
plt.show()

fig, ax = plt.subplots(figsize=(12,6))
plot_importance(xgb_model, max_num_features=20, ax=ax)
plt.show()

#%%

trainX, trainy, testX, testy = get_train_test(uscc, N_TEST)
xgb_model, y_hat = get_model(trainX, trainy)
eval_data(y_hat, testy, uscc.cons, xgb_model)

# %% Make features 

# Time series features mostly relates to time
# Lags, rolling window mean, std, max/min, month flags, cyclical time, year ord count
# Data related, transform target or features, trend

# Data
df_feat = uscc.copy(deep=True)
#df_feat['drug'] = np.log(df_feat['drug'] + 1)

# Lag features
# Make lags
for i in range(N_LAGS):
    df_feat[[f'cons_lags{i+1}', f'inc_lag{i+1}', f'prod_lag{i+1}', f'sav_lag{i+1}', f'unem_lag{i+1}']] = \
        df_feat.iloc[:,i:i+5].shift(1)

display(df_feat)

# %%

trainX, trainy, testX, testy = get_train_test(df_feat, N_TEST)
xgb_model1, y_hat = get_model(trainX, trainy)
eval_data(y_hat, testy,df_feat.cons, xgb_model1)

# %%
