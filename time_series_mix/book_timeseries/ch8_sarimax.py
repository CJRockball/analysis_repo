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

from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor
from IPython import display

dtapath = getdtapath()

# %% Set up df

uscc = pd.read_csv(dtapath + 'USEconomicChange.csv')
uscc.to_csv('USEconomicChange.csv')
timeindex = pd.date_range('1970-03', periods=len(uscc), freq='QE')
uscc.index = timeindex
uscc.rename(columns={'Consumption':"cons", 'Income':"inc", 'Production':'prod', 'Savings':"sav","Unemployment":"unem"},
            inplace=True)
uscc.drop(columns=['Time'], inplace=True)
display(uscc)

uscc['cons'].plot(title='CONS')

# %% EDA

multi_ACFfig(uscc, nlags=12)
plt.show()

# %% Split datta

df_train = uscc.iloc[:-8,:]
trainX = df_train[['inc', 'prod', 'sav','unem']]
trainy = df_train['cons']
df_test = uscc.iloc[-8:,:]
testX = df_test[['inc', 'prod', 'sav','unem']]
testy = df_test['cons']

trainX = sm.add_constant(trainX, prepend=False)
testX = sm.add_constant(testX, prepend=False)

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

display(sarimaxresid)
sarimaxresid.plot(title='SARIMAX Residuals')


# %% Evaluate residuals of squares

acf_pacf_fig(sarimaxresid**2, both=True, lag=20)
plt.show()
plot_LB_pvalue(sarimaxresid**2, noestimatedcoef=2, nolags=20)
plt.show()


# %% Fit GARCH to residuals

archmod = arch_model(sarimaxresid, mean='Zero').fit(disp='off')
print(archmod.summary())

# %% Check GARCH residuals

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
display(g_pred)
g_pred['data'].plot(title='GARCH sigma prediction')

tot_pred = s_pred_frame['mean'] + g_pred.data.values

error_sarimax = mean_absolute_error(testy, s_pred_frame['mean'])
rmse_sarimax = root_mean_squared_error(testy, s_pred_frame['mean'])

print(f'SARIMAX MAE : {error_sarimax:.3f}')
print(f'SARIMAX RMSE: {rmse_sarimax:.3f}')

# %%

plt.figure()
plt.plot(df_test.cons, label='data')
plt.plot(s_pred_frame['mean'], label='Mean Pred')
plt.plot(tot_pred, label='GARCH Pred')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.show()



# %% ------------------------- ARIMA of CONS ---------------------------------------

df_cons = uscc['cons'].copy(deep=True).to_frame()
display(df_cons)

# %% Check if stationary

acf_pacf_fig(df_cons, both=True, lag=30)
plt.show()
sm.tsa.kpss(df_cons, regression='c', nlags=30)

# %% Suggest arima orders

choose_arma2(df_cons, max_p=5, max_q=5, ctrl=1.05)

# %% model arima

arima3 = ARIMA(df_cons, order=(3,0,0), trend='c').fit()
print(arima3.summary())

# %% check residuals

resid = arima3.resid
acf_pacf_fig(resid, both=True, lag=30)
plt.show()

plot_LB_pvalue(resid, noestimatedcoef=3, nolags=30)
plt.show()

# %% Make train test
N_TEST = 12
train = df_cons.iloc[:-N_TEST,:]
test = df_cons.iloc[-N_TEST:,:]

# %% predict, check

arima_3ml = ARIMA(train, order=(3,0,0), trend='c').fit()

y_hat = arima_3ml.get_forecast(steps=N_TEST)
y_hat_frame = y_hat.summary_frame(alpha=0.05)
display(y_hat_frame)

mae = mean_absolute_error(test, y_hat_frame['mean'])
rmse = root_mean_squared_error(test, y_hat_frame['mean'])
print(f'ARIMA MAE : {mae:.3f}')
print(f'ARIMA RMSE: {rmse:.3f}')

# %% -------------------------------- XGB ----------------------------------------
from xgboost import XGBRegressor
from xgboost import plot_importance

df_xgb = uscc.copy(deep=True)
display(df_xgb)

# %% Check correlation

df_corr = df_xgb.corr()
df_n_cols = df_xgb.shape[1]
df_col_names = df_xgb.columns
# plt.figure()
# plt.imshow(df_corr)
# plt.colorbar()
# plt.show()

fig, ax = plt.subplots()
im = ax.imshow(df_corr, cmap='coolwarm')
ax.grid(False)
ax.xaxis.set(ticks=(0, 1, 2,3,4), ticklabels=df_col_names)
ax.yaxis.set(ticks=(0, 1, 2,3,4), ticklabels=df_col_names)
for i in range(df_n_cols):
    for j in range(df_n_cols):
        ax.text(j, i, round(df_corr.iloc[i, j],2), ha='center', va='center',
                color='black')
plt.colorbar(im, ax=ax, format='% .2f')
plt.show()

# %%
# Make features
# Make test_train
# Make one step predict

# Start by raw data

# %%
# Make train/test and train/test/validation
def train_test(df, N_TEST):
    df_temp = df.copy(deep=True)
    df_train = df_temp.iloc[:-N_TEST, :]
    trainX = df_train.iloc[:,1:]
    trainy = df_train.iloc[:,0]
    df_test = df_temp.iloc[-N_TEST:, :]
    testX = df_test.iloc[:,1:]
    testy = df_test.iloc[:,0]
    
    assert df_train.shape[0] == df_temp.shape[0] - N_TEST
    assert df_test.shape[0] == N_TEST
    
    return trainX, trainy, testX, testy

def train_test_val(df, N_TEST, N_VAL):
    df_temp = df.copy(deep=True)
    df_train = df_temp.iloc[:-(N_TEST+N_VAL), :]
    trainX = df_train.iloc[:,1:]
    trainy = df_train.iloc[:,0]
    df_test = df_temp.iloc[-(N_TEST+N_VAL):-N_VAL, :]
    testX = df_test.iloc[:,1:]
    testy = df_test.iloc[:,0]
    df_val = df_temp.iloc[-N_VAL:, :]
    valX = df_val.iloc[:,1:]
    valy = df_val.iloc[:,0]
    
    assert df_train.shape[0] == df_temp.shape[0] - N_TEST - N_VAL
    assert df_test.shape[0] == N_TEST
    assert df_val.shape[0] == N_VAL
    
    return trainX, trainy, testX, testy, valX, valy

def make_lags(df, N_LAGS):
    for i,col in enumerate(df.columns):
        for j in range(N_LAGS):
            df[f'{col}_lag{j+1}'] = df.iloc[:,i].shift(j+1)
            
    df.dropna(inplace=True)
    return df

def make_time_features(df):
    df['quarter'] = df.index.month.astype(int) / 3
    df['year'] = df.index.year.astype(int) - df.index.year.astype(int).min() + 1
    
    return df


def one_step_predictions(train, test, N_LAGS, N_TEST, model):
    
    df_past = train[-N_TEST,:].copy(deep=True)
    df_future = test.copy(deep=True)
    predict_list = []
    
    for i in range(N_TEST):
        pred = model.predict(df_past.iloc[-1,:])


        
    return

def train_test_multiy(df, N_TEST, N_ORG_FEAT):
    df_temp = df.copy(deep=True)
    df_train = df_temp.iloc[:-N_TEST, :]
    trainMX = df_train.iloc[:, N_ORG_FEAT:]
    trainMY = df_train.iloc[:,:N_ORG_FEAT]
    df_test = df_temp.iloc[-N_TEST:, :]
    testMX = df_test.iloc[:, N_ORG_FEAT:]
    testMY = df_test.iloc[:,:N_ORG_FEAT]    
    return trainMX, trainMY, testMX, testMY

#%% Data Processing
df_test = df_xgb.copy(deep=True)
N_ORG_FEAT = df_test.shape[1]
col_names = df_test.columns
# Make lag features. Adding lag features will require one step predictions
N_LAGS = 4
df_lags = make_lags(df_test, N_LAGS)

# Make time features
#df_lags = make_time_features(df_test)
display(df_lags)

# Split data
N_TEST = 8
N_VAL = 8
# Single output
trainX, trainy, testX, testy = train_test(df_lags, N_TEST)
#trainX, trainy, testX, testy, valX, valy = train_test_val(df_xgb, N_TEST, N_VAL)
# Multioutput split data
trainMX, trainMY, testMX, testMY = train_test_multiy(df_lags, N_TEST, N_ORG_FEAT)

#%% Multi out xgb model and fit

eval_metric = ['mae']
xgb_model_m = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
xgb_model_m.fit(trainMX, trainMY)

# %% check predictions
# One step multi output prediction

def one_step_multi_predict(train, test, N_TEST, N_LAGS, N_ORG_FEAT, df_col_names, model):

    df_past = train.iloc[-N_TEST:,:].copy(deep=True)
    df_future = test.copy(deep=True)
    pred_list = []
    for i in range(N_TEST):
        y_hat = model.predict(df_past.iloc[-1,:].to_frame().T)
        pred_list.append(y_hat[0])
        
        df_past = pd.concat([df_past,df_future.iloc[i,:].to_frame().T], axis=0)
        for j in range(N_ORG_FEAT):
            df_temp = df_past.iloc[-1,N_LAGS*j:(N_LAGS*j)+N_LAGS].to_frame().T
            assert df_temp.shape == (1,N_LAGS), print(df_temp.shape)
            df_temp = df_temp.shift(1,axis=1)
            df_temp.iloc[0,0] = y_hat[0][j]
            df_past.iloc[-1, N_LAGS*j:(N_LAGS*j)+N_LAGS] = df_temp.iloc[0,:].values

    df_y_hat = pd.DataFrame(pred_list, columns=df_col_names, index=test.index)        
    return df_y_hat         

y_hat = one_step_multi_predict(trainMX, testMX, N_TEST, N_LAGS, N_ORG_FEAT, df_col_names, xgb_model_m)
display(y_hat)

#%% Check cons predictions

mae  = mean_absolute_error(testMY['cons'], y_hat['cons'])
rmse = root_mean_squared_error(testMY['cons'], y_hat['cons'])
print(f'Multi Predict MAE : {mae:.3f}')
print(f'Multi Predict RMSE: {rmse:.3f}')

# %% Make model and fit
eval_metric = ['mae']
xgb_model1 = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
xgb_model1.fit(trainX, trainy)

#%% Fit hyperparameters
eval_set = [(trainX, trainy), (testX, testy)]
eval_metric = ['mae']

xgb_model1 = XGBRegressor(objective='reg:squarederror', eval_metric=eval_metric,
                          n_estimators=3500, max_depth=5, 
                          learning_rate=0.001, colsample_bytree=0.7)

xgb_model1.fit(trainX, trainy,
               eval_set=eval_set,
               verbose=False)

history = xgb_model1.evals_result()
xx = range(0,len(history['validation_0']['mae']))

plt.figure()
plt.plot(xx,history["validation_0"]['mae'], label="Train")
plt.plot(xx,history['validation_1']['mae'], label='Test')
plt.legend()
plt.show()

# %% Check model wieghts

plot_importance(xgb_model1, max_num_features=20)
plt.show()

# %% check predictions

y_hat = xgb_model1.predict(testX)

mae = mean_absolute_error(testy, y_hat)
rmse = root_mean_squared_error(testy, y_hat)
print(f'MAE : {mae:.3f}')
print(f'RMSE: {rmse:.3f}')

# %%
