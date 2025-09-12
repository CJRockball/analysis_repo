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

dfa = pd.read_csv(dtapath + 'h02July1991June2008.csv', header=0)
dfa.columns = ['drug']

timeindex = pd.date_range('1991-06', periods=len(dfa), freq='M')
dfa.index = timeindex
display(dfa)
dfa.plot(title='Australia Drug Sales Monthly')

# %% Automatic differentiation
# Diff seasonal 12, trend 1

Ddfa = sm.tsa.statespace.tools.diff(dfa, k_diff=0, 
                    k_seasonal_diff=1, seasonal_periods=12)

print(type(Ddfa))
display(Ddfa)

Ddfa.plot()
plt.show()

#Check if stationary
acf_pacf_fig(Ddfa, both=True, lag=36)
plt.show()
sm.tsa.kpss(Ddfa, regression='c', nlags='auto')

#%% Train/test
N_TEST = 24
Ddfa_train = Ddfa.iloc[:-N_TEST,:]
Ddfa_test = Ddfa.iloc[-N_TEST:,:]

Ddfa_train.plot(title='Diff Sales')
plt.show()

#Check if stationary
acf_pacf_fig(Ddfa_train, both=True, lag=36)
plt.show()
sm.tsa.kpss(Ddfa_train, regression='c', nlags='auto')

# %% Term L4 is not significant. Fit again without the term

sarima1 = sm.tsa.SARIMAX(Ddfa_train, order=(0,0,[1,1,1,0,1]),
                                          seasonal_order=(2,1,0,12))
sarimaMod = sarima1.fit(disp=False)
print(sarimaMod.summary())

resid1 = sarimaMod.resid[12:]
plot_ResidDiag(resid1, noestimatedcoef=6, nolags=48, lag=36)
plt.show()

# %% Predict, restore

sarima_prediction = sarimaMod.get_forecast(steps=N_TEST)
df_pred = sarima_prediction.summary_frame(alpha=0.05)

df_pred['mean_recon'] = df_pred['mean'] + dfa.iloc[-N_TEST:,0]
display(df_pred)

error = mean_absolute_error(dfa.iloc[-N_TEST:,0], df_pred['mean_recon'])
print(f'MAE: {error:.3f}')

plt.figure()
plt.plot(dfa.drug, label='Expected')
plt.plot(df_pred.mean_recon, label='Prediction')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.show()

plt.figure()
plt.plot(dfa.iloc[-N_TEST:,0], label='Expected')
plt.plot(df_pred.mean_recon, label='Prediction')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.show()

# %% --------------------- XGB ------------------------------------
# Default test with only lags. Turns out to give the best result
N_IN = 12
dfa1 = dfa.copy()

# Make lag features
def lag_feat(data, n_in, dropnan=True):
    
    for i in range(n_in):
        data[f'drug_lag{i+1}'] = data.iloc[:,i].shift(1)
        
    if dropnan:
        data = data.dropna()
    
    return data

ldfa = lag_feat(dfa1, N_IN)
display(ldfa)

# %% Make train/test
TEST_SIZE = 12

ldf_train = ldfa.iloc[:-TEST_SIZE,:]
trainX = ldf_train.iloc[:,1:]
trainy = ldf_train.iloc[:,0]
ldf_test = ldfa.iloc[-TEST_SIZE:,:]
testX = ldf_test.iloc[:,1:]
testy = ldf_test.iloc[:,0]

#%% fit model

xgb_model = XGBRegressor(objective='reg:squarederror',  eval_metric='rmsle', n_estimators=1000)
xgb_model.fit(trainX,trainy)

# %% Predictions

def one_predict(df_test:pd.DataFrame, n_in:int, test_size:int, model):
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
    return df_y_hat

df_y_hat = one_predict(ldf_test, N_IN, TEST_SIZE, xgb_model)

error = mean_absolute_error(testy, df_y_hat)
print(f'MAE: {error:.3f}') # MAE: 0.072

# plot expected vs preducted
plt.plot(testy, label='Expected')
plt.plot(df_y_hat, label='Predicted')
plt.legend()
plt.xticks(rotation=45, ha='right')
plt.show()

# %% ---------------- XGB Feature engineering --------------------------------------
# Trying out different features. Nothing improves original result
from statsmodels.regression.linear_model import OLS

df_feat = dfa.copy(deep=True)
#df_feat['drug'] = np.log(df_feat['drug'] + 1)
# Make lags
for i in range(12):
    df_feat[f'drug_lag{i+1}'] = df_feat.iloc[:,i].shift(1)

# # Rolling mean
# df_feat['3mo_mean'] = df_feat['drug'].rolling(window = 3).mean()
# df_feat['3mo_std'] = df_feat['drug'].rolling(window = 3).std()
# df_feat['3mo_max'] = df_feat['drug'].rolling(window = 3).max()
# df_feat['3mo_min'] = df_feat['drug'].rolling(window = 3).min()

# # Rolling mean
# df_feat['6mo_mean'] = df_feat['drug'].rolling(window = 6).mean()
# df_feat['6mo_std'] = df_feat['drug'].rolling(window = 6).std()
# df_feat['6mo_max'] = df_feat['drug'].rolling(window = 6).max()
# df_feat['6mo_min'] = df_feat['drug'].rolling(window = 6).min()

df_feat = df_feat.dropna()

#df_feat.drugl.plot(title='Logged drug')

# # Create Month features
# df_feat['month'] = df_feat.index.month
# # Cyclic Month
# df_feat['Month Sin'] = np.sin(2*np.pi*df_feat.month/12)
# df_feat['Month Cos'] = np.cos(2*np.pi*df_feat.month/12)
# # Highlight high vol months
# df_feat['Jan'] = df_feat['month'].apply(lambda x: 1 if x==1 else 0)
# df_feat['Feb'] = df_feat['month'].apply(lambda x: 1 if x==2 else 0)
# df_feat['June'] = df_feat['month'].apply(lambda x: 1 if x==6 else 0)
# df_feat['Dec'] = df_feat['month'].apply(lambda x: 1 if x==12 else 0)
# # Year feature
# df_feat['year'] = df_feat.index.year
# df_feat['year'] = df_feat['year'] - df_feat['year'].min()

# # Add trend fit a + bx
# xx = pd.Series(range(len(df_feat)), index=df_feat.index)
# y = df_feat.drug

# xx = sm.add_constant(xx)
# lin_model = OLS(y,xx).fit()
# #print(lin_model.summary())

# lin_pred = lin_model.predict(xx)
# df_feat['trend'] = lin_pred
# print(lin_pred)

# plt.figure()
# plt.plot(df_feat.drug)
# plt.plot(lin_pred)
# plt.show()

display(df_feat)

# %% Make train/test
TEST_SIZE = 12
df_feat2 = df_feat.copy(deep=True)
# df_feat2 = df_feat2.drop(columns=['month'])
#df_feat2 = df_feat2.drop(columns=['drug_lag6', 'drug_lag8', 'drug_lag5', 'drug_lag4',  'drug_lag7'])

df_feat_train = df_feat2.iloc[:-TEST_SIZE,:]
trainX = df_feat_train.iloc[:,1:]
trainy = df_feat_train.iloc[:,0]
df_feat_test = df_feat2.iloc[-TEST_SIZE:,:]
testX = df_feat_test.iloc[:,1:]
testy = df_feat_test.iloc[:,0]

#%% Make model

xgb_model = XGBRegressor(objective='reg:squarederror',  eval_metric='rmsle', n_estimators=1000)
xgb_model.fit(trainX,trainy)

# %% predict

df_y_hat = one_predict(df_feat_test, N_IN, TEST_SIZE, xgb_model)

error = mean_absolute_error(testy, df_y_hat)
print(f'MAE: {error:.3f}') # MAE: 0.072

# plot expected vs preducted
plt.plot(testy, label='Expected')
plt.plot(df_y_hat, label='Predicted')
plt.legend()
plt.xticks(rotation=45, ha='right')
plt.show()
# %%
from xgboost import plot_importance

fig, ax = plt.subplots(figsize=(12,6))
plot_importance(xgb_model, max_num_features=20, ax=ax)
plt.show();

#%% Get residuals

y_hat_in_sample = xgb_model.predict(trainX)
in_sample_pred_series = pd.Series(y_hat_in_sample, index=trainX.index)
display(in_sample_pred_series)

df_residuals = trainy.copy(deep=True).to_frame()
df_residuals['pred'] = in_sample_pred_series
df_residuals['resid'] = df_residuals['drug'] - df_residuals['pred']
display(df_residuals)

df_residuals['resid'].plot(title='Residuals')

#%% Study residuals

xgb_resid = df_residuals['resid']
plot_ResidDiag(xgb_resid, noestimatedcoef=0, nolags=48, lag=36)
plt.show()

acf_pacf_fig(xgb_resid**2, both=True, lag=36)


# %% --------------------------- XGB PREDICT DIFF SERIES ---------------------------

# df_stat_diff = sm.tsa.statespace.tools.diff(dfa, k_diff=1, 
#                     k_seasonal_diff=1, seasonal_periods=12)

# df_man_diff1 = dfa.diff(12)
# df_man_diff = df_man_diff1.diff(1)

# df_check = pd.concat([df_stat_diff.rename(columns={'drug':'drug_lag'}), df_man_diff], axis=1)
# df_check['org_data'] = dfa.iloc[:,0]

# df_check['reconst_data'] = df_check['drug'] + df_man_diff1['drug'].shift(1)
# df_check['reconst_data2'] = df_check['reconst_data'] + df_check['org_data'].shift(12)
# display(df_check)

# df_man_diff.dropna(inplace=True)
# display(df_man_diff)
# df_man_diff.plot(title='Diff Data')

SEASONAL_DIFF = 12
ORD_DIFF = 1

def diff_df(df, ord_diff=None, seasonal_diff=None):
    if seasonal_diff is not None or seasonal_diff != 0:
        df_diff_seasonal = df.diff(seasonal_diff)
        if ord_diff is not None or ord_diff != 0:
            df_diff_ord = df_diff_seasonal.diff(ord_diff)
            df_diff_ord.dropna()
            return df_diff_seasonal, df_diff_ord
        df_diff_seasonal.dropna()
        return df_diff_seasonal
    
    df_diff_ord = df.diff(ord_diff)
    df_diff_ord.dropna()
    return df_diff_ord

df_man_diff1, df_man_diff = diff_df(dfa, ORD_DIFF, SEASONAL_DIFF)

# %% Create lags

N_LAGS = 12
def get_lags(df, N_LAGS):
    df_lags = df.copy(deep=True)

    for i in range(N_LAGS):
        df_lags[f'drug_lags{i+1}'] = df_lags.iloc[:,i].shift(1)

    df_lags.dropna(inplace=True)    
    #display(df_lags)
    return df_lags

df_lags = get_lags(df_man_diff, N_LAGS)

#%% Train/test split
N_TEST = 12
N_VAL = 0

def train_test(df_in, N_TEST, N_VAL=0):
    df_temp = df_in.copy(deep=True)
    
    df_train = df_temp.iloc[:-(N_TEST + N_VAL), :]
    trainX = df_train.iloc[:,1:]
    trainy = df_train.iloc[:,0]
    df_test = df_temp.iloc[-N_TEST:, :]
    testX = df_test.iloc[:,1:]
    testy = df_test.iloc[:,0]
    
    if N_VAL > 0:
        df_val = df_temp.iloc[-(N_TEST + N_VAL):-N_TEST,:]
        valX = df_val.iloc[:,1:]
        valy = df_val.iloc[:,0]
        return trainX, trainy, valX, valy, testX, testy
    
    return trainX, trainy, testX, testy

trainX, trainy, testX, testy = train_test(df_lags, N_TEST)

# %%

xgb_model1 = XGBRegressor(objective='reg:squarederror', n_estimators=100)
xgb_model1.fit(trainX, trainy)

#%% One step prediction

def one_step_pred(train, N_TEST):
    pred_list = []

    in1 = train.iloc[-1,:].to_frame().T

    for i in range(N_TEST):
        pred1 = xgb_model1.predict(in1)
        pred_list.append(pred1[0])

        in1 = in1.shift(1, axis=1)
        in1['drug_lags1'] = pred1

    y_hat_series = pd.Series(pred_list, index=testy.index)
    return y_hat_series

y_hat_series = one_step_pred(trainX, N_TEST)
display(y_hat_series)

plt.figure()
plt.plot(testy, label='diff Data')
plt.plot(y_hat_series, label='Prediction')
plt.show()


# %% Restore series

y_hat_df = y_hat_series.to_frame()
y_hat_df.columns = ['pred']
y_hat_df['data'] = testy
y_hat_df['lags1'] = df_man_diff1.iloc[-N_TEST-1:,0].shift(1).dropna()
y_hat_df['lags2'] = dfa.iloc[-N_TEST:,0]


y_hat_df['final_pred1'] = y_hat_df['pred'] + y_hat_df['lags1']
y_hat_df['final_pred2'] = y_hat_df['final_pred1'] + y_hat_df['lags2']
#y_hat_df['reconst_data'] = y_hat_df['data'] + y_hat_df['lags']
y_hat_df['org_data'] = dfa.iloc[-N_TEST:,0]
display(y_hat_df)


# %%
error = mean_absolute_error(dfa.iloc[-N_TEST:,0], y_hat_df['final_pred2'])
print(f'MAE: {error:.3f}')

plt.figure()
plt.plot(dfa.iloc[-N_TEST:,0], label='Org Data')
plt.plot(y_hat_df['final_pred2'], label='Pred Data')
plt.legend()
plt.show()

# %% --------------------------- XGB TRY OPTIMIZING HYPERPARAMETERS ---------------------------

# Differentiate series
def get_diff_df(df, ord_diff=0, seasonal_diff=0):
    if seasonal_diff != 0:
        df_diff_seasonal = df.diff(seasonal_diff)
        if ord_diff != 0:
            df_diff_ord = df_diff_seasonal.diff(ord_diff)
            df_diff_ord.dropna(inplace=True)
            return df_diff_seasonal, df_diff_ord
        
        df_diff_seasonal.dropna(inplace=True)
        return df_diff_seasonal
    
    df_diff_ord = df.diff(ord_diff)
    df_diff_ord.dropna(inplace=True)
    return df_diff_ord


# Make lags
def get_lags(df, N_LAGS):
    df_lags = df.copy(deep=True)

    for i in range(N_LAGS):
        df_lags[f'drug_lags{i+1}'] = df_lags.iloc[:,i].shift(1)

    df_lags.dropna(inplace=True)    
    #display(df_lags)
    return df_lags


# Train/test split
def train_test(df_in, N_TEST, N_VAL=0):
    df_temp = df_in.copy(deep=True)
    
    df_train = df_temp.iloc[:-(N_TEST + N_VAL), :]
    trainX = df_train.iloc[:,1:]
    trainy = df_train.iloc[:,0]
    df_test = df_temp.iloc[-N_TEST:, :]
    testX = df_test.iloc[:,1:]
    testy = df_test.iloc[:,0]
    
    if N_VAL > 0:
        df_val = df_temp.iloc[-(N_TEST + N_VAL):-N_TEST,:]
        valX = df_val.iloc[:,1:]
        valy = df_val.iloc[:,0]
        return trainX, trainy, valX, valy, testX, testy
    
    return trainX, trainy, testX, testy


# One step prediction
def one_step_pred(train, N_TEST):
    pred_list = []

    in1 = train.iloc[-1,:].to_frame().T

    for i in range(N_TEST):
        pred1 = xgb_model1.predict(in1)
        pred_list.append(pred1[0])

        in1 = in1.shift(1, axis=1)
        in1['drug_lags1'] = pred1

    y_hat_series = pd.Series(pred_list, index=testy.index)
    return y_hat_series

#%%
df_xgb = dfa.copy(deep=True)
SEASONAL_DIFF = 12
ORD_DIFF = 0
df_diff = get_diff_df(df_xgb, ord_diff=ORD_DIFF, seasonal_diff=SEASONAL_DIFF)

N_LAGS = 12
df_lags = get_lags(df_diff, N_LAGS)

N_TEST = 12
N_VAL = 0
#trainX, trainy, valX, valy, testX, testy = train_test(df_lags, N_TEST, N_VAL)
trainX, trainy, testX, testy = train_test(df_lags, N_TEST, N_VAL)


# %% Def model

# xgb_model1 = XGBRegressor(objective='reg:squarederror', n_estimators=100)
# xgb_model1.fit(trainX, trainy)

eval_set = [(trainX, trainy), (testX, testy)]
eval_metrics = ['mae']

xgb_model1 = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
                        #n_estimators = 100,random_state=42,
                        #tree_method='hist', max_depth=5)

xgb_model1.fit(trainX, trainy,
             eval_set=eval_set,
             eval_metric=eval_metrics,
            #  num_boost_round=5,
             #early_stopping_rounds=5,
             verbose=False)

#print(xgb_model1.get_params())

#%%

history = xgb_model1.evals_result()
xx = range(0,len(history['validation_0']['mae']))

plt.figure()
plt.plot(xx,history["validation_0"]['mae'], label="Train")
plt.plot(xx,history['validation_1']['mae'], label='Test')
plt.legend()
plt.show()

#%% One step prediction

y_hat_series = one_step_pred(trainX, N_TEST)
#isplay(y_hat_series)

# plt.figure()
# plt.plot(testy, label='diff Data')
# plt.plot(y_hat_series, label='Prediction')
# plt.show()

# Restore series
y_hat_df = y_hat_series.to_frame()
y_hat_df.columns = ['pred']
y_hat_df['data'] = testy
y_hat_df['lags'] = df_xgb.iloc[-N_TEST:,0]

y_hat_df['final_pred'] = y_hat_df['pred'] + y_hat_df['lags']
#display(y_hat_df)

# Plot Data and predictions
error = mean_absolute_error(df_xgb.iloc[-N_TEST:,0], y_hat_df['final_pred'])
print(f'MAE: {error:.3f}')

plt.figure()
plt.plot(df_xgb.iloc[-N_TEST:,0], label='Org Data')
plt.plot(y_hat_df['final_pred'], label='Pred Data')
plt.legend()
plt.show()

# %% Get train/test dataset 
