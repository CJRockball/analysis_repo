#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.regression.linear_model import OLS
from statsmodels.formula.api import ols

from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

import lightgbm as lgb
from lightgbm import log_evaluation, early_stopping

seed = 42


from PythonTsa.datadir import getdtapath
dtapath = getdtapath()

#%% ------------------------- Drugs Australia ---------------------------------

df = pd.read_csv(dtapath + 'h02July1991June2008.csv', header=0)
df.columns = ['drug']

timeindex = pd.date_range('1991-06', periods=len(df), freq='M')
df.index = timeindex
display(df)
df.plot(title='Australia Drug Sales Monthly')
#print(df.info())
#df = df.reset_index()


# %% Extract time features from time index

print(df.info())
df['year'] = df.index.year
df['month'] = df.index.month
df['day'] = df.index.day
#display(df)

# Rebase year and make month cyclical
df['year_ord'] = df['year'] - df['year'].min()
df['cos_1'] = np.cos(2*np.pi*(df['month']/12))
df['sin_1'] = np.sin(2*np.pi*(df['month']/12))

display(df)

#%% Functions
# Plot
def plt_fcn(ytest, y_pred, ytrain, Xtrain, len_test=24):

    tl = len(Xtrain)
    xx = range(tl)
    plt.figure()
    plt.plot(xx, ytrain)
    plt.plot(range(tl,tl+len_test), ytest, color='red')
    plt.plot(range(tl,tl+len_test), y_pred, color='orange')
    plt.plot

    plt.figure()
    plt.plot(range(len_test), ytest, color='red')
    plt.plot(range(len_test), y_pred, color='orange')
    plt.show()

    rmse_ = root_mean_squared_error(ytest, y_pred)
    r2 = r2_score(ytest, y_pred)
    mae_ = mean_absolute_error(ytest, y_pred)
    print(f"RMSE: {rmse_}, r2: {r2}, MAE: {mae_}")


# Feature extration
def get_lags(df,cname, N_LAGS):
    df_lags = df.copy(deep=True)

    for i in range(N_LAGS):
        df_lags[f'drug_lag{i+1}'] = df_lags.loc[:,cname].shift(i+1)

    #df_lags.dropna(inplace=True)    
    return df_lags

def get_rolling_features(df, cname, lags=[3,6,12], fcns=['mean', 'std']): #:list=[]):
    df_feature = df.copy(deep=True)
    rolling_df = pd.concat([
        # Rolling window of l
        df_feature[cname].rolling(l)
        # Use agg dict for different functions
        .agg({f'{cname}_{l}lags_{str(f)}': f for f in fcns}) 
        for l in lags], axis=1)
    
    df_feature = df_feature.assign(**rolling_df.to_dict("list"))
    return df_feature

def fourier_terms(df, n_terms):
    df_temp = df.copy(deep=True)
    max_value = 12
    for i in range(1, n_terms+1):
        df_temp[f'sin_{i}'] = np.sin((2*np.pi*df['month'])/max_value)
        df_temp[f'cos_{i}'] = np.cos((2*np.pi*df['month'])/max_value)
    return df_temp


# %% Dataset v1

df_v1 = df.copy(deep=True)

Xtrain = df_v1.iloc[:-24]
ytrain = Xtrain.pop('drug')
Xtest = df_v1.iloc[-24:]
ytest = Xtest.pop('drug')

# %% Basic regression fit, predict all

features = ['year_ord', 'cos_1', 'sin_1']
trn = lgb.Dataset(Xtrain[features], ytrain)
tst = [lgb.Dataset(Xtest[features], ytest)]

params = ({#"device"           : "gpu",
        "boosting_type"     : 'gbdt',
        "objective"        : "regression_l2",
        "metrics"          : "rmse",
        "n_estimators"     : 5000,
        "max_depth"        : 10,
        "learning_rate"    : 0.03,
        "colsample_bytree" : 0.55,
        "subsample"        : 0.80,
        "random_state"     : seed,
        "reg_lambda"       : 1.25,
        "reg_alpha"        : 0.001,
        "verbose"          : -1,
        "n_jobs"           : 4,
         }
      )

rgr1 = lgb.train(params, trn, valid_sets=tst, 
                callbacks=[log_evaluation(100), early_stopping(100, verbose = False)]
                )

y_pred = rgr1.predict(Xtest[features])
plt_fcn(ytest, y_pred, ytrain, Xtrain)

#%% Dataset v2. Add lags, predict all

df_v2 = df.copy(deep=True)
df_v2['drug'] = np.log(df_v2['drug'])
TARGET = 'drug'

df_v2b = get_lags(df_v2, TARGET, 3)
df_v2c = get_rolling_features(df_v2b, TARGET)
df_v2d = fourier_terms(df_v2c, 4)
# display(df_v2d)

Xtrain = df_v2d.iloc[:-24]
ytrain = Xtrain.pop(TARGET)
Xtest = df_v2d.iloc[-24:]
ytest = Xtest.pop(TARGET)



#%%
#, 'drug_6lags_std',, 'drug_6lags_mean', 'drug_3lags_std', ,'drug_12lags_mean', 'drug_12lags_std'
features = ['year_ord', 'cos_1', 'sin_1', 'drug_lag1',
       'drug_lag2', 'drug_lag3', 'drug_3lags_mean']

trn = lgb.Dataset(Xtrain[features], ytrain)
tst = [lgb.Dataset(Xtest[features], ytest)]

rgr2 = lgb.train(params, trn, valid_sets=tst, 
                callbacks=[log_evaluation(100), early_stopping(100, verbose = False)]
                )


y_pred = rgr2.predict(Xtest[features])
plt_fcn(ytest, y_pred, ytrain, Xtrain)

#%% Feature importance

df_features = pd.DataFrame(data=rgr2.feature_importance('gain'), index=features, columns=['Importance'])

display(df_features)
df_features.sort_values(by=['Importance']).plot.bar()
plt.title('Feature Importance by Gain')
plt.show()

#%% Dataset v2, predict one step
features = ['year_ord', 'cos_1', 'sin_1', 'drug_lag1', 'drug_lag2', 'drug_lag3']
df_tmp = pd.concat([Xtrain, ytrain], axis=1)

trn = lgb.Dataset(Xtrain[features], ytrain)
tst = [lgb.Dataset(Xtest[features], ytest)]

predictions = []
for i in range(24):
    
    rgr = lgb.train(params, trn, valid_sets=trn, 
                    callbacks=[log_evaluation(100), early_stopping(100, verbose = False)]
                    )

    y_pred = rgr.predict(Xtest[features].iloc[i,:])
    predictions.append(y_pred[0])

    df_tmp = pd.concat([df_tmp, Xtest.iloc[-24+i,:].to_frame().T]).reset_index(drop=True)
    df_tmp.loc[180+i,'drug'] = y_pred[0]
    df_tmp = get_lags(df_tmp, 'drug', 3)

    trn = lgb.Dataset(df_tmp[features], df_tmp['drug'])


#%% plot predictions

print(predictions)    
        
plt.figure()
plt.plot(range(len(predictions)), predictions)
plt.plot(range(len(predictions)), ytest)
plt.show()

rmse_ = root_mean_squared_error(ytest, np.array(predictions))
r2 = r2_score(ytest, np.array(predictions))
mae_ = mean_absolute_error(ytest, np.array(predictions))
print(f"RMSE: {rmse_}, r2: {r2}, MAE: {mae_}")

# %% Try differentiating

df_diff = df.copy(deep=True)

df_diff['drug_diff'] = df_diff['drug'].diff(3)
#df_diff_1.dropna(inplace=True)

df_diff['drug_diff'].plot()
plt.show()

df_diffb = get_lags(df_diff, 'drug_diff', 12)
df_diffc = get_rolling_features(df_diffb, 'drug_diff')
df_diffd = fourier_terms(df_diffc, 4)

Xtrain = df_diffd[:-24]
ytrain = Xtrain.pop('drug_diff')
Xtest = df_diffd[-24:]
ytest = Xtest.pop('drug_diff')

#%%

print(Xtrain.columns)

#%% differentiatin data, add lagg

features = ['year_ord', 'cos_1', 'sin_1',
       'drug_lag1', 'drug_lag2', 'drug_lag3',
       'drug_lag6', 'drug_lag9', 'drug_lag12', 'drug_diff_3lags_mean',
       'drug_diff_3lags_std']
#['year_ord', 'cos_1', 'sin_1', 'drug_lag1', 'drug_lag2', 'drug_lag3']

trn = lgb.Dataset(Xtrain[features], ytrain)
tst = [lgb.Dataset(Xtest[features], ytest)]

rgr2 = lgb.train(params, trn, valid_sets=tst,
                 callbacks=[log_evaluation(100), early_stopping(100, verbose = False)])


y_pred = rgr2.predict(Xtest[features])

plt_fcn(ytest, y_pred, ytrain,Xtrain)

#%%

df_features = pd.DataFrame(data=rgr2.feature_importance('gain'), index=features, columns=['Importance'])

display(df_features)
df_features.sort_values(by=['Importance']).plot.bar()
plt.show()


# %% differentiating, predict one ahead

features = ['year_ord', 'cos_1', 'sin_1', 'drug_lag1', 'drug_lag2', 'drug_lag3']
df_tmp = pd.concat([Xtrain, ytrain], axis=1)

trn = lgb.Dataset(Xtrain[features], ytrain)
tst = [lgb.Dataset(Xtest[features], ytest)]

predictions = []
for i in range(24):
    # Fit model
    rgr = lgb.train(params, trn, valid_sets=trn, 
                    callbacks=[log_evaluation(500), early_stopping(100, verbose = False)]
                    )
    # Predict one step
    y_pred = rgr.predict(Xtest[features].iloc[i,:])
    # Save prediction
    predictions.append(y_pred[0])

    # Add last prediction data row
    df_tmp = pd.concat([df_tmp, Xtest.iloc[-24+i,:].to_frame().T]).reset_index(drop=True)
    # Add predicted target value
    df_tmp.loc[180+i,'drug_diff'] = y_pred[0]
    # Recalculate lags
    df_tmp = get_lags(df_tmp, 'drug_diff', 3)

    # Make new training dataset
    trn = lgb.Dataset(df_tmp[features], df_tmp['drug_diff'])

#%%

print(predictions)    
        
plt.figure()
plt.plot(range(len(predictions)), predictions)
plt.plot(range(len(predictions)), ytest)
plt.show()

rmse_ = root_mean_squared_error(ytest, np.array(predictions))
r2 = r2_score(ytest, np.array(predictions))
mae_ = mean_absolute_error(ytest, np.array(predictions))
print(f"RMSE: {rmse_}, r2: {r2}, MAE: {mae_}")

# %%
