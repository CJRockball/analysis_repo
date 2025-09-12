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

dfa = pd.read_csv(dtapath + 'h02July1991June2008.csv', header=0)
dfa.columns = ['drug']

timeindex = pd.date_range('1991-06', periods=len(dfa), freq='M')
dfa.index = timeindex
display(dfa)
dfa.plot(title='Australia Drug Sales Monthly')


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
def get_lags(df, N_LAGS, moving_win=False):
    df_lags = df.copy(deep=True)

    for i in range(N_LAGS):
        df_lags[f'drug_lags{i+1}'] = df_lags.iloc[:,i].shift(1)

    if moving_win:
        # When adding rolling mean, need to add predicted ones too
        # Rolling mean
        df_lags['3mo_mean'] = df_lags['drug_lags1'].rolling(window = 3).mean()
        df_lags['3mo_std'] = df_lags['drug_lags1'].rolling(window = 3).std()
        df_lags['3mo_max'] = df_lags['drug_lags1'].rolling(window = 3).max()
        df_lags['3mo_min'] = df_lags['drug_lags1'].rolling(window = 3).min()

        # Rolling mean
        # df_lags['6mo_mean'] = df_lags['drug_lags1'].rolling(window = 6).mean()
        # df_lags['6mo_std'] = df_lags['drug_lags1'].rolling(window = 6).std()
        # df_lags['6mo_max'] = df_lags['drug_lags1'].rolling(window = 6).max()
        # df_lags['6mo_min'] = df_lags['drug_lags1'].rolling(window = 6).min()
    
        # EWMA
        #df_lags['ewma'] = df_lags['drug_lags1'].ewm(alpha=0.95).mean()
            

    df_lags.dropna(inplace=True)    
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
def one_step_pred(train, test, N_TEST, N_LAGS, model, moving_win=False):
    pred_list = []
    df_future = test.copy(deep=True) 
    df_pred = train.iloc[-N_LAGS:,:]
    df_pred = pd.concat([df_pred,df_future.iloc[0,:].to_frame().T])

    
    for i in range(N_TEST):
        # For some reason concat creates a row with nans. This line removes that
        df_pred = df_pred.dropna()
        # Predict and add to save list
        pred1 = model.predict(df_pred.iloc[-1,:].to_frame().T)
        pred_list.append(pred1[0])

        # Add prediction
        df_future.iloc[i, 0] = pred1
        # Make lag
        for j in range(N_LAGS-1):
            df_future.iloc[i,j+1] = df_pred.iloc[-1,j]
        
        df_move = df_future.iloc[i,:].to_frame().T

        #assert df_move.shape == (1,27)
        df_pred = pd.concat([df_pred,df_move])
        #assert df_pred.shape == (N_TEST+i+1, 27)
        
        if moving_win:
            df_pred.loc[-1,'3mo_mean'] = df_pred['drug_lags1'].rolling(window = 3).mean().iloc[-1]
            df_pred.loc[-1,'3mo_std']  = df_pred['drug_lags1'].rolling(window = 3).std().iloc[-1]
            df_pred.loc[-1,'3mo_max']  = df_pred['drug_lags1'].rolling(window = 3).max().iloc[-1]
            df_pred.loc[-1,'3mo_min']  = df_pred['drug_lags1'].rolling(window = 3).min().iloc[-1]
            # # Rolling mean
            # df_pred.loc[-1,'6mo_mean'] = df_pred['drug_lags1'].rolling(window = 6).mean().iloc[-1]
            # df_pred.loc[-1,'6mo_std']  = df_pred['drug_lags1'].rolling(window = 6).std().iloc[-1]
            # df_pred.loc[-1,'6mo_max']  = df_pred['drug_lags1'].rolling(window = 6).max().iloc[-1]
            # df_pred.loc[-1,'6mo_min']  = df_pred['drug_lags1'].rolling(window = 6).min().iloc[-1]
            # # Seasonal mean

    y_hat_series = pd.Series(pred_list, index=testy.index)
    return y_hat_series


        

def get_features(df):
    df_feat = df.copy(deep=True)
    
    # # Create Month features
    df_feat['month'] = df_feat.index.month
    # # Cyclic Month
    # df_feat['Month Sin'] = np.sin(2*np.pi*df_feat.month/12)
    # df_feat['Month Cos'] = np.cos(2*np.pi*df_feat.month/12)
    # # Highlight high vol months
    # df_feat['Jan'] = df_feat['month'].apply(lambda x: 1 if x==1 else 0)
    # df_feat['Feb'] = df_feat['month'].apply(lambda x: 1 if x==2 else 0)
    # df_feat['June'] = df_feat['month'].apply(lambda x: 1 if x==6 else 0)
    # df_feat['Dec'] = df_feat['month'].apply(lambda x: 1 if x==12 else 0)
    # Year count feature
    df_feat['year'] = df_feat.index.year
    df_feat['year'] = df_feat['year'] - df_feat['year'].min()   
    # Month count feature
    df_feat['count'] = range(len(df_feat))

    #df = pred_trend(df_feat)
    
    df_feat.drop(columns=['month'], inplace=True)
    return df_feat


def pred_trend(df):
    df_temp = df.copy(deep=True)
    # # Add trend fit a + bx
    df_temp['xx'] = range(len(df_temp)) 
    df_temp['xx_norm'] = df_temp.xx / df_temp.xx.max()
    df_temp['const'] = 1.0

    df_temp = df_temp.drop(columns=['xx'])
        
    lin_model = OLS.from_formula('drug ~ xx_norm + const + np.power(xx_norm,2)', df_temp).fit()
    print(lin_model.summary())  

    lin_pred = lin_model.predict(df_temp)
    df_temp['trend'] = lin_pred
    
    plt.figure()
    plt.plot(df_temp.drug)
    plt.plot(lin_pred)
    plt.show()
    
    df = pd.concat([df, df_temp.trend.to_frame()], axis=1)
    return df
# df_test = dfa.copy(deep=True)
# pred_trend(dfa)

#%%
df_xgb = dfa.copy(deep=True)
SEASONAL_DIFF = 12
ORD_DIFF = 0
df_diff = get_diff_df(df_xgb, ord_diff=ORD_DIFF, seasonal_diff=SEASONAL_DIFF)

N_LAGS = 18
df_lags = get_lags(df_diff, N_LAGS, moving_win=False)
#df_lags = get_features(df_lags)


N_TEST = 12
N_VAL = 0
#trainX, trainy, valX, valy, testX, testy = train_test(df_lags, N_TEST, N_VAL)
trainX, trainy, testX, testy = train_test(df_lags, N_TEST, N_VAL)

# %% Def model
 
xgb_model1 = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
                        #   n_estimators=3000, max_depth=4, 
                        #   learning_rate=0.001, sub_sample=0.6,
                        #   colsample_bytree=0.6)
xgb_model1.fit(trainX, trainy)

#%%
eval_set = [(trainX, trainy), (testX, testy)]
eval_metrics = ['mae']

xgb_model1 = XGBRegressor(objective='reg:squarederror', eval_metric=eval_metrics,
                          n_estimators=6000, max_depth=4, 
                          learning_rate=0.001,
                          alpha = 0, reg_lambda =40,
                          colsample_bytree=0.6)
                        #n_estimators = 100,random_state=42,
                        #tree_method='hist', , sub_sample=0.6,max_depth=5)

xgb_model1.fit(trainX, trainy,
             eval_set=eval_set,
            #  num_boost_round=5,
             #early_stopping_rounds=5,
             verbose=False)

#print(xgb_model1.get_params())

#%%
#Grid search
from sklearn.model_selection import GridSearchCV

parameters = {'objective':['reg:squarederror'],
              'learning_rate': [0.05, 0.01, 0.005], #so called `eta` value
              'max_depth': [4, 5, 6],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.7, 0.9],
              'colsample_bytree': [0.7, 0.9],
              'n_estimators': [15, 30, 50, 75, 100, 300, 500]}


clf = GridSearchCV(model, parameters, cv=5, scoring='r2', n_jobs=-1, verbose=True)

clf.fit(xtrain_pipe, y)

print(f'best parameters are: {clf.best_params_}, score: {clf.best_score_:.3f}' ) 


#%%

history = xgb_model1.evals_result()
xx = range(0,len(history['validation_0']['mae']))

plt.figure()
plt.plot(xx,history["validation_0"]['mae'], label="Train")
plt.plot(xx,history['validation_1']['mae'], label='Test')
plt.legend()
plt.show()

#%%


fig, ax = plt.subplots(figsize=(12,6))
plot_importance(xgb_model1, max_num_features=20, ax=ax)
plt.show();

#%% One step prediction

def one_step_retrain(trainX, trainy, testX, testy, N_TEST, N_LAGS, moving_win=False):

    Tx = trainX.copy(deep=True)
    Ty = trainy.copy(deep=True)
    df_futureX = testX.copy(deep=True)
    df_futurey = testy.copy(deep=True)
    
    model1 = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
    model1.fit(Tx, Ty)
    
    df_pred = Tx.iloc[-2:,:].copy(deep=True)
    df_pred = pd.concat([df_pred, df_futureX.iloc[0,:].to_frame().T])
    pred_list = []
    for i in range(N_TEST):
        pred_y = model1.predict(df_pred.iloc[i,:].to_frame().T)
        pred_list.append(pred_y)
        
        # Add prediction
        df_futureX.iloc[i, 0] = pred_y
        df_futurey.iloc[i] = pred_y    
        # Make lag
        for j in range(N_LAGS-1):
            df_futureX.iloc[i,j+1] = df_pred.iloc[-1,j]
        
        df_move = df_futureX.iloc[i,:].to_frame().T
        #assert df_move.shape == (1,27)
        df_pred = pd.concat([df_pred,df_move])
        #assert df_pred.shape == (N_TEST+i+1, 27)
        Tx = pd.concat([Tx, df_move])
        df_futurey_df = df_futurey.to_frame()
        Ty = pd.concat([Ty, df_futurey_df.iloc[i,:]])
        
        model1 = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
        model1.fit(Tx, Ty)

    
    y_hat_series = pd.Series(pred_list, index=testy.index)
    return y_hat_series





y_hat_series = one_step_pred(trainX, testX, N_TEST, N_LAGS, xgb_model1, moving_win=False)
#isplay(y_hat_series)

# plt.figure()
# plt.plot(testy, label='diff Data')
# plt.plot(y_hat_series, label='Prediction')
# plt.show()

y_hat_series = one_step_retrain(trainX, trainy, testX, testy, N_TEST, N_LAGS)

# Restore series
y_hat_df = y_hat_series.to_frame()
y_hat_df.columns = ['pred']
y_hat_df['data'] = testy
y_hat_df['lags'] = df_xgb.iloc[-2*N_TEST:-N_TEST,0].values

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


# %%
