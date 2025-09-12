import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS

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
        df_lags[f'lags{i+1}'] = df_lags.iloc[:,i].shift(1)

    if moving_win:
        # When adding rolling mean, need to add predicted ones too
        # Rolling mean
        df_lags['3mo_mean'] = df_lags['lags1'].rolling(window = 3).mean()
        df_lags['3mo_std'] = df_lags['lags1'].rolling(window = 3).std()
        df_lags['3mo_max'] = df_lags['lags1'].rolling(window = 3).max()
        df_lags['3mo_min'] = df_lags['lags1'].rolling(window = 3).min()

        # Rolling mean
        # df_lags['6mo_mean'] = df_lags['lags1'].rolling(window = 6).mean()
        # df_lags['6mo_std'] = df_lags['lags1'].rolling(window = 6).std()
        # df_lags['6mo_max'] = df_lags['lags1'].rolling(window = 6).max()
        # df_lags['6mo_min'] = df_lags['lags1'].rolling(window = 6).min()

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
    df_pred = train.iloc[-N_LAGS:,:]
    df_future = test.copy(deep=True) 
    
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
            df_pred.loc[-1,'3mo_mean'] = df_pred['lags1'].rolling(window = 3).mean().iloc[-1]
            df_pred.loc[-1,'3mo_std']  = df_pred['lags1'].rolling(window = 3).std().iloc[-1]
            df_pred.loc[-1,'3mo_max']  = df_pred['lags1'].rolling(window = 3).max().iloc[-1]
            df_pred.loc[-1,'3mo_min']  = df_pred['lags1'].rolling(window = 3).min().iloc[-1]
            # # Rolling mean
            # df_pred.loc[-1,'6mo_mean'] = df_pred['lags1'].rolling(window = 6).mean().iloc[-1]
            # df_pred.loc[-1,'6mo_std']  = df_pred['lags1'].rolling(window = 6).std().iloc[-1]
            # df_pred.loc[-1,'6mo_max']  = df_pred['lags1'].rolling(window = 6).max().iloc[-1]
            # df_pred.loc[-1,'6mo_min']  = df_pred['lags1'].rolling(window = 6).min().iloc[-1]

    #display(df_pred)
    y_hat_series = pd.Series(pred_list, index=test.index)
    return y_hat_series

def get_features(df):
    df_feat = df.copy(deep=True)
    
    # # Create Month features
    df_feat['month'] = df_feat.index.month
    # # Cyclic Month
    df_feat['Month Sin'] = np.sin(2*np.pi*df_feat.month/12)
    df_feat['Month Cos'] = np.cos(2*np.pi*df_feat.month/12)
    # # Highlight high vol months
    # df_feat['Jan'] = df_feat['month'].apply(lambda x: 1 if x==1 else 0)
    # df_feat['Feb'] = df_feat['month'].apply(lambda x: 1 if x==2 else 0)
    # df_feat['June'] = df_feat['month'].apply(lambda x: 1 if x==6 else 0)
    # df_feat['Dec'] = df_feat['month'].apply(lambda x: 1 if x==12 else 0)
    # Year feature
    df_feat['year'] = df_feat.index.year
    df_feat['year'] = df_feat['year'] - df_feat['year'].min()   
    

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
