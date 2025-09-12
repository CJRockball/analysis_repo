import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit

def RSS(y, y_pred):
    return np.sqrt((y-y_pred)**2).sum()


def fit_data(df, model, param_list):
    data_length = df.shape[0]
    xx = np.arange(0,data_length,1)

    params, cov = curve_fit(model, xdata=xx, ydata=df.iloc[:,0], method='lm')

    print('\n Model 1 \n')
    std_dev = np.sqrt(np.diag(cov))
    for name,p,sd in zip(param_list, params, std_dev):
        print('{0} :  {1:0.3}  CI ~normally [{2:0.2e},{3:0.2e}]'.format(name, p, p-1.96*sd,p+1.96*sd))

    df['model'] = model(xx, *params)
    df_out = df['model'].to_frame()
    
    df.plot(figsize=(12,4), style=['s','^-','k--'], markersize=4, linewidth=2)

    print('Residual Sum of Squares RSS')
    print(f"  RSS model: {round(RSS(df.iloc[:,0], df['model']),2)}")
    return params, df_out
