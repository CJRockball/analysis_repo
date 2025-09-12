#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PythonTsa.plot_multi_ACF import multi_ACFfig
from PythonTsa.plot_multi_Q_pvalue import MultiQpvalue_plot
from PythonTsa.datadir import getdtapath
from PythonTsa.plot_acf_pacf import acf_pacf_fig
from PythonTsa.LjungBoxtest import plot_LB_pvalue

import statsmodels.api as sm
from statsmodels.tsa.api import VAR, VARMAX
# %%

df = pd.read_csv('data/Wei/WW2b.csv')
timeindex = pd.date_range('2009-06', periods=len(df), freq='ME')
df.index = timeindex
df.drop(['Period'], axis=1, inplace=True)
display(df)

# %%
col_names = df.columns

# Plot individually
def mplot(df_):
    fig = plt.figure()
    for i,cname in enumerate(col_names):
        df_[cname].plot(figsize=(8,15), ax=fig.add_subplot(811+i))
        plt.title(f'Raw Data Sales of {cname}')
    plt.tight_layout()
    plt.show()

mplot(df)

# %%

ddf = df.diff(1).dropna()
mplot(ddf)

# %% Check ACF after diff

multi_ACFfig(ddf, nlags=15)
plt.show()

# %%

ddfmod1 = VAR(ddf)
print(ddfmod1.select_order(maxlags=15))

# %%

acf_pacf_fig(ddf['AUT'], both=True, lag=36)
plt.show()

#%%

sddf = df.diff(12).dropna()
acf_pacf_fig(sddf['AUT'], both=True, lag=36)

# %%

stdf = sddf.diff(1).dropna()

#%%

acf_pacf_fig(stdf['GEM'], both=True, lag=36)


# %%
