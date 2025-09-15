# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 10:26:46 2018

@author: CJROCKBALL
"""

import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import pymc3 as pm
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use(['seaborn-colorblind', 'seaborn-darkgrid'])

#%%

d = pd.read_csv('C:/Users/Carlberg-PAT/Documents/Python 3 old/Trees/Bayes course/chimpanzees.csv', sep=';')
# we change "actor" to zero-index
d.actor = d.actor - 1

d.head()

#%%

with pm.Model() as model_intercept:
    a = pm.Normal('a', 0, 10)
    p = pm.math.invlogit(a)    
    pulled_left = pm.Binomial('pulled_left', 1, p, observed=d.pulled_left)

    trace_intercept = pm.sample(2000,njobs=1)

df_trace = pm.trace_to_dataframe(trace_intercept)
    
#%%

pm.sample_ppc(trace_intercept, 1000, model_intercept)

#%%
with pm.Model() as model_prosoc:
    a = pm.Normal('a', 0, 10)
    bp = pm.Normal('bp', 0, 10)
    bpC = pm.Normal('bpC', 0, 10)
    p = pm.math.invlogit(a + (bp + bpC * d.condition) * d.prosoc_left)
    pulled_left = pm.Binomial('pulled_left', 1, p, observed=d.pulled_left)

    trace_prosoc = pm.sample(progressbar=False)
    
#%%    

pm.summary(trace_prosoc).round(2)

pm.traceplot(trace_prosoc)

trace_d2 = pm.trace_to_dataframe(trace_prosoc)
sns.pairplot(trace_d2,vars=["a", "bp", "bpC"],diag_kind="kde", plot_kws=dict(s=2,edgecolor=None))

#%%
with pm.Model() as model_full:
    a = pm.Normal('a', 0, 10)
    bp = pm.Normal('bp', 0, 10)
    bpC = pm.Normal('bpC', 0, 10)
    bC = pm.Normal('bC', 0, 10)
    p = pm.math.invlogit(a + (bp + bpC * d.condition) * d.prosoc_left + bC*d.condition)
    pulled_left = pm.Binomial('pulled_left', 1, p, observed=d.pulled_left)

    trace_full = pm.sample(progressbar=False)
   
#%%
    
comp_df = pm.compare(traces=[trace_intercept, trace_prosoc, trace_full],
                     models=[model_intercept, model_prosoc, model_full],
                     method='pseudo-BMA')

comp_df.loc[:,'model'] = pd.Series(['model_intercept', 'model_prosoc', 'model_full'])
comp_df = comp_df.set_index('model')
comp_df

pm.compareplot(comp_df)

#%% Make prediction with weighted ensemble

traces = [trace_intercept, trace_prosoc, trace_full]
models = [model_intercept, model_prosoc, model_full]

chimp_ensemble = pm.sample_ppc_w(traces=traces, models=models, samples=1000, 
                                 weights=comp_df.weight.sort_index(ascending=True))

#%% Add individual intercept

with pm.Model() as model_10_4:
    a = pm.Normal('alpha', 0, 1, shape=len(d.actor.unique()))
    bp = pm.Normal('bp', 0, 1)
    bpC = pm.Normal('bpC', 0, 1)
    p = pm.math.invlogit(a[d.actor.values] + (bp + bpC * d.condition) * d.prosoc_left)
    pulled_left = pm.Binomial('pulled_left', 1, p, observed=d.pulled_left)

    trace_10_4 = pm.sample(2000,progressbar=False)  

#%%

pm.summary(trace_10_4, alpha=.11).round(2)

#%% Plot marg dist for alpha 1

post = pm.trace_to_dataframe(trace_10_4)
post.head()
pm.kdeplot(post['alpha__1'])

#%%

























