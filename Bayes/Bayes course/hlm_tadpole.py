# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 09:29:29 2018

@author: CJROCKBALL
"""

import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import numpy as np 
import pymc3 as pm
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from scipy.special import expit as logistic

plt.style.use(['seaborn-colorblind', 'seaborn-darkgrid'])

#%%
d = pd.read_csv('C:/Users/Carlberg-PAT/Documents/Python 3 old/Trees/Bayes course/reedfrogs.csv', sep=';')
print(d.shape)
d.head(8)

#%%
# make the tank cluster variable
tank = np.arange(d.shape[0])
# fit
with pm.Model() as m_12_1:
    a_tank = pm.Normal('a_tank', 0, 5, shape=d.shape[0])
    p = pm.math.invlogit(a_tank[tank])
    surv = pm.Binomial('surv', n=d.density, p=p, observed=d.surv)
    trace_12_1 = pm.sample(2000, tune=2000, njobs=1)

#pm.summary(trace_12_1, alpha=.11).round(2)

#%%
with pm.Model() as m_12_2:
    a = pm.Normal('a', 0., 1.)
    sigma = pm.HalfCauchy('sigma', 1.)
    a_tank = pm.Normal('a_tank', a, sigma, shape=d.shape[0])
    p = pm.math.invlogit(a_tank[tank])
    surv = pm.Binomial('surv', n=d.density, p=p, observed=d.surv)
    trace_12_2 = pm.sample(2000, tune=2000, njobs=1)

pm.summary(trace_12_2,['a','sigma'],alpha=.11).round(2)
#%% Calculate correlation of parameters
summary = pm.summary(trace_12_2,alpha=.11)[['mean', 'sd', 'hpd_5.5', 'hpd_94.5']]
trace_cov = pm.trace_cov(trace_12_2,model=m_12_2)
invD = (np.sqrt(np.diag(trace_cov))**-1)[:, None]
trace_corr = pd.DataFrame(invD*trace_cov*invD.T, index=summary.index, columns=summary.index)

summary.join(trace_corr).round(2)
#%%
#%% plot corr
tracedf = pm.trace_to_dataframe(trace_12_2)
tracedf2 = tracedf[['a','sigma']].copy()

grid = (sns.PairGrid(tracedf2,
                     diag_sharey=False)
           .map_diag(sns.kdeplot)
           .map_upper(plt.scatter, alpha=0.1))
 
   
#%%
comp_df = pm.compare(traces=[trace_12_1, trace_12_2],
                     models=[m_12_1, m_12_2],
                     method='pseudo-BMA')

comp_df.loc[:,'model'] = pd.Series(['m12.1', 'm12.2'])
comp_df = comp_df.set_index('model')
comp_df

#%%
# extract PyMC3 samples
post = pm.trace_to_dataframe(trace_12_2, varnames=['a_tank'])

# compute median intercept for each tank
# also transform to probability with logistic
d.loc[:, 'propsurv_est'] = pd.Series(logistic(post.median(axis=0).values), index=d.index)

#%%
_, ax = plt.subplots(1, 2, figsize=(12, 5))
# show first 100 populations in the posterior
xrange = np.linspace(-3, 4, 200)
postcurve = [stats.norm.pdf(xrange, loc=trace_12_2['a'][i], scale=trace_12_2['sigma'][i]) for i in range(100)]
ax[0].plot(xrange, np.asarray(postcurve).T,
           alpha=.1, color='k')
ax[0].set_xlabel('log-odds survive', fontsize=14)
ax[0].set_ylabel('Density', fontsize=14);
# sample 8000 imaginary tanks from the posterior distribution
sim_tanks = np.random.normal(loc=trace_12_2['a'], scale=trace_12_2['sigma'])
# transform to probability and visualize
pm.kdeplot(logistic(sim_tanks), ax=ax[1], color='k')
ax[1].set_xlabel('probability survive', fontsize=14)
ax[1].set_ylabel('Density', fontsize=14)
plt.tight_layout();
# dens( logistic(sim_tanks) , xlab="probability survive" )

#%%

d['pred_dum'] = (d['pred'] == 'pred').astype(np.int64)

with pm.Model() as m_12_3:
    a = pm.Normal('a', 0., 1.)
    b1 = pm.Normal('b1',0.,1.)
    sigma = pm.HalfCauchy('sigma', 1.)
    a_tank = pm.Normal('a_tank', a, sigma, shape=d.shape[0])
    p = pm.math.invlogit(a_tank[tank] + b1*d.pred_dum)
    surv = pm.Binomial('surv', n=d.density, p=p, observed=d.surv)
    trace_12_3 = pm.sample(2000, tune=2000, njobs=1)
#%%
comp_df = pm.compare(traces=[trace_12_1, trace_12_2, trace_12_3],
                     models=[m_12_1, m_12_2, m_12_3],
                     method='pseudo-BMA')

comp_df.loc[:,'model'] = pd.Series(['m12.1', 'm12.2', 'm_12.3'])
comp_df = comp_df.set_index('model')
comp_df
    
















