# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 15:37:19 2018

@author: CJROCKBALL
"""


import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import pymc3 as pm
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit as logistic
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use(['seaborn-colorblind', 'seaborn-darkgrid'])

#%%
d = pd.read_csv('C:/Users/CJ ROCKBALL/Documents/Python Scripts/Python 3 old/Trees/Bayes course/Kline.csv', sep=';')
d.head(9)

#%%
#d_copy = d.copy()
d['log_popu'] = np.log(d.population)
d['contact_high'] = (d.contact == "high").astype(int)
d.head(10).round(2)
#%%
with pm.Model() as m_10_10:
    #priors
    a = pm.Normal('a', 0, 100)
    b = pm.Normal('b', 0, 1, shape=3)
    
    lamb = pm.math.exp(a+b[0]*d.log_popu + b[1] * d.contact_high + b[2] * d.contact_high * d.log_popu)
    #Likelihood
    obs = pm.Poisson('obs', lamb, observed=d.total_tools)
    trace_10_10 = pm.sample(2000, tune=1000,njobs=1)

#%% standard check
pm.summary(trace_10_10).round(2)
pm.forestplot(trace_10_10)
#%% Corr check
summary = pm.summary(trace_10_10, alpha=.11)[['mean', 'sd', 'hpd_5.5', 'hpd_94.5']]
trace_cov = pm.trace_cov(trace_10_10, model=m_10_10)
invD = (np.sqrt(np.diag(trace_cov))**-1)[:, None]
trace_corr = pd.DataFrame(invD*trace_cov*invD.T, index=summary.index, columns=summary.index)

summary.join(trace_corr).round(2)
#%% plot corr
tracedf = pm.trace_to_dataframe(trace_10_10)
grid = (sns.PairGrid(tracedf,
                     diag_sharey=False)
           .map_diag(sns.kdeplot)
           .map_upper(plt.scatter, alpha=0.1))

#%% try centered log_pop
    
d['log_popu_c'] =(d.log_popu-d.log_popu.mean())

    
with pm.Model() as m_10_11:
    #priors
    a = pm.Normal('a', 0, 100)
    b1 = pm.Normal('b1', 0, 1)
    b2 = pm.Normal('b2', 0, 1)
    b3 = pm.Normal('b3', 0, 1)
    
    lamb = pm.math.exp(a+b1*d.log_popu + b2*d.contact_high + b3*d.contact_high*d.log_popu)
    #Likelihood
    obs = pm.Poisson('obs', lamb, observed=d.total_tools)
    trace_10_11 = pm.sample(3000, tune=1000,start = pm.find_MAP(),njobs=1)
    
#%%
pm.summary(trace_10_11).round(2)
pm.forestplot(trace_10_11)   

#%% 
summary = pm.summary(trace_10_11, alpha=.11)[['mean', 'sd', 'hpd_5.5', 'hpd_94.5']]
trace_cov = pm.trace_cov(trace_10_11, model=m_10_11)
invD = (np.sqrt(np.diag(trace_cov))**-1)[:, None]
trace_corr = pd.DataFrame(invD*trace_cov*invD.T, index=summary.index, columns=summary.index)

summary.join(trace_corr).round(2)

#%% plot corr
tracedf = pm.trace_to_dataframe(trace_10_11)
grid = (sns.PairGrid(tracedf,
                     diag_sharey=False)
           .map_diag(sns.kdeplot)
           .map_upper(plt.scatter, alpha=0.1))

#%%
ap = trace_10_11['a'].mean()
b1p = trace_10_11['b1'].mean()
b2p = trace_10_11['b2'].mean()
b3p = trace_10_11['b3'].mean()

log_pop_seq = np.linspace(6, 13, 60)
link_m10_11_ch = np.exp(ap + b1p*log_pop_seq + b2p + b3p*log_pop_seq)
link_m10_11_cl = np.exp(ap + b1p*log_pop_seq)

link_trace_ch = np.exp(trace_10_11['a'] + trace_10_11['b1']*log_pop_seq[:,None] +\
                    trace_10_11['b2'] + trace_10_11['b3']*log_pop_seq[:,None])
lamb_hpd_ch = pm.hpd(link_trace_ch.T)

link_trace_cl = np.exp(trace_10_11['a'] + trace_10_11['b1']*log_pop_seq[:,None])
lamb_hpd_cl = pm.hpd(link_trace_cl.T)

#df.loc[df['column_name'] == some_value]
d_ch = d.loc[d['contact']== 'high']
d_cl = d.loc[d['contact']== 'low']

_, axes = plt.subplots(1, 1, figsize=(5, 5))
axes.scatter(d_ch.log_popu, d_ch.total_tools,color='orange')
axes.scatter(d_cl.log_popu, d_cl.total_tools, facecolors='none',edgecolor='orange')

axes.plot(log_pop_seq, link_m10_11_ch, '--', color='k')
axes.plot(log_pop_seq, link_m10_11_cl, '--', color='b')
axes.fill_between(log_pop_seq,
                      lamb_hpd_ch[:,0], lamb_hpd_ch[:,1], alpha=0.2, color='k')
axes.fill_between(log_pop_seq,
                      lamb_hpd_cl[:,0], lamb_hpd_cl[:,1], alpha=0.2, color='b')

#plt.plot(log_pop_seq, lamb_hpd[:,0], 'k--')
#plt.plot(log_pop_seq, lamb_hpd[:,1], 'k--')

axes.set_xlabel('log-population', fontsize=14)
axes.set_ylabel('total tools', fontsize=14)
axes.set_xlim(6, 12.8)
axes.set_ylim(10, 73);



