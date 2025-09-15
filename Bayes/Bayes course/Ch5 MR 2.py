# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 16:27:04 2018

@author: CJ ROCKBALL
"""


import pymc3 as pm
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

plt.style.use(['seaborn-colorblind', 'seaborn-darkgrid'])

#%%

# load data
d = pd.read_csv('C:/Users/CJ ROCKBALL/Documents/Python Scripts/Python 3 old/Trees/Bayes course/WaffleDivorceraw.csv', header=0)
d.head()
# standardize predictor
d['MedianAgeMarriage_s'] = (d.MedianAgeMarriage - d.MedianAgeMarriage.mean()) / d.MedianAgeMarriage.std()
d['Marriage_s'] = (d.Marriage - d.Marriage.mean()) / d.Marriage.std()

#%%

with pm.Model() as divorce_model:
    
    #priors
    intercept = pm.Normal('intercept', 10,10)
    beta = pm.Normal('beta', 0,1)
    sigma = pm.Uniform('sigma', 0, 10)
    
    # mu
    mu = pm.Deterministic('mu',intercept + beta*d.MedianAgeMarriage_s)
    
    #likelihood
    Divorce = pm.Normal('Divorce', mu=mu, sd=sigma, observed=d.Divorce)

    # Inference!
    #map_estimate = pm.find_MAP(model=divorce_model)
    #trace_divorce = pm.sample(progressbar=False) # draw posterior samples using NUTS sampling%    
    trace_divorce = pm.sample(3000, tune=1000,start = pm.find_MAP(),njobs=3)
    
#%%
    
varnames = ['intercept', 'beta','sigma']

pm.summary(trace_divorce, varnames, alpha=.11).round(3)
pm.forestplot(trace_divorce, varnames=varnames);
pm.traceplot(trace_divorce, varnames);     
    
#%%

mu_mean = trace_divorce['mu']
mu_hpd = pm.hpd(mu_mean)

plt.plot(d.MedianAgeMarriage_s, d.Divorce, 'C0o')
plt.plot(d.MedianAgeMarriage_s, mu_mean.mean(0), 'C2')

idx = np.argsort(d.MedianAgeMarriage_s)
plt.fill_between(d.MedianAgeMarriage_s[idx],
                 mu_hpd[:,0][idx], mu_hpd[:,1][idx], color='C2', alpha=0.25)

plt.xlabel('Meadian Age Marriage', fontsize=14)
plt.ylabel('Divorce', fontsize=14);

















    
    
   