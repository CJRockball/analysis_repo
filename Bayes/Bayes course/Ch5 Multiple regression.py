# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 08:44:37 2018

@author: CJROCKBALL
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
    beta = pm.Normal('beta', 0,1,shape=2)
    sigma = pm.Uniform('sigma', 0, 10)
    
    # mu
    mu = pm.Deterministic('mu',intercept + beta[0]*d.Marriage_s + beta[1]*d.MedianAgeMarriage_s)
    
    #likelihood
    Divorce = pm.Normal('Divorce', mu=mu, sd=sigma, observed=d.Divorce)

    # Inference!
    map_estimate = pm.find_MAP(model=divorce_model)
    #trace_divorce = pm.sample(progressbar=False) # draw posterior samples using NUTS sampling%    
    trace_divorce = pm.sample(3000, tune=1000,start = pm.find_MAP(),njobs=3)
    
#%%

varnames = ['intercept', 'beta','sigma']

pm.summary(trace_divorce, varnames, alpha=.11).round(3)
pm.forestplot(trace_divorce, varnames=varnames);
pm.traceplot(trace_divorce, varnames); 
pm.forestplot(trace_divorce, varnames); 

#trace_divorce = pm.trace_to_dataframe(trace) #Put trace samples in Dataframe
#trace_divorce.corr() #Show parameter correlation

#%% Predict and plot

mu_pred = trace_divorce['mu']
mu_hpd = pm.hpd(mu_pred)

divorce_pred = pm.sample_ppc(trace_divorce, samples=1000, model=divorce_model)['Divorce']
divorce_hpd = pm.hpd(divorce_pred)

#%% plot median age vs divorce with 89% interval 

mu_mean = trace_divorce['mu']
mu_hpd = pm.hpd(mu_mean)

plt.plot(d.MedianAgeMarriage_s, d.Divorce, 'C0o')
plt.plot(d.MedianAgeMarriage_s, mu_mean.mean(0), 'C2')

sortvect = np.argsort(d.MedianAgeMarriage_s)
plt.fill_between(d.MedianAgeMarriage_s[sortvect],
                 mu_hpd[:,0][sortvect], mu_hpd[:,1][sortvect], color='C2', alpha=0.25)

plt.xlabel('Meadian Age Marriage', fontsize=14)
plt.ylabel('Divorce', fontsize=14);

#%%

mu_mean = trace_divorce['mu']
mu_hpd = pm.hpd(mu_mean)

d.plot('Marriage_s', 'Divorce', kind='scatter', xlim = (-2, 3))
plt.plot(d.Marriage_s, mu_mean.mean(0), 'C2')

idx = np.argsort(d.Marriage_s)
plt.fill_between(d.Marriage_s[idx], mu_hpd[:,0][idx], mu_hpd[:,1][idx],
                 color='C2', alpha=0.25);


#%%

mu_hpd = pm.hpd(mu_pred, alpha=0.05)
plt.errorbar(d.Divorce, divorce_pred.mean(0), yerr=np.abs(divorce_pred.mean(0)-mu_hpd.T) , fmt='C0o')
plt.plot(d.Divorce, divorce_pred.mean(0), 'C0o')

plt.xlabel('Observed divorce', fontsize=14)
plt.ylabel('Predicted divorce', fontsize=14)

min_x, max_x = d.Divorce.min(), d.Divorce.max()
plt.plot([min_x, max_x], [min_x, max_x], 'k--');


plt.figure(figsize=(10,12))
residuals = d.Divorce - mu_pred.mean(0)
idx = np.argsort(residuals)
y_label = d.Loc[idx]
y_points = np.linspace(0, 1, 50)
plt.errorbar(residuals[idx], y_points, 
             xerr=np.abs(divorce_pred.mean(0)-mu_hpd.T),
             fmt='C0o',lw=3)

plt.errorbar(residuals[idx], y_points, 
             xerr=np.abs(divorce_pred.mean(0)-divorce_hpd.T),
             fmt='C0o', lw=3, alpha=0.5)

plt.yticks(y_points, y_label);
plt.vlines(0, 0, 1, 'grey');

