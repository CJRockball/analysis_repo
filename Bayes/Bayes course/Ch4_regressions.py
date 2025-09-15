# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 08:34:44 2018

@author: CJROCKBALL
"""

#import os
#os.environ['MKL_THREADING_LAYER'] = 'GNU'

import pymc3 as pm
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import collections
import seaborn as sns

plt.style.use(['seaborn-colorblind', 'seaborn-darkgrid'])

#%%

d = pd.read_csv('C:/Users/CJ ROCKBALL/Documents/Python Scripts/Python 3 old/Trees/Bayes course/Howell1.csv', header=0)
d.head()

d2 = d[d.age >= 18]
d2_norm =(d2-d2.mean())/d2.std()
y_val = d2.loc[:,'height']
x_val = d2.loc[:,'weight']

y_val_norm = d2_norm.loc[:,'height']
x_val_norm = d2_norm.loc[:,'weight']

plt.scatter(x_val, y_val)
plt.hist(y_val)

#%%

with pm.Model() as reg_model:
    
    #Priors
    sigma = pm.Uniform('sigma', 0, 50)
    intercept = pm.Normal('Intercept', 156, sd=100)
    x_coeff = pm.Normal('x_coeff', 0, sd=10)
    
    y_est = intercept + x_coeff * x_val
    
    # Define likelihood
    y = pm.Normal('y', mu=y_est, sd=sigma, observed=y_val)
    
    # Inference!
    map_estimate = pm.find_MAP(model=reg_model)
#    trace = pm.sample(progressbar=False) # draw posterior samples using NUTS sampling%
    trace = pm.sample(3000, tune=1000,start = pm.find_MAP(),njobs=1)

plt.scatter(x_val, y_val)
plt.plot(x_val, map_estimate['Intercept'] + map_estimate['x_coeff'] * x_val, 'C2-')
plt.xlabel(d2.columns[1], fontsize=14)
plt.ylabel(d2.columns[0], fontsize=14)

#%%    
pm.summary(trace, alpha=.11).round(2)

#%%
pm.traceplot(trace)

trace_d2 = pm.trace_to_dataframe(trace) #Put trace samples in Dataframe
trace_d2.corr() #Show parameter correlation

plt.scatter(trace_d2.loc[:,'Intercept'],trace_d2.loc[:,'sigma'], alpha=0.25)

plt.scatter(trace_d2.loc[:,'Intercept'],trace_d2.loc[:,'x_coeff'])

#%% Calculate uncertainty from samples

# For example weight=50

mu_at_50 = trace_d2.loc[:,'Intercept'] + trace_d2.loc[:,'x_coeff']*50
#plt.hist(mu_at_50,25, normed=True)
#plt.show()
#sns.kdeplot(mu_at_50, bw=0.5)
sns.distplot(mu_at_50, hist = False, kde = True)
ma50_hpd = pm.hpd(mu_at_50,0.05)

#%%

weigth_seq = np.arange(25, 71)
# Given that we have a lot of samples we can use less of them for plotting (or we can use all!)
#chain_N_thinned = chain_N[::10]
mu_pred = np.zeros((len(weigth_seq), len(trace_d2)))
for i, w in enumerate(weigth_seq):
    mu_pred[i] = trace_d2.loc[:,'Intercept'] + trace_d2.loc[:,'x_coeff'] * w

plt.plot(weigth_seq, mu_pred, 'C0.', alpha=0.05)
plt.xlabel('weight', fontsize=14)
plt.ylabel('height', fontsize=14);
plt.show()

mu_mean = mu_pred.mean(1)
mu_hpd = pm.hpd(mu_pred.T, alpha=.11) # 89% hpdi

plt.scatter(x_val, y_val)
plt.plot(weigth_seq, mu_mean, 'C2')
plt.fill_between(weigth_seq, mu_hpd[:,0], mu_hpd[:,1], color='C2', alpha=0.25)
plt.xlabel('weight', fontsize=14)
plt.ylabel('height', fontsize=14)
plt.xlim(x_val.min()-5, x_val.max()+5);
plt.show()

#%%

y_valred = d2.loc[1:11,'height']
x_valred = d2.loc[1:11,'weight']

with pm.Model() as reg_modelred:
    
    #Priors
    sigma = pm.Uniform('sigma', 0, 50)
    intercept = pm.Normal('Intercept', 156, sd=100)
    x_coeff = pm.Normal('x_coeff', 0, sd=10)
    
    y_estred = intercept + x_coeff * x_valred
    
    # Define likelihood
    yred = pm.Normal('yred', mu=y_estred, sd=sigma, observed=y_valred)
    
    # Inference!
#    map_estimatered = pm.find_MAP(model=reg_modelred)
#    tracered = pm.sample(model=reg_modelred,progressbar=False) # draw posterior samples using NUTS sampling%
    traced = pm.sample(3000, tune=1000,start = pm.find_MAP(),njobs=1)

trace_dred = pm.trace_to_dataframe(traced) #Put trace samples in Dataframe

#%%
# Given that we have a lot of samples we can use less of them for plotting (or we can use all!)
#chain_N_thinned = chain_N[::10]
mu_predred = np.zeros((len(weigth_seq), len(trace_dred)))
for i, w in enumerate(weigth_seq):
    mu_predred[i] = trace_dred.loc[:,'Intercept'] + trace_dred.loc[:,'x_coeff'] * w

plt.plot(weigth_seq, mu_predred, 'C0.', alpha=0.1)
plt.xlabel('weight', fontsize=14)
plt.ylabel('height', fontsize=14)
plt.show()

mu_meanred = mu_predred.mean(1)
mu_hpdred = pm.hpd(mu_predred.T, alpha=.11)

plt.figure()
plt.scatter(x_valred, y_valred)
plt.plot(weigth_seq, mu_meanred, 'C2')
plt.fill_between(weigth_seq, mu_hpdred[:,0], mu_hpdred[:,1], color='C2', alpha=0.25)
plt.xlabel('weight', fontsize=14)
plt.ylabel('height', fontsize=14)
plt.xlim(x_valred.min(), x_valred.max());
plt.show()

#%%

height_pred = pm.sample_ppc(trace, 2000, reg_model)
height_pred_hpd = pm.hpd(height_pred['y'])

height_pred_quant = pm.quantiles(height_pred['y'],[2.5,30,50,70,97.5])
height_pred_quant_od = collections.OrderedDict(sorted(height_pred_quant.items()))
height_pred_q_list = []
height_pred_q_list = [(v) for k, v in height_pred_quant_od.items()]   

#%% Sort

idx = np.argsort(x_val) # Create sort vector
d2_weight_ord = d2.weight.iloc[idx] #Sort pandas
height_pred_hpd = np.array(height_pred_hpd)[idx] #sort numpy
height_pred_quant1 = np.array(height_pred_q_list[0])[idx]
height_pred_quant2 = np.array(height_pred_q_list[1])[idx]
height_pred_quant3 = np.array(height_pred_q_list[2])[idx]
height_pred_quant4 = np.array(height_pred_q_list[3])[idx]
height_pred_quant5 = np.array(height_pred_q_list[4])[idx]

#np.array(arr)[new_arr]
# Plot 

#plt.scatter(d2.weight, d2.height)
#plt.fill_between(weigth_seq, mu_hpd[:,0], mu_hpd[:,1], color='C2', alpha=0.25)
#plt.fill_between(d2_weight_ord, height_pred_hpd[:,0], height_pred_hpd[:,1], color='C2', alpha=0.25)
#plt.plot(weigth_seq, mu_mean, 'C2')
#plt.xlabel('weight', fontsize=14)
#plt.ylabel('height', fontsize=14)
#plt.xlim(d2.weight[:].min(), d2.weight[:].max());

plt.scatter(d2.weight, d2.height)
plt.fill_between(weigth_seq, mu_hpd[:,0], mu_hpd[:,1], color='C2', alpha=0.25)
#plt.fill_between(d2_weight_ord, height_pred_quant2, height_pred_quant4, color='C2', alpha=0.5)
plt.fill_between(d2_weight_ord, height_pred_quant1, height_pred_quant5, color='C2', alpha=0.25)
plt.plot(weigth_seq, mu_mean, 'C2')
plt.xlabel('weight', fontsize=14)
plt.ylabel('height', fontsize=14)
plt.xlim(d2.weight[:].min(), d2.weight[:].max());

#%%

d.head()

y_valf = d.loc[:,'height']
x_valf = d.loc[:,'weight']
plt.scatter(x_valf, y_valf)
plt.xlabel('weight', fontsize=14)
plt.ylabel('height', fontsize=14)

d_weight_std = (d.weight - d.weight.mean()) / d.weight.std()
d_weight_std2 = d_weight_std**2

#%%

with pm.Model() as quad_model:
    
    # Prior
    alpha = pm.Normal('alpha', mu=178, sd=100)
    beta = pm.Normal('beta', mu=0, sd=10, shape=2)
    sigma = pm.Uniform('sigma', lower=0, upper=50)
    
    
    mu = pm.Deterministic('mu', alpha + beta[0] * d_weight_std + beta[1] * d_weight_std2)
    
    #Likelihood
    height = pm.Normal('height', mu=mu, sd=sigma, observed=y_valf)
#    trace_quad = pm.sample(progressbar=False)

    trace_quad = pm.sample(3000, tune=1000,start = pm.find_MAP(),njobs=1)

#%%

varnames = ['alpha', 'beta', 'sigma']
pm.traceplot(trace_quad, varnames);
pm.summary(trace_quad, varnames, alpha=.11).round(2)

#%%

mu_pred = trace_quad['mu']
idx = np.argsort(d_weight_std)
mu_hpd = pm.hpd(mu_pred, alpha=.11)[idx]

height_pred = pm.sample_ppc(trace_quad, 200, quad_model)
height_pred_hpd = pm.hpd(height_pred['height'], alpha=.11)[idx]

plt.scatter(d_weight_std, d.height, c='C0', alpha=0.3)
plt.fill_between(d_weight_std[idx], mu_hpd[:,0], mu_hpd[:,1], color='C2', alpha=0.25);
plt.fill_between(d_weight_std[idx], height_pred_hpd[:,0], height_pred_hpd[:,1], color='C2', alpha=0.25);

#%%

d_weight_std3 = d_weight_std**3

#%%

with pm.Model() as cube_model:
    
    # Prior
    alpha = pm.Normal('alpha', mu=178, sd=100)
    beta = pm.Normal('beta', mu=0, sd=10, shape=3)
    sigma = pm.Uniform('sigma', lower=0, upper=50)
    
    
    mu = pm.Deterministic('mu', alpha + beta[0] * d_weight_std + beta[1] * d_weight_std2 + beta[2] * d_weight_std3)
    
    #Likelihood
    height = pm.Normal('height', mu=mu, sd=sigma, observed=y_valf)
    #trace_cube = pm.sample(progressbar=False)
    trace_cube = pm.sample(3000, tune=1000,start = pm.find_MAP(),njobs=1)
    
#%%

varnames = ['alpha', 'beta', 'sigma']
pm.traceplot(trace_cube, varnames);
pm.summary(trace_cube, varnames, alpha=.11).round(2)

#%%

mu_pred = trace_cube['mu']
idx = np.argsort(d_weight_std)
mu_hpd = pm.hpd(mu_pred, alpha=.11)[idx]

height_pred = pm.sample_ppc(trace_cube, 200, quad_model)
height_pred_hpd = pm.hpd(height_pred['height'], alpha=.11)[idx]

plt.scatter(d_weight_std, d.height, c='C0', alpha=0.3)
plt.fill_between(d_weight_std[idx], mu_hpd[:,0], mu_hpd[:,1], color='C2', alpha=0.25);
plt.fill_between(d_weight_std[idx], height_pred_hpd[:,0], height_pred_hpd[:,1], color='C2', alpha=0.25);


