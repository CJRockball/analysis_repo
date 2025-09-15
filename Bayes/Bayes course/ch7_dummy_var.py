# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 08:29:49 2018

@author: CJROCKBALL
"""


import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import pymc3 as pm
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

plt.style.use(['seaborn-colorblind', 'seaborn-darkgrid'])

#%%

d = pd.read_csv('C:/Users/Carlberg-PAT/Documents/Python 3 old/Trees/Bayes course/Howell1.csv', sep=',')
d.head()

d_copy = d.copy()
d_adult = d[d.age>20]
# Standardize
#d_adult =(d_adult-d_adult.mean())/d_adult.std()
#%% Plot data
plt.figure()
plt.scatter(d.weight,d.height)

Mh = d_copy[d_copy.male==1]
Mh_a = Mh[Mh.age>20]
Fh = d_copy[d_copy.male==0]
Fh_a = Fh[Fh.age>20]
plt.figure()
plt.hist(Mh_a['height'],color='b', alpha=0.5)
plt.hist(Fh_a['height'],color='r', alpha=0.5)

plt.xlabel('Height [cm]')
plt.ylabel('Frequency')
plt.title('Histogram of Height');

plt.figure()
plt.scatter(Fh_a.weight,Fh_a.height, color='r')
plt.scatter(Mh_a.weight,Mh_a.height, color='b')



#%% Male is a dummy variable for men

with pm.Model() as m5_15:
    
    # Priors
    a = pm.Normal('a', 178, 100)
    bn = pm.Normal('bn', 0,10)
    sigma = pm.Uniform('sigma', 0,50)
    
    mu = pm.Deterministic('mu', a + bn*d_adult['male'])

    height = pm.Normal('height', mu=mu, sd=sigma, observed=d_adult['height'])

    trace5_15 = pm.sample(2000,progressbar=False)
    
#%%

pm.summary(trace5_15,['a','bn','sigma']).round(3)

mu.male = trace5_15['a'] + trace5_15['bn']
pm.hpd(mu.male)

#%% Male is a dummy variable for men

with pm.Model() as height_model:
    
    # Priors
    a = pm.Normal('a', 178, 100)
    b0 = pm.Normal('b0', 0,10)
    b1 = pm.Normal('b1', 0,10)
    sigma = pm.Uniform('sigma', 0,50)
    
    mu = pm.Deterministic('mu', a + b0*d_adult['male']+b1*d_adult['weight'])

    height = pm.Normal('height', mu=mu, sd=sigma, observed=d_adult['height'])

    trace_height = pm.sample(2000,progressbar=False)
    
#%%
pm.summary(trace_height, ['a', 'b0', 'b1', 'sigma']).round(3)
#%%

mu.male = trace_height['a'] + trace_height['b0']
pm.hpd(mu.male)   

#%%
seq = np.linspace(30, 65, 50)
mu_male_pred = trace_height['a'] + trace_height['b0']*1+trace_height['b1']*seq[:,None] 
mu_male_hpd = pm.hpd(mu_male_pred.T)

plt.figure()
plt.scatter(Mh_a.weight,Mh_a.height, color='b')
plt.plot(seq, mu_male_pred.mean(1), 'k')
plt.plot(seq, mu_male_hpd[:,0], 'k--')
plt.plot(seq, mu_male_hpd[:,1], 'k--')

plt.xlabel('Weight')
plt.ylabel('Height');

#height_pred_hpd95 = pm.hpd(stats.norm.rvs(mu_male_pred, trace_height['sigma']).T,alpha=0.05)
#height_pred_hpd80 = pm.hpd(stats.norm.rvs(mu_male_pred, trace_height['sigma']).T,alpha=0.20)
#height_pred_hpd50 = pm.hpd(stats.norm.rvs(mu_male_pred, trace_height['sigma']).T,alpha=0.50)
#plt.fill_between(seq, height_pred_hpd95[:,0], height_pred_hpd95[:,1], color='C2', alpha=0.2)
#plt.fill_between(seq, height_pred_hpd80[:,0], height_pred_hpd80[:,1], color='C2', alpha=0.2)
#plt.fill_between(seq, height_pred_hpd50[:,0], height_pred_hpd50[:,1], color='C2', alpha=0.2)

#%%
seq = np.linspace(30, 65, 50)
mu_female_pred = trace_height['a'] + trace_height['b0']*0+trace_height['b1']*seq[:,None] 
mu_female_hpd = pm.hpd(mu_female_pred.T)

plt.figure()
plt.scatter(Fh_a.weight,Fh_a.height, color='r')
plt.scatter(Mh_a.weight,Mh_a.height, color='b')
plt.plot(seq, mu_male_pred.mean(1), 'k')
plt.plot(seq, mu_female_pred.mean(1), 'k')

#%%Distribution example weight = 50

mu_male_pred50 = trace_height['a'] + trace_height['b0']*1+trace_height['b1']*50
mu_female_pred50 = trace_height['a'] + trace_height['b0']*0+trace_height['b1']*50
fig, ax = plt.subplots()
pm.kdeplot(mu_male_pred50, shade=True, ax=ax)
pm.kdeplot(mu_female_pred50,shade=True, ax=ax)
plt.xlabel('height', fontsize=14)
plt.yticks([])
plt.title('Weigh distribution at 50kg')

#%% Add interaction of Weight and male

with pm.Model() as height_model_int:
    
    # Priors
    a = pm.Normal('a', 178, 100)
    #am = pm.Normal('am', 0,1)
    b0 = pm.Normal('b0', 0,10)
    b1 = pm.Normal('b1', 0,10)
    bWM = pm.Normal('bWM', 0,10)
    sigma = pm.Uniform('sigma', 0,50)
    gamma = b1 + bWM*d_adult['male']
    mu = pm.Deterministic('mu', a + b0*d_adult['male']+gamma*d_adult['weight'])

    height_int = pm.Normal('height_int', mu=mu, sd=sigma, observed=d_adult['height'])

    trace_height_int = pm.sample(2000,progressbar=False)
    
pm.summary(trace_height_int, ['a', 'b0', 'b1', 'bWM', 'sigma'])

#%%

comp_df = pm.compare([trace5_15, trace_height, trace_height_int], 
                     [m5_15, height_model, height_model_int])

comp_df.loc[:,'model'] = pd.Series(['m5_15', 'mheight', 'mheight_int'])
comp_df = comp_df.set_index('model')
comp_df

#%% Plot the line with different slope

seq = np.linspace(30, 65, 50)
mu_female_int_pred = trace_height_int['a'] + trace_height_int['b0']*0+trace_height_int['b1']*seq[:,None] +trace_height_int[bWM]*0*seq[:,None] 
mu_female__int_hpd = pm.hpd(mu_female_int_pred.T)
mu_male_int_pred = trace_height_int['a'] + trace_height_int['b0']*1+trace_height_int['b1']*seq[:,None] +trace_height_int[bWM]*1*seq[:,None] 
mu_male_int_hpd = pm.hpd(mu_male_int_pred.T)

plt.figure()
plt.scatter(Fh_a.weight,Fh_a.height, color='r')
plt.scatter(Mh_a.weight,Mh_a.height, color='b')
plt.plot(seq, mu_male_int_pred.mean(1), 'b')
plt.plot(seq, mu_female_int_pred.mean(1), 'r')

#%% Construct gamma distributions male and not male

gamma_male = trace_height_int['b1']+trace_height_int[bWM]*1
gamma_female = trace_height_int['b1']+trace_height_int[bWM]*0
print("Gamma for men: {:.2f}".format(gamma_male.mean()))
print("Gamma for women: {:.2f}".format(gamma_female.mean()))
fig, ax = plt.subplots()
pm.kdeplot(gamma_male, ax=ax)
pm.kdeplot(gamma_female, ax=ax)
ax.set_xlabel('gamma')
ax.set_ylabel('Density')

#%% Compare difference

diff = gamma_male-gamma_female

pm.kdeplot(diff)

sum(diff[diff < 0]) / len(diff)

#%%
seq = np.linspace(30, 65, 50)
mu_male_pred = trace_height['a'] + trace_height['b0']*1+trace_height['b1']*seq[:,None] 
mu_male_hpd = pm.hpd(mu_male_pred.T)
mu_female_int_pred = trace_height_int['a'] + trace_height_int['b0']*0+trace_height_int['b1']*seq[:,None] +trace_height_int[bWM]*0*seq[:,None] 
mu_female__int_hpd = pm.hpd(mu_female_int_pred.T)
mu_male_int_pred = trace_height_int['a'] + trace_height_int['b0']*1+trace_height_int['b1']*seq[:,None] +trace_height_int[bWM]*1*seq[:,None] 
mu_male_int_hpd = pm.hpd(mu_male_int_pred.T)
mu_male_pred = trace_height['a'] + trace_height['b0']*1+trace_height['b1']*seq[:,None] 
mu_male_hpd = pm.hpd(mu_male_pred.T)

f,(ax1,ax2,ax3) = plt.subplots(1,3,sharey=True, figsize=(8,3))

ax1.scatter(Mh_a.weight,Mh_a.height, color='b',s=4)
ax1.plot(seq, mu_male_pred.mean(1), 'k')
ax1.plot(seq, mu_male_hpd[:,0], 'k--')
ax1.plot(seq, mu_male_hpd[:,1], 'k--')
ax1.set_title('Different intercept')
ax1.set_xlabel('Weight')
ax1.set_ylabel('Height');


ax2.scatter(Fh_a.weight,Fh_a.height, color='r',s=4)
ax2.scatter(Mh_a.weight,Mh_a.height, color='b',s=4)
ax2.plot(seq, mu_male_pred.mean(1), 'k')
ax2.plot(seq, mu_female_pred.mean(1), 'k')
ax2.set_title('Different intercept')
ax2.set_xlabel('Weight')
ax2.set_ylabel('Height');

ax3.scatter(Fh_a.weight,Fh_a.height, color='r',s=4)
ax3.scatter(Mh_a.weight,Mh_a.height, color='b',s=4)
ax3.plot(seq, mu_male_int_pred.mean(1), 'b')
ax3.plot(seq, mu_female_int_pred.mean(1), 'r')
ax3.set_title('Different slope')
ax3.set_xlabel('Weight')
ax3.set_ylabel('Height');



 