# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 09:49:35 2018

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

d = pd.read_csv('C:/Users/Carlberg-PAT/Documents/Python 3 old/Trees/Bayes course/milk.csv', sep=';')
d.head()

dcc = d.dropna().copy()

dcc['log_mass'] = np.log(dcc['mass'])
dcc.head()
sns.pairplot(dcc,vars=["kcal.per.g", "log_mass", "neocortex.perc"],hue="clade")


#%%

with pm.Model() as Model5_5:
    
    # Priors
    a = pm.Normal('a', 10, 100)
    bn = pm.Normal('bn', 0, 1)
    sigma = pm.Uniform('sigma', 0,1)
    
    mu = pm.Deterministic(a + bn*dcc['neocortex.perc'])
    
    # Likelihood
    
    Y = pm.Normal('Y', mu=mu, sd=sigma, observed=dcc['kcal.per.g'])
 
    trace5_5 = pm.sample(2000,progressbar=False)
    
#%%
    
pm.traceplot(trace5_5)
pm.summary(trace5_5, alpha=.11).round(3)  

#%%

d_range = np.linspace(55,76,50)
mean_pred = trace5_5['a']+trace5_5['bn']*d_range[:,None]
mu_hpd = pm.hpd(mean_pred.T)

plt.scatter(dcc['neocortex.perc'],dcc['kcal.per.g'])
plt.plot(d_range, mean_pred.mean(1), 'k')
plt.plot(d_range, mu_hpd[:,0], 'k--')
plt.plot(d_range, mu_hpd[:,1], 'k--')

plt.xlabel('neocortex.perc', fontsize=14)
plt.ylabel('kcal.per.g', fontsize=14);
    
#%%    
# Plot multiple trace fits, the last 100 in the trace
df2 = pm.trace_to_dataframe(trace5_5)[:100]   
df3 = df2[['a', 'bn', 'sigma']]

plt.scatter(dcc['neocortex.perc'],dcc['kcal.per.g'])
plt.plot(d_range, mean_pred.mean(1), 'k')
for i in range(0, 99):
    plt.plot(d_range, df3.iloc[i]['a'] + df3.iloc[i]['bn']  * d_range, 'C2-', alpha=0.5)

plt.xlabel('Neaocortex', fontsize=14)
plt.ylabel('kcal', fontsize=14);

#%% Multivar model

with pm.Model() as model5_7:
    
    #Priors
    a = pm.Normal('a', 0,100)
    bn = pm.Normal('bn', 0,1,shape=2)
    sigma = pm.Uniform('sigma',0,1)
    
    mu = pm.Deterministic('mu',a + bn[0]*dcc['neocortex.perc']+bn[1]*dcc['log_mass'])
    
    kcal = pm.Normal('kcal', mu=mu, sd=sigma, observed=dcc['kcal.per.g'])
    
    trace5_7 = pm.sample(2000,progressbar=False)
    
pm.traceplot(trace5_7, ['a', 'bn', 'sigma'])
pm.summary(trace5_7, ['a', 'bn', 'sigma'], alpha=.11).round(3)  

#%%
    
seq = np.linspace(55, 76, 50)
mu1_pred = trace5_7['a'] + trace5_7['bn'][:,0] * seq[:,None] + trace5_7['bn'][:,1] * dcc['log_mass'].mean()
mu1_hpd = pm.hpd(mu1_pred.T)

plt.figure()
plt.plot(seq, mu1_pred.mean(1), 'k')
plt.plot(seq, mu1_hpd[:,0], 'k--')
plt.plot(seq, mu1_hpd[:,1], 'k--')

plt.xlabel('neocortex.perc')
plt.ylabel('kcal.per.g');

seq_m = np.linspace(-2, 4, 50)
mu2_pred = trace5_7['a'] + trace5_7['bn'][:,0] * dcc['neocortex.perc'].mean() + trace5_7['bn'][:,1] *seq_m[:,None] 
mu2_hpd = pm.hpd(mu2_pred.T)

plt.figure()
plt.plot(seq_m, mu2_pred.mean(1), 'k')
plt.plot(seq_m, mu2_hpd[:,0], 'k--')
plt.plot(seq_m, mu2_hpd[:,1], 'k--')

plt.xlabel('log_mass')
plt.ylabel('kcal.per.g');

#from mpl_toolkits.mplot3d import Axes3D
#mu3_pred = trace5_7['a'] + trace5_7['bn'][:,0] *  seq[:,None] + trace5_7['bn'][:,1] *seq_m[:,None] 
#fig = plt.figure()
#ax = Axes3D(fig)
#ax.plot_surface(seq, seq_m, mu3_pred.mean())

#%% 95% HDP plot of predictired samples

kcal_pred = pm.sample_ppc(trace5_7, 200, model=model5_7) # Gives 200 samples/point
kcal_pred_hpd = pm.hpd(kcal_pred['kcal']) # gives the low 4.5% and high 95.5% (HPD)

#%% Plot fit int and pred int for neocrotex

# Need to sort in ascending order for the plotting
neocort_hpd = pm.hpd(stats.norm.rvs(mu1_pred, trace5_7['sigma']).T)

#Plotting
plt.figure()
# Data points
plt.scatter(dcc['neocortex.perc'], dcc['kcal.per.g'], c='C0', alpha=0.3) 
# Mean fit line
plt.plot(seq, mu1_pred.mean(1), 'k')
# fit line 95HPD
plt.fill_between(seq, mu1_hpd[:,0], mu1_hpd[:,1], color='C2', alpha=0.25);
# Prediction inteval 95% HPD
plt.fill_between(seq, neocort_hpd[:,0], neocort_hpd[:,1], color='C2', alpha=0.25)

plt.xlabel('Neocortex Perc')
plt.ylabel('kcal.per.g')
plt.title('log_mass = mean');







