# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 22:00:55 2018

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

plt.style.use(['seaborn-colorblind', 'seaborn-darkgrid'])

#%% Make dummy data

x_dum = np.linspace(1,100,100)

n = pm.Normal.dist(0,3).random(size=100)

y_dum = 12.5 + 2*(x_dum+n)

plt.plot(x_dum, y_dum, 'C0o')
plt.xlabel('X value', fontsize=14)
plt.ylabel('Y value', fontsize=14)

#%%

with pm.Model() as lin_model:
    
    # Prior
    alfa = pm.Normal('alfa', 0,100)
    beta = pm.Normal('beta', 0,100)
    sigma = pm.HalfCauchy('sigma', 5)

    mu = pm.Deterministic('mu', alfa + beta*x_dum)
    
    #Likelihood
    y = pm.Normal('y', mu=mu, sd=sigma, observed=y_dum)
    
    trace = pm.sample(2000, progressbar=False)
    
#%% Fit summary
    
pm.traceplot(trace, ['alfa', 'beta', 'sigma'])
pm.summary(trace, ['alfa', 'beta', 'sigma'], alpha=.11).round(3)   
pm.forestplot(trace, ['alfa', 'beta', 'sigma']);  
    
#%% Model visualization
# Plot mean trace values
plt.plot(x_dum, y_dum, '.')
plt.plot(x_dum, trace['alfa'].mean() + trace['beta'].mean() * x_dum)
plt.xlabel('X value', fontsize=14)
plt.ylabel('Y value', fontsize=14);

# Plot multiple trace fits, the last 20 in the trace
df2 = pm.trace_to_dataframe(trace)[:20]   
df3 = df2[['alfa', 'beta', 'sigma']]

plt.plot(x_dum, y_dum, '.')
for i in range(0, 19):
    plt.plot(x_dum, df3.iloc[i]['alfa'] + df3.iloc[i]['beta']  * x_dum, 'C2-', alpha=0.5)

plt.xlabel('X value', fontsize=14)
plt.ylabel('Y value', fontsize=14);
plt.xlim(0,20)
plt.ylim(0,50)

#%% 95% HDP plot of predictired samples

y_pred = pm.sample_ppc(trace, 200, model=lin_model) # Gives 200 samples/point
y_pred_hpd = pm.hpd(y_pred['y']) # gives the low 4.5% and high 95.5% (HPD)
#%%

# Need to sort in ascending order for the plotting
idx = np.argsort(x_dum)
x_dum_ord = x_dum[idx]
y_pred_hpd = y_pred_hpd[idx]
mu_pred = trace['mu']
mu_hpd = pm.hpd(mu_pred, alpha=.11)[idx]

#Plotting
plt.scatter(x_dum, y_dum, c='C0', alpha=0.3)
plt.plot(x_dum, trace['alfa'].mean() + trace['beta'].mean() * x_dum)
plt.fill_between(x_dum_ord, mu_hpd[:,0], mu_hpd[:,1], color='C2', alpha=0.25);
plt.fill_between(x_dum_ord, y_pred_hpd[:,0], y_pred_hpd[:,1], color='C2', alpha=0.25)

plt.xlabel('X value', fontsize=14)
plt.ylabel('Y value', fontsize=14);
plt.xlim(0,30)
plt.ylim(0,90)






 

    