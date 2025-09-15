# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 09:42:05 2021

@author: PatCa
"""

import numpy as np
import pandas as pd
import graphviz
import matplotlib.pyplot as plt
import pymc3 as pm
import arviz as az

###---------------------------------------------------------------------------
# Load radon datasets and combine frames

cty = pd.read_csv('ARM_Data/radon/cty.dat')
srrs2 = pd.read_csv('ARM_Data/radon/srrs2.dat')
srrs2.columns = srrs2.columns.map(str.strip)
srrs_mn = srrs2[srrs2['state'] == 'MN']
srrs_mn['fips'] = 1000*srrs_mn.stfips + srrs_mn.cntyfips

cty.columns = cty.columns.map(str.strip)
cty_mn = cty[cty['st'] == 'MN']
cty_mn['fips'] = 1000*cty_mn.stfips + cty_mn.ctfips

srrs_mn = srrs_mn.merge(cty_mn[['fips', 'Uppm']], on='fips')
srrs_mn = srrs_mn.drop_duplicates(subset='idnum')

u = np.log(srrs_mn.Uppm)
u_unique = np.log(srrs_mn.Uppm).unique()
M = len(srrs_mn.Uppm.unique())
n = len(srrs_mn)

srrs_mn.county = srrs_mn.county.map(str.strip)
mn_counties = srrs_mn.county.unique()
counties = len(mn_counties)
county_lookup = dict(zip(mn_counties, range(counties)))

county = srrs_mn['county_code'] = srrs_mn.county.replace(county_lookup).values
radon = srrs_mn.activity
srrs_mn['log_radon'] = log_radon = np.log(radon + 0.1).values
floor = srrs_mn.floor.values
c_num = np.arange(0,85,1)

#plt.hist(log_radon)
#sns.distplot(log_radon, kde=True)
plt.figure()
plt.scatter(county,u, color='purple')

#%% Mouse H model uranium, reparametrized non-centered

with pm.Model() as u_grp2:
    
    # Grp priors
    g0 = pm.Normal('g0', mu=0., sd=100**2)
    g1 = pm.Normal('g1', mu=0., sd=100**2)
    sig_a = pm.HalfCauchy('sig_a', 5.)
    #Var intercept with grp covariate
    alpha = pm.Deterministic('alpha', g0 + g1*u_unique)
    a_off = pm.Normal('a_off', mu=0, sd=1., shape=counties)
    a = pm.Deterministic('a', alpha+a_off*sig_a)
    
    # Model priors
    #Common slope 
    b = pm.Normal('b', mu=0., sd=10.)
    # Model error
    sig = pm.HalfCauchy('sig', 5.)

    #Likelihood
    y_hat = a[county] + b*floor    
    y_like = pm.Normal('y_like', mu=y_hat, sd=sig, observed=log_radon)

    # Sample
    trace2 = pm.sample(3000, tune=4000, n_init=50000, chains=4, cores=1)

df_u_grp = az.summary(trace2)
pm.traceplot(trace2, var_names=['a','a_off','b', 'g1', 'g0','sig_a','sig'])

ug2gv = pm.model_to_graphviz(u_grp2)
ug2gv

#%% Grp var for slop and intercept

with pm.Model() as u_grp2:
    
    # Grp intercept priors
    ga0 = pm.Normal('ga0', mu=0., sd=100**2)
    ga1 = pm.Normal('ga1', mu=0., sd=100**2)
    sigma_a = pm.HalfCauchy('sigma_a', 5.)
    #Var intercept with grp covariate
    a = pm.Deterministic('a', ga0 + ga1*u_unique)
    a_off = pm.Normal('a_off', mu=0, sd=1., shape=counties)
    alpha = pm.Deterministic('alpha', a+a_off*sigma_a)
    
    # Grp slope priors
    gb0 = pm.Normal('gb0', mu=0., sd=100**2)
    gb1 = pm.Normal('gb1', mu=0., sd=100**2)
    sigma_b = pm.HalfCauchy('sigma_b', 5.)
    #Var slope with grp covariate
    b = pm.Deterministic('b', gb0 + gb1*u_unique)
    b_off = pm.Normal('b_off', mu=0, sd=1., shape=counties)
    beta = pm.Deterministic('beta', b+b_off*sigma_b)    
       
    # Model error
    sigma_y = pm.HalfCauchy('sigma_y', 5.)

    #Likelihood
    y_hat = alpha[county] + beta[county]*floor    
    y_like = pm.Normal('y_like', mu=y_hat, sd=sigma_y, observed=log_radon)

    # Sample
    trace3 = pm.sample(3000, tune=4000, n_init=50000, chains=4, cores=1)

df_u_grp = az.summary(trace3)
#pm.traceplot(trace3, var_names=['a','a_off','b', 'g1', 'g0','sig_a','sig'])

ug2gv = pm.model_to_graphviz(u_grp2)
ug2gv
#%% Check correlations

az.plot_pair(trace3,
             var_names=['ga0', 'ga1', 'sigma_a', 'gb0', 'gb1', 'sigma_b', 'sigma_y'],
             plot_kwargs={'alpha': 0.1})



#%% Check grp level values

a_means = trace3['alpha'].mean(axis=0)
a_se = trace3['alpha'].std(axis=0)
plt.figure(dpi=192)
for ui, m, se in zip(u_unique, a_means, a_se):
    plt.plot([ui,ui], [m-se, m+se], 'b-', color='blue', zorder=1)

plt.scatter(u_unique, a_means, color='blue', zorder=2)

g0 = trace3["ga0"].mean()
g1 = trace3['ga1'].mean()
xvals = np.linspace(-1, 0.8)
plt.plot(xvals, g0+g1*xvals, 'k--')
#plt.xlim(-1, 0.8)
plt.xlabel('County-level uranium') 
plt.ylabel('Intercept estimate')


b_means = trace3['beta'].mean(axis=0)
b_se = trace3['beta'].std(axis=0)
plt.figure(dpi=192)
for ui, m, se in zip(u_unique, b_means, b_se):
    plt.plot([ui,ui], [m-se, m+se], 'b-', color='blue', zorder=1)

plt.scatter(u_unique, b_means, color='blue', zorder=2)

g0 = trace3["gb0"].mean()
g1 = trace3['gb1'].mean()
xvals = np.linspace(-1, 0.8)
plt.plot(xvals, g0+g1*xvals, 'k--')
#plt.xlim(-1, 0.8)
plt.xlabel('County-level uranium') 
plt.ylabel('Regression slope')

# Forrest plot
plt.figure(figsize=(6,16))
az.plot_forest(trace3,var_names=['ga0', 'ga1', 'sigma_a', 'gb0', 'gb1', 'sigma_b', 'sigma_y'])

#Plot posterior
plt.figure()
az.plot_posterior(trace3, var_names=['ga0', 'ga1', 'sigma_a', 'gb0', 'gb1', 'sigma_b', 'sigma_y'])











































