# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 11:26:56 2021

@author: PatCa
"""

import numpy as np
import pandas as pd
import graphviz
import seaborn as sns
import matplotlib.pyplot as plt
import pymc3 as pm
import arviz as az

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

u = np.log(srrs_mn.Uppm.unique())
M = len(srrs_mn.Uppm.unique())
n = len(srrs_mn)

#%%

srrs_mn.county = srrs_mn.county.map(str.strip)
mn_counties = srrs_mn.county.unique()
counties = len(mn_counties)
county_lookup = dict(zip(mn_counties, range(counties)))

county = srrs_mn['county_code'] = srrs_mn.county.replace(county_lookup).values
radon = srrs_mn.activity
srrs_mn['log_radon'] = log_radon = np.log(radon + 0.1).values
floor = srrs_mn.floor.values

#plt.hist(log_radon)
sns.distplot(log_radon, kde=True)

#%% Fully pooled model

# Center the data
x = floor
y_data= log_radon

with pm.Model() as pooled_model:
    # Note the M prior parameters for the M groups
    a_pool = pm.Normal('a_pool', mu=2, sd=5)
    b_pool = pm.Normal('b_pool', mu=0, sd=10)
    e_pool = pm.HalfCauchy('e_pool', 5)
    
    y_pred = pm.Normal('y_pred', mu=a_pool + b_pool * x, sd=e_pool, observed=y_data)
    
    map_estimate = pm.find_MAP(model=pooled_model)
    trace_p = pm.sample(2000, tune=1000,start = map_estimate,cores=1)

#%%
df_trace = az.summary(trace_p)
print(df_trace[['mean','sd']]) 
print(pm.traceplot(trace_p))
a_up = trace_p['a_pool'].mean()
b_up = trace_p['b_pool'].mean()
x_pred = np.array([0,1])
ypool_calc = a_up + b_up*x_pred

az.plot_pair(trace_p, var_names=['a_pool', 'b_pool'], plot_kwargs={'alpha': 0.1})

#%%

plt.figure(dpi=96)
plt.scatter(floor+0.05*(np.random.rand(n)-0.5),log_radon)
plt.plot(x_pred, ypool_calc, color='teal', label='pooled')
plt.legend()
plt.show()

#%% Fully unpooled model Normal

idx = county

with pm.Model() as unpooled_model:
    # Note the M prior parameters for the M groups
    a = pm.Normal('a', mu=2, sd=5, shape=M)
    b = pm.Normal('b', mu=0, sd=10, shape=M)
    e = pm.HalfCauchy('e', 5)
    
    y_pred = pm.Normal('y_pred', mu=a[county] + b[county] * x, 
                         sd=e, observed=y_data)
    # Rescale alpha back - after x had been centered the computed alpha is different from the original alpha
    #α = pm.Deterministic('α', α_tmp - β * x_m.mean())
    map_estimate = pm.find_MAP(model=unpooled_model)
    trace_up = pm.sample(2000, tune=1000,start = map_estimate,cores=1)

#az.plot_trace(trace_up)
df_up = az.summary(trace_up)

ngv = pm.model_to_graphviz(unpooled_model)
ngv

#%% 

cnt = 'ST LOUIS' 
c = county_lookup[cnt]
local_radon = srrs_mn.log_radon[srrs_mn['county'] == cnt]
local_floor = srrs_mn.floor[srrs_mn['county'] == cnt]
local_n = len(local_floor)

a_up = df_up.loc[str('a['+str(c)+']'),'mean']     
b_up = df_up.loc[str('b['+str(c)+']'),'mean']
x_pred = np.array([0,1])
yup_pred = a_up + b_up*x_pred

a_p = trace_p['a_pool'].mean()
b_p = trace_p['b_pool'].mean()
x_pred = np.array([0,1])
ypool_pred = a_p + b_p*x_pred

plt.figure(dpi=96)
plt.scatter(local_floor+0.05*(np.random.rand(local_n)-0.5),local_radon)
plt.plot(x_pred, yup_pred, color='black', label='unpooled')
plt.plot(x_pred, ypool_pred, linestyle='--', color='teal', label='pooled')
plt.legend()
plt.show()

az.plot_forest(trace_up, var_names=['a', 'b'], combined=True)



#%% Fully unpooled model Student-t

idx = county

with pm.Model() as unpooled_model:
    # Note the M prior parameters for the M groups
    α_tmp = pm.Normal('α_tmp', mu=2, sd=5, shape=M)
    β = pm.Normal('β', mu=0, sd=10, shape=M)
    ϵ = pm.HalfCauchy('ϵ', 5)
    ν = pm.Exponential('ν', 1/30)
    
    y_pred = pm.StudentT('y_pred', mu=α_tmp[idx] + β[idx] * x, 
                         sd=ϵ, nu=ν, observed=y_data)
    # Rescale alpha back - after x had been centered the computed alpha is different from the original alpha
    #α = pm.Deterministic('α', α_tmp - β * x_m.mean())
    map_estimate = pm.find_MAP(model=unpooled_model)
    trace_up = pm.sample(2000, tune=1000,start = map_estimate,cores=1)

#az.plot_trace(trace_up)
df_up = az.summary(trace_up)

sgv = pm.model_to_graphviz(unpooled_model)
sgv

#%%
with pm.Model() as hierarchical_model:
    # Hyperpriors - we add these instead of setting the prior values to a constant
    # Note that there exists only one hyperprior  for all M groups, shared hyperprior
    α_μ_tmp = pm.Normal('α_μ_tmp', mu=100, sd=1) # try changing these hyperparameters
    α_σ_tmp = pm.HalfNormal('α_σ_tmp', 10) # try changing these hyperparameters
    b_mu = pm.Normal('b_mu', mu=10, sd=2) # reasonable changes do not have an impact
    b_sd = pm.HalfNormal('b_sd', sd=5)
    
    # priors - note that the prior parameters are no longer a constant
    a_hm = pm.Normal('a_hm', mu=α_μ_tmp, sd=α_σ_tmp, shape=M)
    b_hm = pm.Normal('b_hm', mu=b_mu, sd=b_sd, shape=M)
    ϵ = pm.HalfCauchy('ϵ', 5)

    y_pred = pm.Normal('y_pred',
                         mu=a_hm[county] + b_hm[county] * x,
                         sd=ϵ, observed=y_data)
    #α = pm.Deterministic('α', α_tmp - β * x_m.mean())
    #α_μ = pm.Deterministic('α_μ', α_μ_tmp - β_μ *
    #                      x_m.mean())
    #α_σ = pm.Deterministic('α_sd', α_σ_tmp - β_μ * x_m.mean())
    map_estimate = pm.find_MAP(model=hierarchical_model)
    trace_hm = pm.sample(2000, tune=1000,start = map_estimate,cores=1)

df_hm = az.summary(trace_hm)

hgv = pm.model_to_graphviz(hierarchical_model)
hgv
#%%


with pm.Model() as partial_pooling:

    # Priors
    mu_a = pm.Normal('mu_a', mu=0., sd=1e5)
    sigma_a = pm.HalfCauchy('sigma_a', 5)

    # Random intercepts
    a = pm.Normal('a', mu=mu_a, sd=sigma_a, shape=M)

    # Model error
    sigma_y = pm.HalfCauchy('sigma_y',5)

    # Expected value
    y_hat = a[county]

    # Data likelihood
    y_like = pm.Normal('y_like', mu=y_hat, sd=sigma_y, observed=log_radon)


#%%

cnt = 'LAC QUI PARLE' 
c = county_lookup[cnt]
local_radon = srrs_mn.log_radon[srrs_mn['county'] == cnt]
local_floor = srrs_mn.floor[srrs_mn['county'] == cnt]
local_n = len(local_floor)

a_up = df_up.loc[str('a['+str(c)+']'),'mean']     
b_up = df_up.loc[str('b['+str(c)+']'),'mean']
x_pred = np.array([0,1])
yup_pred = a_up + b_up*x_pred

a_p = trace_p['a_pool'].mean()
b_p = trace_p['b_pool'].mean()
x_pred = np.array([0,1])
ypool_pred = a_p + b_p*x_pred

a_hmm = df_hm.loc[str('a_hm['+str(c)+']'),'mean']     
b_hmm = df_hm.loc[str('b_hm['+str(c)+']'),'mean']
x_pred = np.array([0,1])
yhm_pred = a_hmm + b_hmm*x_pred


plt.figure(dpi=96)
plt.scatter(local_floor+0.05*(np.random.rand(local_n)-0.5),local_radon)
plt.plot(x_pred, yup_pred, color='black', label='unpooled')
plt.plot(x_pred, ypool_pred, linestyle='--', color='teal', label='pooled')
plt.plot(x_pred, yhm_pred, color='magenta', label='hirerarchical')
plt.legend()
plt.title(cnt)
plt.show()


#%%

with pm.Model() as hierarchical_model:
    # Hyperpriors
    mu_a = pm.Normal('mu_alpha', mu=0., sigma=1)
    sigma_a = pm.HalfCauchy('sigma_alpha', beta=1)
    mu_b = pm.Normal('mu_beta', mu=0., sigma=1)
    sigma_b = pm.HalfCauchy('sigma_beta', beta=1)
    
    # Intercept for each county, distributed around group mean mu_a
    a_hm = pm.Normal('a_hm', mu=mu_a, sigma=sigma_a, shape=M)
    # Intercept for each county, distributed around group mean mu_a
    b_hm = pm.Normal('b_hm', mu=mu_b, sigma=sigma_b, shape=M)
    
    # Model error
    eps = pm.HalfCauchy('eps', beta=1)
    
    # Expected value
    radon_est = a_hm[idx] + b_hm[idx] * x
    
    # Data likelihood
    y_like = pm.Normal('y_like', mu=radon_est, sigma=eps, observed=y_data)
    trace_hm = pm.sample(10000, tune=2000,start = map_estimate,cores=1)

df_hm = az.summary(trace_hm)

hgv = pm.model_to_graphviz(hierarchical_model)

#%%

cnt = 'ST LOUIS' 
c = county_lookup[cnt]
local_radon = srrs_mn.log_radon[srrs_mn['county'] == cnt]
local_floor = srrs_mn.floor[srrs_mn['county'] == cnt]
local_n = len(local_floor)

a_up = df_up.loc[str('a['+str(c)+']'),'mean']     
b_up = df_up.loc[str('b['+str(c)+']'),'mean']
x_pred = np.array([0,1])
yup_pred = a_up + b_up*x_pred

a_p = trace_p['a_pool'].mean()
b_p = trace_p['b_pool'].mean()
x_pred = np.array([0,1])
ypool_pred = a_p + b_p*x_pred

a_hmm = df_hm.loc[str('a_hm['+str(c)+']'),'mean']     
b_hmm = df_hm.loc[str('b_hm['+str(c)+']'),'mean']
x_pred = np.array([0,1])
yhm_pred = a_hmm + b_hmm*x_pred


plt.figure(dpi=96)
plt.scatter(local_floor+0.05*(np.random.rand(local_n)-0.5),local_radon)
plt.plot(x_pred, yup_pred, color='black', label='unpooled')
plt.plot(x_pred, ypool_pred, linestyle='--', color='teal', label='pooled')
plt.plot(x_pred, yhm_pred, color='magenta', label='hirerarchical')
plt.legend()
plt.title(cnt)
plt.show()























