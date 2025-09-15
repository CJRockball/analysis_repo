# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 23:00:10 2021

@author: PatCa
"""

import numpy as np
import pandas as pd
import graphviz
import seaborn as sns
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

#plt.hist(log_radon)
#sns.distplot(log_radon, kde=True)

#%% Make pooled model. Add everything in one

with pm.Model() as pooled_model:
    # Note the M prior parameters for the M groups
    a_pool = pm.Normal('a_pool', mu=2, sd=5)
    b_pool = pm.Normal('b_pool', mu=0, sd=10)
    e_pool = pm.HalfCauchy('e_pool', 5)
    
    y_pred = pm.Normal('y_pred', mu=a_pool + b_pool * floor, sd=e_pool, observed=log_radon)
    
    map_estimate = pm.find_MAP(model=pooled_model)
    trace_p = pm.sample(2000, tune=1000,start = map_estimate,cores=1)

#az.plot_trace(trace_up)
df_p = az.summary(trace_p)

pgv = pm.model_to_graphviz(pooled_model)
pgv

#%% Make unpooled model
# Creates one regression line per county

with pm.Model() as unpooled_model:
    # Note the M prior parameters for the counties groups
    a = pm.Normal('a', mu=2, sd=5, shape=counties)
    b = pm.Normal('b', mu=0, sd=10)#, shape=counties)
    e = pm.HalfCauchy('e', 5)
    
    y_pred = pm.Normal('y_pred', mu=a[county] + b*floor, #b[county] * floor, 
                         sd=e, observed=log_radon)
    # Rescale alpha back - after floor had been centered the computed alpha is different from the original alpha
    #α = pm.Deterministic('α', α_tmp - β * x_m.mean())
    map_estimate = pm.find_MAP(model=unpooled_model)
    trace_up = pm.sample(2000, tune=1000,start = map_estimate,cores=1)

#az.plot_trace(trace_up)
df_up = az.summary(trace_up)

ngv = pm.model_to_graphviz(unpooled_model)
ngv

ppc_pooled = pm.sample_posterior_predictive(trace_up, samples=500, model=unpooled_model, size=100)

#%% partial pooling model

with pm.Model() as partial_pooling:

    # Priors
    mu_a = pm.Normal('mu_a', mu=0., sd=1e5)
    sigma_a = pm.HalfCauchy('sigma_a', 5)

    # Random intercepts
    a = pm.Normal('a', mu=mu_a, sd=sigma_a, shape=counties)

    # Model error
    sigma_y = pm.HalfCauchy('sigma_y',5)

    # Expected value
    y_hat = a[county]

    # Data likelihood
    y_like = pm.Normal('y_like', mu=y_hat, sd=sigma_y, observed=log_radon)
    
    #Tune
    map_estimate = pm.find_MAP(model=partial_pooling)
    trace_up = pm.sample(2000, tune=1000,start = map_estimate,cores=1)

ppgv = pm.model_to_graphviz(partial_pooling)
ppgv

#%% Varying intercept


with pm.Model() as vi_model:

    # Priors
    mu_a = pm.Normal('mu_a', mu=0., sd=10)
    sigma_a = pm.HalfCauchy('sigma_a', 5)


    # Random intercepts
    a_vi = pm.Normal('a_vi', mu=mu_a, sd=sigma_a, shape=counties)
    # Common slope
    b_vi = pm.Normal('b_vi', mu=0., sd=10)

    # Model error
    sd_y = pm.HalfCauchy('sd_y', 5)

    # Expected value
    y_hat = a_vi[county] + b_vi * floor

    # Data likelihood
    y_like = pm.Normal('y_like', mu=y_hat, sd=sd_y, observed=log_radon)

    #Tune
    map_estimate = pm.find_MAP(model=vi_model)
    trace_up = pm.sample(2000, tune=2000,start = map_estimate,cores=1)

#az.plot_trace(trace_up)
df_vi = az.summary(trace_up)

vigv = pm.model_to_graphviz(vi_model)
vigv

#%% plot function

cnt = 'CLAY' 
c = county_lookup[cnt]
local_radon = srrs_mn.log_radon[srrs_mn['county'] == cnt]
local_floor = srrs_mn.floor[srrs_mn['county'] == cnt]
local_n = len(local_floor)

a_up = df_up.loc[str('a['+str(c)+']'),'mean']     
b_up = df_up.loc['b','mean']
x_pred = np.array([0,1])
yup_pred = a_up + b_up*x_pred

a_p = trace_p['a_pool'].mean()
b_p = trace_p['b_pool'].mean()
x_pred = np.array([0,1])
ypool_pred = a_p + b_p*x_pred

a_vi1 = df_vi.loc[str('a_vi['+str(c)+']'),'mean']     
b_vi1 = df_vi.loc['b_vi','mean']
x_pred = np.array([0,1])
yvi_pred = a_vi1 + b_vi1*x_pred


plt.figure(dpi=96)
plt.scatter(local_floor+0.05*(np.random.rand(local_n)-0.5),local_radon)
plt.plot(x_pred, yup_pred, color='black', label='unpooled')
plt.plot(x_pred, ypool_pred, linestyle='--', color='teal', label='pooled')
plt.plot(x_pred, yvi_pred, color='magenta', label='hirerarchical')
plt.legend()
plt.title(cnt)
plt.show()


#%% Varying slope, intercept


with pm.Model() as vsi_model:

    # Priors
    mu_a = pm.Normal('mu_a', mu=0., sd=10)
    sigma_a = pm.HalfCauchy('sigma_a', 5)
    mu_b = pm.Normal('mu_b', mu=0., sd=10)
    sig_b = pm.HalfCauchy('sig_b', 5)

    # Random intercepts
    a_vsi = pm.Normal('a_vsi', mu=mu_a, sd=sigma_a, shape=M)
    # Random slope
    b_vsi = pm.Normal('b_vsi', mu=mu_b, sd=sig_b, shape=M)

    # Model error
    sd_y = pm.HalfCauchy('sd_y', 5)

    # Expected value
    y_hat = a_vsi[county] + b_vsi[county] * floor

    # Data likelihood
    y_like = pm.Normal('y_like', mu=y_hat, sd=sd_y, observed=log_radon)

    #Tune
    #map_estimate = pm.find_MAP(model=vsi_model)
    #trace_vsi = pm.sample(2000, tune=2000,start = map_estimate,cores=1)
    trace_vsi = pm.sample(5000, tune=2000,n_init=50000,cores=1)

#az.plot_trace(trace_up)
df_vsi = az.summary(trace_vsi)

vsigv = pm.model_to_graphviz(vsi_model)
vsigv

pm.traceplot(trace_vsi, var_names=['mu_a', 'mu_b', 'sigma_a','sig_b','sd_y'])

#%%

#Sample y
#y_pred_vsi = pm.sample_posterior_predictive(trace_vsi, samples=100, model=vsi_model)
ppc = pm.sample_ppc(trace_vsi, samples=500, model=vsi_model, size=100)

#Sample mu and sigma
#ppc = pm.sample_posterior_predictive(
#        trace_g, var_names=['mu','s'], random_seed=123, model=dist_est)



#%% Group level predictors

from pymc3 import Deterministic

with pm.Model() as gl_pred:

    # Priors
    sigma_a = pm.HalfCauchy('sigma_a', 5)
    # County uranium model for slope
    gamma_0 = pm.Normal('gamma_0', mu=0., sd=10)
    gamma_1 = pm.Normal('gamma_1', mu=0., sd=10)


    # Uranium model for intercept
    mu_a = gamma_0 + gamma_1*u
    # County variation not explained by uranium
    eps_a = pm.Normal('eps_a', mu=0., sd=sigma_a, shape=M)
    a = Deterministic('a', mu_a + eps_a[county]) 
    
    
    # Common slope
    b = pm.Normal('b', mu=0., sd=10)

    # Model error
    sigma_y = pm.Uniform('sigma_y', lower=0., upper=100.)

    # Expected value
    y_hat = a + b * floor

    # Data likelihood
    y_like = pm.Normal('y_like', mu=y_hat, sd=sigma_y, observed=log_radon)

    #Tune
    #map_estimate = pm.find_MAP(model=gl_pred)
    #trace_gl_pred = pm.sample(2000, tune=2000,start = map_estimate,cores=1)
    trace_gl_pred = pm.sample(2000, tune=2000,n_init=50000,cores=1)


df_gl_pred = az.summary(trace_gl_pred)

glgv = pm.model_to_graphviz(gl_pred)
glgv

#%% Plot means

a_means = trace_gl_pred['a'].mean(axis=0)
plt.scatter(u, a_means)
g0 = trace_gl_pred['gamma_0'].mean()
g1 = trace_gl_pred['gamma_1'].mean()
xvals = np.linspace(-1, 0.8)
plt.plot(xvals, g0+g1*xvals, 'k--')
plt.xlim(-1, 0.8)

a_se = trace_gl_pred['a'].std(axis=0)
for ui, m, se in zip(u, a_means, a_se):
    plt.plot([ui,ui], [m-se, m+se], 'b-')
plt.plot(u,a_means)
plt.xlabel('County-level uranium') 
plt.ylabel('Intercept estimate')












































