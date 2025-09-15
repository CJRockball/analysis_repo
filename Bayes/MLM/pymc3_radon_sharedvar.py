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
import warnings
warnings.filterwarnings('ignore')

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
u_unique = np.log(srrs_mn.Uppm.unique())
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
    
    d_floor = pm.Data("d_floor", floor)
    y_pred = pm.Normal('y_pred', mu=a_pool + b_pool * d_floor, sd=e_pool, observed=log_radon)
    
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
    
    f_data = pm.Data('f_data', floor)
    c_data = pm.Data('c_data', county)
    r_data = pm.Data('r_data', log_radon)
    
    y_pred = pm.Normal('y_pred', mu=a[c_data] + b*f_data, #b[county] * floor, 
                         sd=e, observed=r_data)
    # Rescale alpha back - after floor had been centered the computed alpha is different from the original alpha
    #α = pm.Deterministic('α', α_tmp - β * x_m.mean())
    map_estimate = pm.find_MAP(model=unpooled_model)
    trace_up = pm.sample(2000, tune=1000,start = map_estimate,cores=1)

#az.plot_trace(trace_up)
df_up = az.summary(trace_up)

ngv = pm.model_to_graphviz(unpooled_model)
ngv
#%% Predict within sample using sharable variables.
# Predicts distribution for basement, floor for existing groups

cnt = 'ST LOUIS' 
c = county_lookup[cnt]

with unpooled_model:
    pm.set_data({'f_data': [0,1], 'c_data':[c,c], 'r_data':[0,0]})
    c_pred = pm.sample_posterior_predictive(trace_up,2000)

c_pred0 = c_pred['y_pred'][:,0]
c_pred1 = c_pred['y_pred'][:,1]
fig,ax = plt.subplots(dpi=192)
az.plot_kde(c_pred0, label='basement', fill_kwargs={"alpha": 0.4, "color": "magenta"})
az.plot_kde(c_pred1, label='floor 1', fill_kwargs={"alpha": 0.4, "color": "lightgreen"})
ax.legend()
ax.set_xlabel("Log_radon")

#%% Varying slope, intercept


with pm.Model() as vsi_model:

    # Priors
    mu_a = pm.Normal('mu_a', mu=0., sd=10)
    sigma_a = pm.HalfCauchy('sigma_a', 5)
    mu_b = pm.Normal('mu_b', mu=0., sd=10)
    sig_b = pm.HalfCauchy('sig_b', 5)

    # Random intercepts
    a_vsi = pm.Normal('a_vsi', mu=mu_a, sd=sigma_a, shape=counties)
    # Random slope
    b_vsi = pm.Normal('b_vsi', mu=mu_b, sd=sig_b, shape=counties)
    # Model error
    sd_y = pm.HalfCauchy('sd_y', 5)

    #Shared variables
    f_data = pm.Data('f_data', floor)
    c_data = pm.Data('c_data', county)
    r_data = pm.Data('r_data', log_radon)
    
    # Expected value
    y_hat = a_vsi[c_data] + b_vsi[c_data] * f_data
    # Data likelihood
    y_like = pm.Normal('y_like', mu=y_hat, sd=sd_y, observed=r_data)

    #Tune
    #map_estimate = pm.find_MAP(model=vsi_model)
    #trace_vsi = pm.sample(2000, tune=2000,start = map_estimate,cores=1)
    trace_vsi = pm.sample(5000, tune=2000,n_init=50000,cores=1)

#az.plot_trace(trace_up)
df_vsi = az.summary(trace_vsi)

vsigv = pm.model_to_graphviz(vsi_model)
vsigv

pm.traceplot(trace_vsi, var_names=['mu_a', 'mu_b', 'sigma_a','sig_b','sd_y'])

#%% Predict within sample using sharable variables.
# Predicts distribution for basement, floor for existing groups

cnt = 'ST LOUIS' 
c = county_lookup[cnt]

with vsi_model:
    pm.set_data({'f_data': [0,1], 'c_data':[c,c], 'r_data':[0,0]})
    vsi_pred = pm.sample_posterior_predictive(trace_vsi,2000)

vsi_pred0 = vsi_pred['y_like'][:,0]
vsi_pred1 = vsi_pred['y_like'][:,1]
fig,ax = plt.subplots(dpi=192)
az.plot_kde(vsi_pred0, label='basement', fill_kwargs={"alpha": 0.4, "color": "magenta"})
az.plot_kde(vsi_pred1, label='floor 1', fill_kwargs={"alpha": 0.4, "color": "lightgreen"})
ax.legend()
ax.set_xlabel("Log_radon")

#%%

with pm.Model() as gl_pred:

    #Shared variables
    f_data = pm.Data('f_data', floor)
    c_data = pm.Data('c_data', county)
    r_data = pm.Data('r_data', log_radon)
    u_data = pm.Data('u_data', u)
    
    # Priors
    sigma_a = pm.HalfCauchy('sigma_a', 5)
    # County uranium model for slope
    g0 = pm.Normal('g0', mu=0., sd=10)
    g1 = pm.Normal('g1', mu=0., sd=10)


    # Uranium model for intercept
    mu_a = g0 + g1*u_data
    # County variation not explained by uranium
    eps_a = pm.Normal('eps_a', mu=0., sd=sigma_a, shape=counties)
    a = pm.Deterministic('a', mu_a + eps_a[c_data]) 
    
    
    # Common slope
    b = pm.Normal('b', mu=0., sd=10)

    # Model error
    sigma_y = pm.Uniform('sigma_y', lower=0., upper=100.)

    # Expected value
    y_hat = a + b * f_data

    # Data likelihood
    y_like = pm.Normal('y_like', mu=y_hat, sd=sigma_y, observed=r_data)

    #Tune
    #map_estimate = pm.find_MAP(model=gl_pred)
    #trace_gl_pred = pm.sample(2000, tune=2000,start = map_estimate,cores=1)
    trace_gl_pred = pm.sample(5000, tune=2000,n_init=50000,cores=1)


df_gl_pred = az.summary(trace_gl_pred)
#pm.traceplot(trace_gl_pred, var_names=['b', 'g1', 'g0','sigma_a','sigma_y'])

glgv = pm.model_to_graphviz(gl_pred)
glgv

#%% Predict within sample using sharable variables.
# Predicts distribution for basement, floor for existing groups

cnt = 'CLAY' 
c = county_lookup[cnt]
u_d = u_unique[c]

with gl_pred:
    pm.set_data({'f_data': [0,1], 'c_data':[c,c], 'r_data':[0,0], 'u_data': [u_d,u_d]})
    gl_wis_pred = pm.sample_posterior_predictive(trace_gl_pred,2000)

gl_wis_pred0 = gl_wis_pred['y_like'][:,0]
gl_wis_pred1 = gl_wis_pred['y_like'][:,1]
bas_mean = round(np.mean(gl_wis_pred0),2)
ff_mean = round(np.mean(gl_wis_pred1),2)
fig,ax = plt.subplots(dpi=192)
az.plot_kde(gl_wis_pred0, label='basement', fill_kwargs={"alpha": 0.4, "color": "magenta"})
az.plot_kde(gl_wis_pred1, label='floor 1', fill_kwargs={"alpha": 0.4, "color": "lightgreen"})
ax.legend()
ax.set_xlabel("Log_radon")
ax.text(0.18,0.9,
    f"B mean: {bas_mean}\nF mean: {ff_mean}",
    fontsize=12,
    horizontalalignment="center",
    verticalalignment="center",
    transform=ax.transAxes,
)
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
ypool_pred = a_p + b_p*x_pred

a_vsi1 = df_vsi.loc[str('a_vsi['+str(c)+']'),'mean']     
b_vsi1 = df_vsi.loc['b_vsi['+str(c)+']','mean']
yvi_pred = a_vsi1 + b_vsi1*x_pred

y0 = bas_mean     
y1 = ff_mean
y_ar = np.array([y0,y1])
x_pred = np.array([0,1])


plt.figure(dpi=96)
plt.scatter(local_floor+0.05*(np.random.rand(local_n)-0.5),local_radon)
plt.plot(x_pred, yup_pred, color='black', label='unpooled')
plt.plot(x_pred, ypool_pred, linestyle='--', color='teal', label='pooled')
plt.plot(x_pred, yvi_pred, color='magenta', label='hirerarchical')
plt.plot(x_pred, y_ar, color='orange', label='group predictor')

plt.legend()
plt.title(cnt)
plt.show()

#%% Mouse H model uranium

with pm.Model() as u_grp:
    
    # Grp priors
    g = pm.Normal('g', mu=0., sd=10., shape=2)
    sig_a = pm.Exponential('sig_a', 1.)
    #Var intercept with grp covariate
    alpha = g[0] + g[1]*u_unique
    a = pm.Normal('a', mu=alpha, sigma=sig_a, shape=counties)
    
    # Model priors
    #Common slope 
    b = pm.Normal('b', mu=0., sd=10.)
    # Model error
    sig = pm.Exponential('sig', 1.)

    #Likelihood
    y_hat = a[county] + b*floor    
    y_like = pm.Normal('y_like', mu=y_hat, sd=sig, observed=log_radon)

    # Sample
    trace2 = pm.sample(4000, tune=3000, n_init=50000, chains=4,cores=1)

df_u_grp = az.summary(trace2)
#pm.traceplot(trace_gl_pred, var_names=['b', 'g1', 'g0','sigma_a','sigma_y'])

ugrpgv = pm.model_to_graphviz(u_grp)
ugrpgv

#%%

#Check funnel

pm.traceplot(trace2, var_names=['sig_a','a'])
#print('Rhat(sig_a) {}'.format(trace3[''])

x = pd.Series(trace2['a'][:,75], name='a')
y = pd.Series(trace2['sig_a'], name='sig_a')

plt.figure()
plt.scatter(x,y, alpha=0.4)
plt.show()

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
    trace3 = pm.sample(3000, tune=4000, n_init=50000, chains=4, cores=1)

df_u_grp = az.summary(trace3)
pm.traceplot(trace3, var_names=['a','a_off','b', 'g1', 'g0','sig_a','sig'])

ug2gv = pm.model_to_graphviz(u_grp2)
ug2gv

#%% Plot

a_means = trace3['a_int'].mean(axis=0)
a_se = trace3['a_int'].std(axis=0)
plt.figure(dpi=192)
for ui, m, se in zip(u_unique, a_means, a_se):
    plt.plot([ui,ui], [m-se, m+se], 'b-', color='blue', zorder=1)

plt.scatter(u_unique, a_means, color='blue', zorder=2)

g0 = trace3["g0"].mean()
g1 = trace3['g1'].mean()
xvals = np.linspace(-1, 0.8)
plt.plot(xvals, g0+g1*xvals, 'k--')
plt.xlim(-1, 0.8)

plt.xlabel('County-level uranium') 
plt.ylabel('Intercept estimate')

#----------------------------------------
#Forest plot for std on parameters

plt.figure(figsize=(6,16))
az.plot_forest(trace3,var_names=['b', 'g1', 'g0','sig_a','sig'])

#---------------------------------------
#Plot posterior

plt.figure()
az.plot_posterior(trace3, var_names=['b'])
plt.figure()
az.plot_posterior(trace3, var_names=['b'], hdi_prob=.75)
plt.figure()
az.plot_posterior(trace3, var_names=['b'], kind='hist')

#------------------------------------------------
#Check funnel

pm.traceplot(trace3, var_names=['sig_a','a_int'])
#print('Rhat(sig_a) {}'.format(trace3[''])

x = pd.Series(trace3['a_int'][:,75], name='a_int')
y = pd.Series(trace3['sig_a'], name='sig_a')

plt.figure()
plt.scatter(x,y, alpha=0.4)
plt.show()


#%%

with pm.Model() as hierarchical_intercept:

    # Priors
    sigma_a = pm.HalfCauchy('sigma_a', 5)

    # County uranium model for slope
    g0 = pm.Normal('g0', mu=0., sd=1e5)
    g1 = pm.Normal('g1', mu=0., sd=1e5)


    # Uranium model for intercept
    mu_a = g0 + g1*u
    # County variation not explained by uranium
    eps_a = pm.Normal('eps_a', mu=0, sd=sigma_a, shape=counties)
    a = pm.Deterministic('a', mu_a + eps_a[county])

    # Common slope
    b = pm.Normal('b', mu=0., sd=1e5)

    # Model error
    sigma_y = pm.Uniform('sigma_y', lower=0, upper=100)

    # Expected value
    y_hat = a + b * floor

    # Data likelihood
    y_like = pm.Normal('y_like', mu=y_hat, sd=sigma_y, observed=log_radon)

    # Sample
    trace_hi = pm.sample(4000, tune=4000, n_init=50000, chains=4, cores=1)


df_hi = az.summary(trace_hi)
pm.traceplot(trace_hi, var_names=['a','b', 'g1', 'g0','sigma_a','sigma_y'])

hi2gv = pm.model_to_graphviz(hierarchical_intercept)
hi2gv

















