# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 10:37:38 2021

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

#%%

with pm.Model() as u_lin:
    
    a = pm.Normal('a', mu=0, sigma=10)
    b = pm.Normal('b', mu=0, sigma=10)
    s = pm.HalfCauchy('s', 5)

    uran = pm.Normal('uran', mu=a + b*county, sigma=s, observed=u)

    trace = pm.sample(2000, tune=1000,n_init=50000,cores=1)

#%%

df_u = az.summary(trace)

plt.figure()
plt.scatter(county,u, color='purple')

#%%

with pm.Model() as grp_u:
    
    # Priors
    g0 = pm.Normal('g0', mu=0, sd=10)
    g1 = pm.Normal('g1', mu=0, sd=10)
    tau = pm.HalfCauchy('tau', 5)
    
    # Uranium model
    mu_a = g0 + g1*u
    
    # County variation not explained by urianium
    a2 = pm.Normal('a2', mu=0, sd=tau, shape=counties)
    a_d = pm.Deterministic('a_d', mu_a+a2[county])

    # Calc
    trace2 = pm.sample(4000, tune=3000, n_init=50000, cores=1, progressbar=1)


#%%

df_u2 = az.summary(trace2)

pm.traceplot(trace2, var_names=['a2', 'g0', 'tau','g1'])
























