# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 20:14:52 2021

@author: PatCa
"""

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import graphviz
import os
import warnings
warnings.filterwarnings('ignore')
import arviz as az
import theano.tensor as tt
import pymc3 as pm
from pymc3 import Model, Normal, HalfNormal
from pymc3 import find_MAP
from pymc3 import NUTS, sample
from scipy import optimize
from pymc3 import traceplot


cty = pd.read_csv('ARM_Data/radon/cty.dat')
srrs2 = pd.read_csv('ARM_Data/radon/srrs2.dat')
srrs2.columns = srrs2.columns.map(str.strip)
srrs_mn = srrs2[srrs2['state'] == 'MN']
srrs_mn['fips'] = 1000*srrs_mn.stfips + srrs_mn.cntyfips

# def rem_space(row):
#     text = row.county
#     ret_text = text.strip()
#     return ret_text

# df3['county1'] = df3.apply(rem_space, axis=1)
#df4 = df3[['state', 'floor', 'activity', 'county']]
#df4 = df4[df4[' activity'] != 0]
#df4['log_radon'] = np.log(df3[' activity'])
#df4.columns = ['state', 'floor', 'activity', 'county', 'log_radon']
#df4['group'] = 1

cty.columns = cty.columns.map(str.strip)
cty_mn = cty[cty['st'] == 'MN']
cty_mn['fips'] = 1000*cty_mn.stfips + cty_mn.ctfips

srrs_mn = srrs_mn.merge(cty_mn[['fips', 'Uppm']], on='fips')
srrs_mn = srrs_mn.drop_duplicates(subset='idnum')

u = np.log(srrs_mn.Uppm.unique())
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

#%%

coords = {'obs_id':np.arange(floor.size),'Level':['Basement', 'Floor']}

with pm.Model(coords=coords) as pooled_model:
    floor_idx = pm.Data('floor_idx', floor, dims='obs_id')
    a = pm.Normal('a', mu=0.0, sigma=10.0, dims='Level')
    
    theta = a[floor_idx]
    sigma = pm.Exponential('sigma',1.0)
    
    y = pm.Normal('y', my=theta, sigma=sigma, observed=log_radon, dims='obs_id')

pm.model_to_graphviz(pooled_model)


#%% Model un-standardized factors

with pm.Model() as monst_model:
    
    #Priors
    sigma = pm.Uniform('sigma', lower=0, upper=50)
    intercept = pm.Normal('intercept', mu=2, sd=5)
    gamma = pm.Normal('gamma', mu=)
    slope = pm.Normal('slope', 0,10)

    eta = gamma*df.county1
    mu_est = intercept + slope*df.floor+eta
    
    #Likelihood
    y = pm.Normal('y',mu=mu_est, sd = sigma, observed = y_height)
    
    #Inference
    map_estimate = pm.find_MAP(model=monst_model)
    trace = pm.sample(2000, tune=1000,start = map_estimate,cores=1)






























