# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 10:58:26 2021

@author: PatCa
"""


import numpy as np
import pandas as pd
import graphviz
import matplotlib.pyplot as plt
import pymc3 as pm
import arviz as az

df = pd.read_csv('supplier.csv')
df2 = pd.read_csv('loom_long.csv')

#%%

x = np.array([0,1,2,3])
x =np.repeat(x, 4, axis=0)
loom=np.array([98,97,99,96,91,90,93,92,96,95,97,95,95,96,99,98])
loom_dict = {'Loom':x, 'Thread':loom}
df3 = pd.DataFrame(loom_dict)


#%% One factor nested variance
looms = 4
Loom = df2.Loom.to_numpy()-1

#%% partial pooling model
# Feels right


with pm.Model() as partial_pooling:

    # Priors
    mu_a = pm.Normal('mu_a', mu=0., sd=1e5)
    sigma_y = pm.HalfCauchy('sigma_y', 5)
    
    # Model error
    sigma_t = pm.HalfCauchy('sigma_t',5)
    tau = pm.Normal('tau', mu=0, sd=sigma_t, shape=looms)

    y_hat = mu_a + tau[Loom]
    # Data likelihood
    y_like = pm.Normal('y_like', mu=y_hat, sd=sigma_y, observed=df2.Thread)
    
    #Tune
    trace_loom_p = pm.sample(5000, tune=3000,n_init=50000, chains=4,cores=1)

df_pp_loom = az.summary(trace_loom_p)

lgv = pm.model_to_graphviz(partial_pooling)
lgv

#%% partial pooling model
#Sort of works

with pm.Model() as partial_pooling:

    # Priors
    mu_a = pm.Normal('mu_a', mu=0., sd=1e5)
    sigma_a = pm.HalfCauchy('sigma_a', 5)

    # Random intercepts
    a = pm.Normal('a', mu=mu_a, sd=sigma_a, shape=looms)

    # Model error
    sigma_y = pm.HalfCauchy('sigma_y',5)

    # Expected value
    y_hat = a[Loom]

    # Data likelihood
    y_like = pm.Normal('y_like', mu=y_hat, sd=sigma_y, observed=df2.Thread)
    
    #Tune
    trace_loom_p = pm.sample(5000, tune=3000,n_init=50000, chains=4,cores=1)

df_loom = az.summary(trace_loom_p)
az.plot_trace(trace_loom_p)

pp1gv = pm.model_to_graphviz(partial_pooling)
pp1gv







