# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 14:07:21 2021

@author: PatCa
"""


import numpy as np
import pandas as pd
import graphviz
import matplotlib.pyplot as plt
import pymc3 as pm
import arviz as az
import pyreadstat    

df, meta = pyreadstat.read_dta('ARM_Data/child.iq/kidiq.dta')


color = ['red', 'blue']
plt.figure(dpi=192)
plt.scatter(df.mom_iq,df.kid_score, color=[color[int(c)] for c in df.mom_hs], alpha=0.4)
plt.xlabel('Mom iq')
plt.ylabel('kid iq')
plt.show()


#%% momiq basic model

iq = df.mom_iq.to_numpy()
hs = df.mom_hs.to_numpy()
kid_iq = df.kid_score.to_numpy()

with pm.Model() as mom_iq:
    
    #Priors
    biq = pm.Normal('iq', mu=0, sigma=100.)
    bhs = pm.Normal('hs', mu=0., sigma=100.)
    bx = pm.Normal('bx', mu=0., sigma=100.)
    sigma_y = pm.HalfCauchy('sigma_y', 5.)
    a = pm.Normal('a', mu=0, sigma=100)
    
    #Likelihood
    y_hat = a + biq*iq + bhs*hs + bx*iq*hs
    l_like = pm.Normal('y_like', mu=y_hat, sigma=sigma_y, observed=kid_iq)
    
    #fit
    trace = pm.sample(3000, tune=3000, n_init=50000, chains=4, cores=1)

df_t = az.summary(trace)

kid_gv = pm.model_to_graphviz(mom_iq)
kid_gv

#%% mom iq normalize data

iq = df.mom_iq.to_numpy() - 100
hs = df.mom_hs.to_numpy() - 0.5
kid_iq = df.kid_score.to_numpy()

with pm.Model() as mom_iq:
    
    # Data
    iq_data = pm.Data('iq_data', iq)
    hs_data = pm.Data('hs_data', hs)
    kid_data = pm.Data('kid_data', kid_iq)
    
    #Priors
    biq = pm.Normal('iq', mu=0, sigma=100.)
    bhs = pm.Normal('hs', mu=0., sigma=100.)
    bx = pm.Normal('bx', mu=0., sigma=100.)
    sigma_y = pm.HalfCauchy('sigma_y', 5.)
    a = pm.Normal('a', mu=0, sigma=100)
    
    #Likelihood
    y_hat = a + biq*iq_data + bhs*hs_data + bx*iq_data*hs_data
    l_like = pm.Normal('y_like', mu=y_hat, sigma=sigma_y, observed=kid_data)
    
    #fit
    trace = pm.sample(3000, tune=3000, n_init=50000, chains=4, cores=1)

df_tn = az.summary(trace)
az.plot_trace(trace)

kid_gv = pm.model_to_graphviz(mom_iq)
kid_gv


















