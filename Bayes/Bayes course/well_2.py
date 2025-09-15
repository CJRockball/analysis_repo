# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 09:45:55 2018

@author: CJROCKBALL
"""

import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import numpy as np
import pymc3 as pm
import pandas as pd
import theano as tt
import scipy.stats as st
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

plt.style.use('seaborn-darkgrid')

#%%
def invlogit(x):
    return np.exp(x) / (1 + np.exp(x))

#%%
d = pd.read_csv('C:/Users/Carlberg-PAT/Documents/Python 3 old/Trees/Bayes course/wells2.csv', sep=',')
d.head(8)

d_copy = d.copy()
d_copy['dist100'] = d.dist/100

d_copy['log_arsenic'] = np.log(d_copy.arsenic)
d_copy.log_arsenic =(d_copy.log_arsenic-d_copy.log_arsenic.mean())

#%%
msk = np.random.rand(len(d_copy)) < 0.8

df_train = d_copy[msk]
df_test = d_copy[~msk]

outcome_test = df_test.switch

#%%
pred1 = tt.shared(np.asarray(df_train.dist100))
pred2 = tt.shared(np.asarray(df_train.arsenic))
pred3 = tt.shared(np.asarray(df_train.educ))

#%% Distance model
with pm.Model() as well_model:
    
    #Priors
    a = pm.Normal('a',0,10)
    b1 = pm.Normal('b1',0,10)
    b2 = pm.Normal('b2',0,10)
    b3 = pm.Normal('b3',0,10)
    b12 = pm.Normal('b12',0,10)
    b13 = pm.Normal('b13',0,10)
    b23 = pm.Normal('b23',0,10)
    p = pm.math.invlogit(a + b1*pred1 + b2*pred2 + b3*pred3 + b12*pred1*pred2 + \
                         b13*pred1*pred3 + b23*pred2*pred3)
    #Likelihood
    well_switch = pm.Binomial('well_switch', 1,p=p,observed=df_train.switch)
    #MCMC
    trace_switch = pm.sample(2000,start = pm.find_MAP(),chains=2,njobs=1)

#%%
pm.traceplot(trace_switch, ['a', 'b1'])
#%%
pm.summary(trace_switch).round(2)
# Supposedly the natural units a,b1....not sure
#invlogit(pm.summary(trace_switch, ['a', 'b1'])[['mean','hpd_2.5','hpd_97.5']])
#%%
pred2_log = tt.shared(np.asarray(df_train.log_arsenic))

#%% Distance model
with pm.Model() as well_log_model:
    
    #Priors
    a = pm.Normal('a',0,10)
    b1 = pm.Normal('b1',0,10)
    b2 = pm.Normal('b2',0,10)
    b3 = pm.Normal('b3',0,10)
    b12 = pm.Normal('b12',0,10)
    b13 = pm.Normal('b13',0,10)
    b23 = pm.Normal('b23',0,10)
    p = pm.math.invlogit(a + b1*pred1 + b2*pred2_log + b3*pred3 + b12*pred1*pred2_log + \
                         b13*pred1*pred3 + b23*pred2_log*pred3)
    #Likelihood
    well_switch = pm.Binomial('well_switch', 1,p=p,observed=df_train.switch)
    #MCMC
    trace_log_switch = pm.sample(2000,chains=2,start = pm.find_MAP(),njobs=1)

#%% Distance model
with pm.Model() as well_log_model_red:
    
    #Priors
    a = pm.Normal('a',0,10)
    b1 = pm.Normal('b1',0,10)
    b2 = pm.Normal('b2',0,10)
    b3 = pm.Normal('b3',0,10)
    p = pm.math.invlogit(a + b1*pred1 + b2*pred2_log + b3*pred3)
    #Likelihood
    well_switch = pm.Binomial('well_switch', 1,p=p,observed=df_train.switch)
    #MCMC
    trace_log_switch_red = pm.sample(2000,chains=2,start = pm.find_MAP(),njobs=1)

#%%

comp_df = pm.compare([trace_switch, trace_log_switch, trace_log_switch_red], 
                     [well_model,well_log_model,well_log_model_red],
                     method='pseudo-BMA')

comp_df.loc[:,'model'] = pd.Series(['well_model','well_log_model','well_log_model_red'])
comp_df = comp_df.set_index('model')
comp_df

#%%
pm.compareplot(comp_df)    
#%%    
# Changing values here will also change values in the model
pred1_test = np.asarray(df_test.dist100)
pred2_test = np.asarray(df_test.arsenic)
pred2_log_test = np.asarray(df_test.log_arsenic)
pred3_test = np.asarray(df_test.educ)
pred1.set_value(pred1_test)
pred2_log.set_value(pred2_log_test)
pred3.set_value(pred3_test)
#%%
# Simply running PPC will use the updated values and do prediction
ppc = pm.sample_ppc(trace_log_switch_red, model=well_log_model_red, samples=100)
#%%

pred_vect = ppc['well_switch'].mean(axis=0)>0.5
(pred_vect == outcome_test).mean()
#%%
confMat = confusion_matrix( outcome_test, pred_vect )
print( "Majority vote estimate of true category:\n" , confMat )

#%%

_, ax = plt.subplots(figsize=(12, 6))

β = st.beta((ppc['well_switch'] == 1).sum(axis=0), (ppc['well_switch'] == 0).sum(axis=0))

# estimated probability
ax.scatter(x=pred2_test, y=β.mean())

# error bars on the estimate
plt.vlines(pred2_test, *β.interval(0.95))

# actual outcomes
ax.scatter(x=pred2_test,
           y=outcome_test, marker='x')

# True probabilities
x = np.linspace(pred2_test.min(), pred2_test.max())
ax.plot(x, invlogit(x), linestyle='-')


ax.set_xlabel('arsenic')
ax.set_ylabel('well_switch');


























