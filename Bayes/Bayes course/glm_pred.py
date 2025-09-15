# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 08:16:01 2018

@author: CJROCKBALL
"""

import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import scipy.stats as st
import theano as tt

plt.style.use('seaborn-darkgrid')

#%%
def invlogit(x):
    return np.exp(x) / (1 + np.exp(x))

n = 4000
coeff = 1.

predictors = np.random.normal(size=n)
# Turn predictor into a shared var so that we can change it later
predictors_shared = tt.shared(predictors)

outcomes = np.random.binomial(1, invlogit(coeff * predictors))

#%% oos data
predictors_out_of_sample = np.random.normal(size=50)
outcomes_out_of_sample = np.random.binomial(1, invlogit(coeff * predictors_out_of_sample))

#%%

with pm.Model() as model:
    m_coeff = pm.Normal('m_coeff', mu=0, sd=1)
    p = pm.math.invlogit(m_coeff*predictors_shared)
    outcome = pm.Bernoulli('outcome', p=p, observed=outcomes)
    trace = pm.sample(5000,njobs=1)

#%%
# Changing values here will also change values in the model
predictors_shared.set_value(predictors_out_of_sample)

#%%
# Simply running PPC will use the updated values and do prediction
ppc = pm.sample_ppc(trace, model=model, samples=100)
#%%
_, ax = plt.subplots(figsize=(12, 6))

β = st.beta((ppc['outcome'] == 1).sum(axis=0), (ppc['outcome'] == 0).sum(axis=0))

# estimated probability
ax.scatter(x=predictors_out_of_sample, y=β.mean())

# error bars on the estimate
plt.vlines(predictors_out_of_sample, *β.interval(0.95))

# actual outcomes
ax.scatter(x=predictors_out_of_sample,
           y=outcomes_out_of_sample, marker='x')

# True probabilities
x = np.linspace(predictors_out_of_sample.min(), predictors_out_of_sample.max())
ax.plot(x, invlogit(x), linestyle='-')


ax.set_xlabel('predictor')
ax.set_ylabel('outcome');



