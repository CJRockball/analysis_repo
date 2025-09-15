# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 14:19:25 2018

@author: CJROCKBALL
"""

import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'


#%%

import pymc3 as pm
from pymc3 import Uniform, Binomial, Model
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

plt.style.use(['seaborn-colorblind', 'seaborn-darkgrid'])

#%% Make data

toss = ['1','0','1','1','1','0','1','0','1']
stats.binom.pmf(6, n=9, p=0.7)

#%% Grid point approximation

def posterior_grid_approx(grid_points=5, success=6, tosses=9):
    """
    """
    # define grid
    p_grid = np.linspace(0, 1, grid_points)

    # define prior
    prior = np.repeat(1, grid_points)  # uniform
    print(prior)
    #prior = (p_grid >= 0.5).astype(int)  # truncated
    #prior = np.exp(- 5 * abs(p_grid - 0.5))  # double exp

    # compute likelihood at each point in the grid
    likelihood = stats.binom.pmf(success, tosses, p_grid)

    # compute product of likelihood and prior
    unstd_posterior = likelihood * prior
    print(unstd_posterior)
    # standardize the posterior, so it sums to 1
    posterior = unstd_posterior / unstd_posterior.sum()
    return p_grid, posterior

points = 10
w, n = 6, 9
p_grid, posterior = posterior_grid_approx(points, w, n)
plt.plot(p_grid, posterior, 'o-', label='success = {}\ntosses = {}'.format(w, n))
plt.xlabel('probability of water', fontsize=14)
plt.ylabel('posterior probability', fontsize=14)
plt.title('{} points'.format(points))
plt.legend(loc=0);

#%%
water = 6
total_draw = 9
niter = 1000
with Model() as test_model:
    
    #Define priors  
    p_prior = Uniform('p_prior', lower=0, upper=1)
    
    #Likelihood
    # Data likelihood
    bin_model = Binomial('bin_model', n=total_draw, p=p_prior, observed=water)
    
    # Inference!
    trace = pm.sample(progressbar=False) # draw posterior samples using NUTS sampling%
    #start = pm.find_MAP() # Use MAP estimate (optimization) as the initial state for MCMC
    #step = pm.NUTS() # Have a choice of samplers
    #trace = pm.sample(niter, step, start, random_seed=123, progressbar=True)
#%%
    
plt.figure(figsize=(7, 7))
pm.traceplot(trace)
plt.tight_layout();

plt.hist(trace['p_prior'], 15, histtype='step', normed=True, label='post');
x = np.linspace(0, 1, 100)
plt.plot(x, stats.uniform.pdf(x, 0, 1), label='prior');
plt.legend(loc='best');

#%%


