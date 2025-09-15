# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 08:09:21 2018

@author: CJROCKBALL
"""

import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import pymc3 as pm
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit as logistic
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use(['seaborn-colorblind', 'seaborn-darkgrid'])

#%%
d_ad = pd.read_csv('C:/Users/Carlberg-PAT/Documents/Python 3 old/Trees/Bayes course/UCBadmit.csv', sep=';')
d_ad['male'] = (d_ad['applicant.gender'] == 'male').astype(np.int64)
d_ad.head(8)

#%%

with pm.Model() as model_10_6:
    
    #Priors
    a = pm.Normal('a',0,10)
    bn = pm.Normal('bn',0,10)
    p = pm.math.invlogit(a + bn * d_ad.male)
    #Likelihood
    admit = pm.Binomial('admit', d_ad.applications,p,observed=d_ad.admit)
    #MCMC
    trace_10_6 = pm.sample(5000,njobs=1)
    
with pm.Model() as model_10_7:
    
    #Priors
    a = pm.Normal('a',0,1)
    p = pm.math.invlogit(a)
    #Likelihood
    admit = pm.Binomial('admit', n=d_ad.applications, p=p, observed=d_ad.admit)
    #MCMC
    trace_10_7 = pm.sample(5000,njobs=1)
    
#%%

pm.summary(trace_10_6, ['a', 'bn'], alpha=0.1) .round(2)
pm.traceplot(trace_10_6)

summary = pm.summary(trace_10_6, alpha=.11)[['mean', 'sd', 'hpd_5.5', 'hpd_94.5']]
trace_cov = pm.trace_cov(trace_10_6, model=model_10_6)
invD = (np.sqrt(np.diag(trace_cov))**-1)[:, None]
trace_corr = pd.DataFrame(invD*trace_cov*invD.T, index=summary.index, columns=summary.index)

summary.join(trace_corr).round(2)

tracedf = pm.trace_to_dataframe(trace_10_6)
grid = (sns.PairGrid(tracedf,
                     diag_sharey=False)
           .map_diag(sns.kdeplot)
           .map_upper(plt.scatter, alpha=0.1))


#%%
    
comp_df = pm.compare([trace_10_6, trace_10_7], 
                     [model_10_6, model_10_7],
                     method='pseudo-BMA')

comp_df.loc[:,'model'] = pd.Series(['m10.6', 'm10.7'])
comp_df = comp_df.set_index('model')
comp_df
    
#%%

post = pm.trace_to_dataframe(trace_10_6)
p_admit_male = logistic(post['a'] + post['bn'])
p_admit_female = logistic(post['a'])
diff_admit = p_admit_male - p_admit_female
diff_admit.describe(percentiles=[.025, .5, .975])[['2.5%', '50%', '97.5%']]

#%%
for i in range(6):
    x = 1 + 2 * i
    y1 = d_ad.admit[x] / d_ad.applications[x]
    y2 = d_ad.admit[x+1] / d_ad.applications[x+1]
    plt.plot([x, x+1], [y1, y2], '-C0o', lw=2)
    plt.text(x + 0.25, (y1+y2)/2 + 0.05, d_ad.dept[x])
plt.ylim(0, 1);








