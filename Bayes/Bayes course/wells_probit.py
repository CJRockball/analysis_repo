# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 12:00:05 2018

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
d = pd.read_csv('C:/Users/Carlberg-PAT/Documents/Python 3 old/Trees/Bayes course/wells2.csv', sep=',')
d.head(8)

d_copy = d.copy()
d_copy['dist100'] = d.dist/100

#%% Distance model
with pm.Model() as well_model:
    
    #Priors
    a = pm.Normal('a',0,10)
    #a1 = pm.StudentT('a1',7,0,2.5)
    #b0 = pm.Normal('b0',0,10)
    b1 = pm.Normal('b1',0,10)
    #b2 = pm.StudentT('b2',7,0,2.5)
    p = pm.math.invlogit(a+b1*d_copy.dist100)# + b0*d.arsenic
    #Likelihood
    well_switch = pm.Binomial('well_switch', 1,p=p,observed=d.switch)
    #MCMC
    trace_switch = pm.sample(2000,chains=2,njobs=1)

#%%
pm.traceplot(trace_switch, ['a', 'b1'])
#%%
pm.summary(trace_switch).round(2)

#%% Distance and Arsenic level
with pm.Model() as model_dist_ars:
    
    #Priors
    a = pm.Normal('a',0,10)
    b1 = pm.Normal('b0',0,10)
    b2 = pm.Normal('b1',0,10)
    p = pm.math.invlogit(a+b1*d_copy.dist100 + b2*d.arsenic)
    #Likelihood
    well_switch = pm.Binomial('well_switch', 1,p=p,observed=d.switch)
    #MCMC
    trace_dist_ars = pm.sample(2000,chains=2,njobs=1)

#%%
pm.summary(trace_dist_ars).round(2)
    
#%%

with pm.Model() as model_interac:
    
    #Priors
    a = pm.Normal('a',0,10)
    b1 = pm.Normal('b0',0,10)
    b2 = pm.Normal('b1',0,10)
    b12 = pm.Normal('b2',0,10)
    p = pm.math.invlogit(a+b1*d_copy.dist100 + b2*d.arsenic+b12*d_copy.dist100*d.arsenic)
    #Likelihood
    well_switch = pm.Binomial('well_switch', 1,p=p,observed=d.switch)
    #MCMC
    trace_interac = pm.sample(2000,chains=2,njobs=1)

#%%
pm.summary(trace_interac).round(2)    
#%%


with pm.Model() as model_full:
    
    #Priors
    a = pm.Normal('a',0,10)
    b1 = pm.Normal('b1',0,10)
    b2 = pm.Normal('b2',0,10)
    b3 = pm.Normal('b3',0,10)
    #b12 = pm.Normal('b12',0,10)
    #b13 = pm.Normal('b13',0,10)
    #b23 = pm.Normal('b23',0,10)

    p = pm.math.invlogit(a+b1*d_copy.dist100 + b2*d.arsenic+b3*d.educ)
    #Likelihood
    well_switch = pm.Binomial('well_switch', 1,p=p,observed=d.switch)
    #MCMC
    trace_full = pm.sample(2000,chains=2,njobs=1)    
    
#%%
pm.summary(trace_full).round(2)     

#%%
with pm.Model() as model_full_inter:
    
    #Priors
    a = pm.Normal('a',0,1)
    b1 = pm.Normal('b1',0,1)
    b2 = pm.Normal('b2',0,1)
    b3 = pm.Normal('b3',0,1)
    b12 = pm.Normal('b12',0,1)
    b13 = pm.Normal('b13',0,1)
    b23 = pm.Normal('b23',0,1)

    p = pm.math.invlogit(a+b1*d_copy.dist100 + b2*d.arsenic+b3*d.educ+b12*d_copy.dist100*d.arsenic+b13*d_copy.dist100*d.educ+b23*d.arsenic*d.educ)
    #Likelihood
    well_switch = pm.Binomial('well_switch', 1,p=p,observed=d.switch)
    #MCMC
    trace_full_inter = pm.sample(2000,chains=2,start = pm.find_MAP(),njobs=1)    
#%%
pm.summary(trace_full_inter).round(2)    
#%%
with pm.Model() as model_full_inter_imp:
    
    #Priors
    a = pm.Normal('a',0,2)
    b1 = pm.Normal('b1',0,2)
    b2 = pm.Normal('b2',0,2)
    b3 = pm.Normal('b3',0,2)
    b13 = pm.Normal('b13',0,2)


    p = pm.math.invlogit(a+b1*d_copy.dist100 + b2*d.arsenic+b3*d.educ+b13*d_copy.dist100*d.educ)
    #Likelihood
    well_switch = pm.Binomial('well_switch', 1,p=p,observed=d.switch)
    #MCMC
    trace_full_inter_imp = pm.sample(2000,chains=2,njobs=1)   
#%%
pm.summary(trace_full_inter_imp).round(2)    

#%%
with pm.Model() as model_full_inter_imp2:
    
    #Priors
    a = pm.Normal('a',0,2)
    b1 = pm.Normal('b1',0,2)
    b2 = pm.Normal('b2',0,2)
    b3 = pm.Normal('b3',0,2)
    b12 = pm.Normal('b12',0,2)
    b13 = pm.Normal('b13',0,2)
    #b23 = pm.Normal('b23',0,1)

    p = pm.math.invlogit(a+b1*d_copy.dist100 + b2*d.arsenic+b3*d.educ+b12*d_copy.dist100*d.arsenic+b13*d_copy.dist100*d.educ)
    #Likelihood
    well_switch = pm.Binomial('well_switch', 1,p=p,observed=d.switch)
    #MCMC
    trace_full_inter_imp2 = pm.sample(2000,chains=2,njobs=1)    
#%%
pm.summary(trace_full_inter_imp2).round(2) 
#%%
with pm.Model() as model_full_inter_imp3:
    
    #Priors
    a = pm.Normal('a',0,2)
    b1 = pm.Normal('b1',0,2)
    b2 = pm.Normal('b2',0,2)
    #b3 = pm.Normal('b3',0,2)
    b12 = pm.Normal('b12',0,2)
    b13 = pm.Normal('b13',0,2)
    #b23 = pm.Normal('b23',0,1)

    p = pm.math.invlogit(a+b1*d_copy.dist100 + b2*d.arsenic+b12*d_copy.dist100*d.arsenic+b13*d_copy.dist100*d.educ)
    #Likelihood
    well_switch = pm.Binomial('well_switch', 1,p=p,observed=d.switch)
    #MCMC
    trace_full_inter_imp3 = pm.sample(2000,chains=2,njobs=1)    
#%%
pm.summary(trace_full_inter_imp3).round(2)    

#%% Center the data

d_copy_norm =(d_copy-d_copy.mean())


with pm.Model() as model_main_b13_centered:
    
    #Priors
    a = pm.Normal('a',0,2)
    b1 = pm.Normal('b1',0,2)
    b2 = pm.Normal('b2',0,2)
    b3 = pm.Normal('b3',0,2)
    b13 = pm.Normal('b13',0,2)


    p = pm.math.invlogit(a+b1*d_copy_norm.dist100 + b2*d_copy_norm.arsenic \
                         +b3*d_copy_norm.educ+b13*d_copy_norm.dist100*d_copy_norm.educ)
    #Likelihood
    well_switch = pm.Binomial('well_switch', 1,p=p,observed=d.switch)
    #MCMC
    trace_main_b13_centered = pm.sample(2000,chains=2,njobs=1)   
#%%
pm.summary(trace_main_b13_centered).round(2)    

#%% Center the data, take log of arsenic data

d_copy_norm['log_arsenic'] = np.log(d_copy.arsenic)
d_copy_norm.log_arsenic =(d_copy_norm.log_arsenic-d_copy_norm.log_arsenic.mean())


with pm.Model() as model_main_b13_centered_logars:
    
    #Priors
    a = pm.Normal('a',0,2)
    b1 = pm.Normal('b1',0,2)
    b2 = pm.Normal('b2',0,2)
    b3 = pm.Normal('b3',0,2)
    b13 = pm.Normal('b13',0,2)


    p = pm.math.invlogit(a+b1*d_copy_norm.dist100 + b2*d_copy_norm.log_arsenic \
                         +b3*d_copy_norm.educ+b13*d_copy_norm.dist100*d_copy_norm.educ)
    #Likelihood
    well_switch = pm.Binomial('well_switch', 1,p=p,observed=d.switch)
    #MCMC
    trace_main_b13_centered_logars = pm.sample(2000,chains=2,njobs=1)   
#%%
pm.summary(trace_main_b13_centered_logars).round(2)  
#%% Center the data, take log of arsenic data

d_copy_norm['log_dist100'] = np.log(d_copy.dist100)
d_copy_norm.log_dist100 =(d_copy_norm.log_dist100-d_copy_norm.log_dist100.mean())


with pm.Model() as model_logars_logdist:
    
    #Priors
    a = pm.Normal('a',0,2)
    b1 = pm.Normal('b1',0,2)
    b2 = pm.Normal('b2',0,2)
    b3 = pm.Normal('b3',0,2)
    #b13 = pm.Normal('b13',0,2)


    p = pm.math.invlogit(a+b1*d_copy_norm.dist100 + b2*d_copy_norm.log_arsenic \
                         +b3*d_copy_norm.educ)
    #Likelihood
    well_switch = pm.Binomial('well_switch', 1,p=p,observed=d.switch)
    #MCMC
    trace_logars_logdist = pm.sample(2000,chains=2,njobs=1)   
#%%
pm.summary(trace_logars_logdist).round(2)  

pm.plot_posterior(trace_logars_logdist, varnames=['a','b1','b2','b3'])


#%%

comp_df = pm.compare([trace_interac,trace_full,trace_full_inter_imp,\
                      trace_full_inter_imp2,trace_full_inter_imp3,\
                      trace_main_b13_centered,trace_main_b13_centered_logars\
                      ,trace_logars_logdist], 
                     [model_interac,model_full,model_full_inter_imp,\
                      model_full_inter_imp2,model_full_inter_imp3,\
                      model_main_b13_centered,model_main_b13_centered_logars,\
                      model_logars_logdist],
                     method='pseudo-BMA')

comp_df.loc[:,'model'] = pd.Series(['model_interac','model_full',\
           'model_full_inter_imp','model_full_inter_imp2','model_full_inter_imp3',\
           'model_main_b13_centered','model_main_b13_centered_logars','model_logars_logdist'])
comp_df = comp_df.set_index('model')
comp_df

#%%
pm.compareplot(comp_df)





