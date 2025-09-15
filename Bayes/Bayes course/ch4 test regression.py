# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 15:03:28 2018

@author: CJ ROCKBALL
"""


import pymc3 as pm
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import collections
import seaborn as sns
import random

plt.style.use(['seaborn-colorblind', 'seaborn-darkgrid'])

#%%
path = 'C:/Users/PatCa/Documents/PythonScripts/Mixed_Folders/Python 3 old/Trees/Bayes course/Howell2.csv'
d = pd.read_csv(path, header=0)

#%%
d.head()

d2 = d[d.height > 0] #Clean data

#%%

d2_norm =(d2-d2.mean())/d2.std()
y_height = d2.loc[:,'height']
x_weight = d2.loc[:,'weight']

y_val_norm = d2_norm.loc[:,'height']
x_val_norm = d2_norm.loc[:,'weight']

plt.scatter(x_weight, y_height)
plt.show()
#plt.hist(y_height, normed = True)
sns.distplot(y_height,hist = False, kde = True)
plt.show()

#%% Model un-standardized factors

with pm.Model() as monst_model:
    
    #Priors
    sigma = pm.Uniform('sigma', lower=0, upper=50)
    intercept = pm.Normal('intercept', 150,40)
    slope = pm.Normal('slope', 0,10)

    mu_est = intercept + slope*x_weight
    
    #Likelihood
    y = pm.Normal('y',mu=mu_est, sd = sigma, observed = y_height)
    
    #Inference
    map_estimate = pm.find_MAP(model=monst_model)
    trace = pm.sample(2000, tune=1000,start = map_estimate,cores=1)
    

#%%
pm.summary(trace).round(2)

#%%
trace_d2 = pm.trace_to_dataframe(trace) #Put trace samples in Dataframe
trace_d2.corr() #Show parameter correlation
#%%
pm.traceplot(trace)

#%%
plt.scatter(x_weight, y_height)
plt.plot(x_weight, map_estimate['intercept'] + map_estimate['slope'] * x_weight, 'C2-')
plt.xlabel(d2.columns[1], fontsize=14)
plt.ylabel(d2.columns[0], fontsize=14)

#%% Uncertainty of mean around 80kg
# For example weight=80
mu_at_80 = trace_d2.loc[:,'intercept'] + trace_d2.loc[:,'slope']*80
sns.distplot(mu_at_80, hist = False, kde = True)
plt.show()
ma80_hpd = pm.hpd(mu_at_80,0.11)
print ('89% hpdi of mu: ', ma80_hpd)
#%% plot MAP with 89% hpdi
# Generate mu sample from trace
weigth_seq = np.arange(45, 120) #mu range
chose_list = random.sample(range(len(trace_d2)), 100) #random list from trace for intercept and slope
mu_pred = np.zeros((len(weigth_seq),100))# len(trace_d2))), pre allocate matrix
for i, w in enumerate(weigth_seq):
    mu_pred[i] = trace_d2.loc[chose_list,'intercept'] + trace_d2.loc[chose_list,'slope'] * w
# Calculate sample mu in all range
plt.plot(weigth_seq, mu_pred, 'C0.', alpha=0.05)
plt.xlabel('weight', fontsize=14)
plt.ylabel('height', fontsize=14);
plt.show()

mu_mean = mu_pred.mean(1) #Calculate mean in each point
mu_hpd = pm.hpd(mu_pred.T, alpha=.11) # 89% hpdi in each point

plt.scatter(x_weight, y_height)
plt.plot(weigth_seq, mu_mean, 'C2')
plt.fill_between(weigth_seq, mu_hpd[:,0], mu_hpd[:,1], color='C2', alpha=0.25)
plt.xlabel('weight', fontsize=14)
plt.ylabel('height', fontsize=14)
plt.xlim(x_weight.min()-5, x_weight.max()+5);
plt.show()

#%% Add prediction inteval. Generate samples with sigma

height_pred = pm.sample_ppc(trace, 200, monst_model) # Generate samples
height_pred_hpd = pm.hpd(height_pred['y']) # Calculate sample HPD
# Sort samples on quantiles
height_pred_quant = pm.quantiles(height_pred['y'],[2.5,25,50,75,97.5])
height_pred_quant_od = collections.OrderedDict(sorted(height_pred_quant.items()))
height_pred_q_list = []
height_pred_q_list = [(v) for k, v in height_pred_quant_od.items()]   

idx = np.argsort(x_weight) # Create sort vector for measurements
d2_weight_ord = d2.weight.iloc[idx] #Sort measurements
# Get quantiles and sort
height_pred_quant1 = np.array(height_pred_q_list[0])[idx]
height_pred_quant2 = np.array(height_pred_q_list[1])[idx]
height_pred_quant3 = np.array(height_pred_q_list[2])[idx]
height_pred_quant4 = np.array(height_pred_q_list[3])[idx]
height_pred_quant5 = np.array(height_pred_q_list[4])[idx]

plt.scatter(d2.weight, d2.height) #Plot measurement
plt.plot(weigth_seq, mu_mean, 'C2') # Plot MAP 
plt.fill_between(weigth_seq, mu_hpd[:,0], mu_hpd[:,1], color='C2', alpha=0.25) # Plot MAP HDP
plt.fill_between(d2_weight_ord, height_pred_quant2, height_pred_quant4, color='C2', alpha=0.5) #Plot 50% interval of predictions
plt.fill_between(d2_weight_ord, height_pred_quant1, height_pred_quant5, color='C2', alpha=0.25) # Plot 95% inteval of prediction
plt.xlabel('weight', fontsize=14)
plt.ylabel('height', fontsize=14)
plt.xlim(d2.weight[:].min(), d2.weight[:].max());
































