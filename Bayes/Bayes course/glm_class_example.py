# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 13:46:46 2018

@author: CJROCKBALL
"""

import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import numpy as np 
import pymc3 as pm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.style.use(['seaborn-colorblind', 'seaborn-darkgrid'])

#%% class 0:
# covariance matrix and mean
cov0 = np.array([[5,-4],[-4,4]])
mean0 = np.array([2.,3])
# number of data points
m0 = 80

# class 1
# covariance matrix
cov1 = np.array([[5,-3],[-3,3]])
mean1 = np.array([1.,1])
m1 = 80

# generate m0 gaussian distributed data points with
# mean0 and cov0.
r0 = np.random.multivariate_normal(mean0, cov0, m0)
r1 = np.random.multivariate_normal(mean1, cov1, m1)

def plot_data(r0, r1):
    plt.figure(figsize=(7.,7.))
    plt.scatter(r0[...,0], r0[...,1], c='b', marker='o', label="Klasse 0")
    plt.scatter(r1[...,0], r1[...,1], c='r', marker='x', label="Klasse 1")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

plot_data(r0, r1)


X_train = np.concatenate((r0,r1))
y_train = np.zeros(len(r0)+len(r1))
y_train[:len(r0)] = 1

df = pd.DataFrame({"x1":X_train[:,0], "x2":X_train[:,1], "y":y_train})
df.head(3)

# we add an extra column 0 with 1 for convience 
X = np.concatenate((np.ones((len(X_train),1)), X_train), axis=1)
y = y_train

r0_test = np.random.multivariate_normal(mean0, cov0, 500)
r1_test = np.random.multivariate_normal(mean1, cov1, 500)

X_test_ = np.concatenate((r0_test, r1_test))
y_test = np.zeros(len(r0_test)+len(r1_test))
y_test[:len(r0_test)] = 1
n_test = len(y_test)
# we add an extra column 0 with 1 for convience 
X_test = np.concatenate((np.ones((len(X_test_),1)), X_test_), axis=1)
#%%

with pm.Model() as class_model:
    
    #Priors
    a = pm.Normal('a', 0, 10)
    bx = pm.Normal('bx', 0, 10)
    by = pm.Normal('by', 0 , 10)

    p = pm.math.invlogit(a + bx*df.x1+by*df.x2)
    y_class = pm.Binomial('y_class',n=1,p=p,observed=y)
    
    trace_class = pm.sample(5000, tune=1000,start = pm.find_MAP(),njobs=1)

#%%
pm.summary(trace_class).round(2)
#%%
#%% 
summary = pm.summary(trace_class, alpha=.11)[['mean', 'sd', 'hpd_5.5', 'hpd_94.5']]
trace_cov = pm.trace_cov(trace_class, model=class_model)
invD = (np.sqrt(np.diag(trace_cov))**-1)[:, None]
trace_corr = pd.DataFrame(invD*trace_cov*invD.T, index=summary.index, columns=summary.index)

summary.join(trace_corr).round(2)

#%%
pm.forestplot(trace_class) 
#%%
pm.traceplot(trace_class)
#%% plot corr
tracedf = pm.trace_to_dataframe(trace_class)
grid = (sns.PairGrid(tracedf,
                     diag_sharey=False)
           .map_diag(sns.kdeplot)
           .map_upper(plt.scatter, alpha=0.1))

#%% Plot decision boundary
    
trace_data = trace_class[:1000]
x_seq = np.linspace(-6,8,100)

ap = trace_class['a'].mean()
bxp = trace_class['bx'].mean()
byp = trace_class['by'].mean()

center_est = -(ap + bxp*x_seq)/byp

df1 = pm.trace_to_dataframe(trace_class)
df2 = pm.trace_to_dataframe(trace_class)[:20] 
df3 = df1[['a', 'bx', 'by']]

    
#fig,ax = plt.subplots(1, 1, figsize=(5, 5))
plt.figure()
plt.scatter(r0[...,0], r0[...,1], c='b', marker='o', label="Klasse 0")
plt.scatter(r1[...,0], r1[...,1], c='r', marker='x', label="Klasse 1") 
plt.plot(x_seq,center_est, '--', color='k')  
for i in range(0, 19):
    plt.plot(x_seq, -(df2.iloc[i]['a'] + df2.iloc[i]['bx']  * x_seq)/df2.iloc[i]['by'],\
             color='g', alpha=0.3)

#%%
def logistic_function_(z):
    return 1./(1.+np.exp(-z))  

r = np.outer(X_test[:,0],df3.a[:500])
r = r + np.outer(X_test[:,1],df3.bx[:500])
r = r + np.outer(X_test[:,2],df3.by[:500])

predictions_1 = logistic_function_(r).mean(axis=1)>0.5    
(predictions_1 == y_test).mean()

#%%

x_coord = np.linspace(-6,8,100)
y_coord = np.linspace(-6,8,100)
m = np.zeros((100,100))

for i in range(99):
    for j in range(99):
        m[i,j] = ap+x_coord[i]*bxp+y_coord[j]*byp
        
statmap = logistic_function_(m)

plt.figure()

#plt.imshow(statmap)        
#plt.colorbar()
cmap = sns.diverging_palette(250, 12, s=85, l=25, as_cmap=True)
plt.contourf(x_coord, y_coord, statmap,
                  cmap=cmap)
plt.scatter(r0[...,0], r0[...,1], c='b', marker='o', label="Klasse 0")
plt.scatter(r1[...,0], r1[...,1], c='r', marker='x', label="Klasse 1") 

#%%

# Creater posterior predictive samples
ppc = pm.sample_ppc(trace_class, model=class_model, samples=500)





