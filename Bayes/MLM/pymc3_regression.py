# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 10:16:11 2021

@author: PatCa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import pymc3 as pm

#%% Estimating distribution

data = np.array([55.12, 53.73, 50.24, 52.05, 56.4 , 48.45, 52.34, 55.65, 51.49,
       51.86, 63.43, 53.  , 56.09, 51.93, 52.31, 52.33, 57.48, 57.44,
       55.14, 53.93, 54.62, 56.09, 68.58, 51.36, 55.47, 50.73, 51.94,
       54.95, 50.39, 52.91, 51.5 , 52.68, 47.72, 49.73, 51.82, 54.99,
       52.84, 53.19, 54.52, 51.46, 53.73, 51.61, 49.81, 52.42, 54.3 ,
       53.84, 53.16])

plt.plot(data)
plt.show()
sns.histplot(data=data, kde=True)
plt.show()

with pm.Model() as dist_est:
    mu = pm.Uniform('mu', lower=40, upper=70)
    s = pm.HalfNormal('s', sd=10)
    y = pm.Normal('y', mu=mu, sd=s, observed=data)
    trace_g = pm.sample(draws=2000, tune=1000,cores=1) 

az.plot_trace(trace_g)
gv = pm.model_to_graphviz(dist_est)   
gv

print('mu= ', trace_g['mu'].mean())
print('s= ', trace_g['s'].mean())
#%%
#Sample y
y_pred_g = pm.sample_posterior_predictive(trace_g, samples=100, model=dist_est)

#Sample mu and sigma
ppc = pm.sample_posterior_predictive(
        trace_g, var_names=['mu','s'], random_seed=123, model=dist_est)

az.plot_pair(trace_g, kind='kde', fill_last=False)
az.summary(trace_g)

data_ppc = az.from_pymc3(trace=trace_g, posterior_predictive=y_pred_g)
ax = az.plot_ppc(data_ppc, figsize=(12, 6), mean=True)
ax[0].legend(fontsize=15)

#%% Student-t estimate

with pm.Model() as model_t:
    mu = pm.Uniform('my', 40, 75) # mean
    s = pm.HalfNormal('s', sd=10)
    ny = pm.Exponential('ny', 1/30)
    y = pm.StudentT('y', mu=mu, sd=s, nu=ny, observed=data)
    trace_t = pm.sample(draws=2000, tune=1000,cores=1)

az.plot_trace(trace_t)
gv = pm.model_to_graphviz(model_t)  
gv




#%% Regression ----------------------------------------------------------------
np.random.seed(1)
N = 100

alpha_real = 2.5
beta_real = 0.9
eps_real = np.random.normal(0,0.5, size=N)

x = np.random.normal(10,1,N)
y_real = alpha_real + beta_real*x

y = y_real + eps_real

plt.figure()
plt.scatter(x,y)
plt.show()

az.plot_kde(y)

#%%


with pm.Model() as model_g:
    a = pm.Normal('a', mu=0, sd=10)
    b = pm.Normal('b', mu=0, sd=1)
    e = pm.HalfCauchy('e',5)
    mux = pm.Deterministic('mu', a+b*x)
    y_pred = pm.Normal('y_pred', mu=mux, sd=e, observed=y)
    trace_g = pm.sample(2000,tune=1000,cores=1)

#%%
df1 = trace_g    
pm.traceplot(trace_g)    
df = az.summary(trace_g)    
az.plot_pair(trace_g, kind='kde', fill_last=False )
az.plot_pair(trace_g, var_names=['a','b'], plot_kwargs={'alpha':0.1} )
gv = pm.model_to_graphviz(model_g)
gv

#%%
#Visualize uncertainty
plt.figure()
plt.scatter(x,y)

#Get mean inferred values 
inter = trace_g['a'].mean()
b_m = trace_g['b'].mean()

#Plot all every 10 draw to show variance of model
draws10 = range(0,len(trace_g['a']), 10)
plt.plot(x,trace_g['a'][draws10] + trace_g['b'][draws10]*x[:,np.newaxis], 
         c='lightblue', alpha=0.05)

# Plot the mean regression line
plt.plot(x, inter+b_m*x, c='teal', label = f"y = {inter:.2f} + {b_m:.2f}*x")
plt.legend()
plt.show()

#%% Sampling

ppc = pm.sample_posterior_predictive(trace_g, samples=2000, model=model_g)

plt.scatter(x,y)

az.plot_hdi(x,ppc['y_pred'], color='lightblue', fill_kwargs={"alpha": .2})

plt.plot(x, inter+b_m*x, c='teal')
plt.show()

#%%
# Centering data

x_c = x - x.mean()

with pm.Model() as model_c:
    a = pm.Normal('a', mu=0, sd=10)
    b=pm.Normal('b', mu=0, sd=1)
    e = pm.HalfCauchy('e', 5)
    mux = pm.Deterministic('mux', a+b*x_c)
    y_pred = pm.Normal('y_pred', mu=mux, sd=e, observed=y)
    a_rec = pm.Deterministic('a_rec', a-b*x.mean())
    trace_c = pm.sample(2000, tune=1000, cores=1)



df1 = trace_c    
pm.traceplot(trace_c)    
df = az.summary(trace_c)    
az.plot_pair(trace_c, kind='kde', fill_last=False )
az.plot_pair(trace_c, var_names=['a','b'], plot_kwargs={'alpha':0.1} )

#%%

gv = pm.model_to_graphviz(model_c)
#gv.format = 'png'
#gv.render(filename='model_graph', view=True)

gv

#%% Logistic regression -------------------------------------------------------

import pymc3 as pm
import sklearn
import numpy as np
import graphviz
from matplotlib import pyplot as plt
import seaborn
from sklearn import datasets

df = datasets.load_iris()
iris_data = pd.DataFrame(df['data'], columns=df['feature_names'])
iris_data['target'] = df['target']
seaborn.stripplot(x='target', y='sepal length (cm)', data=iris_data, jitter=False)
plt.figure()
seaborn.stripplot(x='target', y='petal length (cm)', data=iris_data, jitter=False)
plt.figure()
seaborn.pairplot(iris_data, hue='target', diag_kind='kde')
plt.figure()
corr = iris_data.query("target == (0,1)").loc[:, iris_data.columns != 'target'].corr() 
mask = np.tri(*corr.shape).T 
seaborn.heatmap(corr.abs(), mask=mask, annot=True)
plt.show()

# Select the first two classes for a binary classification problem
df = iris_data.query("target == (0,1)")
y_0 = df.target
x_n = 'sepal length (cm)' 
x_0 = df[x_n].values
x_c = x_0 - x_0.mean()


with pm.Model() as model_0:
    α = pm.Normal('α', mu=0, sd=10)
    β = pm.Normal('β', mu=0, sd=10)
    μ = α + pm.math.dot(x_c, β)    
    θ = pm.Deterministic('θ', pm.math.sigmoid(μ))
    bd = pm.Deterministic('bd', -α/β)
    yl = pm.Bernoulli('yl', p=θ, observed=y_0)
    trace_0 = pm.sample(2000, tune=1000, cores=1)

gv = pm.model_to_graphviz(model_0)
gv
az.summary(trace_0, var_names=["α","β","bd"])

#VIsualize decision boundary
theta = trace_0['θ'].mean(axis=0)
idx = np.argsort(x_c)

# Plot the fitted theta
plt.plot(x_c[idx], theta[idx], color='teal', lw=3)
# Plot the HPD for the fitted theta
az.plot_hpd(x_c, trace_0['θ'], color='teal')
plt.xlabel(x_n)
plt.ylabel('θ', rotation=0)

# Plot the decision boundary
plt.vlines(trace_0['bd'].mean(), 0, 1, color='steelblue')
# Plot the HPD for the decision boundary
bd_hpd = az.hpd(trace_0['bd'])
plt.fill_betweenx([0, 1], bd_hpd[0], bd_hpd[1], color='steelblue', alpha=0.5)
plt.scatter(x_c, np.random.normal(y_0, 0.02),
            marker='.', color=[f'C{x}' for x in y_0])

# use original scale for xticks
locs, _ = plt.xticks()
plt.xticks(locs, np.round(locs + x_0.mean(), 1))

#%% Multiple logisitc regression

# Select the first two classes for a binary classification problem
df = iris_data.query("target == (0,1)")
y_0 = df.target
x_n = ['sepal length (cm)', 'sepal width (cm)']
# Center the data by subtracting the mean from both columns
df_c = df - df.mean() 
x_c = df_c[x_n].values

with pm.Model() as model_1: 
    α = pm.Normal('α', mu=0, sd=10) 
    β = pm.Normal('β', mu=0, sd=2, shape=len(x_n)) 
    μ = α + pm.math.dot(x_c, β) 
    θ = pm.Deterministic('θ', 1 / (1 + pm.math.exp(-μ))) 
    bd = pm.Deterministic('bd', -α/β[1] - β[0]/β[1] * x_c[:,0])
    yl = pm.Bernoulli('yl', p=θ, observed=y_0) 
    trace_0 = pm.sample(2000, tune=1000, cores=1)
    
    
gv=pm.model_to_graphviz(model_1)
gv

idx = np.argsort(x_c[:,0]) 
bd = trace_0['bd'].mean(0)[idx] 
plt.scatter(x_c[:,0], x_c[:,1], c=[f'C{x}' for x in y_0]) 
plt.plot(x_c[:,0][idx], bd, color='steelblue'); 
az.plot_hpd(x_c[:,0], trace_0['bd'], color='steelblue')
plt.xlabel(x_n[0]) 
plt.ylabel(x_n[1])

#%% Multiclass classification

df = datasets.load_iris()
iris_data = pd.DataFrame(df['data'], columns=df['feature_names'])
iris_data['target'] = df['target']
y_s = iris_data.target
x_n = iris_data.columns[:-1]
x_s = iris_data[x_n]
x_s = (x_s - x_s.mean()) / x_s.std()
x_s = x_s.values

import theano as tt
tt.config.gcc.cxxflags = "-Wno-c++11-narrowing"

with pm.Model() as model_mclass:
    alpha = pm.Normal('alpha', mu=0, sd=5, shape=3)
    beta = pm.Normal('beta', mu=0, sd=5, shape=(4,3))
    μ = pm.Deterministic('μ', alpha + pm.math.dot(x_s, beta))
    θ = tt.tensor.nnet.softmax(μ)
    #θ = pm.math.exp(μ)/pm.math.sum(pm.math.exp(μ), axis=0)
    yl = pm.Categorical('yl', p=θ, observed=y_s)
    trace_s = pm.sample(2000, tune=1000, cores=1)

data_pred = trace_s['μ'].mean(0)
y_pred = [np.exp(point)/np.sum(np.exp(point), axis=0) for point in data_pred]
az.plot_trace(trace_s, var_names=['alpha'])
f'{np.sum(y_s == np.argmax(y_pred, axis=1)) / len(y_s):.2f}'








































