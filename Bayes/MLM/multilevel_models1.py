# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 14:12:30 2021

@author: PatCa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.formula.api import ols
from statsmodels.graphics.api import interaction_plot, abline_plot
from statsmodels.stats.anova import anova_lm

url = 'http://stats191.stanford.edu/data/salary.csv'
salary_table = pd.read_csv('salary.csv')


df = salary_table.rename(columns = {'S':"Salary", 'X':"Experience", 'E':'Education', 'P':'Management'})
df.Education.replace({"B":0, "M":1, "P":2}, inplace=True)
df.Management.replace({"M":1,"L":0}, inplace=True)

#S=Salary
#E=Education (1=Bach, 2=Mast, 3=PhD)
#X=Experience (years)
#M=Management (1=Managment, 0=Labour)

#%% Plot data grouped by Education and Management

plt.figure(figsize=(6,6))
symbols = ['D', '^']
colors = ['r', 'g', 'blue']
factor_groups = df.groupby(['Education','Management'])
for values, group in factor_groups:
    i,j = values
    plt.scatter(group['Experience'], group['Salary'],marker=symbols[j], 
                color=colors[i-1],s=144)
plt.xlabel('Experience')
plt.ylabel('Salary')

#%% Fit OLS

model = ols('Salary ~ C(Education)+C(Management)+Experience', data=df)
result = model.fit()
print(result.summary())
print(result.model.data.orig_exog[:5])

#%%
#Plot res by experience
y_pred2 = result.fittedvalues
res = result.resid
plt.figure()
plt.scatter(df.Experience, res/res.std()) #y_pred2, res)

#%%
#Plt res by management*Education categories
def mult_fcn(row):   
    if row.Education == 0 and row.Management == 0:
        return 0
    elif row.Education == 0 and row.Management == 1:
        return 1
    elif row.Education == 1 and row.Management == 0:
        return 2
    elif row.Education == 1 and row.Management == 1:
        return 3
    elif row.Education == 2 and row.Management == 0:
        return 4
    else:
        return 5
        
df['Ed_Man'] = df.apply(mult_fcn,axis=1)

colors = ['blue', 'red', 'green', 'orange', 'magenta', 'lightgreen']
factor_g = df.groupby(['Education','Management'])
plt.figure()
for factor, group in factor_g:
    i,j = factor
    group_num = 2*i + j
    plt.scatter(group.Ed_Man, res[group.index], color=colors[group_num])

plt.xlabel('Group')
plt.ylabel("Residuals")
plt.show()

plt.figure()
for factor, group in factor_g:
    i,j = factor
    group_num = 2*i + j
    plt.scatter(group.Experience, res[group.index], color=colors[group_num])

plt.xlabel('Group')
plt.ylabel("Residuals")
plt.show()


#%% Plot fits
from statsmodels.sandbox.predict_functional import predict_functional

plt.figure(figsize=(6,6))
factor_groups = df.groupby(['Education','Management'])
for values, group in factor_groups:
    i,j = values
    plt.scatter(group['Experience'], group['Salary'], s=144)
plt.xlabel('Experience')
plt.ylabel('Salary')

for i in range(3):
    for j in range(2):
        values = {'Education': i, 'Management':j}
        pr, cb, fv = predict_functional(result, 'Experience', values=values, ci_method='simultaneous')
        
        plt.plot(fv,pr)
        #plt.fill_between(fv, cb[:,0],cb[:,1], color='grey', alpha=0.4)
        
plt.xlabel('Experience')
plt.ylabel('Salary')

#%% residuals within each group

resid = result.resid
plt.figure(figsize=(6,6));
for values, group in factor_groups:
    i,j = values
    group_num = i*2 + j - 1  # for plotting purposes
    x = [group_num] * len(group)
    plt.scatter(x, resid[group.index], marker=symbols[j], color=colors[i-1],
            s=144, edgecolors='black')
plt.xlabel('Group')
plt.ylabel('Residuals')

#%% Make interaction ols regression

model = ols('Salary ~ C(Education)+C(Management)+Experience+C(Education)*C(Management) \
                        ', data=df)
interact_lm = model.fit()
print(interact_lm.summary())

#%%

plt.figure(figsize=(6,6))
factor_groups = df.groupby(['Education','Management'])
for values, group in factor_groups:
    i,j = values
    plt.scatter(group['Experience'], group['Salary'], s=144)
plt.xlabel('Experience')
plt.ylabel('Salary')

for i in range(3):
    for j in range(2):
        values = {'Education': i, 'Management':j}
        pr, cb, fv = predict_functional(interact_lm, 'Experience', values=values, ci_method='simultaneous')
        
        plt.plot(fv,pr)
        #plt.fill_between(fv, cb[:,0],cb[:,1], color='grey', alpha=0.4)
        
plt.xlabel('Experience')
plt.ylabel('Salary')



#%% Same with ANOVE 
from statsmodels.stats.api import anova_lm

table1 = anova_lm(result, interact_lm)
print(table1)
































