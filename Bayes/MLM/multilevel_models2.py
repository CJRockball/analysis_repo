# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 15:44:43 2021

@author: PatCa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.graphics.api import interaction_plot, abline_plot
from statsmodels.stats.anova import anova_lm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.sandbox.predict_functional import predict_functional

# Import jobtest data
# TEST = aptitude test score
# MINORITY = 1 if minority, 0 itherwise
# PERF = Job performance evaluation

jobtest_table = pd.read_table('jobtest.table', sep='\t')
test_max = jobtest_table['TEST'].max()
test_min = jobtest_table['TEST'].min()
test_ones = np.ones(2).reshape(2,1)
test_pred = np.linspace(test_min, test_max, 2).reshape(2,1)
test = np.concatenate((test_ones, test_pred), axis=1)
df_pred = pd.DataFrame(data=test, columns=['Intercept', 'TEST'])

factor_group = jobtest_table.groupby(['MINORITY'])

# fig, ax = plt.subplots(figsize=(6,6))
colors = ['purple', 'green']
markers = ['o', 'v']
# for factor, group in factor_group:
#     ax.scatter(group['TEST'], group['JPERF'], color=colors[factor],
#                 marker=markers[factor], s=12**2)
# ax.set_xlabel('TEST')
# ax.set_ylabel('JPERF')

# df1 = jobtest_table[jobtest_table['MINORITY'] == 0]
# df2 = jobtest_table[jobtest_table['MINORITY'] == 1]
# plt.figure(figsize=(6,6),dpi=300)
# plt.scatter(df1.TEST, df1.JPERF, color='purple', marker='o', s=144)
# plt.scatter(df2.TEST, df2.JPERF, color='green', marker='v', s=144)
# plt.xlabel("TEST")
# plt.ylabel("JPERF")
# plt.show()

plt.figure(figsize=(6,6),dpi=300)
for factor, df in factor_group:
    plt.scatter(df['TEST'], df['JPERF'], color=colors[factor], 
                marker=markers[factor], s=144)
plt.xlabel("TEST")
plt.ylabel("JPERF")
plt.show()

#%% OLS regression

model = sm.OLS.from_formula('JPERF ~ TEST', data=jobtest_table)
min_lm = model.fit()
print(min_lm.summary())

# Statsmodels plot tools
# fig, ax = plt.subplots(figsize=(6,6));
# for factor, group in factor_group:
#     ax.scatter(group['TEST'], group['JPERF'], color=colors[factor],
#                 marker=markers[factor], s=12**2)
# ax.set_xlabel('TEST')
# ax.set_ylabel('JPERF')
# fig = abline_plot(model_results = min_lm, ax=ax)

#built in, within sample, prediction functions
prstd, iv_l, iv_u = wls_prediction_std(min_lm)
pr, cb, fv = predict_functional(min_lm, 'TEST')
#Out of sample prediction functions
min_pred = min_lm.predict(df_pred)
test = min_lm.get_prediction(df_pred)
print(test.summary_frame(alpha=0.05))

plt.figure(figsize=(6,6),dpi=300)
for factor, df in factor_group:
    plt.scatter(df['TEST'], df['JPERF'], color=colors[factor], 
                marker=markers[factor], s=144)
plt.xlabel("TEST")
plt.ylabel("JPERF")
plt.plot(test_pred, min_pred.to_numpy())
plt.fill_between(fv, cb[:,0],cb[:,1], color='grey', alpha=0.4)
plt.show()

#%% Variable intercept

model = ols('JPERF ~ TEST + MINORITY', data=jobtest_table)
intercept_lm = model.fit()
print(intercept_lm)

fig, ax = plt.subplots(figsize=(6,6));
for factor, group in factor_group:
    ax.scatter(group['TEST'], group['JPERF'], color=colors[factor],
                marker=markers[factor], s=12**2)

fig = abline_plot(intercept = intercept_lm.params['Intercept'],
                 slope = intercept_lm.params['TEST'], ax=ax, color='purple');
fig = abline_plot(intercept = intercept_lm.params['Intercept'] + intercept_lm.params['MINORITY'],
        slope = intercept_lm.params['TEST'], ax=ax, color='green');


#%% Variable slope

model = ols('JPERF ~ TEST + TEST:MINORITY', data=jobtest_table)
slope_lm = model.fit()
print(slope_lm.summary())

fig, ax = plt.subplots(figsize=(6,6));
for factor, group in factor_group:
    ax.scatter(group['TEST'], group['JPERF'], color=colors[factor],
                marker=markers[factor], s=12**2)

fig = abline_plot(intercept = slope_lm.params['Intercept'],
                 slope = slope_lm.params['TEST'], ax=ax, color='purple');
fig = abline_plot(intercept = slope_lm.params['Intercept'],
        slope = slope_lm.params['TEST'] + slope_lm.params['TEST:MINORITY'],
        ax=ax, color='green');

#%%Var intercept, var slope

model = ols("JPERF ~ TEST + MINORITY + TEST:MINORITY", data=jobtest_table)
var_lm = model.fit()
print(var_lm.summary())

fig, ax = plt.subplots(figsize=(8,6));
for factor, group in factor_group:
    ax.scatter(group['TEST'], group['JPERF'], color=colors[factor],
                marker=markers[factor], s=12**2)

fig = abline_plot(intercept = var_lm.params['Intercept'],
                 slope = var_lm.params['TEST'], ax=ax, color='purple');
fig = abline_plot(intercept = var_lm.params['Intercept'] + var_lm.params['MINORITY'],
        slope = var_lm.params['TEST'] + var_lm.params['TEST:MINORITY'],
        ax=ax, color='green');


#%% Use ANOVA to select models
# min_lm, intercept_lm, slope_lm, var_lm

# Is there any effect of MINORITY in slope and intercept
anova_var = anova_lm(min_lm, var_lm)
print("Base vs free slope and intercept \n", anova_var)

# Is there an effect of variable intercept and fixed slopes
anova_inter = anova_lm(min_lm, intercept_lm)
print("Base vs free intercept \n",anova_inter)

# Is there an effect of variable slope and fixed intercept 
anova_slope = anova_lm(min_lm, slope_lm)
print("Base vs free slope \n",anova_slope)

# Comparing the variable models to each other
# Free intercept and free slope
#anova_1free = anova_lm(intercept_lm, slope_lm)
#print("Free intercept vs free slope \n", anova_1free)

#Free intercept vs both free
anova_inter_both = anova_lm(intercept_lm, var_lm)
print('Free intercept vs both free\n', anova_inter_both)

#Free slope vs both free
anova_slope_both = anova_lm(slope_lm, var_lm)
print('Free slope vs both free \n', anova_slope_both)

#%% MixedLM is a random effects method

model = sm.MixedLM.from_formula('JPERF ~ TEST', groups='MINORITY', data=jobtest_table)
mixed_lm = model.fit()
print(mixed_lm.summary())

#built in, within sample, prediction functions
#prstd, iv_l, iv_u = wls_prediction_std(min_lm)
#pr, cb, fv = predict_functional(min_lm, 'TEST')
#Out of sample prediction functions
min_pred = mixed_lm.predict(df_pred)
#test = mixed_lm.get_prediction(df_pred)
#print(test.summary_frame(alpha=0.05))

plt.figure(figsize=(6,6),dpi=300)
for factor, df in factor_group:
    plt.scatter(df['TEST'], df['JPERF'], color=colors[factor], 
                marker=markers[factor], s=144)
plt.xlabel("TEST")
plt.ylabel("JPERF")
plt.plot(test_pred, min_pred.to_numpy())
plt.title("MixedLM model")
#plt.fill_between(fv, cb[:,0],cb[:,1], color='grey', alpha=0.4)
plt.show()

#%% GEE test

model = sm.GEE.from_formula('JPERF ~ TEST', groups='MINORITY', data=jobtest_table)
gee_lm = model.fit()
print(gee_lm.summary())

min_pred = gee_lm.predict(df_pred)

plt.figure(figsize=(6,6),dpi=300)
for factor, df in factor_group:
    plt.scatter(df['TEST'], df['JPERF'], color=colors[factor], 
                marker=markers[factor], s=144)
plt.xlabel("TEST")
plt.ylabel("JPERF")
plt.plot(test_pred, min_pred.to_numpy())
plt.title("GEE Model")
plt.show()























































