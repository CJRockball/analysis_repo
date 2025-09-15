# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 11:44:17 2021

@author: PatCa
"""

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import pyreadstat    
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.sandbox.predict_functional import predict_functional


df, meta = pyreadstat.read_dta('ARM_Data/child.iq/kidiq.dta')
df['group'] = 1

model = smf.ols('kid_score ~ C(mom_hs)+mom_iq+mom_iq:C(mom_hs)', data=df)
result = model.fit()
print(result.summary())
print(result.scale)
#%%

model = sm.MixedLM.from_formula('kid_score ~ mom_iq', data=df, groups='group', re_formula='0+C(mom_hs)')
lm = model.fit()
print(lm.summary())


#%%

color = ['green', 'purple']

plt.figure()
for i in range(2):
    df2 = df[df.mom_hs == i]
    plt.scatter(df2.mom_iq, df2.kid_score, color=color[i], alpha=0.4)
    values = {'mom_hs': i}
    pr, cb, fv = predict_functional(result, 'mom_iq', values=values, ci_method='simultaneous')
    plt.plot(fv,pr, color=color[i])
    plt.fill_between(fv, cb[:,0],cb[:,1], color=color[i], alpha=0.4)


plt.show()

#%%


color = ['green', 'purple']

plt.figure()
for i in range(2):
    df2 = df[df.mom_hs == i]
    plt.scatter(df2.mom_iq, df2.kid_score, color=color[i], alpha=0.4)
    values = {'mom_hs': i}
    pr, cb, fv = predict_functional(result, 'mom_iq', values=values, ci_method='simultaneous')
    plt.plot(fv,pr, color=color[i])

iq_max = df.mom_iq.max()

test_ones = np.ones(2).reshape(2,1)
test_pred = np.linspace(0, iq_max+5, 2).reshape(2,1)
test = np.concatenate((test_ones, test_pred), axis=1)

df_pred = pd.DataFrame(data=test, columns=['Intercept', 'mom_iq'])

for i in range(2):
    df_pred['mom_hs'] = i
    min_pred = result.predict(df_pred)
    plt.plot(test_pred, min_pred.to_numpy(), color=color[i], label="hs="+str(i))

plt.legend()
plt.show()

#%%

#Plot res by experience
y_pred2 = result.fittedvalues
res = result.resid
plt.figure()
plt.scatter(df.mom_iq, res/res.std()) #y_pred2, res)
plt.hlines(0, df.mom_iq.min(), df.mom_iq.max(), color='black')


#%%






















