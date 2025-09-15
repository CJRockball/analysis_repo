# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 18:50:25 2021

@author: PatCa
"""


import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt  
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.sandbox.predict_functional import predict_functional
from patsy import dmatrices


df = pd.read_csv('ARM_Data/radon/cty.dat')

df2 = pd.read_csv('ARM_Data/radon/srrs2.dat')
df3 = df2[df2[' state'] == 'MN']

df4 = df3[[' state', ' floor', ' activity', ' county']]

df4 = df4[df4[' activity'] != 0]
df4['log_radon'] = np.log(df3[' activity'])
df4.columns = ['state', 'floor', 'activity', 'county', 'log_radon']
df4['group'] = 1

def rem_space(row):
    text = row.county
    ret_text = text.strip()
    return ret_text

df4['county1'] = df4.apply(rem_space, axis=1)
#%%

model = smf.ols('log_radon ~ C(floor)', data=df4)
lm_mean = model.fit()
print(lm_mean.summary())
print(lm_mean.scale)

#%%

#model = smf.mixedlm('log_radon ~ C(floor)', data=df4, groups='county')
model = smf.ols('log_radon ~ C(county1) + C(floor)-1', data=df4)
lm_np = model.fit()
print(lm_np.summary())
print(lm_np.scale)

#%%

vc = {"county1": "0 + C(county1)", }
formula = 'log_radon ~ C(county1) + C(floor) - 1'
model = sm.MixedLM.from_formula('log_radon ~ C(county1) + C(floor) - 1', groups='county1', 
                                vc_formula = vc ,data=df4)
#model = sm.MixedLM.from_formula('log_radon ~ C(county1) + C(floor) - 1', groups='county1', data=df4)
lm_h = model.fit()

print(lm_h.summary())

fe_params = pd.DataFrame(lm_h.fe_params, columns=['LMM'])
random_effects = pd.DataFrame(lm_h.random_effects)
random_effects = random_effects.transpose()
random_effects = random_effects.rename(index=str, columns={'Group': 'LMM'})


# Generate Design Matrix for later use
Y, X   = dmatrices(formula, data=df4, return_type='matrix')
Terms  = X.design_info.column_names
_, Z   = dmatrices("log_radon ~ county", data=df4, return_type='matrix')
X      = np.asarray(X) # fixed effect
Z      = np.asarray(Z) # mixed effect
Y      = np.asarray(Y).flatten()
nfixed = np.shape(X)
nrandm = np.shape(Z)


# print("fixed effects parameters: ",lm_h.fe_params)
# print("random effects cov mat: ", lm_h.cov_re)
# print('Standard error of fitted random effects: ', lm_h.bse_re)

#print(lm_h.random_effects)
#a = lm_h.random_effects.get('AITKIN')
#print(a)

county_name = 'LAC QUI PARLE'

df5 = df4[df4['county1'] == county_name]
df6 = df5[['county1', 'floor']]
plt.figure()
plt.scatter(df5.floor+ 0.1*(np.random.rand(df5.shape[0])-0.5), df5.log_radon)

values = {'county1': county_name}
pr, cb, fv = predict_functional(lm_np, 'floor', values=values, ci_method='simultaneous')
plt.plot(fv,pr, color='limegreen', label='no-pooling')

pr, cb, fv = predict_functional(lm_mean, 'floor', ci_method='simultaneous')
plt.plot(fv,pr, '--',color='purple', label='full pooling')


pred = lm_h.predict(df6)
#pr, cb, fv = predict_functional(lm_h, 'floor', values=values, ci_method='simultaneous')
plt.plot(df6.floor,pred,color='orange', label='rand intercpt')

plt.ylim([-1,3])
plt.legend()
plt.title(county_name)
plt.show()





















