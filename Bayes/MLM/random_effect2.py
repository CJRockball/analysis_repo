# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 08:46:29 2021

@author: PatCa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf


# Sire data
weight = [[61,100,56,113,99,103,75,62], [75,102,95,103,98,115,98,94],
          [58,60,60,57,57,59,54,100], [57,56,67,59,58,121,101,101],[59,46,120,115,115,93,105,75]]
ar_w = np.array(weight)
ar_w2 = ar_w.reshape(40,)
calf_mean = np.mean(ar_w2)

df_w = pd.DataFrame(data=ar_w2,columns=['weight'])

# Make sire id
count_l = []
for i in range(5):
    for j in range(8):
        count_l.append(i+1)

df_w['sire'] = count_l

#%% Plot

plt.boxplot(weight)
df_w.boxplot(column='weight', by='sire')

#%%

model = sm.MixedLM.from_formula('weight ~ 1', groups='sire', data=df_w)
mixed_lm = model.fit()
print(mixed_lm.summary())

#%%

model = sm.MixedLM.from_formula('weight ~ sire', groups=df_w['sire'], data=df_w)
mixed_lm = model.fit()
print(mixed_lm.summary())

#%% sm.MixedLM 
## #Model fits fixed effects for columns in endog and random intercepts for group

lm_group = df_w['sire'].to_numpy()
lm_endog = df_w['weight'].to_numpy()
lm_exog = np.ones((40,))

model = sm.MixedLM(lm_endog, lm_exog, lm_group)
lm = model.fit()
print(lm.summary())


























