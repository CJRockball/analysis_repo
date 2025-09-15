# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 15:47:08 2021

@author: PatCa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf


# Sire data
df = pd.read_csv('grnr.csv')
df['group'] = 1

#%% Plot

#plt.boxplot(weight)
df.boxplot(column='data', by='operator')

#%% Model

vcf = {'operator':'0+C(operator)', 'part':'0+C(part)', 'operator:part':'0+C(operator):C(part)'}
model = sm.MixedLM.from_formula('data ~ 1',groups='group', vc_formula=vcf, 
                                re_formula = '0', data=df)
grnr = model.fit()
print(grnr.summary())

#%%









































