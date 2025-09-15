# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 15:58:47 2021

@author: PatCa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.sandbox.predict_functional import predict_functional


df = sm.datasets.get_rdataset('dietox', 'geepack').data

not_used = ['Evit','Cu','Litter','Start','Feed']
df2 = df.drop(not_used, axis=1)

model = smf.mixedlm("Weight ~ Time", df2, groups="Pig")
pig_model = model.fit(method=["lbfgs"])
print(pig_model.summary())

#%%

plt.figure()
plt.scatter(df2.Time, df2.Weight, alpha=0.4)

pigs = df2.Pig.unique()


for pig in pigs:
    values = {'Pig': pig}
    print(pig)
    pr, cb, fv = predict_functional(pig_model,'Time', values=values, ci_method='simultaneous')
    plt.plot(fv,pr)



































