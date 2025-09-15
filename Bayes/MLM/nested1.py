# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 12:36:43 2021

@author: PatCa
"""

import pandas as pd                                                                                                        
import statsmodels.api as sm   

df = pd.read_csv('supplier.csv')

#%%

vc = {"Batch": "0 + C(Batch)"}

model1 = sm.MixedLM.from_formula("Data ~ 1",re_formula="1",
                                 vc_formula=vc, groups="Supplier", data=df)
result1 = model1.fit()
print(result1.summary())





















