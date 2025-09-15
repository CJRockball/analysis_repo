# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 14:59:18 2021

@author: PatCa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm


df = pd.read_csv("C:/Users/PatCa/Documents/PythonScripts/tensorflow_progs/Loom.csv")

#%%

df2 = df.stack().reset_index()
df2 = df2.drop('level_0',1)
df2.columns = ['Loom', 'Data']
df2 = df2.replace({"Loom1":1, "Loom2":2, "Loom3":3, "Loom4":4})

#df2.boxplot(column='Data', by='Loom')

df2_list = []
for i in range(4):
    df2_list.append(df2.Data[df2.Loom == i+1].to_list())

plt.figure(dpi=120)
plt.boxplot(df2_list)
plt.grid(True)
plt.show()

#%%

loom_mean = df2.Data.mean()
loom_std = df2.Data.std()


#%%

model = sm.MixedLM.from_formula('Data ~ 1', groups='Loom', data=df2)
mixed_lm = model.fit()
print(mixed_lm.summary())





























