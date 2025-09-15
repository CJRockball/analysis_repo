# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 09:01:57 2018

@author: CJROCKBALL
"""

import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import pymc3 as pm
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

plt.style.use(['seaborn-colorblind', 'seaborn-darkgrid'])

#%% Make dummy data

x_dum = np.ones((12, 2))
x_dum[:,0] = x_dum[:,0]+3
x_dum[:,1] = x_dum[:,1]+5
n = pm.Normal.dist(0,0.5).random(size=12)
x_dum[:,0] = x_dum[:,0]-n
n = pm.Normal.dist(0,0.5).random(size=12)
x_dum[:,1] = x_dum[:,1]+n
y_dum = np.linspace(1,12,12)
x_int = x_dum[:,0]

plt.plot(y_dum, x_int, 'C0o',marker='+')
plt.plot(y_dum, x_dum[:,1], 'C0o',marker='+')
plt.xlabel('X value', fontsize=14)
plt.ylabel('Y value', fontsize=14)
