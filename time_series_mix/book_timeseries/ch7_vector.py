#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PythonTsa.plot_multi_ACF import multi_ACFfig
from PythonTsa.plot_multi_Q_pvalue import MultiQpvalue_plot
from PythonTsa.datadir import getdtapath

import statsmodels.api as sm
from statsmodels.tsa.api import VAR, VARMAX

dtapath = getdtapath()

#%%

mda = pd.read_csv(dtapath + 'USQgdpunemp.csv', header=0)

mda = mda[['gdp','rate']]
timeindex = pd.date_range('1948-01', periods=len(mda), freq='ME')
mda.index = timeindex

mda['gdp'] = np.log(mda['gdp'])
mda.columns = ['lgdp', 'rate']

display(mda)
mda.plot(title='df data')

#%% Check ACF for the "raw" data

multi_ACFfig(mda, nlags=16)
plt.show()

# %%

dmda = mda.diff(1).dropna()
dmda.colums = ['dlgdp', 'drate']
dmda.plot(title='plot diff data')
plt.show()

# %%
# ACF shows quick fall off, which means stationary
# LB test show significance, which means residual is not white noise. So we model

multi_ACFfig(dmda, nlags=16)
plt.show()
qs, pv = MultiQpvalue_plot(dmda, nolags=16)
plt.show()

# %% ------------------------------- GDR MACRO -----------------------------

geco = pd.read_csv(dtapath + 'EconGermany.dat', header=0, sep='\s+')
display(geco)
timeindex = pd.date_range('1960-03', periods=len(geco), freq='Q')
geco.index = timeindex
geco = geco[['inv.', 'inc.','cons.']]

display(geco)
geco.plot(title='GDR Macro data raw')

# %% Get log data

dlge = np.log(geco).diff(1).dropna()
dlge.plot(title='Log data')
plt.show()

# %% Make test/train save last (1982) year for testing

dlgem = dlge.loc['1960-06-30':'1981-12-31',:]
display(dlgem)

# %% Check data

multi_ACFfig(dlgem, nlags=10)
plt.show()


# %% More check order

dlgemMod = VAR(dlgem)
print(dlgemMod.select_order(maxlags=4))
print(dlgemMod.select_order(maxlags=9))

# %%  Fit data

dlgemRes = dlgemMod.fit(maxlags=2, ic=None, trend='c')
print(dlgemRes.summary())
dlgemRes.is_stable()

# %% Check residauls

resid = dlgemRes.resid
multi_ACFfig(resid, nlags=10)
plt.show()

q,p = MultiQpvalue_plot(resid, p=2, q=0, noestimatedcoef=18, nolags=24)
plt.show()

# %% Check corr

sigma_u = dlgemRes.sigma_u
print(sigma_u)

# %% for forcast move data np.array

dlgem = dlgem.values
fore_interval = dlgemRes.forecast_interval(dlgem, steps=4)
point,lower,upper = dlgemRes.forecast_interval(dlgem, steps=4)
dlgemRes.plot_forecast(steps=4)
plt.show()

# %% Check causality

g1 = dlgemRes.test_causality(caused='cons.', causing='inc.',
                             kind='f', signif=0.05)
print(g1)

g2 = dlgemRes.test_causality(caused='inc.', causing='cons.',
                             kind='f', signif=0.05)
print(g2)

# %%

irf = dlgemRes.irf(periods=10)
irf.plot()
plt.show()

irf.plot_cum_effects()
plt.show()


# %% ------------------------- US MACRO -------------------------------------------

mdata = sm.datasets.macrodata.load_pandas().data
mdata = mdata[['realgdp', 'realcons','realinv']]
timeindex = pd.date_range('1959-01', periods=len(mdata), freq='Q')
mdata.index = timeindex

display(mdata)
mdata.plot(title='US Macro')
plt.show()

# %% Diff data

dldata = np.log(mdata).diff(1).dropna()
dldata.plot(title='Logged data')
plt.show()

# Plot individually
fig = plt.figure()
dldata['realgdp'].plot(ax=fig.add_subplot(311))
plt.title('Differenced log of real gdp')
dldata['realcons'].plot(ax=fig.add_subplot(312))
plt.title('Differenced log of real consumption')
dldata['realinv'].plot(ax=fig.add_subplot(313))
plt.title('Differenced log of real investment')
plt.show()


# %% train/test

myd = dldata.loc['1959-06-30':'2008-12-31',:]
display(myd)

# %% check stionary

multi_ACFfig(myd, nlags=10)
plt.show()

# %% use VAR to estimate orders

mydmod1 = VAR(myd)
print(mydmod1.select_order(maxlags=10))
print(mydmod1.select_order(maxlags=11))
print(mydmod1.select_order(maxlags=24))

# %% Fit VAR(3)

mydmod = VARMAX(myd, order=(3,0), enforce_stationarity=True)
modfit = mydmod.fit(disp=False)
print(modfit.summary())

# %% Check residuals
#from PythonTsa.plot_multi_Q_pvalue import MultiQpvalue_plot

resid = modfit.resid
multi_ACFfig(resid, nlags=12)
plt.show()

qs, pv = MultiQpvalue_plot(resid, p=3, q=0, noestimatedcoef=27, nolags=24)
plt.show()
# %% resid fails LB test

param = mydmod.param_names
mydmodf = VARMAX(myd, order=(3,0), enforce_stationarity=False)
with mydmodf.fix_params({param[0]:0.0, param[5]:0.0, param[6]:0.0,
                         param[8]:0.0, param[9]:0.0, param[11]:0.0,
                         param[12]:0.0, param[14]:0.0, param[15]:0.0,
                         param[17]:0.0, param[24]:0.0, param[26]:0.0,
                         param[27]:0.0, param[28]:0.0, param[29]:0.0}):
    modff = mydmodf.fit(method='bfgs')
print(modff.summary())


# %% Check new residuals

residf = modff.resid
multi_ACFfig(residf, nlags=10)
plt.show()

qs, pv = MultiQpvalue_plot(residf, p=3, q=0, noestimatedcoef=13, nolags=24)
plt.show()

# %% Residuals are ok. Predict

fore = modff.predict(end='2009-09-30')
realgdpFitgpd = pd.DataFrame({'realgdp':dldata['realgdp'],
                              'fittedgdp':fore['realgdp']})
realconsFitcons = pd.DataFrame({'realcons':dldata['realcons'],
                              'fittedcons':fore['realcons']})
realinvFitinv = pd.DataFrame({'realinv':dldata['realinv'],
                              'fittedinv':fore['realinv']})

fig = plt.figure()
realgdpFitgpd.plot(style=['-','--'], ax=fig.add_subplot(311))
realconsFitcons.plot(style=['-','--'], ax=fig.add_subplot(312))
realinvFitinv.plot(style=['-','--'], ax=fig.add_subplot(313))
plt.show()


# %%
