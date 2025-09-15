# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 09:49:44 2021

@author: PatCa
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm

print(f"Running on PyMC3 v{pm.__version__}")

#%% Make city temperature data
RANDOM_SEED = 8927
np.random.seed(RANDOM_SEED)
az.style.use("arviz-darkgrid")

df_data = pd.DataFrame(columns=["date"]).set_index("date")
dates = pd.date_range(start="2020-05-01", end="2020-05-20")

for city, mu in {"Berlin": 15, "San Marino": 18, "Paris": 16}.items():
    df_data[city] = np.random.normal(loc=mu, size=len(dates))

df_data.index = dates
df_data.index.name = "date"
print(df_data.head())

#%% Use coord to make model aware of input and output

# The data has two dimensions: date and city
coords = {"date": df_data.index, "city": df_data.columns}

with pm.Model(coords=coords) as city_model:
    europe_mean = pm.Normal("europe_mean_temp", mu=15.0, sd=3.0)
    city_offset = pm.Normal("city_offset", mu=0.0, sd=3.0, dims="city")
    city_temperature = pm.Deterministic("city_temperature", europe_mean + city_offset, dims="city")

    data = pm.Data("data", df_data, dims=("date", "city"))
    y_temp = pm.Normal("y_temp", mu=city_temperature, sd=0.5, observed=data)

    trace_city = pm.sample(
        2000,
        tune=2000,
        target_accept=0.85,
        return_inferencedata=True,
        random_seed=RANDOM_SEED,
        cores=1
    )

#%%

print(city_model.coords)
print(trace_city.posterior.coords)
#az.plot_trace(idata, var_names=["europe_mean_temp", "city_temperature"]);

gr_city = pm.model_to_graphviz(city_model)
gr_city

#print("basic model: ",city_model.basic_RVs)
#print("free RVs: ",city_model.free_RVs)
#print("observed RVs:", city_model.observed_RVs)

#%% Use shared variable to make model predict unseen data ---------------------

x = np.random.randn(100)
y = x > 0

with pm.Model() as model:
    x_shared = pm.Data("x_shared", x)
    coeff = pm.Normal("x", mu=0, sigma=1)

    logistic = pm.math.sigmoid(coeff * x_shared)
    y_like = pm.Bernoulli("obs", p=logistic, observed=y)

    # fit the model
    trace = pm.sample(return_inferencedata=False, cores=1)


#%% predict

new_values = [-1, 0, 1.0]
with model:
    # Switch out the observations and use `sample_posterior_predictive` to predict
    pm.set_data({"x_shared": new_values})
    post_pred = pm.sample_posterior_predictive(trace, samples=500)


#%% Poly-regression example. Baby height data----------------------------------

data = pd.read_csv(pm.get_data("babies.csv"))
data.plot.scatter("Month", "Length", alpha=0.4)

with pm.Model() as model_babies:
    a = pm.Normal("a", sigma=10)
    b = pm.Normal("b", sigma=10)
    sd = pm.HalfNormal("sd", sigma=10)
    sd2 = pm.HalfNormal("sd2", sigma=10)

    month = pm.Data("month", data.Month.values.astype(float))

    my = pm.Deterministic("my", a + b * month ** 0.5)
    stdev = pm.Deterministic("stdev", sd + sd2 * month)

    length = pm.Normal("length", mu=my, sigma=stdev, observed=data.Length)

    trace_babies = pm.sample(tune=2000, return_inferencedata=False, cores=1)

#%% Plot result

with model_babies:
    pp_length = pm.sample_posterior_predictive(trace_babies)["length"]
    

plt.plot(data.Month, data.Length, "k.", alpha=0.15)

plt.plot(data.Month, trace_babies["μ"].mean(0))
az.plot_hdi(data.Month, pp_length, hdi_prob=0.6, fill_kwargs={"alpha": 0.8})
az.plot_hdi(data.Month, pp_length, fill_kwargs={"alpha": 0.4})

plt.xlabel("Month")
plt.ylabel("Length");

#%% predict

with model_babies:
    pm.set_data({"month": [17.]})
    length_ppc = pm.sample_posterior_predictive(trace_babies, 2000)

# need to get rid of second column because of this bug
# when predicting only one new value:
# https://github.com/pymc-devs/pymc3/issues/3640
length_ppc = length_ppc["length"][:, 0]


ref_length = 80
percentile = np.mean(length_ppc <= ref_length).round(2)

ax = az.plot_kde(length_ppc, quantiles=[percentile], fill_kwargs={"alpha": 0.4})
ax.text(
    0.18,
    0.9,
    f"Ref Length: {ref_length}\nPercentile: {percentile}",
    fontsize=12,
    horizontalalignment="center",
    verticalalignment="center",
    transform=ax.transAxes,
)
ax.set_xlabel("Length");
#%%



























































