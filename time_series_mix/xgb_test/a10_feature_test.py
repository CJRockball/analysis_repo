#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from PythonTsa.datadir import getdtapath
%load_ext autoreload
%autoreload 2

import autoregressive_features as arf
import temporal_features as tmpf

dtapath = getdtapath()
 
#%% ------------------------- Drugs Australia ---------------------------------

df = pd.read_csv(dtapath + 'h02July1991June2008.csv', header=0)
df.columns = ['drug']

timeindex = pd.date_range('1991-06', periods=len(df), freq='M')
df.index = timeindex
display(df)
df.plot(title='Australia Drug Sales Monthly')

# %% Test Lags. Use new df
from autoregressive_features import add_lags

lags = (np.arange(16) + 1).tolist()

full_df, added_features = add_lags(df, lags=lags, column='drug')

# display(full_df)
# print(added_features)

#%% Test rolling feature
from autoregressive_features import add_rolling_features

full_df, added_features = add_rolling_features(full_df, rolls=[6, 12, 18],
            column='drug', agg_funcs=['mean', 'std', 'max', 'min'])

# display(full_df)
# print(added_features)

# #%% Test seasonal rolling  XXX Doesn't work XXX
# from autoregressive_features import add_seasonal_rolling_features

# dfa = df.copy(deep=True)
# full_df, added_features = add_seasonal_rolling_features(
#     dfa, rolls=[3], seasonal_periods=[12,24], column='drug', agg_funcs=['mean', 'std'])

# display(full_df)
# print(added_features)


#%% Test ewma
from autoregressive_features import add_ewma

#dfb = df.copy(deep=True)
full_df, added_feature = add_ewma(full_df, spans=[6, 12, 18], column='drug')

# display(full_df)
# print(added_feature)

# %% Test temporal features use new df
from temporal_features import add_temporal_features

#dfa = df.copy(deep=True)
full_df['time_index'] = timeindex
full_df, added_features = add_temporal_features(full_df, field_name='time_index',
            frequency='M', add_elapsed=True, drop=False)
full_df.drop(columns='time_index', inplace=True)

# display(full_df)
# print(added_features)

# %% Test fourier transform use df with time features
from temporal_features import bulk_add_fourier_features

full_dfa = full_df.copy(deep=True)

full_dfb, added_features = bulk_add_fourier_features(full_dfa,
            ['time_index_Month', 'time_index_Quarter'],
            max_values=[12, 4], n_fourier_terms=4)

display(full_dfb)
print(added_features)

# %% make train test

N_TEST = 12
df_temp = full_df.copy(deep=True)

df_train = df_temp.iloc[:-(N_TEST), :]
trainX = df_train.iloc[:,1:]
trainy = df_train.iloc[:,0]
df_test = df_temp.iloc[-N_TEST:, :]
testX = df_test.iloc[:,1:]
testy = df_test.iloc[:,0]
    
#%%
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from xgboost import plot_importance

eval_set = [(trainX, trainy), (testX, testy)]
eval_metric = ['mae']

model = XGBRegressor(objective='reg:squarederror', eval_metric=eval_metric,
                          n_estimators=5000, max_depth=5, 
                          learning_rate=0.001,
                          alpha = 1, reg_lambda =0,
                          colsample_bytree=0.7)

history = model.fit(trainX, trainy,
                    eval_set=eval_set,
                    verbose=False)


history = model.evals_result()
xx = range(0,len(history['validation_0']['mae']))

plt.figure()
plt.plot(xx,history["validation_0"]['mae'], label="Train")
plt.plot(xx,history['validation_1']['mae'], label='Test')
plt.legend()
plt.show()

fig, ax = plt.subplots(figsize=(12,6))
plot_importance(model, max_num_features=20, ax=ax)
plt.show();

# %%

y_hat = model.predict(testX)
y_hat_df = pd.Series(data=y_hat, index=testy.index)
error = mean_absolute_error(y_hat_df, testy)
print(f'MAE: {error:.4f}')

plt.figure()
plt.plot(testy, label='Data')
plt.plot(y_hat_df, label='Predict')
plt.legend()
plt.show()

#%% ---------------------------- USE ALGO ----------------------------------------------------
from ml_forecast import (
    FeatureConfig,
    MissingValueConfig,
    MLForecast,
    ModelConfig,
    calculate_metrics
)

import time
import humanize

class LogTime:
    def __init__(self, verbose=True, **humanize_kwargs) -> None:
        if "minimum_unit" not in humanize_kwargs.keys():
            humanize_kwargs["minimum_unit"] = 'microseconds'
        self.humanize_kwargs = humanize_kwargs
        self.elapsed = None
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        """
        Exceptions are captured in *args, we’ll handle none, since failing can be timed anyway
        """
        self.elapsed = time.time() - self.start
        self.elapsed_str = humanize.precisedelta(self.elapsed, **self.humanize_kwargs)
        if self.verbose:
            print(f"Time Elapsed: {self.elapsed_str}")


def evaluate_model(model_config, feature_config, missing_config, 
                   train_features, train_target, test_features, test_target):
    ml_model = MLForecast(model_config=model_config, feature_config=feature_config, missing_config=missing_config)
    ml_model.fit(train_features, train_target)

    y_pred = ml_model.predict(test_features)
    #y_pred.index = test_target.index

    feat_df = ml_model.feature_importance()
    metrics = calculate_metrics(test_target, y_pred, model_config.name, train_target)
    return y_pred, metrics, feat_df


#%% Set up FeatutrConfig

def difference_list(list1, list2):
    return list(set(list1)- set(list2))

xgb_full = full_df.copy(deep=True)
xgb_full = xgb_full.reset_index().rename(columns={'index':'time_index'})
# Remove the target name
continuous_list = xgb_full.select_dtypes([np.number]).columns.tolist()[1:]
bool_list = xgb_full.select_dtypes(['bool']).columns.tolist()
# All columns NOT cont or bool are category
cols = xgb_full.columns
categorical_list = difference_list(cols, continuous_list+bool_list)

feat_config = FeatureConfig(
    date                    ="time_index",
    target                  ="drug",
    continuous_features     =continuous_list,
    categorical_features    =[],
    boolean_features        =bool_list,
    index_cols              =["time_index"],
    exogenous_features      =[],)

#%% Get train test

# Get train/test
sample_train_df = xgb_full.iloc[:-12,:]
sample_test_df = xgb_full.iloc[-12:,:]

train_features, train_target, train_original_target = feat_config.get_X_y(
    sample_train_df, categorical=False, exogenous=False
)
test_features, test_target, test_original_target = feat_config.get_X_y(
    sample_test_df, categorical=False, exogenous=False
)

#del sample_train_df, sample_test_df

#%% set up missing_value_config

nc = train_features.isnull().sum()
get_na_cols = nc[nc>0].index.to_list()

missing_value_config = MissingValueConfig(
    bfill_columns=get_na_cols,
    ffill_columns=[],
    zero_fill_columns=[],
)

# %%
metric_record = []
pred_df = pd.concat([train_target, test_target])

#%% Run fitting and predict

model_config = ModelConfig(
    model=XGBRegressor(random_state=42, max_depth=4),
    name="XGB Random Forest",
    # XGB is not affected by normalization
    normalize=False,
    # XGB handles missing values
    fill_missing=False,
)
with LogTime() as timer:
    y_pred, metrics, feat_df = evaluate_model(
        model_config,
        feat_config,
        missing_value_config,
        train_features,
        train_target,
        test_features,
        test_target,
    )
metrics["Time Elapsed"] = timer.elapsed
metric_record.append(metrics)
pred_df = pred_df.join(y_pred)

# %%

display(metric_record)

plt.figure()
plt.plot(test_target, label="Data")
plt.plot(y_pred, label='Predict')
plt.show()

#%%

display(feat_df)


# %%----------------------- RUN HARNESS WITH TRANSFORMED TARGET --------------
#data_features = xgb_full.
data_target = pd.Series(xgb_full['drug'].tolist(), index=xgb_full['time_index'])

display(data_target)
print(data_target.info())

#%%
#---------------------- EXPLORE TARGET ---------------------------------------
from stationary_utils import check_unit_root

res = check_unit_root(data_target, confidence=0.05)

print(f"Stationary: {res.stationary} | p-value: {res.results[1]}")

# %% Try differencing 
from target_transformations import AdditiveDifferencingTransformer

diff_transformer = AdditiveDifferencingTransformer()
# [1:] because differencing reduces the length by 1
y_diff = diff_transformer.fit_transform(data_target, freq='1M')[1:]
display(y_diff)

fig, axes = plt.subplots(2,1)
data_target.plot(title='Data', ax=axes[0])
y_diff.plot(title='1_diff', ax=axes[1])
plt.tight_layout()
plt.show()

check_unit_root(y_diff)


# %% Check trend

from stationary_utils import check_trend, check_deterministic_trend

check_deterministic_trend(data_target)

# %%
kendall_tau_res = check_trend(data_target, confidence=0.05)
mann_kendall_res = check_trend(data_target, confidence=0.05, mann_kendall=True)
mann_kendall_seas_res = check_trend(data_target, confidence=0.05, mann_kendall=True, seasonal_period=25)
print(f"Kendalls Tau: Trend: {kendall_tau_res.trend} | Direction: {kendall_tau_res.direction} | Deterministic: {kendall_tau_res.deterministic}")
print(f"Mann-Kendalls: Trend: {mann_kendall_res.trend} | Direction: {mann_kendall_res.direction} | Deterministic: {mann_kendall_res.deterministic}")
print(f"Mann-Kendalls Seasonal Trend: {mann_kendall_seas_res.trend} | Direction: {mann_kendall_seas_res.direction} | Deterministic: {mann_kendall_seas_res.deterministic}")


# %% detrend
from target_transformations import DetrendingTransformer

detrending_transformer = DetrendingTransformer()
y_diff = detrending_transformer.fit_transform(data_target, freq='1M')

fig, axes = plt.subplots(2)
data_target.plot(title='Data', ax=axes[0])
y_diff.plot(title='1_diff', ax=axes[1])
plt.tight_layout()
plt.show()

kendall_tau_res = check_trend(y_diff, confidence=0.05)
mann_kendall_res = check_trend(y_diff, confidence=0.05, mann_kendall=True)
print(f"Kendalls Tau: Trend: {kendall_tau_res.trend}")
print(f"Mann-Kendalls: Trend: {mann_kendall_res.trend}")

# %% Check seasonality
from stationary_utils import check_seasonality

seasonality_res = check_seasonality(y_diff, max_lag=38, seasonal_period=12, confidence=0.05)
print(f"Seasonality Test for 25th lag: {seasonality_res.seasonal}")
seasonality_id_res = check_seasonality(y_diff, max_lag=60, confidence=0.05)
print(f"Seasonality identified for: {seasonality_res.seasonal_periods}")

# %% De seasonalizing XXX Doesn't work still seasonal XXX
from target_transformations import DeseasonalizingTransformer

deseas_transformer = DeseasonalizingTransformer(seasonality_extraction="period_averages",seasonal_period=12)
y_deseas = deseas_transformer.fit_transform(y_diff, freq='1M')

fig,axes = plt.subplots(2)
y_diff.plot(title='Seasonal y_diff', ax=axes[0])
y_deseas.plot(title='Deseasonal', ax=axes[1])
plt.tight_layout()
plt.show()

seasonality_res = check_seasonality(y_deseas, seasonal_period=12, max_lag=26)
print(f"Seasonality at {seasonality_res.seasonal_periods}: {seasonality_res.seasonal}")

# %% Try autotransform
from target_transformations import AutoStationaryTransformer
N_TEST = 12
train_target = data_target[:-N_TEST]

auto_stationary = AutoStationaryTransformer(seasonal_period=12)
y_stat = auto_stationary.fit_transform(train_target, freq='1ME')
print(f"Transformations applied: {[p.__class__.__name__ for p in auto_stationary._pipeline]}")

fig, axs = plt.subplots(2)
train_target.plot(title="Original",ax=axs[0])
y_stat.plot(title="Auto Stationary",ax=axs[1])
plt.tight_layout()
plt.show()

unit_root = check_unit_root(y_stat, confidence=0.05)
print(f"Unit Root: {unit_root.stationary} with a p-value of {unit_root.results[1]}")
y_inv = auto_stationary.inverse_transform(y_stat)
print(f"Inverse == Original @ precision of 2 decimal points: {np.all(y_inv.round(2)==train_target.round(2))}")

# %% Save transformed targets and pipline
import joblib
os.chdir('/home/patrick/Python/timeseries/weather/xgb_book')

train_target = y_stat.to_frame()
train_target.columns = ['drug_auto_stat']
#train_target.rename(columns={"drug":"drug_auto_stat"}, inplace=True)
# Save train target (i.e. minus last 12 columns) 
train_target.to_parquet("data/xgb_data/train_target_auto_stat.parquet")

#print(auto_stationary)
joblib.dump(auto_stationary, "data/xgb_data/auto_transformer_pipeline.pkl")

# %% Load data and try fitting
# Load train, join mod target

df_temp = full_df.copy(deep=True)
# Full train inc, orig target
df_train = df_temp.iloc[:-(N_TEST), :]
# Load transformed targ
auto_stat_target = pd.read_parquet('data/xgb_data/train_target_auto_stat.parquet')
# Full test inc orig target
df_test = df_temp.iloc[-N_TEST:, :]
df_test = df_test.reset_index().rename(columns={'index':'time_index'})
# Add transformed target to train.
# Train df now contains date, original_target, transformed target
df_train = df_train.join(auto_stat_target).reset_index().rename(columns={'index':'time_index'})
transformer_pipelines = joblib.load('data/xgb_data/auto_transformer_pipeline.pkl')

print(df_train.columns)

#%%

feat_config1 = FeatureConfig(
    date                    ="time_index",
    target                  ="drug_auto_stat",
    original_target         ="drug",
    continuous_features     =continuous_list,
    categorical_features    =[],
    boolean_features        =bool_list,
    index_cols              =["time_index"],
    exogenous_features      =[],)

# Use missing_value set up from original run
missing_value_config1 = MissingValueConfig(
    bfill_columns=get_na_cols,
    ffill_columns=[],
    zero_fill_columns=[],
)

# %% Set up eval model with target_transformer

def evaluate_model(model_config, feat_config, missing_value_config, target_transformer, train_features, train_target, test_features, test_target, train_target_original=None):
    ml_model = MLForecast(model_config=model_config, feature_config=feat_config, missing_config=missing_value_config, target_transformer=target_transformer)
    ml_model.fit(train_features, train_target, is_transformed=True)

    y_pred = ml_model.predict(test_features)
    #y_pred.index = test_target.index
  
    feat_df = ml_model.feature_importance()
    metrics = calculate_metrics(test_target, y_pred, model_config.name, train_target_original)
    return y_pred, metrics, feat_df

#%% Batch run models
from sklearn.linear_model import LassoCV
from xgboost import XGBRFRegressor
from lightgbm import LGBMRegressor
import warnings

models_to_run = [
    ModelConfig(model = LassoCV(), name="Lasso Regression", normalize=True, fill_missing=True),
    ModelConfig(model = XGBRFRegressor(random_state=42, max_depth=4), name="XGB Random Forest", normalize=False, fill_missing=False),
    ModelConfig(model = LGBMRegressor(random_state=42), name="LightGBM", normalize=False, fill_missing=False)
]

# %%

all_preds = []
all_metrics = []
#We can parallelize this loop to run this faster
for model_config in models_to_run:
    model_config = model_config.clone()
    X_train, y_train, y_train_orig = feat_config1.get_X_y(df_train, categorical=False, exogenous=False)
    X_test, y_test_trans, y_test_orig = feat_config1.get_X_y(df_test, categorical=False, exogenous=False)
    transformer = transformer_pipelines
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred, metrics, feat_df = evaluate_model(model_config, feat_config1, missing_value_config1, transformer, X_train, y_train, X_test, y_test_orig, y_train_orig)
    
    y_pred.name = "predictions"
    y_pred = y_pred.to_frame()
    y_pred['Algorithm'] = model_config.name + "_auto_stat"
    metrics["Algorithm"] = model_config.name + "_auto_stat"
    y_pred['energy_consumption'] = y_test_orig.values
    all_preds.append(y_pred)
    all_metrics.append(metrics)

#%% Consolidate metrics

metrics_df = pd.DataFrame(all_metrics)
display(metrics_df)

# %% Consolidate predictions

pred_df = pd.concat(all_preds)
display(pred_df)

#%% Plot predictions
fname = pred_df.Algorithm.unique()

plt.figure()
plt.plot(pred_df.iloc[0:11, 2], label='Data')
for name in fname:
    data_series = pred_df.loc[pred_df.Algorithm == name,:]
    plt.plot(data_series['predictions'], label=name)
plt.legend()
plt.xticks(rotation=45, ha='right')
plt.show()




# %% Save data
from pathlib import Path

os.makedirs("data/xgb_data/output", exist_ok=True)
output = Path("data/xgb_data/output")

pred_df.to_pickle(output / "ml_single_step_prediction_auto_stationary_test_df.pkl")
metrics_df.to_pickle(output / "ml_single_step_metrics_auto_stationary_test_df.pkl")

# %% ----------------------------  MAKE ENSEMBLE  ---------------------------------------------
from IPython.display import display, HTML
# Read in predictions and metrics
try:
    pred_val_df = pd.read_pickle(output / "ml_single_step_prediction_auto_stationary_test_df.pkl")
    metrics_val_df = pd.read_pickle(output / "ml_single_step_metrics_auto_stationary_test_df.pkl")
except FileNotFoundError:
    display(HTML("""
    <div class="alert alert-block alert-warning">
    <b>Warning!</b> File not found. Please make sure you have run all notebooks in Chapter08 and 02-Baseline Forecasts using darts.ipynb in Chapter04
    </div>
    """))
    
#%%
# Get all predictions on the same row
pred_wide_val = pd.pivot(pred_val_df.reset_index(),
                         index = ['time_index'],
                         columns= 'Algorithm',
                         values = 'predictions')
# Add y_val data to the rows
pred_wide_val = pred_wide_val.join(pred_val_df
                    .loc[pred_val_df.Algorithm == 'LightGBM_auto_stat','energy_consumption'])

#display(pred_wide_val)

# %%

def evaluate_ensemble(pred_wide, model, target):
    metric_l = []
    # Get y_pred and y_test
    test_target = pred_wide.loc[:, target]
    y_pred = pred_wide.loc[:, model]
    # Calculate metrics and make dataframe
    metric_l.append(
        calculate_metrics(test_target, y_pred, name=model))
    eval_metrics_df = pd.DataFrame(metric_l)
    return eval_metrics_df


def highlight_abs_min(s, props=""):
    return np.where(s == np.nanmin(np.abs(s.values)), props, "")

#%% Calculate average and meadian ensembles

ensemble_forecasts = ['Lasso Regression_auto_stat','LightGBM_auto_stat','XGB Random Forest_auto_stat']

pred_wide_val['average_ensemble'] = pred_wide_val[ensemble_forecasts].mean(axis=1)
pred_wide_val['median_ensemble'] = pred_wide_val[ensemble_forecasts].median(axis=1)

#%%

agg_metric = evaluate_ensemble(pred_wide_val, 'average_ensemble', 'energy_consumption')
metrics_val_df = pd.concat([metrics_df,agg_metric], axis=0).reset_index(drop=True)

agg_metric = evaluate_ensemble(pred_wide_val, 'median_ensemble', 'energy_consumption')
metrics_val_df = pd.concat([metrics_df,agg_metric], axis=0).reset_index(drop=True)


# %%

display(metrics_val_df)
# %% Try stacking models with linear fit
from sklearn.linear_model import (
    HuberRegressor,
    LassoCV,
    LinearRegression,
    RidgeCV)

#Linear Regression, Fit and print coef
stacking_model = LinearRegression(positive=True, fit_intercept=False)
stacking_model.fit(pred_wide_val[ensemble_forecasts], pred_wide_val["energy_consumption"])

stack_df = pd.DataFrame({"Forecast": ensemble_forecasts, "Weights": stacking_model.coef_}) \
                .round(4).sort_values("Weights", ascending=False)

display(stack_df)

# %% Predict and add to df

pred_wide_val["linear_reg_blending"] = stacking_model.predict(
    pred_wide_val[ensemble_forecasts]
)
