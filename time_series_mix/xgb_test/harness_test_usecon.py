#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PythonTsa.datadir import getdtapath
from PythonTsa.plot_acf_pacf import acf_pacf_fig
from PythonTsa.plot_multi_ACF import multi_ACFfig
from PythonTsa.LjungBoxtest import plot_LB_pvalue
from statsmodels.graphics.tsaplots import plot_predict

from PythonTsa.Selecting_arma2 import choose_arma2
from scipy import stats
from PythonTsa.ModResidDiag import plot_ResidDiag

import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from arch import arch_model

from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from xgboost import plot_importance

from tqdm.autonotebook import tqdm
from IPython.display import display, HTML

dtapath = getdtapath()


# %% Set up df
# Get df with data column and time column, index is running numbers

df = pd.read_csv(dtapath + 'USEconomicChange.csv')
timeindex = pd.date_range('1970-03', periods=len(df), freq='QE')
df.index = timeindex
df.rename(columns={'Consumption':"cons", 'Income':"inc", 'Production':'prod', 'Savings':"sav","Unemployment":"unem"},
            inplace=True)
df.drop(columns=['Time'], inplace=True)

#display(df)
#df['cons'].plot()

list_to_sum = ['cons', 'inc', 'prod','sav']
sum_list = ['cons_sum', 'inc_sum', 'prod_sum', 'sav_sum', 'unem']
# Use cumprod to sum data
for cname in list_to_sum:
    df[f'{cname}_p1'] = (df[cname]/100) + 1
    df[f'{cname}_sum'] = df[f"{cname}_p1"].cumprod()

display(df)
for cname in list_to_sum:
    df[f'{cname}_sum'].plot(label=cname)
plt.legend()

#df_cons = df['cons_sum'].copy(deep=True).to_frame()
df_cons = df.copy(deep=True)
df_cons = df_cons[list_to_sum + ['unem']]
df_cons.rename(columns={'cons':'cons_sum'}, inplace=True)

df_cons['time_index'] = df_cons.index
df_cons = df_cons.reset_index(drop=True)
display(df_cons)

# %% Make lag and temporal features
import helpers.autoregressive_features as arf
import helpers.temporal_features as tmpf
from helpers.autoregressive_features import add_lags

full_df = df_cons.copy(deep=True)

# Add lag features
lags = (np.arange(6) + 1).tolist()
full_df, added_features = add_lags(full_df, lags=lags, column='cons_sum')

# Add rolling features
from helpers.autoregressive_features import add_rolling_features
full_df, added_features = add_rolling_features(full_df, rolls=[4, 8],
            column='cons_sum', agg_funcs=['mean', 'std', 'max', 'min'])

# Add ewma features
from helpers.autoregressive_features import add_ewma
full_df, added_feature = add_ewma(full_df, spans=[4, 8], column='cons_sum')

# Add temporal features
from helpers.temporal_features import add_temporal_features
full_df, added_features = add_temporal_features(full_df, field_name='time_index',
            frequency='QE', add_elapsed=True, drop=False)

# Add fourier transforms
from helpers.temporal_features import bulk_add_fourier_features

full_df, added_features = bulk_add_fourier_features(full_df,
            ['time_index_Quarter'], max_values=[4], n_fourier_terms=4)


display(full_df)
print(full_df.info())

# %% make train test

N_TEST = 8
df_temp = full_df.copy(deep=True)
df_temp.index = df_temp['time_index']
df_temp.drop(columns=['time_index'], inplace=True)

df_train = df_temp.iloc[:-(N_TEST), :]
trainX = df_train.iloc[:,1:]
trainy = df_train.iloc[:,0]
df_test = df_temp.iloc[-N_TEST:, :]
testX = df_test.iloc[:,1:]
testy = df_test.iloc[:,0]

#%% XGB baseline MAE = 0.1314
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from xgboost import plot_importance

eval_set = [(trainX, trainy), (testX, testy)]
eval_metric = ['mae']

model = XGBRegressor(objective='reg:squarederror', eval_metric=eval_metric,
                     n_estimators=50)

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

y_hat = model.predict(testX)
y_hat_df = pd.Series(data=y_hat, index=testy.index)
error = mean_absolute_error(y_hat_df, testy)
print(f'MAE: {error:.4f}')

plt.figure()
plt.plot(testy, label='Data')
plt.plot(y_hat_df, label='Predict')
plt.legend()
plt.show()

# %% --------------------------  USE ALGO  -------------------------------
from lightgbm import LGBMRegressor
from helpers.mixed_utils import LogTime
from helpers.ml_forecast import (
    FeatureConfig,
    MissingValueConfig,
    MLForecast,
    ModelConfig,
    calculate_metrics
)

def evaluate_model(model_config, feat_config, missing_value_config, 
            train_features, train_target, test_features, test_target,
            target_transformer=None, fit_kwargs={}):
    
    ml_model = MLForecast(model_config=model_config, feature_config=feat_config, 
            missing_config=missing_value_config, target_transformer=target_transformer)
    ml_model.fit(train_features, train_target, is_transformed=True, fit_kwargs=fit_kwargs)

    y_pred = ml_model.predict(test_features)
  
    feat_df = ml_model.feature_importance()
    metrics = calculate_metrics(test_target, y_pred, model_config.name, train_target)
    return y_pred, metrics, feat_df

#%% Set up FeatutrConfig

def difference_list(list1, list2):
    return list(set(list1)- set(list2))

xgb_full = full_df.copy(deep=True)
#print(xgb_full.info())
# Check continuous list. Might need to remove target and time_index
continuous_list = xgb_full.select_dtypes([np.number]).columns.tolist()[1:]
#print(continuous_list)
bool_list = xgb_full.select_dtypes(['bool']).columns.tolist()
# print(bool_list)
# All columns NOT cont or bool are category
cols = xgb_full.columns
categorical_list = difference_list(cols, continuous_list+bool_list)
# print(categorical_list)

feat_config = FeatureConfig(
    date                    ="time_index",
    target                  ="cons_sum",
    continuous_features     =continuous_list,
    categorical_features    =[],
    boolean_features        =bool_list,
    index_cols              =["time_index"],
    exogenous_features      =['inc', 'prod', 'sav','unem'],)

#%% Set up train, val, test data
VAL_SIZE = 8
TEST_SIZE = 8
NO_TRAIN = VAL_SIZE + TEST_SIZE

# Get train/val
sample_train_df = xgb_full.iloc[:-NO_TRAIN,:]
sample_val_df = xgb_full.iloc[-NO_TRAIN:-TEST_SIZE,:]

train_features, train_target, train_original_target = feat_config.get_X_y(
    sample_train_df, categorical=False, exogenous=False
)
test_features, test_target, test_original_target = feat_config.get_X_y(
    sample_val_df, categorical=False, exogenous=False
)

#%% set up missing_value_config

nc = train_features.isnull().sum()
get_na_cols = nc[nc>0].index.to_list()
#print(nc[nc>0])

missing_value_config = MissingValueConfig(
    bfill_columns=get_na_cols,
    ffill_columns=[],
    zero_fill_columns=[],
)
# %%
metric_val_record = []
pred_val_df = pd.concat([train_target, test_target])

#%% Run fitting and predict

model_config = ModelConfig(
    model=LGBMRegressor(random_state=42, max_depth=4),
    name="LGBM Regressor",
    # XGB is not affected by normalization
    normalize=False,
    # XGB handles missing values
    fill_missing=False,)
with LogTime() as timer:
    y_pred, metrics, feat_df = evaluate_model(model_config, feat_config, missing_value_config,
        train_features, train_target, test_features, test_target)
metrics["Time Elapsed"] = timer.elapsed
metric_val_record.append(metrics)
pred_val_df = pred_val_df.join(y_pred)

#%% Consolidate data and display
df_metric_val_record = pd.DataFrame(metric_val_record)
display(df_metric_val_record)

plt.figure()
plt.plot(test_target, label="Data")
plt.plot(y_pred, label='Predict')
plt.legend()
plt.show()

#%%

display(feat_df)

# %% ------------------  Use target transformations ------------------------------
from helpers.target_transformations import AutoStationaryTransformer
from helpers.stationary_utils import check_unit_root

# Autotransform takes target as a pandas series with time_index as index
train_target = pd.Series(sample_train_df['cons_sum'].tolist(), index=sample_train_df['time_index'])

#%%
SEASONAL_PERIOD = 4

def run_target_transform(train_target, SEASONAL_PERIOD:int):
    # Run transform
    auto_stationary = AutoStationaryTransformer(seasonal_period=SEASONAL_PERIOD)
    y_stat = auto_stationary.fit_transform(train_target, freq='1QE')
    print(f"Transformations applied: {[p.__class__.__name__ for p in auto_stationary._pipeline]}")
    #plot result
    fig, axs = plt.subplots(2)
    train_target.plot(title="Original",ax=axs[0])
    y_stat.plot(title="Auto Stationary",ax=axs[1])
    plt.tight_layout()
    plt.show()

    # Run tests
    unit_root = check_unit_root(y_stat, confidence=0.05)
    print(f"Unit Root: {unit_root.stationary} with a p-value of {unit_root.results[1]}")
    y_inv = auto_stationary.inverse_transform(y_stat)
    print(f"Inverse == Original @ precision of 2 decimal points: {np.all(y_inv.round(2)==train_target.round(2))}")
    return auto_stationary, y_stat

auto_stationary, y_stat = run_target_transform(train_target, SEASONAL_PERIOD)
#display(y_stat.to_frame('cons_sum_auto_stat'))

#%%
# Add transformed target to train.
# Train df now contains date, original_target, transformed target
df_train = sample_train_df.merge(y_stat.to_frame('cons_sum_auto_stat'), left_on='time_index', right_on='time_index')
transformer_pipelines = auto_stationary

#%% Batch run models
from sklearn.linear_model import LassoCV
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRFRegressor

import warnings

models_to_run = [
    ModelConfig(model = LassoCV(), name="Lasso Regression", normalize=True, fill_missing=True),
    ModelConfig(model = MLPRegressor(random_state=1, max_iter=500), name="Sklearn MLP", normalize=True, fill_missing=True),
    ModelConfig(model = HistGradientBoostingRegressor(), name='Sklearn GBR', normalize=False, fill_missing=True),
    ModelConfig(model = AdaBoostRegressor(random_state=0, n_estimators=100), name='AdaBoostRegressor', normalize=True, fill_missing=True),
    ModelConfig(model = XGBRFRegressor(random_state=42, max_depth=4), name="XGB Random Forest", normalize=False, fill_missing=False),
    ModelConfig(model = LGBMRegressor(random_state=42, verbose=-1), name="LightGBM", normalize=False, fill_missing=False)
]

#%% Set up data classes

feat_config = FeatureConfig(
    date                    ="time_index",
    target                  ="cons_sum_auto_stat",
    original_target         ="cons_sum",
    continuous_features     =continuous_list,
    categorical_features    =[],
    boolean_features        =bool_list,
    index_cols              =["time_index"],
    exogenous_features      =[],)

# Use missing_value set up from original run
missing_value_config = MissingValueConfig(
    bfill_columns=get_na_cols,
    ffill_columns=[],
    zero_fill_columns=[],
)

# %% Fit, predict

#We can parallelize this loop to run this faster
def fit_predict_fcn(df_train, df_test, transformer, model_config, feat_config):
    all_preds = []
    all_metrics = []
    for model_config in models_to_run:
        model_config = model_config.clone()
        X_train, y_train, y_train_orig = feat_config.get_X_y(df_train, categorical=False, exogenous=False)
        X_test, _, y_test_orig = feat_config.get_X_y(df_test, categorical=False, exogenous=False)
        
        #transformer = transformer_pipelines
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_pred, metrics, feat_df = evaluate_model(model_config, feat_config, 
                missing_value_config, X_train, y_train, X_test, y_test_orig, transformer)
        
        y_pred.name = "predictions"
        y_pred = y_pred.to_frame()
        y_pred['Algorithm'] = model_config.name + "_auto_stat"
        metrics["Algorithm"] = model_config.name + "_auto_stat"
        y_pred['consumption'] = y_test_orig.values
        all_preds.append(y_pred)
        all_metrics.append(metrics)
    return all_preds, all_metrics, feat_df

all_val_preds, all_val_metrics, feat_df = fit_predict_fcn(df_train, sample_val_df, transformer_pipelines,
                                model_config=model_config, feat_config=feat_config)

#%% Consolidate metrics display data
# metrics
val_metrics_df = pd.DataFrame(all_val_metrics)
df_metric_val_record2 = pd.concat([df_metric_val_record, val_metrics_df]).reset_index(drop=True)

# Highlight high and low value
subset_ = ['MAE', 'MSE', 'MAPE', 'MASE']
df_metric_val_record2 = df_metric_val_record2 \
            .style.highlight_min(axis=0, props='background-color: lightgreen;', subset=subset_) \
             .highlight_max(axis=0, props='background-color:rgb(255 204 203);', subset=subset_)
display(df_metric_val_record2)

# Consolidate predictions
pred_val_df = pd.concat(all_val_preds)
#display(pred_val_df)

# Plot predictions
fname = pred_val_df.Algorithm.unique()

plt.figure()
plt.plot(pred_val_df.loc[pred_val_df.Algorithm == fname[0],'consumption'], label='Data')
for name in fname:
    data_series = pred_val_df.loc[pred_val_df.Algorithm == name,:]
    plt.plot(data_series['predictions'], label=name)
plt.legend()
plt.xticks(rotation=45, ha='right')
plt.show()

# %% Set up train, val, test and run ensemble
# Train base on train, predict on val 
# Train ensemble with base on val, predict on test
# predict all the base on test to use as benchmark



#%% Set up training data with train as train+val and test on test
sample_train_df = xgb_full.iloc[:-TEST_SIZE,:]
sample_test_df = xgb_full.iloc[-TEST_SIZE:,:]

# Train a new target pipeline on new train set
# Autotransform takes target as a pandas series with time_index as index
train_target = pd.Series(sample_train_df['cons_sum'].tolist(), index=sample_train_df['time_index'])
auto_stationary, y_stat = run_target_transform(train_target, SEASONAL_PERIOD)

# Add transformed target to train.
# Train df now contains date, original_target, transformed target
df_train = sample_train_df.merge(y_stat.to_frame('cons_sum_auto_stat'), left_on='time_index', right_on='time_index')
transformer_pipelines = auto_stationary

all_test_preds, all_test_metrics, feat_df = fit_predict_fcn(df_train, sample_test_df, transformer_pipelines,
                                model_config=model_config, feat_config=feat_config)


#%% Consolidate metrics display data
# metrics
df_metric_test_record = pd.DataFrame(all_test_metrics)
display(df_metric_test_record)

# Consolidate predictions
pred_test_df = pd.concat(all_test_preds)
display(pred_test_df)

# Plot predictions
fname = pred_test_df.Algorithm.unique()

plt.figure()
plt.plot(pred_test_df.loc[pred_test_df.Algorithm == fname[0],'consumption'], label='Data')
for name in fname:
    data_series = pred_test_df.loc[pred_test_df.Algorithm == name,:]
    plt.plot(data_series['predictions'], label=name)
plt.legend()
plt.xticks(rotation=45, ha='right')
plt.show()

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

ensemble_forecasts = pred_val_df.Algorithm.unique()

# Get all predictions on the same row
pred_wide_val = pd.pivot(pred_val_df.reset_index(),
                         index = ['time_index'],
                         columns= 'Algorithm',
                         values = 'predictions')

# # Add y_val data to the rows
pred_wide_val = pred_wide_val.join(pred_val_df
                    .loc[pred_val_df.Algorithm == 'LightGBM_auto_stat','consumption'])

# Get all predictions on the same row
pred_wide_test = pd.pivot(pred_test_df.reset_index(),
                         index = ['time_index'],
                         columns= 'Algorithm',
                         values = 'predictions')

# # Add y_val data to the rows
pred_wide_test = pred_wide_test.join(pred_test_df
                    .loc[pred_test_df.Algorithm == 'LightGBM_auto_stat','consumption'])

# display(pred_wide_val)
# display(pred_wide_test)

# %% Calculate mean and meadian ensembles

pred_wide_test['average_ensemble'] = pred_wide_test[ensemble_forecasts].mean(axis=1)
pred_wide_test['median_ensemble'] = pred_wide_test[ensemble_forecasts].median(axis=1)

#%%
agg_metric = evaluate_ensemble(pred_wide_test, 'average_ensemble', 'consumption')
df_metric_test_record = pd.concat([df_metric_test_record,agg_metric], axis=0).reset_index(drop=True)

agg_metric = evaluate_ensemble(pred_wide_test, 'median_ensemble', 'consumption')
df_metric_test_record = pd.concat([df_metric_test_record,agg_metric], axis=0).reset_index(drop=True)

display(df_metric_test_record)
# %% Stacking using 2nd level models
from sklearn.linear_model import (
    HuberRegressor,
    LassoCV,
    LinearRegression,
    RidgeCV
)

#%% Try linear regression
# Fit
stacking_model = LinearRegression(positive=True, fit_intercept=False)
stacking_model.fit(pred_wide_val[ensemble_forecasts], pred_wide_val["consumption"])

# Display weights
forecast_weights = pd.DataFrame({"Forecast": ensemble_forecasts, "Weights": stacking_model.coef_}) \
        .round(4).sort_values("Weights", ascending=False)
display(forecast_weights)

# Predict add to metrics list
pred_wide_test["linear_reg_blending"] = stacking_model.predict(pred_wide_test[ensemble_forecasts])

# Calculate metrics
agg_metric = evaluate_ensemble(pred_wide_test, 'linear_reg_blending', 'consumption')
df_metric_test_record = pd.concat([df_metric_test_record,agg_metric], axis=0).reset_index(drop=True)
display(df_metric_test_record)

#%% Try ridge regression
# Fit
stacking_model = RidgeCV()
stacking_model.fit(pred_wide_val[ensemble_forecasts], pred_wide_val["consumption"])

# Display weights
forecast_weights = pd.DataFrame({"Forecast": ensemble_forecasts, "Weights": stacking_model.coef_}) \
        .round(4).sort_values("Weights", ascending=False)
display(forecast_weights)

# Predict add to metrics list
pred_wide_test["ridge_reg_blending"] = stacking_model.predict(pred_wide_test[ensemble_forecasts])

# Calculate metrics
agg_metric = evaluate_ensemble(pred_wide_test, 'ridge_reg_blending', 'consumption')
df_metric_test_record = pd.concat([df_metric_test_record,agg_metric], axis=0).reset_index(drop=True)
display(df_metric_test_record)

#%% Try Huber regression
# Fit
stacking_model = HuberRegressor()
stacking_model.fit(pred_wide_val[ensemble_forecasts], pred_wide_val["consumption"])

# Display weights
forecast_weights = pd.DataFrame({"Forecast": ensemble_forecasts, "Weights": stacking_model.coef_}) \
        .round(4).sort_values("Weights", ascending=False)
display(forecast_weights)

# Predict add to metrics list
pred_wide_test["huber_reg_blending"] = stacking_model.predict(pred_wide_test[ensemble_forecasts])

# Calculate metrics
agg_metric = evaluate_ensemble(pred_wide_test, 'huber_reg_blending', 'consumption')
df_metric_test_record = pd.concat([df_metric_test_record,agg_metric], axis=0).reset_index(drop=True)
display(df_metric_test_record)
#%%

corrs = pred_wide_test[ensemble_forecasts].corr()
display(corrs)

x_label_names = corrs.columns
y_label_names = corrs.index
NO_FACTORS = 3

plt.figure()
plt.pcolormesh(corrs, edgecolors='k', linewidth=2)
plt.xticks(np.arange(NO_FACTORS), x_label_names, rotation=45, ha='right')
plt.yticks(np.arange(NO_FACTORS), y_label_names)
plt.colorbar()
ax = plt.gca()
ax.set_aspect('equal')
plt.show()

# %% get train and val for optimization --------------------------------------------------
sample_train_df = xgb_full.iloc[:-NO_TRAIN,:]
sample_val_df = xgb_full.iloc[-NO_TRAIN:-TEST_SIZE,:]

train_features, train_target, train_original_target = feat_config.get_X_y(
    sample_train_df, categorical=False, exogenous=False
)
test_features, test_target, test_original_target = feat_config.get_X_y(
    sample_val_df, categorical=False, exogenous=False
)

display(test_target)

cat_features = [] #set(train_features.columns).intersection(_feat_config.categorical_features)

#%% Get baseline
# Can use PredefinedSplit along with GridSearchCV to have the search done faster using multi-processing
# Or we can parallelize the loop ourselves
scores = []
parameter_space = [1]
for p in tqdm(parameter_space, desc="Performing Grid Search"):
    _model_config = ModelConfig(
        model=LGBMRegressor(verbose=-1),
        name="Global Meta LightGBM Tuning",
        # LGBM is not sensitive to normalized data
        normalize=False,
        # LGBM can handle missing values
        fill_missing=False,
    )
    y_pred, metrics, feat_df = evaluate_model(
        _model_config,
        feat_config,
        missing_value_config,
        train_features,
        train_target,
        test_features,
        test_target,
#        fit_kwargs=dict(categorical_feature=cat_features),
    )
    scores.append(metrics)

print(scores)
# 'MAE': 0.4067309056184675

#%%

print(feat_df)

# %% Get params 
from sklearn.model_selection import ParameterGrid

grid_params = {
    "num_leaves": [16, 31, 63],
    "objective": ["regression", "regression_l1", "huber"],
    "random_state": [42],
    "colsample_bytree": [0.5, 0.8, 1.0],
}
# List of dicts with all combinations
parameter_space = list(ParameterGrid(grid_params))


# %%
# Can use PredefinedSplit along with GridSearchCV to have the search done faster using multi-processing
# Or we can parallelize the loop ourselves
scores = []
for p in tqdm(parameter_space, desc="Performing Grid Search"):
    _model_config = ModelConfig(
        model=LGBMRegressor(**p, verbose=-1),
        name="Global Meta LightGBM Tuning",
        # LGBM is not sensitive to normalized data
        normalize=False,
        # LGBM can handle missing values
        fill_missing=False,
    )
    y_pred, metrics, feat_df = evaluate_model(
        _model_config,
        feat_config,
        missing_value_config,
        train_features,
        train_target,
        test_features,
        test_target,
#        fit_kwargs=dict(categorical_feature=cat_features),
    )
    scores.append(metrics)

mae_list = []
for i,v in enumerate(scores):
    mae_list.append(scores[i]['MAE'])

# %% Move result to pd and display

grid_search_trials = pd.DataFrame({"params":parameter_space, "score":mae_list}).sort_values("score")
best_params_gs = grid_search_trials.iloc[0,0]
best_score_gs = grid_search_trials.iloc[0,1]
display(grid_search_trials)

best_grid = grid_search_trials.iloc[0].to_list
# best 0.307150

# %% Random search
# Set up parameter space
import scipy
from sklearn.model_selection import ParameterSampler
N_ITER = 100

random_search_params = {
    # A uniform distribution between 10 and 100, but only integers
    "num_leaves": scipy.stats.randint(10,100),
    # A list of categorical string values
    "objective": ["regression", "regression_l1", "huber"],
    "random_state": [42],
    # List of floating point numbers between 0.3 and 1.0 with a resolution of 0.05
    "colsample_bytree": np.arange(0.3,1.0,0.05),
    # List of floating point numbers between 0 and 10 with a resolution of 0.1
    "lambda_l1":np.arange(0,10,0.1),
    # List of floating point numbers between 0 and 10 with a resolution of 0.1
    "lambda_l2":np.arange(0,10,0.1)
}
# Sampling from the search space number of iterations times
parameter_space = list(ParameterSampler(random_search_params, n_iter=N_ITER, random_state=42))

#%%
# Can use PredefinedSplit along with GridSearchCV to have the search done faster using multi-processing
# Or we can parallelize the loop ourselves
scores = []
for p in tqdm(parameter_space, desc="Performing Grid Search"):
    _model_config = ModelConfig(
        model=LGBMRegressor(**p, verbose=-1),
        name="Global Meta LightGBM Tuning",
        # LGBM is not sensitive to normalized data
        normalize=False,
        # LGBM can handle missing values
        fill_missing=False,
    )
    y_pred, metrics, feat_df = evaluate_model(
        _model_config,
        feat_config,
        missing_value_config,
        train_features,
        train_target,
        test_features,
        test_target,
#        fit_kwargs=dict(categorical_feature=cat_features),
    )
    scores.append(metrics)

mae_list = []
for i,v in enumerate(scores):
    mae_list.append(scores[i]['MAE'])

# %% Move result to pd and display

random_search_trials = pd.DataFrame({"params":parameter_space, "score":mae_list}).sort_values("score")
best_params_gs = random_search_trials.iloc[0,0]
best_score_gs = random_search_trials.iloc[0,1]
display(random_search_trials.head(10))

#%%
best_random = random_search_trials.iloc[0].to_list()
# best 0.204

# %% Run Optuna, bayesian optimization lib
import optuna

# Define an objective functions which takes in trial as a parameter 
# and evaluates the model with the generated params
N_TRIALS = 100
def objective(trial):
    params = {
        # Sample an integer between 10 and 100
        "num_leaves": trial.suggest_int("num_leaves", 10, 100),
        # Sample a categorical value from the list provided
        "objective": trial.suggest_categorical(
            "objective", ["regression", "regression_l1", "huber"]
        ),
        "random_state": [42],
        # Sample from a uniform distribution between 0.3 and 1.0
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.3, 1.0),
        # Sample from a uniform distribution between 0 and 10
        "lambda_l1": trial.suggest_uniform("lambda_l1", 0, 10),
        # Sample from a uniform distribution between 0 and 10
        "lambda_l2": trial.suggest_uniform("lambda_l2", 0, 10),
    }
    _model_config = ModelConfig(
        # Use the sampled params to initialize the model
        model=LGBMRegressor(**params, verbose=-1),
        name="Global Meta LightGBM Tuning",
        # LGBM is not sensitive to normalized data
        normalize=False,
        # LGBM can handle missing values
        fill_missing=False,
    )
    y_pred, metric, feat_df = evaluate_model(
        _model_config,
        feat_config,
        missing_value_config,
        train_features,
        train_target,
        test_features,
        test_target,
#        fit_kwargs=dict(categorical_feature=cat_features),
    )
    # Return the MAE metric as the value
    # ts_utils.mae(test_target["energy_consumption"], y_pred)
    return metric['MAE']

#%%
# Create a sampler and set seed for repeatability. 
# Set startup trials as 5 because out total trials is lower.
sampler = optuna.samplers.TPESampler(n_startup_trials=5, seed=42)
# Create a study
study = optuna.create_study(direction="minimize", sampler=sampler)
# Start the optimization run
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

# %% Summarize and display

bo_search_trials = study.trials_dataframe()
best_params_bo = study.best_params
best_score_bo = study.best_value
bo_search_trials.sort_values("value").head()

best_optuna = bo_search_trials.iloc[0].to_list()
# best 0.2024
# %%
