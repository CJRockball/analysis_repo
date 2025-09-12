import copy
import warnings
from dataclasses import MISSING, dataclass, field
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.base import BaseEstimator, clone
from sklearn.preprocessing import StandardScaler
from darts.metrics import mase
from darts import TimeSeries
# from darts.metrics import mae, mase, mse
# from utils import darts_metrics_adapter

def intersect_list(list1, list2):
    return list(set(list1).intersection(set(list2)))

def difference_list(list1, list2):
    return list(set(list1)- set(list2))

def union_list(list1, list2):
    return list(set(list1).union(set(list2)))


@dataclass
class FeatureConfig:
    # Define vars
    date: list = field(default=MISSING, 
                    metadata={"help":"Column name of the date column"}),

    target: str = field(default=MISSING, 
                    metadata={"help":"Column name of target column"}),

    original_target: str = field(default=None,
                    metadata={"help":"Column name of unprocessed target. \
                              If None it will be assigned as 'target'"})

    continuous_features: list[str] = field(default_factory=list,
                    metadata={'help':"Column names of numeric fields, default []"})

    categorical_features: list[str] = field(default_factory=list,
                    metadata={"help":"Column names of categorical features, default []"})

    boolean_features: list[str] = field(default_factory=list,
                    metadata={'help':"Column names of boolean features, default []"})

    index_cols: list[str] = field(default_factory=list,
                    metadata={'help':"Column name needed to set index for X and Y dataframes"})

    exogenous_features: list[str] = field(default_factory=list,
                    metadata={"help":"Column names of exogenous features. \
                        Must be a subset of categorical and continuous features"})
    feature_list: list[str] = field(init=False)
    
    def __post_init__(self):
        assert (len(self.categorical_features) + len(self.continuous_features) > 0),\
            "There should be at least one continuous or categorical feature defined"
            
        self.feature_list = self.categorical_features + self.continuous_features + self.boolean_features

        assert (self.target not in self.feature_list), f"'target'({self.target}) should\
            not be present in either in categorical, continuous or boolean feature list"
        assert (self.date not in self.feature_list), f"'date' ({self.date}) should not be in \
            categorical, continuous or boolean features"
        extra_exog = set(self.exogenous_features) - set(self.feature_list)
        assert (len(extra_exog) == 0), f"These exogenous features are no present in the feature list: {extra_exog}"
        intersection = (set(self.continuous_features).intersection(self.categorical_features + 
                        self.boolean_features).union(set(self.boolean_features).intersection(
                            self.continuous_features + self.categorical_features))
                        .union(set(self.boolean_features).intersection(
                            self.continuous_features + self.categorical_features)))
        assert (len(intersection) == 0), f"There should not be any overlaps between the categorical, \
            continuous and boolean features. {intersection} are present in more than one definition"
        if self.original_target is None:
            self.original_target = self.target
            
    def get_X_y(self, df: pd.DataFrame, categorical:bool=False, exogenous:bool=False):
        feature_list = copy.deepcopy(self.continuous_features)
        if categorical:
            feature_list += self.categorical_features + self.boolean_features
        if not exogenous:
            feature_list = list(set(feature_list) - set(self.exogenous_features))
        
        feature_list = list(set(feature_list))
        delete_index_cols = list(set(self.index_cols) - set(self.feature_list))
        X      = df.loc[:,list(set(feature_list + self.index_cols))] \
                .set_index(self.index_cols, drop=False).drop(columns=delete_index_cols)
        y      = df.loc[:,[self.target] + self.index_cols].set_index(self.index_cols, drop=True)\
                    if self.target in df.columns else None
        y_orig = df.loc[:,[self.original_target] + self.index_cols].set_index(self.index_cols, drop=True)\
                    if self.original_target in df.columns else None
        return X, y, y_orig
            


@dataclass
class MissingValueConfig:
    
    bfill_columns: list = field(default_factory=list,
            metadata={'help':"Column names which should be filled using 'bfill'"})
    ffill_columns: list = field(default_factory=list,
            metadata={'help':"Column names which should be filled using 'ffill'"})
    zero_fill_columns:list = field(default_factory=list,
            metadata={'help':"Column names which should be filled using zeros"})
    
    def impute_missing_values(self, df:pd.DataFrame):
        df = df.copy()

        assert (set(self.bfill_columns).issubset(df.columns)), "All columns in bfill_columns should be in df"
        df[self.bfill_columns] = df[self.bfill_columns].bfill()
        assert (set(self.ffill_columns).issubset(df.columns)), "All columns in ffill_columns should be in df "
        df[self.ffill_columns] = df[self.ffill_columns].ffill()
        assert (set(self.zero_fill_columns).issubset(df.columns)), "All columns in zero_fill_columns should be in df"

        check = df.isnull().any()
        missing_cols = check[check].index.tolist()
        missing_numeric_cols = intersect_list(
            missing_cols, df.select_dtypes([np.number]).columns.tolist())
        missing_object_cols = intersect_list(
            missing_cols, df.select_dtypes(['object']).columns.tolist())
        # Fill with mean and NA as default fillna strategy
        df[missing_numeric_cols] = df[missing_numeric_cols].fillna(df[missing_numeric_cols].mean())
        df[missing_object_cols] = df[missing_object_cols].fillna('NA')

        return df
    
@dataclass
class ModelConfig:
    
    model: BaseEstimator = field(default=MISSING,
        metadata={'help':"Sci-kit Learn compatible model instance"})
    name: str = field(default=None,
        metadata={'help':"Name or identifier of the model. If None will use string rep of model"})
    normalize: bool = field(default=True,
        metadata={'help':"Flag whether to normalize input or not before fitting"})
    fill_missing:bool=field(default=True,
        metadata={'help':"Flag whether to fill missing values before fitting"})
    encode_categorical: bool = field(default=False,
        metadata={'help':"Flag wheter to encode categorical before fitting"})
    categorical_encoder: BaseEstimator = field(default=None,
        metadata={'help':"Categorical encoder to be used"})
    
    def __post_init__(self):
        assert not(self.encode_categorical and self.categorical_encoder is None),\
            "'categorical_encoder cannot be None if 'encode_categorical' is True"

    def clone(self):
        self.model = clone(self.model)
        return self
        

class MLForecast:
    def __init__(self, model_config:ModelConfig, feature_config:FeatureConfig, 
                 missing_config:MissingValueConfig=None, target_transformer:object=None
    ) -> None:
        """Convenient wrapper around scikit-learn style estimators

            Args:
                model_config (ModelConfig): Instance of the ModelConfig object defining the model
                feature_config (FeatureConfig): Instance of the FeatureConfig object defining the features
                missing_config (MissingValueConfig, optional): Instance of the MissingValueConfig object
                    defining how to fill missing values. Defaults to None.
                target_transformer (object, optional): Instance of target transformers from src.transforms.
                    Should support `fit`, `transform`, and `inverse_transform`. It should also
                    return `pd.Series` with datetime index to work without an error. Defaults to None.
            """
        self.model_config = model_config
        self.feature_config = feature_config
        self.missing_config = missing_config
        self.target_transformer = target_transformer
        self._model = clone(model_config.model)
        if self.model_config.normalize:
            self._scaler = StandardScaler()
        if self.model_config.encode_categorical:
            self._cat_encoder = self.model_config.categorical_encoder
            self._encoded_cateforical_features = copy.deepcopy(self.feature_config.categorical_features)

    def fit(self, X:pd.DataFrame, y:Union[pd.Series, np.ndarray], 
            is_transformed:bool=False, fit_kwargs:Dict={}):
        """Handles standardization, missing value handling, and training the model

        Args:
            X (pd.DataFrame): The dataframe with the features as columns
            y (Union[pd.Series, np.ndarray]): Dataframe, Series, or np.ndarray with the targets
            is_transformed (bool, optional): Whether the target is already transformed.
            If `True`, fit wont be transforming the target using the target_transformer
                if provided. Defaults to False.
            fit_kwargs (Dict, optional): The dictionary with keyword args to be passed to the
                fit funciton of the model. Defaults to {}.
        """
        
        missing_feats = difference_list(X.columns, self.feature_config.feature_list)
        if len(missing_feats) > 0:
            warnings.warn(f"Some features in FeatureConfig are not present in the dataframe.\
                         Ignoring these features: {missing_feats}")
        self._continuous_feats = intersect_list(
            self.feature_config.continuous_features, X.columns)
        self._categorical_feats = intersect_list(
            self.feature_config.categorical_features, X.columns)
        self._boolean_feats = intersect_list(
            self.feature_config.boolean_features, X.columns)
        if self.model_config.fill_missing:
            X = self.missing_config.impute_missing_values(X)
        if self.model_config.encode_categorical:
            missing_cat_cols = difference_list(
                self._categorical_feats, self.model_config.categorical_encoder.cols)
            assert (len(missing_cat_cols) == 0), \
            f"These categorical features are not handeled by the categorical encoder: {missing_cat_cols}"
            
            try:
                feature_names = self.model_config.categorical_encoder.get_feature_names_out()
            except: "The sklearn you're using if of older model. Use get_feature_names"
            
            X = self._cat_encoder.fit_transform(X, y)
            self._encoded_categorical_features = difference_list(feature_names, 
                self.feature_config.continuous_features + self.feature_config.boolean_features)
        else:
            self._encoded_categorical_features = []
        
        if self.model_config.normalize:
            X[self._continuous_feats + self._encoded_categorical_features] = \
                self._scaler.fit_transform(X[self._continuous_feats + self._encoded_categorical_features])
        
        self._train_features = X.columns.tolist()
        if not is_transformed and self.target_transformer is not None:
            y = self.target_transformer.fit_transform(y)
        self._model.fit(X, y, **fit_kwargs)
        return self
    
    def predict(self, X:pd.DataFrame) -> pd.Series:
        """Predicts on the given dataframe using the trained model

        Args:
            X (pd.DataFrame): The dataframe with the features as columns. The index is passed on to the prediction series

        Returns:
            pd.Series: predictions using the model as a pandas Series with datetime index
        """
        assert len(intersect_list(self._train_features, X.columns)) == len(self._train_features),\
            f"All the features during training is not available to while predicting: {difference_list(self._train_features, X.columns)}"
        if self.model_config.fill_missing:
            X = self.missing_config.impute_missing_values(X)
        if self.model_config.encode_categorical:
            X = self._cat_encoder.transform(X)
        if self.model_config.normalize:
            X[self._continuous_feats + self._encoded_categorical_features] =\
                self._scaler.transform(X[self._continuous_feats + self._encoded_categorical_features])
        y_pred = pd.Series(self._model.predict(X).ravel(), index=X.index, name=f"{self.model_config.name}") # Changed the index to dates
        if self.target_transformer is not None:
            y_pred = self.target_transformer.inverse_transform(y_pred)
            y_pred.name = f"{self.model_config.name}"
        return y_pred
    
    
    def feature_importance(self) -> pd.DataFrame:
        """Generates the feature importance dataframe, if available. For linear
            models the coefficients are used and tree based models use the inbuilt
            feature importance. For the rest of the models, it returns an empty dataframe.

        Returns:
            pd.DataFrame: Feature Importance dataframe, sorted in descending order of its importances.
        """
        if hasattr(self._model, "coef_") or hasattr(self._model, "feature_importances_"):
            feat_df = pd.DataFrame({
                                    "feature":self._train_features,
                                    "importance":self._model.coef_.ravel()
                                    if hasattr(self._model, "coef_") 
                                    else self._model.feature_importances_.ravel()
                                    }
                                )
            feat_df['_abs_imp'] = np.abs(feat_df.importance)
            feat_df = feat_df.sort_values("_abs_imp", ascending=False).drop(columns="_abs_imp")
        else:
            feat_df = pd.DataFrame()
            
        return feat_df
    
def calculate_metrics(y:pd.Series, y_pred:pd.Series, name:str, y_train:pd.Series=None):
    """Method to calculate the metrics given the actual and predicted series

    Args:
        y (pd.Series): Actual target with datetime index
        y_pred (pd.Series): Predictions with datetime index
        name (str): Name or identification for the model
        y_train (pd.Series, optional): Actual train target to calculate MASE with datetime index. Defaults to None.

    Returns:
        Dict: Dictionary with MAE, MSE, MASE, and Forecast Bias
    """
    y_ts        = TimeSeries.from_series(y)
    y_pred_ts   = TimeSeries.from_series(y_pred)
    y_train_ts  = TimeSeries.from_series(y_train)
    return {
        "Algorithm": name,
        "MAE": mean_absolute_error(y, y_pred),
        "MSE": mean_squared_error(y, y_pred),
        "MAPE": mean_absolute_percentage_error(y, y_pred),
        "MASE": mase(y_ts,y_pred_ts, y_train_ts),
    }