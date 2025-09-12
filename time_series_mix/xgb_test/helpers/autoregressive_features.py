import pandas as pd
import warnings

from window_ops.rolling import (
    seasonal_rolling_max,
    seasonal_rolling_mean,
    seasonal_rolling_min,
    seasonal_rolling_std,
)

SEASONAL_ROLLING_MAP = {
    "mean": seasonal_rolling_mean,
    "min": seasonal_rolling_min,
    "max": seasonal_rolling_max,
    "std": seasonal_rolling_std,
}

# Make lags
# TODO Add list for columns for lags of exog vars
def add_lags(df:pd.DataFrame, 
    lags: list[int], 
    column:str,
    ts_id:str = None
    ) -> tuple[pd.DataFrame, list]:
    """ Make lags for columns provided and adds them to the provided dataframe
    Args:
        df (pd.DataFrame): The dataframe in which features needs to be created
        lags (list[int]): List of lags
        column (str): Name of collumn to be lagged
    Returns:
        Typle(pd.DataFrame, list): Return the df with new laggs and a list of col names
    """
    
    assert isinstance(lags, list), "'lag' should be a list"
    assert (column in df.columns), "'column' should be in the df"
    
    if ts_id is None:
        warnings.warn("Assuming just one unique time seres in dataset. If there are multiple, provide 'ts_id")
        col_dict = {f"{column}_lag_{l}": df[column].shift(l) for l in lags}
    else:
        assert (ts_id in df.columns), "'ts_id' should be in dataframe "
        col_dict = {f'{column}_lag_{l}': 
            df.groupby([ts_id], observable=False)[column].shift(l) for l in lags}
        
    df = df.assign(**col_dict)
    added_features = list(col_dict.keys())

    return df, added_features


def add_rolling_features(df:pd.DataFrame,
    rolls: list[int],
    column: str,
    agg_funcs: list[int] = ['mean', 'std'],
    ts_id: str = None,
    n_shift: int = 1
    ) -> tuple[pd.DataFrame, list]:
    """ Add rolling statistics, from the column provided, to the dataframe
    
    Args:
        df (pd.DataFrame): The dataframe to which to add features
        rolls (list[int]): DIfferent windows over which the rolling aggregation is done
        column (str): The name of the column to aggregate
        ts_id (str, optional): Unique id for time series if multiple in df
        n_shift (int, optional): Steps to shift column to avoid data leakage
    """

    assert isinstance(rolls, list), "'rolls' should be a list"
    assert (column in df.columns), "'column' should be in df"
    #assert (agg_funcs)

    if ts_id is None:
        warnings.warn("Assuming just one unique time series in dataset. If there are multiple, provide 'ts_id' argument")

        rolling_df = pd.concat([
            df[column].shift(n_shift).rolling(l).agg({f"{column}_rolling_{l}_{agg}":
                agg for agg in agg_funcs}) for l in rolls], axis=1)
    else:
        assert (ts_id in df.columns), "'ts_id' should be in df"
        rolling_df = pd.concat([
            df.groupby(ts_id, observed=False)[column].shift(n_shift).rolling(l)
            .agg({f"{column}_rolling_{l}_{agg}": agg for agg in agg_funcs}) for l in rolls], axis=1)

    df = df.assign(**rolling_df.to_dict("list"))
    added_features = rolling_df.columns.tolist()
    return df, added_features


def add_seasonal_rolling_features(df:pd.DataFrame,
    seasonal_periods:list[int],
    rolls:list[int],
    column:str,
    agg_funcs:list[str]=['mean','std'],
    ts_id:str=None,
    n_shift:int=1
    ) -> tuple[pd.DataFrame, list]:
    """ Add seasonal rolling statistics, from columns provided, to dataframe provided
    
    Args:
        df (pd.DataFrame): Dataframe to which to add features
        seasonal_periods (list[int]): List of seasonal periods over which to roll agg functions
        rolls (list[int]): List of seasonal rolling windows
        column (str): Name of column in df to create features from
        add_funcs (list[str]): agg functions to roll
        ts_id (str, optional): column in df to sort over if there are multiple time series in the df
        n_shift (int, optional): number of steps to shift to avoid data leakage
    Returns:
        tuple[pd.DataFrame, list]: Returns updated dataframe and a list of columns names added    
    """

    assert isinstance(rolls, list), "'rolls' should be a list"
    assert isinstance(seasonal_periods, list), "'seasonal_periods' should be a list"
    assert (column in df.columns), "'column' shoudl be in df"
    
    agg_funcs = {agg: SEASONAL_ROLLING_MAP[agg] for agg in agg_funcs}
    added_features = []
    for sp in seasonal_periods:
        if ts_id is None:
            warnings.warn("Assuming just one unique time series in dataset. If there are multiple, provide 'ts_id' argument")
            col_dict = {f"{column}_{sp}_seasonal_rolling_{l}_{name}":df[column]
                .transform(lambda x: agg(x.shift(n_shift*sp).values, season_length=sp, window_size=l))
                for (name, agg) in agg_funcs.items() for l in rolls}
        else:
            assert (ts_id in df.columns), "'ts_id' should be in df"
            col_dict = {f"{column}_{sp}_seasonal_rolling_{l}_{name}": 
                        df.groupby(ts_id, observed=False)[column]
                        .transform(lambda x: agg(x.shift(n_shift*sp).values,
                                    season_length=sp, window_size=l))
                        for (name, agg) in agg_funcs.items() for l in rolls}
        
            
        df = df.assign(**col_dict)
        added_features += list(col_dict.keys())
    return df, added_features


def add_ewma(df: pd.DataFrame, column:str, alphas:list[float]=[0.5], spans:list[float]=None,
            ts_id:str=None, n_shift:int=1) -> tuple[pd.DataFrame, list]:
    """ Create expanding window features 
    
    Args:
        df (pd.DataFrame): Dataframe with org feature and to witch they are added
        column (str): Name of column from which to make features
        alpha (list[float]): Smoothing parameters to be used
        spans (list[float]):  List of length of ewma windows alpha = 2/(1+span). If span is given alpha is ignored
        ts_id (str, optional): Unique ID if there are multiple timeseries
        n_shift (int, optional): Shift to avoid data leakage
    
    Returns:
        tuple(pd.DataFrame, list): Returns tuple of the new dataframe and a list of names of added features
    """
    
    if spans is not None:
        assert isinstance(spans, list), "'spans' should be a list"
        use_spans = True
    if alphas is not None:
        assert isinstance(alphas, list), "'alphas' should be a list"
    if spans is None and alphas is None:
        raise ValueError("Either 'alpha' or 'spans' has to be provided")    
    assert (column in df.columns), "'columns' should be in df"
    
    if ts_id is None:
        warnings.warn("Assuming just one unique time series. If there are multiple, provide 'ts_id' argument")
        col_dict = {f"{column}_ewma_{'span' if use_spans else 'alpha'}_{param}":
                    df[column].shift(n_shift).ewm(
                        alpha=None if use_spans else param, 
                        span=param if use_spans else None, 
                        adjust=False).mean() for param in (spans if use_spans else alphas)}
    else:
        col_dict = {f"{column}_ewma_{'span' if use_spans else 'alpha'}_{param}": 
                    df.groupby([ts_id], observed=False)[column].shift(n_shift)
                    .ewm(
                    alpha=None if use_spans else param,
                    span=param if use_spans else None,
                    adjust=False,
                    ).mean()for param in (spans if use_spans else alphas)}
    
    df = df.assign(**col_dict)
    return df, list(col_dict.keys())            
        
 