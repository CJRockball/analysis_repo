import re
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset

# adapted from gluonts
def time_features_from_frequency_str(freq_str: str) -> List[str]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.

    Parameters
    ----------

    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.

    """

    features_by_offsets = {
        offsets.YearBegin: [],
        offsets.YearEnd: [],
        offsets.QuarterBegin: [
            "Quarter",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",],
        offsets.QuarterEnd: [
            "Quarter",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",],
        offsets.MonthBegin: [
            "Month",
            "Quarter",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
        ],
        offsets.MonthEnd: [
            "Month",
            "Quarter",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
        ],
        offsets.Week: [
            "Month",
            "Quarter",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
            "Is_month_start",
            "Week",
        ],
        offsets.Day: [
            "Month",
            "Quarter",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
            "Is_month_start",
            "Week" "Day",
            "Dayofweek",
            "Dayofyear",
        ],
        offsets.BusinessDay: [
            "Month",
            "Quarter",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
            "Is_month_start",
            "Week" "Day",
            "Dayofweek",
            "Dayofyear",
        ],
        offsets.Hour: [
            "Month",
            "Quarter",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
            "Is_month_start",
            "Week" "Day",
            "Dayofweek",
            "Dayofyear",
            "Hour",
        ],
        offsets.Minute: [
            "Month",
            "Quarter",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
            "Is_month_start",
            "Week" "Day",
            "Dayofweek",
            "Dayofyear",
            "Hour",
            "Minute",
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return feature

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}

    The following frequencies are supported:

        Y, YS   - yearly
            alias: A
        QE   - quarterly
        M, MS   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
    """
    raise RuntimeError(supported_freq_msg)

def add_temporal_features(df:pd.DataFrame, field_name:str, frequency:str,
                          add_elapsed:bool=True, prefix:str=None, drop:bool=True
                          ) -> tuple[pd.DataFrame, list]:
    """ Adds temporal features relevant to date in the column 'field_name' of 'df'
    
    Args:
        df (pd.DataFrame): Dataframe from which and to which features are engeineered
        field_name (str): The date column which should be encoded using temporal features
        frequency (str): The frequency of the date column so that only relevant features are added
            If frequency is 'Weekly', then temporal feautres like hour, minutes etc doesn't make sense.
        add_elapsed (bool, optional): Add time elapsed as a monotonically increasing function
        prefix (str, optional): Prefix to newly created columns. if None, will use the field name
        drop (bool, optional): Flag to drop the data column after features are created\
    
    Returns: tuple[pd.Dataframe, list]: Returns updated dataframe and a list of columns added
    """
    
    field = df[field_name]
    prefix = (re.sub("[Dd]ate$", "", field_name) if prefix is None else prefix) + "_"
    attr = time_features_from_frequency_str(frequency)
    added_features = []
    for n in attr:
        if n == "Week":
            continue
        df[prefix + n] = getattr(field.dt, n.lower())
        added_features.append(prefix + n)
    
    if "Week" in attr:
        week = (field.dt.isocalender().week if hasattr(field.dt, 'isocalender') else field.dt.week)
        df.insert(3, prefix + 'Week', week)
        added_features.append(prefix + 'Week')
    
    if add_elapsed:
        mask = ~field.isna()
        df[prefix + 'Elapsed'] = np.where(mask, field.values.astype(np.int64) // 10**9, None)
        df[prefix + 'Elapsed'] = df[prefix + 'Elapsed'].astype('float32')
        added_features.append(prefix + 'Elapsed')
        
    if drop:
        df.drop(field_name, axis=1, inplace=True)
    return df, added_features



# Fourier Terms
def _calculate_fourier_terms(seasonal_cycle:np.ndarray, max_cycle:int, n_fourier_terms:int):
    """ Calculates Fourier terms given the seasonal cycle and max cycle"""

    sin_X = np.empty((len(seasonal_cycle), n_fourier_terms), dtype='float64')
    cos_X = np.empty((len(seasonal_cycle), n_fourier_terms), dtype='float64')
    for i in range(1, n_fourier_terms + 1):
        sin_X[:, i - 1] = np.sin((2 * np.pi * seasonal_cycle * i) / max_cycle)
        cos_X[:, i - 1] = np.sin((2 * np.pi * seasonal_cycle * i) / max_cycle)
    return np.hstack([sin_X, cos_X])


def add_fourier_features(df: pd.DataFrame, column_to_encode:str, max_value:Optional[int]=None,
                         n_fourier_terms:int=1) -> tuple[pd.DataFrame, list]:
    """ Adds Fourier Terms for the specified seasonal cycle column, like month, week, hour, etc.

    Args:
        df (pd.DataFrame): The dataframe which has the seasonal cyycles which has to be encoded
        column_to_encode (str): The column name which has the seasonal cycle
        max_value (int): The maximum value the seasonal cycle can attain. for eg. for month, max_value is 12.
            If not given, it will be inferred from the data, but if the data does not have at least a
            single full cycle, the inferred max value will not be appropriate. Defaults to None
        n_fourier_terms (int): Number of fourier terms to be added. Defaults to 1
        use_32_bit (bool, optional): Flag to use float32 or int32 to reduce memory. Defaults to False.
    Raises:
        warnings.warn: Raises a warning if max_value is None

    Returns:
        [Tuple[pd.DataFrame, List]]: Returns a tuple of the new dataframe and a list of features which were added
    """
    assert (column_to_encode in df.columns), "'column_to_encode should be in the df"
    assert is_numeric_dtype(df[column_to_encode]), "'column_to_encode' should be numeric"
    if max_value is None:
        max_value = df[column_to_encode].max()
        raise warnings.warn("Inferring max cycle as {} from the data. This may not be accurate")

    fourier_features = _calculate_fourier_terms(df[column_to_encode].astype(int).values,
                            max_cycle=max_value, n_fourier_terms=n_fourier_terms)
    feature_names = [f"{column_to_encode}_sin_{i}" for i in range(1, n_fourier_terms + 1)] + \
        [f"{column_to_encode}_cos_{i}" for i in range(1, n_fourier_terms + 1)]
        
    df[feature_names] = fourier_features
    return df, feature_names


def bulk_add_fourier_features(
    df: pd.DataFrame,
    columns_to_encode: List[str],
    max_values: List[int],
    n_fourier_terms: int = 1,
) -> Tuple[pd.DataFrame, List]:
    """Adds Fourier Terms for all the specified seasonal cycle columns, like month, week, hour, etc.

    Args:
        df (pd.DataFrame): The dataframe which has the seasonal cyycles which has to be encoded
        columns_to_encode (List[str]): The column names which has the seasonal cycle
        max_values (List[int]): The list of maximum values the seasonal cycles can attain in the
            same order as the columns to encode. for eg. for month, max_value is 12.
            If not given, it will be inferred from the data, but if the data does not have at least a
            single full cycle, the inferred max value will not be appropriate. Defaults to None
        n_fourier_terms (int): Number of fourier terms to be added. Defaults to 1
        use_32_bit (bool, optional): Flag to use float32 or int32 to reduce memory. Defaults to False.
    Raises:
        warnings.warn: Raises a warning if max_value is None

    Returns:
        [Tuple[pd.DataFrame, List]]: Returns a tuple of the new dataframe and a list of features which were added
    """
    
    assert len(columns_to_encode) == len(max_values), "'columns_to_encode' and 'max_values' should be same length"
    
    added_features = []
    for column_to_encode, max_value in zip(columns_to_encode, max_values):
        df, features = add_fourier_features(df, column_to_encode, max_value, n_fourier_terms=n_fourier_terms)
        added_features += features
    return df, added_features
