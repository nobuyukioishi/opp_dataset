import pandas as pd
from zipfile import ZipFile


def get_file_list_in_zip(zip_path: str):
    with ZipFile(zip_path, 'r') as zf:
        return [path for path in zf.namelist()]


def replace_nans(df: pd.DataFrame, fill_val: float = 0.0) -> pd.DataFrame:
    """ Replace nan values in the given dataframe with linear interpolation. The first and last nan values are filled \
    with zeros.
    :param df:
    :param fill_val: float value to fill the nan values
    :return:
    """

    # Replace trailing NaN values with 0.0
    for column in df.columns:
        first_valid_idx = df[column].first_valid_index()
        last_valid_idx = df[column].last_valid_index()
        df.loc[:first_valid_idx, column] = df.loc[:first_valid_idx, column].fillna(fill_val)
        df.loc[last_valid_idx:, column] = df.loc[last_valid_idx:, column].fillna(fill_val)

    # Select sensor values (float64)
    data_cols = df.select_dtypes(include=['float64']).columns
    df[data_cols] = df[data_cols].interpolate(method='linear', axis=0)

    return df
