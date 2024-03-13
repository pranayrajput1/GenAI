import pandas as pd

def pre_process(df, target_column):
    """
    Splits the DataFrame `df` into features (X) and target (y).

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        target_column (str): The name of the target column.

    Returns:
        pd.DataFrame, pd.Series: Features (X) and target (y).
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y
