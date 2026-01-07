import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def prepare_features(df, target_column):
    """
    Splits features and target, and encodes categorical variables
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=["object"]).columns
    numeric_cols = X.select_dtypes(exclude=["object"]).columns

    # One-hot encode categorical features
    X_encoded = pd.get_dummies(
        X,
        columns=categorical_cols,
        drop_first=True
    )

    return X_encoded, y

