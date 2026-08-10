"""
material_preprocessing.py

Preprocessing utilities for hydrogel datasets.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def auto_select_feature_columns(df, target_columns):
    """
    Select all columns experc targets as input features
    
    """

    return [col for col in df.columns if col not in target_columns]

def encode_input_features(X):
    """
    One hot encoding of categorical features.
    """

    return pd.get_dummies(X, drop_first=False)

def prepare_regression_data(df, target_columns, feature_columns=None, test_size=0.2, random_state=42, scale_X=True, scale_y=False):
    """
    Prepare dataset for regression task with single/multi target support.
    """

    if isinstance(target_columns, str):
        target_columns = [target_columns]

    if feature_columns is None:
        feature_columns = auto_select_feature_columns(df, target_columns)

    X = df[feature_columns].copy()
    y = df[target_columns].copy()

    X = encode_input_features(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    x_scaler = None
    y_scaler = None

    if scale_X:
        x_scaler = StandardScaler()
        X_train = x_scaler.fit_transform(X_train)
        X_test = x_scaler.transform(X_test)

    if scale_y:
        y_scaler = StandardScaler()
        y_train = y_scaler.fit_transform(y_train)
        y_test = y_scaler.transform(y_test)

    return {
        "task" : "regression",
        "feature_columns" : feature_columns,
        "encoded_feature_columns" : list(pd.get_dummies(df[feature_columns], drop_first=False).columns),
        "target_columns" : target_columns,
        "X_train" : X_train,
        "X_test" : X_test,
        "y_train" : y_train,
        "y_test" : y_test,
        "x_scaler" : x_scaler,
        "y_scaler" : y_scaler
    }                

def prepare_classification_data(df, target_column, feature_columns=None, test_size=0.2, random_state=42, scale_X=True):
    """
    Prepare dataset for classification tasks.
    """

    if feature_columns is None:
        feature_columns = auto_select_feature_columns(df, [target_column])

    X = df[feature_columns].copy()
    y = df[target_column].copy()     

    X = encode_input_features(X)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y_encoded if len(set(y_encoded)) > 1 else None)

    x_scaler = None

    if scale_X:
        x_scaler = StandardScaler()
        X_train = x_scaler.fit_transform(X_train)
        X_test = x_scaler.transform(X_test)

    return {
        "task" : "classification",
        "feature_columns" : feature_columns,
        "encoded_feature_columns" : list(pd.get_dummies(df[feature_columns], drop_first=False).columns),
        "target_column" : target_column,
        "class_names" : list(label_encoder.classes_),
        "X_train" : X_train,
        "X_test" : X_test,
        "y_train" : y_train,
        "y_test" : y_test,
        "x_scaler" : x_scaler,
        "label_encoder" : label_encoder
    }   

def prepare_generation_data(df, condition_columns, output_columns, scale_conditions=True, scale_outputs=True):
    """
    Prepare dataset for generative models.
    """

    X_cond = df[condition_columns].copy()
    Y_out = df[output_columns].copy()

    X_cond = encode_input_features(X_cond)
    Y_out = encode_input_features(Y_out)

    cond_scaler = None
    out_scaler = None

    if scale_conditions:
        cond_scaler = StandardScaler()
        X_cond = cond_scaler.fit_transform(X_cond)

    if scale_outputs:
        out_scaler = StandardScaler()
        Y_out = out_scaler.fit_transform(Y_out)

    return {
        "task" : "generation",
        "condition_columns" : condition_columns,
        "output_columns" : output_columns,
        "encoded_condition_columns" : list(pd.get_dummies(df[condition_columns], drop_first=False).columns),
        "encoded_output_columns" : list(pd.get_dummies(df[output_columns], drop_first=False).columns),
        "X_condition" : X_cond,
        "Y_output" : Y_out,
        "condition_scaler" : cond_scaler,
        "output_scaler" : out_scaler
    }


if __name__ == "__main__":
    from utils.material_dataset_utils import preview_dataset
    from utils.material_schema_discovery import build_schema_report

    df = preview_dataset("paa_hydrogel")
    report = build_schema_report(df)

    print("\nTesting regression processing.............")
    reg_target = report["suggested_regression_targets"][:2]

    reg_data = prepare_regression_data(df, target_columns=reg_target)

    print(f"Regression target columns: {reg_data["target_columns"]}")
    print(f"X_train shape: {reg_data["X_train"].shape}")
    print(f"X_test shape: {reg_data["X_test"].shape}")

    print("\nTesting generation preprocessing...............")
    gen_info = report["generation_suggestions"]

    gen_data = prepare_generation_data(df, condition_columns=gen_info["condition_columns"], output_columns=gen_info["output_columns"])

    print(f"Condition columns: {gen_data["condition_columns"]}")
    print(f"Output columns: {gen_data["output_columns"]}")
    print(f"X_condition shape: {gen_info["X_condition"].shape}")
    print(f"Y_output shape: {gen_data["Y_output"].shape}")
