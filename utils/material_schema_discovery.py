""" material_schema_discovery.py 

Schema discovery utilities for hydrogel material workflow. 
""" 
import json 
import numpy as np 
import pandas as pd 

CATEGORICAL_NAME_KEYWORDS = [ 
    "label", 
    "outcome", 
    "class", 
    "category", 
    "categorical", 
    "encoded", 
    "formation", 
    "shape", 
    "stability", 
    "stable", 
    "unstable", 
    "success", 
    "failure", 
    "resuspension", 
    "method", 
    "state" 
    ] 

CLASSIFICATION_TARGET_KEYWORDS = [ 
    "label", 
    "outcome", 
    "class", 
    "formation", 
    "shape", 
    "stability", 
    "resuspension", 
    "encoded" 
    ] 

PHYSICAL_NUMERIC_KEYWORDS = [ 
    "diam", 
    "diameter", 
    "length", 
    "height", 
    "width", 
    "distance", 
    "pressure", 
    "temp", 
    "temperature", 
    "conc", 
    "concentration", 
    "modulus", 
    "viscosity", 
    "volume", 
    "fraction", 
    "stress", 
    "strain", 
    "force", 
    "speed", 
    "time", 
    "frequency", 
    "rate", 
    "pore", 
    "nozzle", 
    "displacement", 
    "plunger", 
    "syringe", 
    "crosslinker", 
    "alginate", 
    "layer", 
    "exposure", 
    "ca2", 
    "na+", 
    "calcium", 
    "sodium", 
    "microgel", 
    "fluid", 
    "solid" 
    ] 

GENERATION_CONDITION_KEYWORDS = [ 
    "frequency", 
    "oscillation strain", 
    "oscillation stress", 
    "storage modulus", 
    "loss modulus", 
    "complex modulus", 
    "g'", 
    'g"', 
    "g’", 
    "g”" 
    ] 

def normalize_column_name(column_name): 
    """ 
    Normalize column name for keyword matching. 
    """ 
    return str(column_name).strip().lower() 

def contains_any_keyword(column_name, keywords): 
    """ 
    Check whether a column name contains any keyword. 
    """ 
    col_lower = normalize_column_name(column_name) 

    return any(keyword.lower() in col_lower for keyword in keywords) 

def is_numeric_series(series): 
    """ 
    Check whether a series is numeric or can be converted to numeric. 
    """ 
    if pd.api.types.is_numeric_dtype(series): 
        return True
     
    converted = pd.to_numeric(series, errors="coerce") 
    non_missing_original = series.notna().sum()

    if non_missing_original == 0: 
        return False 
    
    converted_fraction = converted.notna().sum() / non_missing_original 
    
    return converted_fraction >= 0.95 


def is_integer_like_series(series, tolerance=1e-8): 
    """ 
    Check whether non-missing values are integer-like. 
    """ 
    
    values = pd.to_numeric(series, errors="coerce").dropna().values 
    
    if len(values) == 0: 
        return False 
    
    return np.all(np.abs(values - np.round(values)) < tolerance) 


def get_unique_numeric_values(series): 
    """ 
    Return sorted unique numeric values. 
    """ 
    
    values = pd.to_numeric(series, errors="coerce").dropna().values 
    
    if len(values) == 0: 
        return [] 
    
    unique_values = np.unique(values) 
    
    return sorted(unique_values.tolist())
    
def is_small_consecutive_integer_code(series, max_classes=20): 
    """ 
    Detect numeric class codes. 
    """ 
    if not is_integer_like_series(series): 
        return False 
    
    unique_values = get_unique_numeric_values(series) 
    
    if len(unique_values) < 2: 
        return False 
    
    if len(unique_values) > max_classes: 
        return False 
    
    unique_ints = [int(round(value)) for value in unique_values] 
    
    if min(unique_ints) < 0: 
        return False 
    
    if max(unique_ints) > max_classes: 
        return False 
    
    starts_at_zero = unique_ints == list(range(0, len(unique_ints))) 
    starts_at_one = unique_ints == list(range(1, len(unique_ints) + 1)) 
    
    return starts_at_zero or starts_at_one 

def is_categorical_column(column_name, series): 
    """ 
    Determine whether a column should be treated as categorical. 
    """ 
    
    col_lower = normalize_column_name(column_name) 
    
    if pd.api.types.is_bool_dtype(series): 
        return True 
    
    if pd.api.types.is_categorical_dtype(series): 
        return True 
    
    if pd.api.types.is_object_dtype(series): 
        return True 
    
    if contains_any_keyword(col_lower, CATEGORICAL_NAME_KEYWORDS): 
        return True 
    
    if is_small_consecutive_integer_code(series): 
        return True 
    
    return False 

def is_regression_numeric_column(column_name, series): 
    """ 
    Determine whether a column is numeric for regression/model input. 
    """ 
    
    if not is_numeric_series(series): 
        return False 
    
    if is_categorical_column(column_name, series): 
        return False 
    return True 

def is_classification_target_candidate(column_name, series, max_classes=20): 
    """ 
    Determine if a column is a good classification target candidate. 
    """ 
    n_unique = series.dropna().nunique() 
    
    if n_unique < 2 or n_unique > max_classes: 
        return False 
    
    if contains_any_keyword(column_name, CLASSIFICATION_TARGET_KEYWORDS): 
        return True 
    
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_bool_dtype(series): 
        return True 
    
    if is_small_consecutive_integer_code(series, max_classes=max_classes): 
        return True 
    return False 


def get_column_summary(df): 
    """ 
    Build a column-level schema summary. 
    """ 
    
    rows = [] 
    
    for col in df.columns: 
        series = df[col] 
        non_missing = int(series.notna().sum()) 
        n_unique = int(series.dropna().nunique()) 
        is_numeric = is_numeric_series(series) 
        is_categorical = is_categorical_column(col, series) 
        is_regression_numeric = is_regression_numeric_column(col, series) 
        is_classification_candidate = is_classification_target_candidate(col, series) 
        
        rows.append( 
            { "column": col, 
             "dtype": str(series.dtype), 
             "non_missing": non_missing, 
             "n_unique": n_unique, 
             "is_numeric": bool(is_numeric), 
             "is_categorical": bool(is_categorical), 
             "is_regression_numeric": bool(is_regression_numeric), 
             "is_classification_candidate": bool(is_classification_candidate), 
             "example_values": series.dropna().unique()[:10].tolist() 
             } 
             ) 
        
    return pd.DataFrame(rows) 
    
def suggest_regression_targets(df): 
    """ 
    Suggest continuous regression targets. 
    """ 
    
    candidates = [] 
    
    for col in df.columns: 
        if is_regression_numeric_column(col, df[col]): 
            candidates.append(col) 
    
    return candidates 

def suggest_classification_targets(df): 
    """ 
    Suggest classification target candidates. 
    """ 
    candidates = [] 

    for col in df.columns: 
        if is_classification_target_candidate(col, df[col]): 
            candidates.append(col) 
    
    return candidates 

def suggest_generation_candidates(df): 
    """ 
    Suggest condition/output columns for conditional generation. 
    """ 
    condition_columns = [] 
    output_columns = [] 

    for col in df.columns: 
        col_lower = normalize_column_name(col) 
        is_condition = any( keyword.lower() in col_lower for keyword in GENERATION_CONDITION_KEYWORDS ) 

        if is_condition: 
            condition_columns.append(col) 
        else: 
            output_columns.append(col) 

    return { "condition_columns": condition_columns, "output_columns": output_columns } 

def is_generation_allowed(df=None, dataset_key=None, csv_filename=None):
    """
    Decide whether conditional generation should be allowed.
    """

    if isinstance(df, str) and isinstance(dataset_key, str) and csv_filename is None:
        csv_filename = dataset_key
        dataset_key = df
        df = None

    if isinstance(df, str) and dataset_key is None and csv_filename is None:
        dataset_key = df
        df = None

    if dataset_key == "paa_hydrogel":
        return True

    if dataset_key == "granular_hydrogel":
        if csv_filename is None:
            return False

        filename_lower = str(csv_filename).lower()

        allowed_by_name = (
            "rheo_multioutput_oscstraincurves" in filename_lower
            or "rheo_multioutput_oscstresscurves" in filename_lower
        )

        return bool(allowed_by_name)

    if df is not None and isinstance(df, pd.DataFrame):
        generation_candidates = suggest_generation_candidates(df)

        has_conditions = len(generation_candidates["condition_columns"]) > 0
        has_outputs = len(generation_candidates["output_columns"]) > 0

        return bool(has_conditions and has_outputs)

    return False

def build_schema_report(df): 
    """ 
    Build full schema report for a dataset. 
    """ 
    column_summary_df = get_column_summary(df) 
    numeric_columns = column_summary_df.loc[ column_summary_df["is_numeric"] == True, "column" ].tolist() 
    categorical_columns = column_summary_df.loc[ column_summary_df["is_categorical"] == True, "column" ].tolist() 
    regression_numeric_columns = column_summary_df.loc[ column_summary_df["is_regression_numeric"] == True, "column" ].tolist() 

    classification_candidates = suggest_classification_targets(df) 
    regression_candidates = suggest_regression_targets(df) 
    generation_candidates = suggest_generation_candidates(df) 

    generation_allowed = (
    len(generation_candidates["condition_columns"]) > 0
    and len(generation_candidates["output_columns"]) > 0
        )

    report = {
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "regression_numeric_columns": regression_numeric_columns,
        "suggested_regression_targets": regression_candidates,
        "suggested_classification_targets": classification_candidates,
        "suggested_generation_candidates": generation_candidates,
        "generation_allowed": bool(generation_allowed),
        "column_summary": column_summary_df.to_dict(orient="records")
    
        }

    return report 

def print_schema_report(report): 
    """ 
    Print a readable schema report. 
    """ 
    print("Dataset schema report") 
    print("-" * 70) 
    print(f"Rows: {report['n_rows']}") 
    print(f"Columns: {report['n_columns']}") 

    print("\nNumeric columns:") 
    for col in report["numeric_columns"]: 
        print(f" - {col}") 

    print("\nCategorical columns:") 
    for col in report["categorical_columns"]: 
        print(f" - {col}") 

    print("\nRegression target candidates:") 
    for col in report["suggested_regression_targets"]: 
        print(f" - {col}") 

    print("\nClassification target candidates:") 
    for col in report["suggested_classification_targets"]: 
        print(f" - {col}") 

    print("\nGeneration condition columns:") 
    for col in report["suggested_generation_candidates"]["condition_columns"]: 
        print(f" - {col}") 

    print("\nGeneration output columns:") 
    for col in report["suggested_generation_candidates"]["output_columns"]: 
        print(f" - {col}")


if __name__ == "__main__": 
    from utils.material_dataset_utils import preview_dataset 

    df = preview_dataset("paa_hydrogel") 
    schema_report = build_schema_report(df) 
    print_schema_report(schema_report)