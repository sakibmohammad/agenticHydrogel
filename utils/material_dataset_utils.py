"""
material_dataset_utils.py

Dataset loading utilities for hydrogel workflows.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "paa_hydrogel": {
        "name": "PAA Hydrogel Dataset",
        "directory": REPO_ROOT / "data" / "PAA_Hydrogel_Dataset"
    },
    "granular_hydrogel": {
        "name": "Granular Hydrogel Dataset",
        "directory": REPO_ROOT / "data" / "Granular_Hydrogel_Dataset"
    }
}


def list_available_datasets():
    """
    Return available dataset keys and names.
    """
    return {
        key: value["name"]
        for key, value in DATASETS.items()
    }


def list_csv_files(dataset_key):
    """
    List CSV files inside the selected dataset directory.
    """
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset key: {dataset_key}")

    data_dir = DATASETS[dataset_key]["directory"]
    data_dir.mkdir(parents=True, exist_ok=True)

    return sorted(list(data_dir.glob("*.csv")))


def clean_column_name(column_name):
    """
    Clean one column name.
    """
    column_name = str(column_name)

    column_name = column_name.replace("\ufeff", "")
    column_name = column_name.replace("\xa0", " ")
    column_name = column_name.replace("\n", " ")
    column_name = column_name.replace("\r", " ")
    column_name = column_name.replace("\t", " ")

    column_name = re.sub(r"\s+", " ", column_name)

    return column_name.strip()


def clean_column_names(df):
    """
    Clean column names without modifying the original DataFrame.
    """
    df = df.copy()
    df.columns = [clean_column_name(col) for col in df.columns]

    return df


def is_row_number_column_name(col):
    """
    Check whether the column name looks like an exported row-number/index column.
    """
    col_lower = str(col).strip().lower()

    index_like_names = {
        "index",
        "row",
        "row number",
        "row_number",
        "row id",
        "row_id",
        "unnamed: 0",
    }

    return (
        col_lower in index_like_names
        or col_lower.startswith("unnamed")
    )


def is_index_like_column(series):
    """
    Detect exported row-index columns.
    """
    values = pd.to_numeric(series.dropna(), errors="coerce").reset_index(drop=True)

    if len(values) == 0:
        return False

    if values.isnull().any():
        return False

    if not (values == values.round()).all():
        return False

    values = values.astype(int)

    zero_based = pd.Series(range(len(values)))
    one_based = pd.Series(range(1, len(values) + 1))

    return values.equals(zero_based) or values.equals(one_based)


def is_numeric_unnamed_column(column_name, series):
    """
    Detect numeric unnamed columns in exported granular hydrogel CSV files.
    """
    col_lower = str(column_name).strip().lower()

    if not col_lower.startswith("unnamed"):
        return False

    numeric_values = pd.to_numeric(series, errors="coerce")
    non_missing = series.notna().sum()

    if non_missing == 0:
        return False

    numeric_fraction = numeric_values.notna().sum() / non_missing

    return numeric_fraction >= 0.95


def is_curve_metadata_column(series):
    """
    Detect granular rheology curve-type metadata columns.
    """
    values = (
        series
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
        .tolist()
    )

    if len(values) == 0:
        return False

    allowed_values = {
        "stress",
        "strain",
        "flow"
    }

    return set(values).issubset(allowed_values)


def is_stress_strain_metadata_column(series):
    """
    Backward-compatible wrapper.
    """
    return is_curve_metadata_column(series)


def is_category_metadata_column(column_name, series):
    """
    Detect renamed Category_* metadata columns.
    """
    col_lower = str(column_name).strip().lower()

    if not col_lower.startswith("category_"):
        return False

    return is_curve_metadata_column(series)


def detect_granular_metadata_columns(df):
    """
    Detect granular hydrogel metadata columns to remove.
    """
    metadata_columns = []

    for col in df.columns:
        if is_numeric_unnamed_column(col, df[col]):
            metadata_columns.append(col)

        elif is_row_number_column_name(col) and is_index_like_column(df[col]):
            metadata_columns.append(col)

        elif is_curve_metadata_column(df[col]):
            metadata_columns.append(col)

        elif is_category_metadata_column(col, df[col]):
            metadata_columns.append(col)

    return list(dict.fromkeys(metadata_columns))


def detect_row_number_columns(df):
    """
    Detect columns that are likely row-number or index columns.
    """
    row_number_columns = []

    for col in df.columns:
        if is_row_number_column_name(col) and is_index_like_column(df[col]):
            row_number_columns.append(col)

    return row_number_columns


def rename_unnamed_categorical_columns(df):
    """
    Rename non-numeric 'Unnamed' columns.
    """
    df = df.copy()
    rename_dict = {}
    category_count = 0

    for col in df.columns:
        col_lower = str(col).strip().lower()

        if col_lower.startswith("unnamed") and not is_numeric_unnamed_column(col, df[col]):
            rename_dict[col] = f"Category_{category_count}"
            category_count += 1

    if rename_dict:
        print("Renaming unnamed categorical columns:")
        for old_name, new_name in rename_dict.items():
            print(f" - {old_name} -> {new_name}")

        df = df.rename(columns=rename_dict)

    return df


def drop_unwanted_columns(
        df,
        columns_to_drop=None,
        auto_drop_row_numbers=True
):
    """
    Drop unwanted columns from the DataFrame, including detected metadata columns.
    """
    df = df.copy()

    detected_columns = []

    if auto_drop_row_numbers:
        detected_columns.extend(detect_row_number_columns(df))
        detected_columns.extend(detect_granular_metadata_columns(df))

    manual_columns = columns_to_drop or []

    final_columns_to_drop = list(dict.fromkeys(detected_columns + manual_columns))
    final_columns_to_drop = [
        col for col in final_columns_to_drop
        if col in df.columns
    ]

    if final_columns_to_drop:
        print("Dropping columns:")
        for col in final_columns_to_drop:
            print(f" - {col}")

        df = df.drop(columns=final_columns_to_drop)

    return df


def load_material_dataset(
        dataset_key,
        csv_filename=None,
        columns_to_drop=None,
        auto_drop_row_numbers=True
):
    """
    Load a CSV dataset file.
    """
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset key: {dataset_key}")

    data_dir = DATASETS[dataset_key]["directory"]
    csv_files = list_csv_files(dataset_key)

    if len(csv_files) == 0:
        raise FileNotFoundError(
            f"No CSV files found in dataset directory: {data_dir}"
        )

    if csv_filename is None:
        csv_path = csv_files[0]
    else:
        csv_path = data_dir / csv_filename

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    df = clean_column_names(df)

    df = rename_unnamed_categorical_columns(df)

    df = drop_unwanted_columns(
        df,
        columns_to_drop=columns_to_drop,
        auto_drop_row_numbers=auto_drop_row_numbers
    )

    df = clean_column_names(df)

    return df, csv_path


def preview_dataset(
        dataset_key,
        csv_filename=None,
        n_rows=5,
        columns_to_drop=None,
        auto_drop_row_numbers=True
):
    """
    Load and preview the dataset.
    """
    df, csv_path = load_material_dataset(
        dataset_key,
        csv_filename=csv_filename,
        columns_to_drop=columns_to_drop,
        auto_drop_row_numbers=auto_drop_row_numbers
    )

    print(f"\nLoaded dataset from: {csv_path}")
    print(f"Shape after cleaning: {df.shape}")

    print("\nColumns:")
    for col in df.columns:
        print(f" - {col}")

    print("\nPreview:")
    print(df.head(n_rows))

    print("\nMissing values per column:")
    print(df.isnull().sum())

    return df


if __name__ == "__main__":
    print("Available datasets:")
    print(list_available_datasets())

    preview_dataset("paa_hydrogel")