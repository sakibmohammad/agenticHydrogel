"""
material_visualization.py
Visualization utilities for hydrogel workflows.

This creates basic plots for regression, classification and generation data exploration.

Plots are save to: output/plots/
"""

from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = REPO_ROOT/"outputs"/"plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def sanitize_filename(text):
    """
    Convert column names or plot titles into safe filenames.
    """
    text = str(text)
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "_", text)

    return text[:100]

def save_current_plot(filename):
    """
    Save the current matplotlib figure.
    """

    save_path = PLOTS_DIR / filename
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return save_path

def plot_correlation_heatmap(df, filename="correleation_heatmap.png"):
    """
    Plot correlation heatmap for numeric columns.
    """

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        print("Not enough numeric columns for correlation heatmap")
        return None
    
    corr = numeric_df.corr()

    plt.figure(figsize=(10, 8))
    plt.imshow(corr, aspect="auto")
    plt.colorbar(label="Pearson correlation")

    plt.xticks(
        range(len(corr.columns)),
        corr.columns,
        rotation=90,
        fontsize=8
    )

    plt.yticks(
        range(len(corr.index)),
        corr.index,
        fontsize=8
    )

    plt.title("Correlation Heatmap")
    
    if corr.shape[0]<=10:
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                plt.text(
                    j,
                    i,
                    f"{corr.iloc[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6
                )

    return save_current_plot(filename)

def plot_target_distribution(df, target_column):
    """
    Plot distribution of a target column.

    Numeric target: histogram
    Categorical target: bar plot
    """            

    if target_column not in df.columns:
        raise ValueError(f"Target column not found: {target_column}")
    
    filename = f"target_distribution_{sanitize_filename(target_column)}.png"

    plt.figure(figsize=(8, 5))

    if pd.api.types.is_numeric_dtype(df[target_column]):
        plt.hist(df[target_column].dropna(), bins=30)
        plt.xlabel(target_column)
        plt.ylabel("Count")
        plt.title(f"Distribution of {target_column}")
    else:
        counts = df[target_column].value_counts()
        plt.bar(counts.index.astype(str), counts.values)
        plt.xlabel(target_column)
        plt.ylabel("Count")
        plt.title(f"Class distribution of {target_column}")
        plt.xticks(rotation=45, ha="right")

    return save_current_plot(filename)        

def plot_feature_vs_target(df, feature_column, target_column):
    """
    Plot one feature against one target.
    Numeric feature and numeric target: scatter plot
    Categorical feature and numeric target: boxplot like group scatter
    Any categorical target: feature distibution by class
    """

    if feature_column not in df.columns:
        raise ValueError(f"Feature column not found: {feature_column}")

    if target_column not in df.columns:
        raise ValueError(f"Target column not found: {target_column}")

    filename = (
        f"feature_vs_target_"
        f"{sanitize_filename(feature_column)}_vs_"
        f"{sanitize_filename(target_column)}.png"
    ) 

    plt.figure(figsize=(7, 5))

    x = df[feature_column]
    y = df[target_column]

    if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
        plt.scatter(x, y, alpha=0.7)
        plt.xlabel(feature_column)
        plt.ylabel(target_column)
        plt.title(f"{feature_column} vs {target_column}")
    else:
        x_codes = x.astype("category").cat.codes if not pd.api.types.is_numeric_dtype(x) else x
        y_codes = y.astype("category").cat.codes if not pd.api.types.is_numeric_dtype(y) else y    

        plt.scatter(x_codes, y_codes, alpha=0.7)
        plt.xlabel(feature_column)
        plt.ylabel(target_column)
        plt.title(f"{feature_column} vs {target_column}")

    return save_current_plot(filename)

def make_regression_plots(df, target_columns, max_features=5):
    """
    Make standard regression plots.
    """

    if isinstance(target_columns, str):
        target_columns = [target_columns]

    saved_paths = []

    heatmap_path = plot_correlation_heatmap(df)

    if heatmap_path is not None:
        saved_paths.append(heatmap_path)

    for target in target_columns:
        saved_paths.append(plot_target_distribution(df, target))

    numeric_features = [
        col for col in df.select_dtypes(include="number").columns if col not in target_columns
    ]             

    for target in target_columns:
        for feature in numeric_features[:max_features]:
            saved_paths.append(plot_feature_vs_target(df, feature, target))

    return saved_paths

def make_classification_plots(df, target_column, max_features=5):
    """
    Make standard classification plots.
    """        
    saved_paths = []

    heatmap_path = plot_correlation_heatmap(df)
    if heatmap_path is not None:
        saved_paths.append(heatmap_path)

    numeric_features = [
        col for col in df.select_dtypes(include="number").columns if col != target_column
    ]    

    for feature in numeric_features[:max_features]:
        saved_paths.append(plot_feature_vs_target(df, feature, target_column))

    return saved_paths    

def make_generation_plots(df, condition_columns, output_columns, max_outputs=5):
    """
    Make simple generation plots.
    """

    saved_paths = []

    heatmap_path = plot_correlation_heatmap(df)

    if heatmap_path is not None:
        saved_paths.append(heatmap_path)

    for col in condition_columns:
        saved_paths.append(plot_target_distribution(df, col))

    for col in output_columns[:max_outputs]:
        saved_paths.append(plot_target_distribution(df, col))

    return saved_paths

def make_basic_plots(df, task, target_columns=None, target_column=None, condition_columns=None, output_columns=None):
    """
    Main plotting function used by agents.
    Parameters
    -----------
    task: str
    "regression", "classification", or "generation"
    """            

    task = task.lower().strip()

    if task == "regression":
        if target_columns is None:
            raise ValueError("target_columns is required for regression plots")
        return make_regression_plots(df, target_columns)
    
    if task =="classification":
        if target_column is None:
            raise ValueError("target_column is required for classification plots")
        return make_classification_plots(df, target_column)
    
    if task == "generation":
        if condition_columns is None or output_columns is None:
            raise ValueError("condition_columns and output_columns are required for generation plots")
        return make_generation_plots(df, condition_columns, output_columns)
    
    raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    from utils.material_dataset_utils import preview_dataset
    from utils.material_schema_discovery import build_schema_report

    df = preview_dataset("paa_hydrogel")
    report = build_schema_report(df)

    target_columns = report["suggested_regression_targets"][:2]
    print("Creating regression plots...............")

    saved = make_basic_plots(
        df,
        task="regression",
        target_columns=target_columns
    )

    print("Saved plots: ")
    for path in saved:
        print(f" - {path}")

