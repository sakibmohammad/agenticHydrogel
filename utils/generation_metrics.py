"""
generation_metrics.py

Evaluation utilities for conditional tabular generation.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

from scipy.stats import ks_2samp
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, mean_absolute_error


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "outputs" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def align_numeric_columns(real_df, synthetic_df, columns):
    """
    Select shared numeric columns from real and synthetic dataframes.
    """

    shared_columns = [
        col for col in columns
        if col in real_df.columns and col in synthetic_df.columns
    ]

    real_numeric = real_df[shared_columns].apply(pd.to_numeric, errors="coerce")
    synth_numeric = synthetic_df[shared_columns].apply(pd.to_numeric, errors="coerce")

    return real_numeric, synth_numeric, shared_columns


def allign_numeric_columns(real_df, synthetic_df, columns):
    """
    Select shared numeric columns from real and synthetic dataframes.
    """

    return align_numeric_columns(real_df, synthetic_df, columns)


def ks_distance_report(real_df, synthetic_df, output_columns):
    """
    Compute KS distance for each output column.
    """

    report = {}

    for col in output_columns:
        if col not in real_df.columns or col not in synthetic_df.columns:
            continue

        real_values = pd.to_numeric(real_df[col], errors="coerce").dropna()
        synth_values = pd.to_numeric(synthetic_df[col], errors="coerce").dropna()

        if len(real_values) == 0 or len(synth_values) == 0:
            continue

        stat, p_value = ks_2samp(real_values, synth_values)

        report[col] = {
            "ks_statistic": float(stat),
            "p_value": float(p_value),
            "real_mean": float(real_values.mean()),
            "synthetic_mean": float(synth_values.mean()),
            "real_std": float(real_values.std()),
            "synthetic_std": float(synth_values.std())
        }

    if len(report) == 0:
        mean_ks = None
    else:
        mean_ks = float(
            np.mean([item["ks_statistic"] for item in report.values()])
        )

    return {
        "per_column": report,
        "mean_ks_statistic": mean_ks
    }


def correlation_frobenius_report(real_df, synthetic_df, columns):
    """
    Compare real vs synthetic correlation matrices using Frobenius norm.
    """

    real_numeric, synth_numeric, shared_columns = align_numeric_columns(
        real_df,
        synthetic_df,
        columns
    )

    if len(shared_columns) < 2:
        return {
            "frobenius_norm": None,
            "message": "Not enough shared numeric columns."
        }

    real_numeric = real_numeric.dropna()
    synth_numeric = synth_numeric.dropna()

    if len(real_numeric) < 2 or len(synth_numeric) < 2:
        return {
            "frobenius_norm": None,
            "columns_used": shared_columns,
            "message": "Not enough valid rows after dropping missing values."
        }

    real_corr = real_numeric.corr().fillna(0).values
    synth_corr = synth_numeric.corr().fillna(0).values

    frobenius_norm = np.linalg.norm(real_corr - synth_corr, ord="fro")

    return {
        "columns_used": shared_columns,
        "frobenius_norm": float(frobenius_norm)
    }


def encode_condition_columns_for_nn(real_df, synthetic_df, condition_columns):
    """
    Encode condition columns for nearest-neighbor matching.

    Numeric columns are kept numeric.
    Categorical columns are one-hot encoded.
    Real and synthetic columns are aligned after encoding.
    """

    real_conditions = real_df[condition_columns].copy()
    synth_conditions = synthetic_df[condition_columns].copy()

    combined = pd.concat(
        [real_conditions, synth_conditions],
        axis=0,
        ignore_index=True
    )

    combined_encoded = pd.get_dummies(combined, drop_first=False)
    combined_encoded = combined_encoded.apply(pd.to_numeric, errors="coerce")

    real_encoded = combined_encoded.iloc[:len(real_conditions)].reset_index(drop=True)
    synth_encoded = combined_encoded.iloc[len(real_conditions):].reset_index(drop=True)

    return real_encoded, synth_encoded


def conditional_nearest_neighbor_report(
        real_df,
        synthetic_df,
        condition_columns,
        output_columns,
        n_neighbors=1
):
    """
    Evaluate conditional fidelity.
    """

    condition_columns = [
        col for col in condition_columns
        if col in real_df.columns and col in synthetic_df.columns
    ]

    output_columns = [
        col for col in output_columns
        if col in real_df.columns and col in synthetic_df.columns
    ]

    if len(condition_columns) == 0:
        return {
            "conditional_nn_rmse_scaled": None,
            "conditional_nn_mae_scaled": None,
            "message": "No shared condition columns found."
        }

    if len(output_columns) == 0:
        return {
            "conditional_nn_rmse_scaled": None,
            "conditional_nn_mae_scaled": None,
            "message": "No shared output columns found."
        }

    real_conditions, synth_conditions = encode_condition_columns_for_nn(
        real_df,
        synthetic_df,
        condition_columns
    )

    real_outputs = real_df[output_columns].apply(pd.to_numeric, errors="coerce")
    synth_outputs = synthetic_df[output_columns].apply(pd.to_numeric, errors="coerce")

    real_valid = pd.concat([real_conditions, real_outputs.reset_index(drop=True)], axis=1)
    synth_valid = pd.concat([synth_conditions, synth_outputs.reset_index(drop=True)], axis=1)

    real_valid = real_valid.dropna().reset_index(drop=True)
    synth_valid = synth_valid.dropna().reset_index(drop=True)

    if len(real_valid) == 0 or len(synth_valid) == 0:
        return {
            "conditional_nn_rmse_scaled": None,
            "conditional_nn_mae_scaled": None,
            "message": "No valid rows remained after numeric conversion and missing-value removal."
        }

    real_conditions = real_valid[real_conditions.columns]
    synth_conditions = synth_valid[synth_conditions.columns]

    real_outputs = real_valid[output_columns]
    synth_outputs = synth_valid[output_columns]

    if len(real_conditions) < n_neighbors:
        n_neighbors = len(real_conditions)

    condition_scaler = StandardScaler()
    real_conditions_scaled = condition_scaler.fit_transform(real_conditions)
    synth_conditions_scaled = condition_scaler.transform(synth_conditions)

    output_scaler = StandardScaler()
    real_outputs_scaled = output_scaler.fit_transform(real_outputs)
    synth_outputs_scaled = output_scaler.transform(synth_outputs)

    nn_model = NearestNeighbors(n_neighbors=n_neighbors)
    nn_model.fit(real_conditions_scaled)

    distances, indices = nn_model.kneighbors(synth_conditions_scaled)

    matched_real_outputs = []

    for neighbor_indices in indices:
        neighbor_outputs = real_outputs_scaled[neighbor_indices]
        matched_real_outputs.append(neighbor_outputs.mean(axis=0))

    matched_real_outputs = np.vstack(matched_real_outputs)

    rmse = np.sqrt(mean_squared_error(matched_real_outputs, synth_outputs_scaled))
    mae = mean_absolute_error(matched_real_outputs, synth_outputs_scaled)

    return {
        "condition_columns_used": condition_columns,
        "output_columns_used": output_columns,
        "n_neighbors": int(n_neighbors),
        "conditional_nn_rmse_scaled": float(rmse),
        "conditional_nn_mae_scaled": float(mae),
        "mean_condition_distance_scaled": float(np.mean(distances))
    }


def range_violation_report(real_df, synthetic_df, output_columns):
    """
    Count how many synthetic values fall outside the real training range.
    """

    report = {}

    for col in output_columns:
        if col not in real_df.columns or col not in synthetic_df.columns:
            continue

        real_values = pd.to_numeric(real_df[col], errors="coerce").dropna()
        synth_values = pd.to_numeric(synthetic_df[col], errors="coerce").dropna()

        if len(real_values) == 0 or len(synth_values) == 0:
            continue

        min_val = real_values.min()
        max_val = real_values.max()

        below = int((synth_values < min_val).sum())
        above = int((synth_values > max_val).sum())
        total = int(len(synth_values))

        report[col] = {
            "real_min": float(min_val),
            "real_max": float(max_val),
            "below_range_count": below,
            "above_range_count": above,
            "total_count": total,
            "violation_fraction": float((below + above) / total)
        }

    return report


def evaluate_conditional_generation(
    real_df,
    synthetic_df,
    condition_columns,
    output_columns,
    run_name="generation_evaluation",
    save_report=True
):
    """
    Run all conditional generation evaluation metrics.
    """

    ks_report = ks_distance_report(
        real_df=real_df,
        synthetic_df=synthetic_df,
        output_columns=output_columns
    )

    corr_report = correlation_frobenius_report(
        real_df=real_df,
        synthetic_df=synthetic_df,
        columns=output_columns
    )

    conditional_nn = conditional_nearest_neighbor_report(
        real_df=real_df,
        synthetic_df=synthetic_df,
        condition_columns=condition_columns,
        output_columns=output_columns,
        n_neighbors=1
    )

    range_report = range_violation_report(
        real_df=real_df,
        synthetic_df=synthetic_df,
        output_columns=output_columns
    )

    report = {
        "run_name": run_name,
        "condition_columns": condition_columns,
        "output_columns": output_columns,
        "ks_distance": ks_report,
        "correlation_frobenius": corr_report,
        "conditional_nearest_neighbor": conditional_nn,
        "range_violations": range_report
    }

    if save_report:
        report_path = REPORTS_DIR / f"{run_name}_generation_metrics.json"

        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)

        report["report_path"] = str(report_path)

    return report


if __name__ == "__main__":
    from utils.material_dataset_utils import preview_dataset
    from utils.material_schema_discovery import build_schema_report
    from utils.generation_cvae import run_cvae_generation

    df = preview_dataset("paa_hydrogel")
    schema_report = build_schema_report(df)

    condition_columns = schema_report["suggested_generation_candidates"]["condition_columns"]
    output_columns = schema_report["suggested_generation_candidates"]["output_columns"]

    generation_result = run_cvae_generation(
        df,
        condition_columns=condition_columns,
        output_columns=output_columns,
        run_name="paa_hydrogel_cvae_metric_test",
        epochs=10,
        patience=3
    )

    metrics = evaluate_conditional_generation(
        real_df=df,
        synthetic_df=generation_result["synthetic_df"],
        condition_columns=condition_columns,
        output_columns=output_columns,
        run_name="paa_hydrogel_cvae_metric_test"
    )

    print(metrics)