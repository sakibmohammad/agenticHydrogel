"""
generation_postprocessing.py

Postprocessing utilities for generated hydrogel material data.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIR = REPO_ROOT / "outputs" / "generated_data"
REPORTS_DIR = REPO_ROOT / "outputs" / "reports"

GENERATED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def is_encoded_column(column_name):
    """
    Detect encoded categorical columns.
    """
    col_lower = str(column_name).strip().lower()

    return (
        col_lower.endswith("_encoded")
        or "_encoded_" in col_lower
        or col_lower.startswith("encoded_")
    )


def is_likely_nonnegative_column(column_name):
    """
    Detect columns that should generally be nonnegative.
    """
    col_lower = str(column_name).strip().lower()

    nonnegative_keywords = [
        "conc",
        "concentration",
        "size",
        "height",
        "time",
        "exposure",
        "frequency",
        "strain",
        "stress",
        "modulus",
        "volume fraction",
        "ca2",
        "na+",
        "fluid phase",
        "loss",
        "storage",
        "complex"
    ]

    return any(keyword in col_lower for keyword in nonnegative_keywords)


def get_real_column_bounds(real_df, columns):
    """
    Get min/max values for selected real-data columns.
    """
    bounds = {}

    for col in columns:
        if col not in real_df.columns:
            continue

        values = pd.to_numeric(real_df[col], errors="coerce").dropna()

        if len(values) == 0:
            continue

        bounds[col] = {
            "min": float(values.min()),
            "max": float(values.max())
        }

    return bounds


def apply_custom_bounds(df, custom_bounds):
    """
    Apply user-provided column bounds.
    """
    df = df.copy()
    applied = {}

    if custom_bounds is None:
        return df, applied

    for col, bounds in custom_bounds.items():
        if col not in df.columns:
            continue

        min_val = bounds.get("min", None)
        max_val = bounds.get("max", None)

        before_min = pd.to_numeric(df[col], errors="coerce").min()
        before_max = pd.to_numeric(df[col], errors="coerce").max()

        if min_val is not None:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=min_val)

        if max_val is not None:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(upper=max_val)

        after_min = pd.to_numeric(df[col], errors="coerce").min()
        after_max = pd.to_numeric(df[col], errors="coerce").max()

        applied[col] = {
            "custom_min": min_val,
            "custom_max": max_val,
            "before_min": None if pd.isna(before_min) else float(before_min),
            "before_max": None if pd.isna(before_max) else float(before_max),
            "after_min": None if pd.isna(after_min) else float(after_min),
            "after_max": None if pd.isna(after_max) else float(after_max)
        }

    return df, applied


def clip_to_real_range(df, real_df, columns):
    """
    Clip generated columns to the real training-data range.
    """
    df = df.copy()
    report = {}

    bounds = get_real_column_bounds(real_df, columns)

    for col, bound in bounds.items():
        if col not in df.columns:
            continue

        values = pd.to_numeric(df[col], errors="coerce")

        before_below = int((values < bound["min"]).sum())
        before_above = int((values > bound["max"]).sum())

        df[col] = values.clip(bound["min"], bound["max"])

        report[col] = {
            "real_min": bound["min"],
            "real_max": bound["max"],
            "values_below_before_clip": before_below,
            "values_above_before_clip": before_above
        }

    return df, report


def enforce_nonnegative_values(df, columns=None):
    """
    Clip likely nonnegative columns at zero.
    """
    df = df.copy()
    report = {}

    if columns is None:
        columns = [
            col for col in df.columns
            if is_likely_nonnegative_column(col)
        ]

    for col in columns:
        if col not in df.columns:
            continue

        values = pd.to_numeric(df[col], errors="coerce")

        negative_count = int((values < 0).sum())

        if negative_count > 0:
            df[col] = values.clip(lower=0)

        report[col] = {
            "negative_values_before_clip": negative_count
        }

    return df, report


def round_selected_columns(df, columns):
    """
    Round selected columns to nearest integer.
    """
    df = df.copy()
    report = {}

    for col in columns:
        if col not in df.columns:
            continue

        before_unique_preview = (
            pd.to_numeric(df[col], errors="coerce")
            .dropna()
            .head(10)
            .tolist()
        )

        df[col] = pd.to_numeric(df[col], errors="coerce").round()

        after_unique_preview = (
            pd.to_numeric(df[col], errors="coerce")
            .dropna()
            .head(10)
            .tolist()
        )

        report[col] = {
            "before_preview": before_unique_preview,
            "after_preview": after_unique_preview
        }

    return df, report


def round_encoded_columns(df):
    """
    Round all encoded categorical columns.
    """
    encoded_columns = [
        col for col in df.columns
        if is_encoded_column(col)
    ]

    return round_selected_columns(df, encoded_columns)


def summarize_postprocessing(original_df, postprocessed_df, output_columns):
    """
    Summarize changes before and after postprocessing.
    """
    summary = {}

    for col in output_columns:
        if col not in original_df.columns or col not in postprocessed_df.columns:
            continue

        original_values = pd.to_numeric(original_df[col], errors="coerce")
        post_values = pd.to_numeric(postprocessed_df[col], errors="coerce")

        changed_count = int(
            (
                np.abs(original_values.fillna(0) - post_values.fillna(0)) > 1e-12
            ).sum()
        )

        summary[col] = {
            "changed_count": changed_count,
            "total_count": int(len(postprocessed_df)),
            "changed_fraction": float(changed_count / max(1, len(postprocessed_df))),
            "post_min": None if post_values.dropna().empty else float(post_values.min()),
            "post_max": None if post_values.dropna().empty else float(post_values.max()),
            "post_mean": None if post_values.dropna().empty else float(post_values.mean())
        }

    return summary


def postprocess_generated_dataframe(
    synthetic_df,
    real_df=None,
    output_columns=None,
    condition_columns=None,
    clip_real_range=True,
    enforce_nonnegative=True,
    round_encoded=True,
    integer_columns=None,
    custom_bounds=None
):
    """
    Main postprocessing function for generated synthetic data.
    """
    original_df = synthetic_df.copy()
    postprocessed_df = synthetic_df.copy()

    if output_columns is None:
        if condition_columns is None:
            output_columns = postprocessed_df.columns.tolist()
        else:
            output_columns = [
                col for col in postprocessed_df.columns
                if col not in condition_columns
            ]

    report = {
        "output_columns": output_columns,
        "condition_columns": condition_columns,
        "clip_real_range": clip_real_range,
        "enforce_nonnegative": enforce_nonnegative,
        "round_encoded": round_encoded,
        "steps": {}
    }

    if enforce_nonnegative:
        nonnegative_columns = [
            col for col in output_columns
            if is_likely_nonnegative_column(col)
        ]

        postprocessed_df, nonnegative_report = enforce_nonnegative_values(
            postprocessed_df,
            columns=nonnegative_columns
        )

        report["steps"]["nonnegative_clipping"] = nonnegative_report

    if clip_real_range and real_df is not None:
        postprocessed_df, range_report = clip_to_real_range(
            postprocessed_df,
            real_df=real_df,
            columns=output_columns
        )

        report["steps"]["real_range_clipping"] = range_report

    postprocessed_df, custom_report = apply_custom_bounds(
        postprocessed_df,
        custom_bounds=custom_bounds
    )

    report["steps"]["custom_bounds"] = custom_report

    if round_encoded:
        encoded_columns = [
            col for col in output_columns
            if is_encoded_column(col)
        ]

        postprocessed_df, encoded_round_report = round_selected_columns(
            postprocessed_df,
            encoded_columns
        )

        report["steps"]["encoded_rounding"] = encoded_round_report

    if integer_columns is not None:
        postprocessed_df, integer_round_report = round_selected_columns(
            postprocessed_df,
            integer_columns
        )

        report["steps"]["integer_rounding"] = integer_round_report

    report["summary"] = summarize_postprocessing(
        original_df=original_df,
        postprocessed_df=postprocessed_df,
        output_columns=output_columns
    )

    return postprocessed_df, report


def save_postprocessed_generation(
    synthetic_df,
    real_df=None,
    output_columns=None,
    condition_columns=None,
    run_name="postprocessed_generation",
    clip_real_range=True,
    enforce_nonnegative=True,
    round_encoded=True,
    integer_columns=None,
    custom_bounds=None
):
    """
    Postprocess generated data and save CSV/report.
    """
    postprocessed_df, report = postprocess_generated_dataframe(
        synthetic_df=synthetic_df,
        real_df=real_df,
        output_columns=output_columns,
        condition_columns=condition_columns,
        clip_real_range=clip_real_range,
        enforce_nonnegative=enforce_nonnegative,
        round_encoded=round_encoded,
        integer_columns=integer_columns,
        custom_bounds=custom_bounds
    )

    csv_path = GENERATED_DIR / f"{run_name}_postprocessed.csv"
    report_path = REPORTS_DIR / f"{run_name}_postprocessing_report.json"

    postprocessed_df.to_csv(csv_path, index=False)

    report["postprocessed_csv_path"] = str(csv_path)

    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    return {
        "postprocessed_df": postprocessed_df,
        "postprocessed_csv_path": str(csv_path),
        "postprocessing_report_path": str(report_path),
        "postprocessing_report": report
    }


if __name__ == "__main__":
    from utils.material_dataset_utils import preview_dataset

    real_df = preview_dataset("paa_hydrogel")

    synthetic_df = real_df.sample(10, random_state=42).copy()
    synthetic_df["Acrylamide Conc. %"] = synthetic_df["Acrylamide Conc. %"] * -1

    result = save_postprocessed_generation(
        synthetic_df=synthetic_df,
        real_df=real_df,
        output_columns=[
            "Acrylamide Conc. %",
            "Bis-acrylamide conc %",
            "Photo-initiator conc. %",
            "Layer Height. (micron)",
            "Bottom Layer exposure time (s)",
            "Exposure time (s)"
        ],
        condition_columns=[
            "Frequency (Hz)",
            "Storage modulus (Pa)",
            "Loss modulus (Pa)"
        ],
        run_name="postprocessing_test"
    )

    print(f"Postprocessed CSV: {result['postprocessed_csv_path']}")
    print(f"Report: {result['postprocessing_report_path']}")