"""
material_report_writer.py

Markdown report writer for hydrogel material workflows.

"""

from pathlib import Path
from datetime import datetime
import json
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "outputs" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path):
    """
    Load JSON file.
    """
    path = Path(path)

    if not path.is_absolute():
        path = REPO_ROOT / path

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_get(dictionary, keys, default=None):
    """
    Safely get nested dictionary value.
    """
    current = dictionary

    for key in keys:
        if not isinstance(current, dict):
            return default

        if key not in current:
            return default

        current = current[key]

    return current


def format_value(value, digits=4):
    """
    Format numeric values for reports.
    """
    if value is None:
        return "N/A"

    if isinstance(value, float):
        if np.isnan(value):
            return "N/A"
        return f"{value:.{digits}f}"

    if isinstance(value, int):
        return str(value)

    return str(value)


def markdown_table(headers, rows):
    """
    Create Markdown table.
    """
    lines = []

    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")

    return "\n".join(lines)


def build_header(title):
    """
    Build report header.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"# {title} Generated on: {timestamp}"
    

def summarize_plan(plan):
    """
    Summarize workflow plan.
    """
    task = plan.get("task", "unknown")

    lines = []
    lines.append("## Workflow Plan\n")
    lines.append(f"- Dataset key: `{plan.get('dataset_key')}`")
    lines.append(f"- CSV file: `{plan.get('csv_filename')}`")
    lines.append(f"- Task: `{task}`")

    if task == "regression":
        lines.append("\n### Regression Target Columns\n")
        for col in plan.get("target_columns", []):
            lines.append(f"- {col}")

    elif task == "classification":
        lines.append("\n### Classification Target Column\n")
        lines.append(f"- {plan.get('target_column')}")

    elif task == "generation":
        lines.append("\n### Generation Condition Columns\n")
        for col in plan.get("condition_columns", []):
            lines.append(f"- {col}")

        lines.append("\n### Generation Output Columns\n")
        for col in plan.get("output_columns", []):
            lines.append(f"- {col}")

    lines.append("\n### Workflow Steps\n")
    for step in plan.get("workflow", []):
        lines.append(f"- {step}")

    return "\n".join(lines) + "\n"


def summarize_regression_result(summary):
    """
    Summarize regression result.
    """
    lines = []
    lines.append("## Regression Results\n")

    metrics = summary.get("metrics", {})

    rows = [
        ["R2", format_value(metrics.get("r2"))],
        ["MAE", format_value(metrics.get("mae"))],
        ["MSE", format_value(metrics.get("mse"))],
        ["RMSE", format_value(metrics.get("rmse"))],
        ["MAPE (%)", format_value(metrics.get("mape_percent"))],
    ]

    lines.append(markdown_table(["Metric", "Value"], rows))

    lines.append("\n### Output Files\n")
    lines.append(f"- Model path: `{summary.get('model_path', 'N/A')}`")
    lines.append(f"- Analysis report: `{summary.get('analysis_report_path', 'N/A')}`")

    return "\n".join(lines) + "\n"


def summarize_classification_result(summary):
    """
    Summarize classification result.
    """
    lines = []
    lines.append("## Classification Results\n")

    metrics = summary.get("metrics", {})

    rows = [
        ["Accuracy", format_value(metrics.get("accuracy"))],
        ["Precision macro", format_value(metrics.get("precision_macro"))],
        ["Recall macro", format_value(metrics.get("recall_macro"))],
        ["F1 macro", format_value(metrics.get("f1_macro"))],
    ]

    lines.append(markdown_table(["Metric", "Value"], rows))

    lines.append("\n### Output Files\n")
    lines.append(f"- Model path: `{summary.get('model_path', 'N/A')}`")
    lines.append(f"- Analysis report: `{summary.get('analysis_report_path', 'N/A')}`")

    if "classification_report" in metrics:
        lines.append("\n### Classification Report\n")
        lines.append("```text")
        lines.append(metrics["classification_report"])
        lines.append("```")

    return "\n".join(lines) + "\n"


def summarize_generation_result(summary):
    """
    Summarize generation result.
    """
    lines = []
    lines.append("## Conditional Generation Results\n")

    lines.append(f"- Selected model: `{summary.get('selected_model', 'N/A')}`")
    lines.append(f"- Training samples: `{summary.get('training_n_samples', 'N/A')}`")
    lines.append(f"- Condition input rows: `{summary.get('condition_input_rows', 'N/A')}`")
    lines.append(f"- Candidates per condition: `{summary.get('candidates_per_condition', 'N/A')}`")

    metrics = summary.get("metrics_summary", {})

    rows = [
        ["Mean KS statistic", format_value(metrics.get("mean_ks_statistic"))],
        ["Correlation Frobenius norm", format_value(metrics.get("correlation_frobenius_norm"))],
        ["Conditional NN RMSE scaled", format_value(metrics.get("conditional_nn_rmse_scaled"))],
        ["Conditional NN MAE scaled", format_value(metrics.get("conditional_nn_mae_scaled"))],
    ]

    lines.append("\n### Generation Quality Metrics\n")
    lines.append(markdown_table(["Metric", "Value"], rows))

    lines.append("\n### Output Files\n")
    lines.append(f"- Synthetic data: `{summary.get('synthetic_path', 'N/A')}`")
    lines.append(f"- Model path: `{summary.get('model_path', 'N/A')}`")
    lines.append(f"- Final report: `{summary.get('final_report_path', 'N/A')}`")

    return "\n".join(lines) + "\n"


def summarize_artifacts(summary):
    """
    Summarize common artifacts.
    """
    lines = []
    lines.append("## Notes\n")

    task = summary.get("task", "unknown")

    if task == "generation":
        lines.append(
            "The generated data should be inspected before using it as final "
            "design recommendations. Encoded columns may require "
            "postprocessing depending on whether continuous or discrete outputs "
            "are desired."
        )

    elif task in ["regression", "classification"]:
        lines.append(
            "The reported model performance is based on the current train/test "
            "split and should be verified with additional validation if used in "
            "design."
        )

    else:
        lines.append("No additional notes.")

    return "\n".join(lines) + "\n"


def build_markdown_report(summary):
    """
    Build full Markdown report from workflow summary.
    """
    task = summary.get("task", "unknown")

    title = f"Hydrogel Material MAS Report - {task.title()}"

    report = []
    report.append(build_header(title))

    plan = summary.get("plan", {})
    report.append(summarize_plan(plan))

    if task == "regression":
        report.append(summarize_regression_result(summary))

    elif task == "classification":
        report.append(summarize_classification_result(summary))

    elif task == "generation":
        report.append(summarize_generation_result(summary))

    else:
        report.append("## Results\n\nUnknown task type.\n")

    report.append(summarize_artifacts(summary))

    return "\n".join(report)


def write_markdown_report(summary, output_filename=None):
    """
    Write Markdown report from workflow summary dictionary.
    """
    task = summary.get("task", "workflow")
    dataset_key = summary.get("dataset_key", "dataset")

    if output_filename is None:
        output_filename = f"{dataset_key}_{task}_material_report.md"

    output_path = REPORTS_DIR / output_filename

    markdown = build_markdown_report(summary)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    return output_path


def write_markdown_report_from_json(json_path, output_filename=None):
    """
    Write Markdown report from saved workflow summary JSON.
    """
    summary = load_json(json_path)

    return write_markdown_report(
        summary=summary,
        output_filename=output_filename
    )


if __name__ == "__main__":
    example_summary = {
        "dataset_key": "paa_hydrogel",
        "csv_filename": "paa_hydrogel_data.csv",
        "task": "generation",
        "selected_model": "cvae",
        "training_n_samples": 2336,
        "condition_input_rows": 10,
        "candidates_per_condition": 1,
        "synthetic_path": "outputs/generated_data/test.csv",
        "model_path": "outputs/models/test.pt",
        "final_report_path": "outputs/reports/test.json",
        "metrics_summary": {
            "mean_ks_statistic": 0.12,
            "correlation_frobenius_norm": 0.85,
            "conditional_nn_rmse_scaled": 0.72,
            "conditional_nn_mae_scaled": 0.51
        },
        "plan": {
            "dataset_key": "paa_hydrogel",
            "csv_filename": "paa_hydrogel_data.csv",
            "task": "generation",
            "condition_columns": [
                "Frequency (Hz)",
                "Storage modulus (Pa)",
                "Loss modulus (Pa)"
            ],
            "output_columns": [
                "Acrylamide Conc. %",
                "Bis-acrylamide conc %",
                "Photo-initiator conc. %",
                "Layer Height. (micron)",
                "Bottom Layer exposure time (s)",
                "Exposure time (s)"
            ],
            "workflow": [
                "load_data",
                "discover_schema",
                "prepare_generation_data",
                "run_generation",
                "evaluate_generation",
                "save_report"
            ]
        }
    }

    path = write_markdown_report(example_summary)
    print(f"Saved report to: {path}")