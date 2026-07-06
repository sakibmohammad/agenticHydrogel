"""
main_materials_llm.py

Main CLI entry point for the hydrogel material workflow.
"""

from pathlib import Path
import argparse
import json
import pandas as pd
import numpy as np

from agents.materials_planner_agent import MaterialsPlannerAgent
from agents.materials_analysis_agent import MaterialsAnalysisAgent
from agents.materials_generation_agent import MaterialsGenerationAgent

from utils.material_dataset_utils import list_available_datasets
from utils.material_visualization import make_basic_plots
from utils.ollama_llm_client import summarize_workflow_with_ollama
from utils.material_report_writer import write_markdown_report


REPO_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = REPO_ROOT / "outputs" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Hydrogel material MAS workflow CLI"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset key: paa_hydrogel or granular_hydrogel"
    )

    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="CSV filename inside selected dataset directory"
    )

    parser.add_argument(
        "--task",
        type=str,
        default=None,
        choices=["regression", "classification", "generation"],
        help="Workflow task"
    )

    parser.add_argument(
        "--target_columns",
        nargs="*",
        default=None,
        help="Regression target columns"
    )

    parser.add_argument(
        "--target_column",
        type=str,
        default=None,
        help="Classification target column"
    )

    parser.add_argument(
        "--condition_columns",
        nargs="*",
        default=None,
        help="Generation condition columns"
    )

    parser.add_argument(
        "--output_columns",
        nargs="*",
        default=None,
        help="Generation output columns"
    )

    parser.add_argument(
        "--columns_to_drop",
        nargs="*",
        default=None,
        help="Columns to drop before analysis"
    )

    parser.add_argument(
        "--condition_csv",
        type=str,
        default=None,
        help="CSV file containing user-provided condition rows for generation"
    )

    parser.add_argument(
        "--candidates_per_condition",
        type=int,
        default=1,
        help="Number of generated candidate compositions per condition row"
    )

    parser.add_argument(
        "--forced_model",
        type=str,
        default=None,
        choices=["cvae", "ddpm"],
        help="Force generation model instead of automatic selection"
    )

    parser.add_argument(
        "--round_encoded_columns",
        action="store_true",
        help="Round generated *_encoded columns after generation"
    )

    parser.add_argument(
        "--cvae_epochs",
        type=int,
        default=300,
        help="CVAE training epochs"
    )

    parser.add_argument(
        "--ddpm_epochs",
        type=int,
        default=500,
        help="DDPM training epochs"
    )

    parser.add_argument(
        "--ddpm_timesteps",
        type=int,
        default=1000,
        help="DDPM diffusion timesteps"
    )

    parser.add_argument(
        "--ddpm_sampling_steps",
        type=int,
        default=None,
        help="DDPM sampling steps"
    )

    parser.add_argument(
        "--llm_provider",
        type=str,
        default="none",
        choices=["none", "ollama"],
        help="Optional LLM provider for summary"
    )

    parser.add_argument(
        "--ollama_model",
        type=str,
        default="qwen2.5:1.5b",
        help="Ollama model name"
    )

    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip human confirmation"
    )

    return parser.parse_args()

def ask_user_if_missing(value, message, allowed_values=None):
    """
    Ask user for a value if command line argument was missing
    """

    if value is not None:
        return value
    
    while True:
        user_value = input(message).strip()

        if allowed_values is None:
            if user_value:
                return user_value

        elif user_value in allowed_values:
            return user_value

        print(f"Invalid choice. Allowed values: {allowed_values}")

def print_plan_summary(plan, df, model_preview=None):
    """
    Print human-readable workflow summary before execution.
    """
    print("\n" + "." * 100)
    print("HYDROGEL MATERIAL MAS WORKFLOW PLAN")
    print("." * 100)

    print(f"Dataset key: {plan.get('dataset_key')}")
    print(f"CSV file: {plan.get('csv_filename')}")
    print(f"Task: {plan.get('task')}")
    print(f"Rows after cleaning: {len(df)}")
    print(f"Columns after cleaning: {len(df.columns)}")

    if plan["task"] == "regression":
        print("\nRegression target columns:")
        for col in plan["target_columns"]:
            print(f" - {col}")

    elif plan["task"] == "classification":
        print("\nClassification target column:")
        print(f" - {plan['target_column']}")

    elif plan["task"] == "generation":
        print("\nGeneration condition columns:")
        for col in plan["condition_columns"]:
            print(f" - {col}")

        print("\nGeneration output columns:")
        for col in plan["output_columns"]:
            print(f" - {col}")

        if model_preview is not None:
            print(f"\nSelected generation model preview: {model_preview.upper()}")

    print("\nWorkflow steps:")
    for step in plan["workflow"]:
        print(f" - {step}")

    print("-" * 100)            


def confirm_or_exit(skip_confirmation=False):
    """
    Ask user to confirm workflow execution.
    """
    if skip_confirmation:
        return

    answer = input("\nProceed with this workflow? [y/n]: ").strip().lower()

    if answer not in ["y", "yes"]:
        print("Workflow cancelled by user.")
        raise SystemExit(0)
    
def make_json_serializable(obj):
    """
    Convert common Python/numpy objects to JSON-safe objects.
    """
    if isinstance(obj, dict):
        return {
            key: make_json_serializable(value)
            for key, value in obj.items()
        }

    if isinstance(obj, list):
        return [make_json_serializable(value) for value in obj]

    if isinstance(obj, tuple):
        return [make_json_serializable(value) for value in obj]

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return float(obj)

    return obj    

def save_cli_summary(summary, filename):
    """
    Save CLI workflow summary.
    """
    save_path = REPORTS_DIR / filename

    with open(save_path, "w") as f:
        json.dump(make_json_serializable(summary), f, indent=4)

    return save_path


def run_regression_or_classification(planning_result):
    """
    Run regression or classification using MaterialsAnalysisAgent.
    """
    analysis_agent = MaterialsAnalysisAgent()

    result = analysis_agent.run_plan(
        df=planning_result["df"],
        plan=planning_result["plan"]
    )

    return result

def run_generation(planning_result, args):
    """
    Run generation using MaterialsGenerationAgent.
    """
    generation_agent = MaterialsGenerationAgent()

    df = planning_result["df"]
    plan = planning_result["plan"]

    plot_paths = make_basic_plots(
        df,
        task="generation",
        condition_columns=plan["condition_columns"],
        output_columns=plan["output_columns"]
    )

    result = generation_agent.run_generation(
        df=df,
        plan=plan,
        condition_csv_path=args.condition_csv,
        candidates_per_condition=args.candidates_per_condition,
        forced_model=args.forced_model,
        round_encoded_columns=args.round_encoded_columns,
        cvae_epochs=args.cvae_epochs,
        ddpm_epochs=args.ddpm_epochs,
        ddpm_timesteps=args.ddpm_timesteps,
        ddpm_sampling_steps=args.ddpm_sampling_steps
    )

    result["plot_paths"] = [str(path) for path in plot_paths]

    return result

def build_workflow_summary(planning_result, execution_result):
    """
    Build final workflow summary.
    """
    plan = planning_result["plan"]

    summary = {
        "dataset_key": plan.get("dataset_key"),
        "csv_filename": plan.get("csv_filename"),
        "task": plan.get("task"),
        "plan": plan,
        "execution_result_keys": list(execution_result.keys()),
    }

    if plan["task"] in ["regression", "classification"]:
        summary["analysis_report_path"] = execution_result.get("analysis_report_path")

        if "mlp_result" in execution_result:
            summary["model_path"] = execution_result["mlp_result"].get("model_path")
            summary["metrics"] = execution_result["mlp_result"].get("best_metrics")
            summary["best_score"] = execution_result["mlp_result"].get("best_score")

    elif plan["task"] == "generation":
        summary["selected_model"] = execution_result.get("selected_model")
        summary["training_n_samples"] = execution_result.get("training_n_samples")
        summary["condition_input_rows"] = execution_result.get("condition_input_rows")
        summary["candidates_per_condition"] = execution_result.get("candidates_per_condition")
        summary["synthetic_path"] = execution_result["generation_result"].get("synthetic_path")
        summary["model_path"] = execution_result["generation_result"].get("model_path")
        summary["final_report_path"] = execution_result.get("final_report_path")
        summary["metrics_summary"] = execution_result["final_report"].get("metrics_summary")

    return summary

def main():
    """
    Main CLI workflow.
    """

    args = parse_args()

    available_datasets = list_available_datasets()

    dataset_key = ask_user_if_missing(
        args.dataset,
        message=f"Dataset key {list(available_datasets.keys())}",
        allowed_values=list(available_datasets.keys())
    )

    task = ask_user_if_missing(
        args.task,
        message="Task [regression/classification/generation]: ",
        allowed_values=["regression", "classifcation", "generation"]
    )

    planner = MaterialsPlannerAgent()

    planning_result = planner.create_plan(
        dataset_key=dataset_key,
        csv_filename=args.file,
        task=task,
        target_columns=args.target_columns,
        target_column=args.target_column,
        condition_columns=args.condition_columns,
        output_columns=args.output_columns,
        columns_to_drop=args.columns_to_drop

    )

    model_preview = None

    if task == "generation":
        generation_agent = MaterialsGenerationAgent()
        model_preview = generation_agent.choose_generation_model(
            training_n_samples=len(planning_result['df']),
            forced_model=args.forced_model
        )
    
    print_plan_summary(
        plan=planning_result['plan'],
        df=planning_result['df'],
        model_preview=model_preview
    )

    confirm_or_exit(skip_confirmation=args.yes)

    if task in ["regression", "classification"]:
        execution_result = run_regression_or_classification(planning_result)
    elif task == "generation":
        execution_result = run_generation(planning_result, args)
    else:
        raise ValueError(f"Unknown task : {task}")

    workflow_summary = build_workflow_summary(
        planning_result=planning_result,
        execution_result=execution_result
    )        

    cli_summary_path = save_cli_summary(
        workflow_summary,
        filename=f"{dataset_key}_{task}_cli_memory.json"
    )
    markdown_report_path = write_markdown_report(workflow_summary)

    print("\nWorkflow completed.")
    print(f"CLI summary saved to : {cli_summary_path}")
    print(f"Markdown report saved to: {markdown_report_path}")

    if args.llm_provider == "ollama":
        print("\nGenerating ollama summary............")
        llm_summary = summarize_workflow_with_ollama(
            workflow_summary=workflow_summary,
            model_name=args.ollama_model
        )

        print("\nLLM summary: ")
        print(llm_summary)

        llm_summary_path = REPORTS_DIR / f"{dataset_key}_{task}_llm_summary.txt"

        with open(llm_summary_path, "w") as f:
            f.write(llm_summary)

        print(f"LLM summary saved to: {llm_summary_path}")

    if task == "generation":
        print(f"Synthetic data: {workflow_summary.get('synthetic_path')}") 

    if "model_path" in workflow_summary:
        print(f"Model path: {workflow_summary.get('model_path')}")       


if __name__ == "__main__":
    main()        