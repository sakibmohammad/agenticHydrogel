"""
materials_planner_agent.py

Planner agent for hydrogel workflow.
"""

from pathlib import Path
import json

from utils.material_dataset_utils import (
    list_available_datasets, list_csv_files, load_material_dataset
)

from utils.material_schema_discovery import (
    build_schema_report, is_generation_allowed
)

class MaterialsPlannerAgent:
    """
    Planner agent for hydrogel material dataset workflow.
    """

    def __init__(self, use_hf=False, hf_client=None):
        self.use_hf = use_hf
        self.hf_client = hf_client

    def list_datasets(self):
        """
        List know dataset groups.
        """
        return list_available_datasets()
        
    def list_files_for_dataset(self, dataset_key):
        """
        List CSV files for a dataset group.
        """
        csv_files = list_csv_files(dataset_key)

        return [path.name for path in csv_files]
        
    def load_and_inspect_dataset(self, dataset_key, csv_filename=None, columns_to_drop=None):
        """
        Load dataset and build schema report
        """
        df, csv_path = load_material_dataset(dataset_key=dataset_key, csv_filename=csv_filename, columns_to_drop=columns_to_drop)

        schema_report = build_schema_report(df)

        return df, csv_path, schema_report
        
    def get_available_tasks(self, dataset_key, csv_filename=None):
        """
        Decide which datasets are avilable.

        All datasets:
            - regression
            - classification

        Generation:
            - allowed for PAA hydrogel dataset
            - allowed only for selected granular hydrogel files    
        """

        tasks = ["regression", "classification"]

        if is_generation_allowed(csv_filename=csv_filename, dataset_key=dataset_key):
            tasks.append("generation")

        return tasks

    def create_rule_based_plan(self, dataset_key, csv_filename, task, schema_report, target_columns=None, target_column=None, condition_columns=None, output_columns=None):
        """
        Create a deterministic JSON-like plan.

        This is safet than asking the LLM agent to invent columns.
        The user or the agent should pass confirmed columns when possible.
        """    

        task = task.lower().strip()

        available_tasks = self.get_available_tasks(dataset_key=dataset_key, csv_filename=csv_filename)

        if task not in available_tasks:
            raise ValueError(f"Task {task} is not avialable for this file"
                            f"Available tasks: {available_tasks}")
            
        if task == "regression":
            if target_columns is None:
                target_columns = schema_report["suggested_regression_targets"][:1]

            if isinstance(target_columns, str):
                target_columns = [target_columns]

            workflow = [
                    "load_data",
                    "discover_schema",
                    "prepare_regression_data",
                    "make_regression_plots",
                    "run_adaptive_mlp_regression",
                    "save_report"
                ]        

            plan = {
                    "dataset_key" : dataset_key,
                    "csv_filename" : csv_filename,
                    "task" : "regression",
                    "target_columns" : target_columns,
                    "workflow" : workflow
                }

        elif task == "classification":
            if target_column is None:
                candidates = schema_report["suggested_classification_targets"]

                if len(candidates) == 0:
                    raise ValueError(f"No classification target was found. "
                                         "Please provide target column manually.")

                target_column = candidates[0]

            workflow = [
                    "load_data",
                    "discover_schema",
                    "prepare_classification_data"
                    "make_classification_plots",
                    "run_adaptive_mlp_classification",
                    "save_report"
                ]        

            plan = {
                    "dataset_key" : dataset_key,
                    "csv_filename" : csv_filename,
                    "task" : "classification",
                    "target_column" : target_column,
                    "workflow" : workflow
                }                 

        elif task == "generation":
            suggestions = schema_report["suggested_generation_candidates"]

            if condition_columns is None:
                condition_columns = suggestions["condition_columns"]

            if output_columns is None:
                output_columns = suggestions["output_columns"]    

            if len(condition_columns) == 0:
                raise ValueError("No generation condition columns were selected")

            if len(output_columns) == 0:
                raise ValueError("No generation output columns were selected.")

            workflow = [
                    "load_data",
                    "discover_schema",
                    "prepare_generation_data",
                    "make_generation_plots",
                    "generation_model_not_added_yet",
                    "save_report"
                ]    

            plan = {
                    "dataset_key" : dataset_key,
                    "csv_filename" : csv_filename,
                    "task" : "generation",
                    "condition_columns" : condition_columns,
                    "output_columns" : output_columns,
                    "workflow" : workflow
                }

        else:
            raise ValueError(f"Unknown task: {task}")

        return plan

    def create_plan(
                self,
                dataset_key,
                csv_filename=None,
                task="regression",
                target_columns=None,
                target_column=None,
                condition_columns=None,
                output_columns=None,
                columns_to_drop=None
    ):
        """
        Main planning method.
        Returns:
            - dataframe
            - csv_path
            - schema_report
            - plan
        """        

        df, csv_path, schema_report = self.load_and_inspect_dataset(
                dataset_key=dataset_key,
                csv_filename=csv_filename,
                columns_to_drop=columns_to_drop
            )

        if csv_filename is None:
                csv_filename = Path(csv_path).name

        plan = self.create_rule_based_plan(
                dataset_key=dataset_key,
                csv_filename=csv_filename,
                task=task,
                schema_report=schema_report,
                target_columns=target_columns,
                target_column=target_column,
                condition_columns=condition_columns,
                output_columns=output_columns
            )    

        return {
                "df" : df,
                "csv_path" : csv_path,
                "schema_report" : schema_report,
                "plan" : plan
            }
        
    def save_plan(self, plan, filename="materials_plan.json"):
        """
            Save workflow plans to outputs/reports.
        """

        repo_root = Path(__file__).resolve().parents[1]
        reports_dir = repo_root/"outputs"/"reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        save_path = reports_dir/filename

        with open(save_path, "w") as f:
            json.dump(plan, f, indent=4)

        return save_path




if __name__ == "__main__":
    planner = MaterialsPlannerAgent()

    result = planner.create_plan(
        dataset_key = "paa_hydrogel",
        task = "regression",
        target_columns=[
            "Storage modulus (Pa)",
            "Loss modulus (Pa)"
        ]
    )

    print("\nGenerated plan: ")
    print(json.dump(result['plan'], indent=4))

    save_path = planner.save_plan(result['plan'])
    print(f"\nPlan saves to: {save_path}")