"""
materials_analysis_agent.py


Analysis agent for hydrogel material workflow.


"""

from pathlib import Path
import json

from utils.material_preprocessing import (
    prepare_regression_data,
    prepare_classification_data,
    prepare_generation_data
)

from utils.material_visualization import make_basic_plots
from utils.adaptive_mlp import run_adaptive_mlp
from agents.materials_generation_agent import MaterialsGenerationAgent


class MaterialsAnalysisAgent:
    """
    Executes MAS analysis plans.
    """

    def __init__(self):
        repo_root = Path(__file__).resolve().parents[1]
        self.reports_dir = repo_root/"outputs"/"reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def run_regression(self, df, plan, scale_y=True):
        """
        Run regression workflow: preprocessing -> plots -> model training.
        """

        target_columns = plan["target_columns"]

        print("\nRunning regression workflow..........")
        print(f"Targets : {target_columns}")

        prepared_data = prepare_regression_data(df, target_columns=target_columns, scale_y=scale_y)

        plot_paths = make_basic_plots(df, task="regression", target_columns=target_columns)

        run_name = self._make_run_name(plan)

        mlp_result = run_adaptive_mlp(prepared_data, threshold=0.60, run_name=run_name)

        result = {
            "task" : "regression",
            "plan" : plan,
            "plot_paths" : [str(path) for path in plot_paths],
            "mlp_result" : mlp_result
        }

        report_path = self.save_analysis_result(
            result, filename=f"{run_name}_analysis_report.json"
        )

        result["analysis_report_path"] = str(report_path)

        return result
    
    def run_classification(self, df, plan):
        """
        Run classification workflow: preprocessing -> plots -> training model.
        """

        target_column = plan["target_column"]

        print("\nRunning classification workflow.........")
        print(f"Target column : {target_column}")

        prepared_data = prepare_classification_data(df, target_column=target_column)

        plot_paths = make_basic_plots(df, task="classification", target_column=target_column)

        run_name = self._make_run_name(plan)

        mlp_result = run_adaptive_mlp(prepared_data, threshold=0.60, run_name=run_name)

        result = {
            "task" : "classification",
            "plan" : plan,
            "plot_paths" : [str(path) for path in plot_paths],
            "mlp_result" : mlp_result
        }

        report_path = self.save_analysis_result(result, filename=f"{run_name}_analysis_report.json")

        result["analysis_report_path"] = str(report_path)

        return result
    
    def run_generation_preparation(
        self,
        df,
        plan,
        requested_n_samples=1000,
        forced_model=None,
        round_encoded_columns=False,
        cvae_epochs=300,
        ddpm_epochs=500,
        ddpm_timesteps=1000,
        ddpm_sampling_steps=None
        ):
        """
        Prepare generation data and plots
        """

        condition_columns = plan["condition_columns"]
        output_columns = plan["output_columns"]

        print("\nRunning generation workflow...........")
        print(f"Condition columns : {condition_columns}")
        print(f"Output columns: {output_columns}")

        prepared_data = prepare_generation_data(
            df,
            condition_columns=condition_columns,
            output_columns=output_columns
            )

        plot_paths = make_basic_plots(
            df,
            task="generation",
            condition_columns=condition_columns,
            output_columns=output_columns
            )


        generation_agent = MaterialsGenerationAgent()

        generation_result = generation_agent.run_generation(
            df=df,
            plan=plan,
            requested_n_samples=requested_n_samples,
            forced_model=forced_model,
            round_encoded_columns=round_encoded_columns,
            cvae_epochs=cvae_epochs,
            ddpm_epochs=ddpm_epochs,
            ddpm_timesteps=ddpm_timesteps,
            ddpm_sampling_steps=ddpm_sampling_steps
            )

        result = {
            "task": "generation",
            "status": "generation_completed",
            "plan": plan,
            "plot_paths": [str(path) for path in plot_paths],
            "prepared_shapes": {
                "X_condition": prepared_data["X_condition"].shape,
                "Y_output": prepared_data["Y_output"].shape
            },
            "condition_columns": prepared_data["condition_columns"],
            "output_columns": prepared_data["output_columns"],
            "encoded_condition_columns": prepared_data["encoded_condition_columns"],
            "encoded_output_columns": prepared_data["encoded_output_columns"],
            "selected_model": generation_result["selected_model"],
            "synthetic_path": generation_result["generation_result"]["synthetic_path"],
            "model_path": generation_result["generation_result"]["model_path"],
            "generation_report_path": generation_result["generation_result"]["report_path"],
            "metrics_report_path": generation_result["metrics"].get("report_path"),
            "final_generation_agent_report": generation_result["final_report_path"],
            "metrics_summary": generation_result["final_report"]["metrics_summary"]
        }

        run_name = self._make_run_name(plan)

        report_path = self.save_analysis_result(
            result,
            filename=f"{run_name}_generation_report.json"
        )

        result["analysis_report_path"] = str(report_path)

        return result
    
    def run_plan(
        self,
        df,
        plan,
        requested_n_samples=1000,
        forced_model=None,
        round_encoded_columns=False,
        cvae_epochs=300,
        ddpm_epochs=500,
        ddpm_timesteps=1000,
        ddpm_sampling_steps=None
    ):
        """
        Execute a full plan.
        """
        task = plan["task"]

        if task == "regression":
            return self.run_regression(df, plan)

        if task == "classification":
            return self.run_classification(df, plan)

        if task == "generation":
            return self.run_generation_preparation(
                df=df,
                plan=plan,
                requested_n_samples=requested_n_samples,
                forced_model=forced_model,
                round_encoded_columns=round_encoded_columns,
                cvae_epochs=cvae_epochs,
                ddpm_epochs=ddpm_epochs,
                ddpm_timesteps=ddpm_timesteps,
                ddpm_sampling_steps=ddpm_sampling_steps
            )

        raise ValueError(f"Unknown task: {task}")
    
    def save_analysis_result(self, result, filename):
        """
        Save final analysis result as JSON.
        """

        save_path = self.reports_dir/filename

        json_ready_result = self._make_json_serializable(result)

        with open(save_path, "w") as f:
            json.dump(json_ready_result, f, indent=4)

        return save_path

    def _make_run_name(self, plan):
        """
        Create a clean run name from dataset, file and task.
        """    

        dataset_key = plan.get("dataset_key", "dataset")
        task = plan.get("task", "task")
        csv_filename = plan.get("csv_filename", "file")

        csv_base = Path(csv_filename).stem
        csv_base = csv_base.replace(" ", "_")

        return f"{dataset_key}_{csv_base}_{task}"
    
    def _make_json_serializable(self, obj):
        """
        Convert tuples and Paths recursively so JSON can save them.
        """
        if isinstance(obj, dict):
            return {
                key : self._make_json_serializable(value) for key, value in obj.items()
            }
        
        if isinstance(obj, list):
            return [self._make_json_serializable(value) for value in obj]
        
        if isinstance(obj, tuple):
            return list(obj)
        
        if isinstance(obj, Path):
            return str(obj)
        
        return obj
    


if __name__ == "__main__":
    from agents.materials_planner_agent import MaterialsPlannerAgent

    planner = MaterialsPlannerAgent()
    analysis_agent = MaterialsAnalysisAgent()


    planning_result = planner.create_plan(
            dataset_key = "paa_hydrogel",
            task = "regression",
            target_columns=[
            "Storage modulus (Pa)",
            "Loss modulus (Pa)"
        ]
        )

    result = analysis_agent.run_plan(
            df = planning_result["df"],
            plan = planning_result["plan"]
        )

    print("\nAnalysis complete.")
    print(f"Report was saved to: {result['analysis_report_path']}")
