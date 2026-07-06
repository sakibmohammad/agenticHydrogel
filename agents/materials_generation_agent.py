"""
materials_analysis_agent.py

Generation agent for hydrogel material workflow.
"""

from pathlib import Path
import json
import math
import pandas as pd


from utils.generation_cvae import run_cvae_generation
from utils.generation_ddpm import run_ddpm_generation
from utils.generation_metrics import evaluate_conditional_generation

class MaterialsGenerationAgent:
    """
    Agent for conditional tabular generation.
    """

    def __init__(self, ddpm_threshold=2000):
        self.ddpm_threshold = ddpm_threshold

        repo_root = Path(__file__).resolve().parents[1]
        self.reports_dir = repo_root / "outputs" / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def choose_generation_model(self, training_n_samples, forced_model=None):
        """
        Select VAE/DDPM or make a choice using forced_model variable.
        """     

        if forced_model is not None:
            forced_model = forced_model.lower().strip()

            if forced_model not in ["cvae", "ddpm"]:
                raise ValueError("Model must be either 'cvae' or 'ddpm' ")
            return forced_model
        
        if training_n_samples < self.ddpm_threshold:
            return "cvae"
        
        return "ddpm"
    
    def _make_run_name(self, plan, model_name):
        """
        Create a clean run name.
        """
        dataset_key = plan.get("dataset_key", "dataset")
        task = plan.get("task", "generation")
        csv_filename = plan.get("csv_filename", "file")

        csv_base = Path(csv_filename).stem
        csv_base = csv_base.replace(" ", "_")

        return f"{dataset_key}_{csv_base}_{task}_{model_name}"
        
    def run_generation(
    self,
    df,
    plan,
    condition_input_df=None,
    condition_csv_path=None,
    candidates_per_condition=1,
    forced_model=None,
    round_encoded_columns=False,
    cvae_epochs=300,
    ddpm_epochs=500,
    ddpm_timesteps=1000,
    ddpm_sampling_steps=None
):
        """
        Run full generation workflow.
        """

        if plan["task"] != "generation":
            raise ValueError("MaterialsGenerationAgent only runs generation plans.")

        condition_columns = plan["condition_columns"]
        output_columns = plan["output_columns"]

        training_n_samples = len(df)

        model_name = self.choose_generation_model(
            training_n_samples=training_n_samples,
            forced_model=forced_model
        )

        run_name = self._make_run_name(plan, model_name)

        if condition_csv_path is not None:
            generation_condition_df = pd.read_csv(condition_csv_path)
        elif condition_input_df is not None:
            generation_condition_df = condition_input_df.copy()
        else:
            generation_condition_df = None

        print("\nRunning materials generation agent...")
        print(f"Training dataset samples: {training_n_samples}")
        print(f"DDPM threshold: {self.ddpm_threshold}")
        print(f"Selected model: {model_name.upper()}")
        print(f"Condition columns: {condition_columns}")
        print(f"Output columns: {output_columns}")
        print(f"Candidates per condition: {candidates_per_condition}")

        if generation_condition_df is not None:
            print(f"User-provided condition rows: {len(generation_condition_df)}")
        else:
            print("No user condition file provided. Using held-out test conditions.")

        if model_name == "cvae":
            generation_result = run_cvae_generation(
                df=df,
                condition_columns=condition_columns,
                output_columns=output_columns,
                run_name=run_name,
                generation_condition_df=generation_condition_df,
                epochs=cvae_epochs,
                patience=20,
                num_samples_per_condition=candidates_per_condition,
                round_encoded_columns=round_encoded_columns
            )

        elif model_name == "ddpm":
            generation_result = run_ddpm_generation(
                df=df,
                condition_columns=condition_columns,
                output_columns=output_columns,
                run_name=run_name,
                generation_condition_df=generation_condition_df,
                epochs=ddpm_epochs,
                patience=20,
                timesteps=ddpm_timesteps,
                sampling_steps=ddpm_sampling_steps,
                num_samples_per_condition=candidates_per_condition,
                round_encoded_columns=round_encoded_columns
            )

        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        metrics = evaluate_conditional_generation(
            real_df=df,
            synthetic_df=generation_result["synthetic_df"],
            condition_columns=condition_columns,
            output_columns=output_columns,
            run_name=run_name
        )

        final_report = {
            "agent": "MaterialsGenerationAgent",
            "selected_model": model_name,
            "training_n_samples": training_n_samples,
            "ddpm_threshold": self.ddpm_threshold,
            "condition_input_rows": None if generation_condition_df is None else len(generation_condition_df),
            "candidates_per_condition": candidates_per_condition,
            "total_generated_rows": len(generation_result["synthetic_df"]),
            "plan": plan,
            "generation_training_report": generation_result["training_report"],
            "synthetic_path": generation_result["synthetic_path"],
            "model_path": generation_result["model_path"],
            "generation_report_path": generation_result["report_path"],
            "metrics_report_path": metrics.get("report_path"),
            "metrics_summary": {
                "mean_ks_statistic": metrics["ks_distance"]["mean_ks_statistic"],
                "correlation_frobenius_norm": metrics["correlation_frobenius"]["frobenius_norm"],
                "conditional_nn_rmse_scaled": metrics["conditional_nearest_neighbor"]["conditional_nn_rmse_scaled"],
                "conditional_nn_mae_scaled": metrics["conditional_nearest_neighbor"]["conditional_nn_mae_scaled"]
            }
        }

        final_report_path = self.reports_dir / f"{run_name}_agent_report.json"

        with open(final_report_path, "w") as f:
            json.dump(final_report, f, indent=4)

        return {
            "selected_model": model_name,
            "training_n_samples": training_n_samples,
            "condition_input_rows": None if generation_condition_df is None else len(generation_condition_df),
            "candidates_per_condition": candidates_per_condition,
            "generation_result": generation_result,
            "metrics": metrics,
            "final_report": final_report,
            "final_report_path": str(final_report_path)
        }



if __name__ == "__main__":
    from agents.materials_planner_agent import MaterialsPlannerAgent

    planner = MaterialsPlannerAgent()
    generation_agent = MaterialsGenerationAgent()

    planning_result = planner.create_plan(
        dataset_key="paa_hydrogel",
        task="generation"
    )

    result = generation_agent.run_generation(
        df=planning_result["df"],
        plan=planning_result["plan"],
        requested_n_samples=1000,
        forced_model=None,
        cvae_epochs=10,
        ddpm_epochs=3,
        ddpm_timesteps=50,
        ddpm_sampling_steps=50
    )

    print("\nGeneration complete.")
    print(f"Training samples, {result['training_n_samples']}")
    print(f"Selected model: {result['selected_model']}")
    print(f"Synthetic data: {result['generation_result']['synthetic_path']}")
    print(f"Final report: {result['final_report_path']}")