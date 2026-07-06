"""
materials_orchestrator_agent.py

High-level orchestrator agent for hydrogel material workflows.
"""

from pathlib import Path
from datetime import datetime
import json
import os
import urllib.request
import urllib.error
import numpy as np
import pandas as pd

from agents.materials_planner_agent import MaterialsPlannerAgent
from agents.materials_analysis_agent import MaterialsAnalysisAgent
from agents.materials_generation_agent import MaterialsGenerationAgent

from utils.material_report_writer import write_markdown_report
from utils.workflow_state import WorkflowState, make_json_serializable


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "outputs" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class LLMFallbackRouter:
    """
    Lightweight LLM router.

    """

    def __init__(
        self,
        llm_provider="auto",
        ollama_model="qwen2.5:1.5b",
        hf_model=None,
        hf_token=None,
        hf_router_url="https://router.huggingface.co/v1/chat/completions",
        timeout=120
    ):
        self.llm_provider = llm_provider
        self.ollama_model = ollama_model
        self.hf_model = hf_model or os.getenv(
            "HF_MODEL_ID",
            "Qwen/Qwen3-4B-Thinking-2507"
        )
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.hf_router_url = hf_router_url
        self.timeout = timeout

    def generate_summary(self, workflow_summary):
        """
        Generate workflow summary using selected provider.

        Falls back to rule-based summary if LLM is unavailable.
        """
        provider = self.llm_provider.lower().strip()

        if provider in ["none", "rule_based"]:
            return self.rule_based_summary(workflow_summary)

        if provider == "ollama":
            return self.try_ollama_summary(workflow_summary)

        if provider == "huggingface":
            return self.try_huggingface_summary(workflow_summary)

        if provider == "auto":
            ollama_summary = self.try_ollama_summary(
                workflow_summary,
                return_none_on_failure=True
            )

            if ollama_summary is not None:
                return ollama_summary

            hf_summary = self.try_huggingface_summary(
                workflow_summary,
                return_none_on_failure=True
            )

            if hf_summary is not None:
                return hf_summary

            return self.rule_based_summary(workflow_summary)

        return self.rule_based_summary(workflow_summary)

    def rule_based_summary(self, workflow_summary):
        """
        Deterministic summary without any LLM.
        """
        task = workflow_summary.get("task", "unknown")
        dataset_key = workflow_summary.get("dataset_key", "unknown")
        csv_filename = workflow_summary.get("csv_filename", "unknown")

        lines = []
        lines.append("Rule-based workflow summary")
        lines.append(f"Dataset: {dataset_key}")
        lines.append(f"CSV file: {csv_filename}")
        lines.append(f"Task: {task}")

        if task in ["regression", "classification"]:
            lines.append(f"Best score: {workflow_summary.get('best_score', 'N/A')}")
            lines.append(f"Model path: {workflow_summary.get('model_path', 'N/A')}")
            lines.append(
                f"Analysis report: {workflow_summary.get('analysis_report_path', 'N/A')}"
            )

        elif task == "generation":
            lines.append(
                f"Selected model: {workflow_summary.get('selected_model', 'N/A')}"
            )
            lines.append(
                f"Training samples: {workflow_summary.get('training_n_samples', 'N/A')}"
            )
            lines.append(
                f"Candidates per condition: {workflow_summary.get('candidates_per_condition', 'N/A')}"
            )
            lines.append(
                f"Synthetic data: {workflow_summary.get('synthetic_path', 'N/A')}"
            )
            lines.append(
                f"Metrics summary: {workflow_summary.get('metrics_summary', {})}"
            )

        return "\n".join(lines)

    def try_ollama_summary(self, workflow_summary, return_none_on_failure=False):
        """
        Try Ollama summary.
        """
        try:
            from utils.ollama_llm_client import OllamaLLMClient

            client = OllamaLLMClient(model_name=self.ollama_model)

            if not client.is_available():
                if return_none_on_failure:
                    return None
                return self.rule_based_summary(workflow_summary)

            prompt = self._build_summary_prompt(workflow_summary)

            system_prompt = (
                "You are a concise scientific workflow assistant. "
                "Summarize the workflow result clearly. "
                "Do not invent results. Only use the provided JSON."
            )

            return client.safe_generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.2,
                max_tokens=512,
                fallback_text=self.rule_based_summary(workflow_summary)
            )

        except Exception:
            if return_none_on_failure:
                return None
            return self.rule_based_summary(workflow_summary)
    def clean_llm_output(self, text):
        """
        Clean unwanted reasoning-style prefixes from LLM output.
        """
        if text is None:
            return ""

        text = text.strip()

        unwanted_starts = [
            "Hmm,",
            "Okay,",
            "The user wants",
            "I need to",
            "Let's"
        ]

        lines = text.splitlines()
        cleaned_lines = []

        for line in lines:
            line_strip = line.strip()

            if any(line_strip.startswith(prefix) for prefix in unwanted_starts):
                continue

            cleaned_lines.append(line)

        cleaned_text = "\n".join(cleaned_lines).strip()

        if cleaned_text == "":
            return text

        return cleaned_text    

    def try_huggingface_summary(self, workflow_summary, return_none_on_failure=False):
        """
        Try Hugging Face chat completion summary.

        Requires:
        - HF_TOKEN environment variable
        - HF_MODEL_ID environment variable optionally
        """
        if not self.hf_token:
            if return_none_on_failure:
                return None
            return self.rule_based_summary(workflow_summary)

        prompt = self._build_summary_prompt(workflow_summary)

        payload = {
            "model": self.hf_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                                "You are a concise scientific workflow assistant. "
                                "Summarize the workflow result clearly for a researcher. "
                                "Do not show reasoning. "
                                "Do not explain your thought process. "
                                "Do not invent results. "
                                "Only use the provided JSON. "
                                "Return only the final summary."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2,
            "max_tokens": 512
        }

        data = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            url=self.hf_router_url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.hf_token}",
                "Content-Type": "application/json"
            },
            method="POST"
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_data = json.loads(response.read().decode("utf-8"))

            raw_text = (
                    response_data
                    .get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                        )

            return self.clean_llm_output(raw_text)

        except Exception:
            if return_none_on_failure:
                return None
            return self.rule_based_summary(workflow_summary)

    def _build_summary_prompt(self, workflow_summary):
        """
        Build summary prompt.
        """
        return (
        "Write a concise final summary of this hydrogel/material MAS workflow.\n"
        "Do not include reasoning or internal analysis.\n"
        "Do not start with phrases like 'Hmm' or 'The user wants'.\n"
        "Only summarize the workflow result.\n\n"
        f"{json.dumps(make_json_serializable(workflow_summary), indent=2)}"
        )


class MaterialsOrchestratorAgent:
    """
    High-level agent that coordinates the full workflow.
    """

    def __init__(
        self,
        llm_provider="auto",
        ollama_model="qwen2.5:1.5b",
        hf_model=None,
        hf_token=None,
        require_human_confirmation=True
    ):
        self.planner_agent = MaterialsPlannerAgent()
        self.analysis_agent = MaterialsAnalysisAgent()
        self.generation_agent = MaterialsGenerationAgent()

        self.llm_router = LLMFallbackRouter(
            llm_provider=llm_provider,
            ollama_model=ollama_model,
            hf_model=hf_model,
            hf_token=hf_token
        )

        self.require_human_confirmation = require_human_confirmation

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
        Create workflow plan using rule-based planner.
        """
        return self.planner_agent.create_plan(
            dataset_key=dataset_key,
            csv_filename=csv_filename,
            task=task,
            target_columns=target_columns,
            target_column=target_column,
            condition_columns=condition_columns,
            output_columns=output_columns,
            columns_to_drop=columns_to_drop
        )

    def print_plan_for_confirmation(self, planning_result):
        """
        Print plan summary before execution.
        """
        df = planning_result["df"]
        plan = planning_result["plan"]

        print("\n" + "-" * 50)
        print("MATERIALS ORCHESTRATOR PLAN")
        print("-" * 50)
        print(f"Dataset: {plan.get('dataset_key')}")
        print(f"CSV file: {plan.get('csv_filename')}")
        print(f"Task: {plan.get('task')}")
        print(f"Rows: {len(df)}")
        print(f"Columns: {len(df.columns)}")

        if plan["task"] == "regression":
            print("\nRegression targets:")
            for col in plan["target_columns"]:
                print(f" - {col}")

        elif plan["task"] == "classification":
            print("\nClassification target:")
            print(f" - {plan['target_column']}")

        elif plan["task"] == "generation":
            selected_model = self.generation_agent.choose_generation_model(
                training_n_samples=len(df),
                forced_model=None
            )

            print("\nGeneration condition columns:")
            for col in plan["condition_columns"]:
                print(f" - {col}")

            print("\nGeneration output columns:")
            for col in plan["output_columns"]:
                print(f" - {col}")

            print(f"\nAutomatic model choice: {selected_model.upper()}")

        print("\nWorkflow steps:")
        for step in plan.get("workflow", []):
            print(f" - {step}")

        print("-" * 50)

    def confirm_or_stop(self):
        """
        Human-in-the-loop confirmation.
        """
        if not self.require_human_confirmation:
            return

        answer = input("\nProceed with this workflow? [y/n]: ").strip().lower()

        if answer not in ["y", "yes"]:
            print("Workflow cancelled by user.")
            raise SystemExit(0)

    def execute_plan(
        self,
        planning_result,
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
        Execute a workflow plan.
        """
        df = planning_result["df"]
        plan = planning_result["plan"]
        task = plan["task"]

        if task in ["regression", "classification"]:
            return self.analysis_agent.run_plan(
                df=df,
                plan=plan
            )

        if task == "generation":
            return self.generation_agent.run_generation(
                df=df,
                plan=plan,
                condition_input_df=condition_input_df,
                condition_csv_path=condition_csv_path,
                candidates_per_condition=candidates_per_condition,
                forced_model=forced_model,
                round_encoded_columns=round_encoded_columns,
                cvae_epochs=cvae_epochs,
                ddpm_epochs=ddpm_epochs,
                ddpm_timesteps=ddpm_timesteps,
                ddpm_sampling_steps=ddpm_sampling_steps
            )

        raise ValueError(f"Unknown task: {task}")

    def build_workflow_summary(self, planning_result, execution_result):
        """
        Build compact workflow summary.
        """
        plan = planning_result["plan"]
        task = plan["task"]

        summary = {
            "created_at": datetime.now().isoformat(),
            "dataset_key": plan.get("dataset_key"),
            "csv_filename": plan.get("csv_filename"),
            "task": task,
            "plan": plan
        }

        if task in ["regression", "classification"]:
            summary["analysis_report_path"] = execution_result.get("analysis_report_path")

            if "mlp_result" in execution_result:
                summary["model_path"] = execution_result["mlp_result"].get("model_path")
                summary["metrics"] = execution_result["mlp_result"].get("best_metrics")
                summary["best_score"] = execution_result["mlp_result"].get("best_score")

        elif task == "generation":
            summary["selected_model"] = execution_result.get("selected_model")
            summary["training_n_samples"] = execution_result.get("training_n_samples")
            summary["condition_input_rows"] = execution_result.get("condition_input_rows")
            summary["candidates_per_condition"] = execution_result.get(
                "candidates_per_condition"
            )
            summary["synthetic_path"] = execution_result["generation_result"].get(
                "synthetic_path"
            )
            summary["model_path"] = execution_result["generation_result"].get(
                "model_path"
            )
            summary["final_report_path"] = execution_result.get("final_report_path")
            summary["metrics_summary"] = execution_result["final_report"].get(
                "metrics_summary"
            )

        return summary

    def save_orchestrator_summary(self, workflow_summary):
        """
        Save orchestrator summary JSON and Markdown report.
        """
        dataset_key = workflow_summary.get("dataset_key", "dataset")
        task = workflow_summary.get("task", "task")

        json_path = REPORTS_DIR / f"{dataset_key}_{task}_orchestrator_summary.json"

        with open(json_path, "w") as f:
            json.dump(make_json_serializable(workflow_summary), f, indent=4)

        markdown_path = write_markdown_report(workflow_summary)

        return {
            "json_path": str(json_path),
            "markdown_path": str(markdown_path)
        }

    def run_workflow(
        self,
        dataset_key,
        csv_filename=None,
        task="regression",
        target_columns=None,
        target_column=None,
        condition_columns=None,
        output_columns=None,
        columns_to_drop=None,
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
        Run full orchestrated workflow.
        """
        workflow_state = WorkflowState(workflow_name="materials_orchestrator")

        workflow_state.save_config(
            {
                "dataset_key": dataset_key,
                "csv_filename": csv_filename,
                "task": task,
                "target_columns": target_columns,
                "target_column": target_column,
                "condition_columns": condition_columns,
                "output_columns": output_columns,
                "condition_csv_path": condition_csv_path,
                "candidates_per_condition": candidates_per_condition,
                "forced_model": forced_model
            }
        )

        planning_result = self.create_plan(
            dataset_key=dataset_key,
            csv_filename=csv_filename,
            task=task,
            target_columns=target_columns,
            target_column=target_column,
            condition_columns=condition_columns,
            output_columns=output_columns,
            columns_to_drop=columns_to_drop
        )

        workflow_state.add_step(
            step_name="create_plan",
            status="completed",
            details=planning_result["plan"]
        )

        self.print_plan_for_confirmation(planning_result)
        self.confirm_or_stop()

        execution_result = self.execute_plan(
            planning_result=planning_result,
            condition_input_df=condition_input_df,
            condition_csv_path=condition_csv_path,
            candidates_per_condition=candidates_per_condition,
            forced_model=forced_model,
            round_encoded_columns=round_encoded_columns,
            cvae_epochs=cvae_epochs,
            ddpm_epochs=ddpm_epochs,
            ddpm_timesteps=ddpm_timesteps,
            ddpm_sampling_steps=ddpm_sampling_steps
        )

        workflow_state.add_step(
            step_name="execute_plan",
            status="completed",
            details={"result_keys": list(execution_result.keys())}
        )

        workflow_summary = self.build_workflow_summary(
            planning_result=planning_result,
            execution_result=execution_result
        )

        llm_summary = self.llm_router.generate_summary(workflow_summary)
        workflow_summary["llm_or_rule_summary"] = llm_summary

        saved_paths = self.save_orchestrator_summary(workflow_summary)

        workflow_state.add_artifact(
            artifact_name="orchestrator_summary_json",
            artifact_path=saved_paths["json_path"],
            artifact_type="json",
            description="Orchestrator workflow summary JSON"
        )

        workflow_state.add_artifact(
            artifact_name="orchestrator_markdown_report",
            artifact_path=saved_paths["markdown_path"],
            artifact_type="markdown",
            description="Human-readable workflow report"
        )

        workflow_state.update_status("completed")

        return {
            "planning_result": planning_result,
            "execution_result": execution_result,
            "workflow_summary": workflow_summary,
            "saved_paths": saved_paths,
            "workflow_state": workflow_state.get_summary()
        }


if __name__ == "__main__":
    orchestrator = MaterialsOrchestratorAgent(
        llm_provider="rule_based",
        require_human_confirmation=False
    )

    result = orchestrator.run_workflow(
        dataset_key="paa_hydrogel",
        task="generation",
        forced_model="cvae",
        cvae_epochs=5,
        candidates_per_condition=1
    )

    print("\nWorkflow complete.")
    print(json.dumps(result["saved_paths"], indent=4))