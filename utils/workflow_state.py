"""
workflow_state.py

Workflow state manager for hydrogel material workflows.
"""

from pathlib import Path
from datetime import datetime
import json
import uuid
import shutil
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS_DIR = REPO_ROOT / "outputs" / "workflows"
WORKFLOWS_DIR.mkdir(parents=True, exist_ok=True)


def make_json_serializable(obj):
    """
    Convert common Python, numpy, pandas, and pathlib objects into JSON-safe objects.
    """
    if isinstance(obj, dict):
        return {
            str(key): make_json_serializable(value)
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

    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")

    if isinstance(obj, pd.Series):
        return obj.tolist()

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    return obj


def create_run_id(prefix="materials_mas"):
    """
    Create a unique run ID.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]

    return f"{prefix}_{timestamp}_{short_uuid}"


class WorkflowState:
    """
    Workflow state recorder.
    """

    def __init__(self, run_id=None, workflow_name="materials_mas"):
        if run_id is None:
            run_id = create_run_id(prefix=workflow_name)

        self.run_id = run_id
        self.workflow_name = workflow_name
        self.run_dir = WORKFLOWS_DIR / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.state_path = self.run_dir / "state.json"
        self.events_path = self.run_dir / "events.jsonl"
        self.config_path = self.run_dir / "config.json"
        self.artifacts_path = self.run_dir / "artifacts.json"

        self.state = {
            "run_id": self.run_id,
            "workflow_name": self.workflow_name,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "status": "initialized",
            "steps": [],
            "artifacts": {}
        }

        self.save_state()

    def save_config(self, config):
        """
        Save workflow configuration.
        """
        config = make_json_serializable(config)

        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=4)

        self.log_event(
            event_type="config_saved",
            message="Workflow configuration saved.",
            data={"config_path": str(self.config_path)}
        )

        return self.config_path

    def log_event(self, event_type, message, data=None):
        """
        Append an event to the workflow event log.
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "message": message,
            "data": make_json_serializable(data or {})
        }

        with open(self.events_path, "a") as f:
            f.write(json.dumps(event) + "\n")

        return event

    def add_step(self, step_name, status="completed", details=None):
        """
        Add a workflow step to the state.
        """
        step = {
            "step_name": step_name,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "details": make_json_serializable(details or {})
        }

        self.state["steps"].append(step)
        self.state["updated_at"] = datetime.now().isoformat()

        self.log_event(
            event_type="step_update",
            message=f"Step '{step_name}' marked as {status}.",
            data=step
        )

        self.save_state()

        return step

    def add_artifact(self, artifact_name, artifact_path, artifact_type=None, description=None):
        """
        Register an output artifact.
        """
        artifact_path = Path(artifact_path)

        artifact = {
            "path": str(artifact_path),
            "type": artifact_type,
            "description": description,
            "exists": artifact_path.exists(),
            "registered_at": datetime.now().isoformat()
        }

        self.state["artifacts"][artifact_name] = artifact
        self.state["updated_at"] = datetime.now().isoformat()

        self.log_event(
            event_type="artifact_registered",
            message=f"Artifact '{artifact_name}' registered.",
            data=artifact
        )

        self.save_state()
        self.save_artifacts()

        return artifact

    def save_artifacts(self):
        """
        Save artifact registry.
        """
        artifacts = make_json_serializable(self.state.get("artifacts", {}))

        with open(self.artifacts_path, "w") as f:
            json.dump(artifacts, f, indent=4)

        return self.artifacts_path

    def update_status(self, status):
        """
        Update workflow status.
        """
        self.state["status"] = status
        self.state["updated_at"] = datetime.now().isoformat()

        self.log_event(
            event_type="status_update",
            message=f"Workflow status updated to {status}.",
            data={"status": status}
        )

        self.save_state()

    def save_state(self):
        """
        Save current workflow state.
        """
        self.state["updated_at"] = datetime.now().isoformat()

        with open(self.state_path, "w") as f:
            json.dump(make_json_serializable(self.state), f, indent=4)

        return self.state_path

    def save_json(self, filename, data):
        """
        Save arbitrary JSON data inside the workflow run directory.
        """
        save_path = self.run_dir / filename

        with open(save_path, "w") as f:
            json.dump(make_json_serializable(data), f, indent=4)

        self.add_artifact(
            artifact_name=Path(filename).stem,
            artifact_path=save_path,
            artifact_type="json",
            description=f"Saved JSON file: {filename}"
        )

        return save_path

    def save_text(self, filename, text):
        """
        Save text file inside the workflow run directory.
        """
        save_path = self.run_dir / filename

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(str(text))

        self.add_artifact(
            artifact_name=Path(filename).stem,
            artifact_path=save_path,
            artifact_type="text",
            description=f"Saved text file: {filename}"
        )

        return save_path

    def copy_artifact(self, source_path, artifact_name=None, artifact_type=None, description=None):
        """
        Copy an existing artifact into the workflow run directory.
        """
        source_path = Path(source_path)

        if not source_path.exists():
            raise FileNotFoundError(f"Artifact not found: {source_path}")

        destination_path = self.run_dir / source_path.name
        shutil.copy2(source_path, destination_path)

        if artifact_name is None:
            artifact_name = source_path.stem

        self.add_artifact(
            artifact_name=artifact_name,
            artifact_path=destination_path,
            artifact_type=artifact_type,
            description=description
        )

        return destination_path

    def get_summary(self):
        """
        Return compact workflow summary.
        """
        return {
            "run_id": self.run_id,
            "workflow_name": self.workflow_name,
            "run_dir": str(self.run_dir),
            "status": self.state.get("status"),
            "n_steps": len(self.state.get("steps", [])),
            "n_artifacts": len(self.state.get("artifacts", {})),
            "state_path": str(self.state_path),
            "events_path": str(self.events_path),
            "config_path": str(self.config_path),
            "artifacts_path": str(self.artifacts_path)
        }


if __name__ == "__main__":
    workflow = WorkflowState(workflow_name="materials_mas_test")

    workflow.save_config(
        {
            "dataset": "paa_hydrogel",
            "task": "generation",
            "model": "cvae"
        }
    )

    workflow.add_step(
        step_name="load_dataset",
        status="completed",
        details={"rows": 2336}
    )

    workflow.add_step(
        step_name="train_model",
        status="completed",
        details={"best_val_loss": 0.123}
    )

    workflow.update_status("completed")

    print("Workflow summary:")
    print(json.dumps(workflow.get_summary(), indent=4))