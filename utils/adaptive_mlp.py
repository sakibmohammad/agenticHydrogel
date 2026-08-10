"""
adaptive_mlp.py

Adaptive MLP training utilities for hydrogel workflow.
"""

from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT/"outputs"/"models"
REPORTS_DIR = REPO_ROOT/"outputs"/"reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def get_device():
    """
    Use GPU if available
    """
    return "cuda" if torch.cuda.is_available() else "cpu"

def make_hidden_layers(base_neurons=64, base_layers=2, attempt_index=0):
    """
    Create hidden layer sizes for each adaptive attempt.
    """
    n_layers = base_layers + attempt_index
    neurons = int(round(base_neurons*(2**attempt_index)))
    return tuple([neurons]*n_layers)

def to_numpy_array(data):
    """
    Convert pandas/numpy list to numpy array
    """

    if hasattr(data, "values"):
        data = data.values

    data = np.asarray(data)

    return data

def make_json_serializable(obj):
    """
    Convert numpy/torch objects into normal Python objects for JSON saving.
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

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()

    return obj

def prepare_regression_arrays(X_train, X_test, y_train, y_test):
    """
    Convert regression arrays to float32 numpy arrays.
    """    

    X_train = to_numpy_array(X_train).astype(np.float32)
    X_test = to_numpy_array(X_test).astype(np.float32)

    y_train = to_numpy_array(y_train).astype(np.float32)
    y_test = to_numpy_array(y_test).astype(np.float32)

    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)

    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)

    return X_train, X_test, y_train, y_test

def prepare_classification_arrays(X_train, X_test, y_train, y_test):
    """
    Convert classification arrays to correct numpy datatypes.
    """
    X_train = to_numpy_array(X_train).astype(np.float32)
    X_test = to_numpy_array(X_test).astype(np.float32)

    y_train = to_numpy_array(y_train).astype(np.int64)
    y_test = to_numpy_array(y_test).astype(np.int64)

    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    original_classes = np.array(
        sorted(np.unique(np.concatenate([y_train, y_test])))
    )

    class_to_index = {
        original_class: index
        for index, original_class in enumerate(original_classes)
    }

    y_train_encoded = np.array(
        [class_to_index[value] for value in y_train],
        dtype=np.int64
    )

    y_test_encoded = np.array(
        [class_to_index[value] for value in y_test],
        dtype=np.int64
    )

    return X_train, X_test, y_train_encoded, y_test_encoded, original_classes.tolist()

class TabularMLP(nn.Module):
    """
    Simple fully connected MLP for tabular regression/classification.
    """       

    def __init__(self, input_dim, output_dim, hidden_layer_sizes=(64, 64), dropout=0.0):
        super().__init__()

        layers = []
        previous_dim = input_dim

        for hidden_dim in hidden_layer_sizes:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def safe_mape(y_true, y_pred):
    """
    Mean absolute percentage error w/ 0 protection.
    """                

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    epsilon = 1e-8

    denominator = np.where(np.abs(y_true)<epsilon, epsilon, np.abs(y_true))

    return np.mean(np.abs((y_true-y_pred)/denominator))*100

def evaluate_regression(y_true, y_pred):
    """
    Evaluate regression precedure.
    """

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    metrics = {
        "r2" : float(r2_score(y_true, y_pred, multioutput="uniform_average")),
        "mae" : float(mean_absolute_error(y_true, y_pred)),
        "mse" : float(mse),
        "rmse" : float(rmse),
        "mape_percent" : float(safe_mape(y_true, y_pred))
    }

    return metrics

def evaluate_classification(y_true, y_pred):
    """
    Evaluate classification procedure.
    """
    metrics = {
        "accuracy" : float(accuracy_score(y_true, y_pred)),
        "precision_macro" : float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        "recall_macro" : float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        "f1_macro" : float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        "confusion_matrix" : confusion_matrix(y_true, y_pred).tolist(),
        "classification_report" : classification_report(y_true, y_pred, zero_division=0)
    }

    return metrics

def train_one_regression_model(X_train, y_train, X_test, y_test, hidden_layer_sizes, batch_size=32, learning_rate=1e-3, weight_decay=0.01, epochs=100, patience=20, dropout=0.2, random_state=42):
    """
    Train one MLP regression model.
    """

    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)
    np.random.seed(random_state)

    device = get_device()

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    model = TabularMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_layer_sizes=hidden_layer_sizes,
        dropout=dropout
    ).to(device)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    best_state = None
    best_val_loss = np.inf
    patience_counter = 0

    history = {
        "train_loss" : [],
        "val_loss" : []
    }

    for epoch in range(epochs):
        model.train()
        batch_losses = []

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses))

        model.eval()    
        with torch.no_grad():
            val_preds = model(X_test_tensor)
            val_loss = criterion(val_preds, y_test_tensor).item()
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(float(val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).cpu().numpy()

    return model, y_pred, history

def train_one_classification_model(X_train, y_train, X_test, y_test, hidden_layer_sizes, batch_size=32, learning_rate=1e-3, weight_decay=0.01, epochs=100, patience=20, dropout=0.2, random_state=42):
    """
    Train one MLP classification model.
    """

    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)
    np.random.seed(random_state)

    device = get_device()

    input_dim = X_train.shape[1]
    n_classes = int(len(np.unique(y_train)))

    model = TabularMLP(
        input_dim=input_dim,
        output_dim=n_classes,
        hidden_layer_sizes=hidden_layer_sizes,
        dropout=dropout
    ).to(device)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    best_state = None
    best_val_loss = np.inf
    patience_counter = 0

    history = {
        "train_loss" : [],
        "val_loss" : []
    }

    for epoch in range(epochs):
        model.train()
        batch_losses = []

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses))

        model.eval()    
        with torch.no_grad():
            val_logits = model(X_test_tensor)
            val_loss = criterion(val_logits, y_test_tensor).item()
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(float(val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor)
        y_pred = torch.argmax(logits, dim=1).cpu().numpy()

    return model, y_pred, history

def inverse_transformation_regression(y_test, y_pred, y_scaler):
    """
    If y was scaled during training then bring it back to regular values.
    """

    if y_scaler is None:
        return y_test, y_pred
    
    y_test_orginial = y_scaler.inverse_transform(y_test)
    y_pred_original = y_scaler.inverse_transform(y_pred)

    return y_test_orginial, y_pred_original

def adaptive_regression_training(
        prepared_data,
        threshold=0.60,
        max_retries=3,
        base_neurons=64,
        base_layers=2,
        batch_size=32,
        learning_rate=1e-3,
        epochs=100,
        patience=40,
        dropout=0.2,
        run_name="adaptive_regression"
):
    """
    Train adaptive regression model.
    """

    X_train, X_test, y_train, y_test = prepare_regression_arrays(
        prepared_data["X_train"],
        prepared_data["X_test"],
        prepared_data["y_train"],
        prepared_data["y_test"]
    )

    y_scaler = prepared_data.get("y_scaler")

    attempts = []
    best_model = None
    best_metrics = None
    best_score = -np.inf
    best_hidden_layers = None
    best_history = None

    for attempt_index in range(max_retries+1):
        hidden_layers = make_hidden_layers(
            base_neurons=base_neurons,
            base_layers=base_layers,
            attempt_index=attempt_index
        )

        print(f"\nRegression attempt: {attempt_index+1}")
        print(f"Hidden layers : {hidden_layers}")
        print(f"Device: {get_device()}")

        model, y_pred, history = train_one_regression_model(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            hidden_layer_sizes=hidden_layers,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            patience=patience,
            dropout=dropout
        )

        y_test_eval, y_pred_eval = inverse_transformation_regression(y_test, y_pred, y_scaler)

        metrics = evaluate_regression(y_test_eval, y_pred_eval)

        attempt_results = {
            "attempt" : attempt_index + 1,
            "hidden_layer_sizes": hidden_layers,
            "epochs_ran" : len(history["train_loss"]),
            "metrics" : metrics
        }

        attempts.append(attempt_results)

        print(f"Epochs ran: {len(history['train_loss'])}")
        print(f"R2: {metrics['r2']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")

        if metrics["r2"] > best_score:
            best_score = metrics["r2"]
            best_model = model
            best_metrics = metrics
            best_hidden_layers = hidden_layers
            best_history = history

        if metrics["r2"] > threshold:
            print("Regression threhold reached!")
            break

    result = {
        "task" : "regression",
        "threshold" : threshold,
        "threshold_metrics" : "r2",
        "threshold_reached" : best_score >= threshold,
        "best_score" : float(best_score),
        "best_hidden_layer_sizes" : best_hidden_layers,
        "best_metrics" : best_metrics,
        "attempts" : attempts,
        "feature_columns" : prepared_data.get("feature_columns"),
        "encoded_feature_columns" : prepared_data.get("encoded_feature_columns"),
        "target_columns" : prepared_data.get("target_columns"),
        "best_history" : best_history
    }        

    model_path = MODELS_DIR / f"{run_name}_best_model.pt"
    report_path = REPORTS_DIR / f"{run_name}.metrics.json"

    torch.save ({
        "model_state_dict" : best_model.state_dict(),
        "input_dim" : X_train.shape[1],
        "output_dim" : y_train.shape[1],
        "hidden_layer_sizes" : best_hidden_layers,
        "task" : "regression",
        "feature_columns" : prepared_data.get("feature_columns"),
        "encoded_feature_columns" : prepared_data.get("encoded_feature_columns"),
        "target_columns" : prepared_data.get("target_columns"),
    }, model_path
    )

    with open(report_path, "w") as f:
        json.dump(make_json_serializable(result), f, indent=4)

    result["model_path"] = str(model_path)
    result["report_path"] = str(report_path)

    return result


def adaptive_classification_training(
        prepared_data,
        threshold=0.60,
        max_retries=3,
        base_neurons=64,
        base_layers=2,
        batch_size=32,
        learning_rate=1e-3,
        epochs=100,
        patience=40,
        dropout=0.2,
        run_name="adaptive_classification"
):
    """
    Train adaptive classification model.
    """

    X_train, X_test, y_train, y_test, original_classes = prepare_classification_arrays(
        prepared_data["X_train"],
        prepared_data["X_test"],
        prepared_data["y_train"],
        prepared_data["y_test"]
    )

    n_classes = int(len(original_classes))

    attempts = []
    best_model = None
    best_metrics = None
    best_score = -np.inf
    best_hidden_layers = None
    best_history = None
    best_y_pred = None
    best_y_test = None

    for attempt_index in range(max_retries + 1):
        hidden_layers = make_hidden_layers(
            base_neurons=base_neurons,
            base_layers=base_layers,
            attempt_index=attempt_index
        )

        print(f"\nClassification attempt: {attempt_index + 1}")
        print(f"Hidden layers : {hidden_layers}")
        print(f"Device: {get_device()}")

        model, y_pred, history = train_one_classification_model(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            hidden_layer_sizes=hidden_layers,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            patience=patience,
            dropout=dropout
        )

        metrics = evaluate_classification(y_test, y_pred)

        attempt_results = {
            "attempt": attempt_index + 1,
            "hidden_layer_sizes": list(hidden_layers),
            "epochs_ran": len(history["train_loss"]),
            "metrics": metrics
        }

        attempts.append(attempt_results)

        print(f"Epochs ran: {len(history['train_loss'])}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 macro: {metrics['f1_macro']:.4f}")

        if metrics["accuracy"] > best_score:
            best_score = metrics["accuracy"]
            best_model = model
            best_metrics = metrics
            best_hidden_layers = hidden_layers
            best_history = history
            best_y_pred = y_pred.copy()
            best_y_test = y_test.copy()

        if metrics["accuracy"] > threshold:
            print("Classification threshold reached!")
            break

    if best_model is None:
        raise RuntimeError("Classification training failed: no model was trained.")

    original_classes_array = np.asarray(original_classes)

    best_y_test_original = original_classes_array[best_y_test]
    best_y_pred_original = original_classes_array[best_y_pred]

    predictions = {
        "actual_encoded": best_y_test.tolist(),
        "predicted_encoded": best_y_pred.tolist(),
        "actual_original": best_y_test_original.tolist(),
        "predicted_original": best_y_pred_original.tolist()
    }

    result = {
        "task": "classification",
        "threshold": threshold,
        "threshold_metrics": "accuracy",
        "threshold_reached": best_score >= threshold,
        "best_score": float(best_score),
        "best_hidden_layer_sizes": list(best_hidden_layers),
        "best_metrics": best_metrics,
        "attempts": attempts,
        "feature_columns": prepared_data.get("feature_columns"),
        "encoded_feature_columns": prepared_data.get("encoded_feature_columns"),
        "target_columns": prepared_data.get("target_columns"),
        "class_names": prepared_data.get("class_names"),
        "original_classes": original_classes,
        "best_history": best_history,
        "predictions": predictions
    }

    model_path = MODELS_DIR / f"{run_name}_best_model.pt"
    report_path = REPORTS_DIR / f"{run_name}.metrics.json"

    torch.save(
        {
            "model_state_dict": best_model.state_dict(),
            "input_dim": int(X_train.shape[1]),
            "output_dim": n_classes,
            "hidden_layer_sizes": list(best_hidden_layers),
            "task": "classification",
            "feature_columns": prepared_data.get("feature_columns"),
            "encoded_feature_columns": prepared_data.get("encoded_feature_columns"),
            "target_columns": prepared_data.get("target_columns"),
            "class_names": prepared_data.get("class_names"),
            "original_classes": original_classes,
        },
        model_path
    )

    with open(report_path, "w") as f:
        json.dump(make_json_serializable(result), f, indent=4)

    result["model_path"] = str(model_path)
    result["report_path"] = str(report_path)

    return result

def run_adaptive_mlp(
        prepared_data,
        threshold=0.60,
        max_retries=3,
        base_neurons=64,
        base_layers=2,
        batch_size=32,
        learning_rate=1e-3,
        epochs=100,
        patience=40,
        dropout=0.2,
        run_name=None
):
    """
    General wrapper for MLP training.
    Task is inferred from prepared_data["task"].
    """
    task = prepared_data.get("task")

    if task == "regression":
        return adaptive_regression_training(
            prepared_data=prepared_data,
            threshold=0.60 if threshold is None else threshold,
            max_retries=max_retries,
            base_neurons=base_neurons,
            base_layers=base_layers,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            patience=patience,
            dropout=dropout,
            run_name="adaptive_regression" if run_name is None else run_name
        )

    if task == "classification":
        return adaptive_classification_training(
            prepared_data=prepared_data,
            threshold=0.60 if threshold is None else threshold,
            max_retries=max_retries,
            base_neurons=base_neurons,
            base_layers=base_layers,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            patience=patience,
            dropout=dropout,
            run_name="adaptive_classification" if run_name is None else run_name
        )

    raise ValueError(
        f"Unsupported task for adaptive MLP: {task}. "
        "Expected: 'regression' or 'classification'."
    )




if __name__ == "__main__":
    from utils.material_dataset_utils import preview_dataset
    from utils.material_preprocessing import prepare_regression_data

    df = preview_dataset("paa_hydrogel")

    target_columns = [
        "Storage modulus (Pa)",
        "Loss modulus (Pa)"
    ]
    reg_data = prepare_regression_data(
        df, 
        target_columns=target_columns
    )

    result = run_adaptive_mlp(reg_data, threshold=0.60, run_name="paa_hydrogel_regression_test")

    print(f"\nFinal results:")
    print(f"Best score: {result['best_score']}")
    print(f"Threshold reached: {result['threshold_reached']}")
    print(f"Model path : {result['model_path']}")
    print(f"Report path: {result["report_path"]}")





       
