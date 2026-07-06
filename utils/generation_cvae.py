"""
generation_cvae.py

Conditional Variational Autoencoder for hydrogel genration.
"""

from pathlib import Path
import copy
import json
import random
import numpy as np
import pandas as pd

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "outputs" / "models"
GENERATED_DIR = REPO_ROOT / "outputs" / "generated_data"
REPORTS_DIR = REPO_ROOT / "outputs" / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    """
    Get GPU if available, else get CPU
    """

    return "cuda" if torch.cuda.is_available() else "cpu"

class ConditionalVAE(nn.Module):
    """
    Conditional VAE for tabular generation.
    """

    def __init__(self,
                 x_dim,
                 condition_dim,
                 latent_dim=30,
                 hidden_dims=(512, 256, 128),
                 dropout=0.2):
        
        super().__init__()
        self.x_dim = x_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout

        encoder_layers = []
        input_dim = x_dim + condition_dim

        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(input_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        decoder_layers = []
        decoder_input_dim = latent_dim + condition_dim

        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(decoder_input_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(dropout))
            decoder_input_dim = hidden_dim

        decoder_layers.append(nn.Linear(hidden_dims[0], x_dim))

        self.decoder = nn.Sequential(*decoder_layers) 

    def encode(self, x , condition):
        """
        Encode x and condition into latent distribution parameters.
        """           

        combined = torch.cat([x, condition], dim=1)
        h = self.encoder(combined)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick.
        """

        std = torch.exp(0.5*logvar)
        epsilon = torch.randn_like(std)

        return mu + epsilon * std
    
    def decode(self, z, condition):
        """
        Decode latent vecotr and condition into generated output variables.
        """

        combined = torch.cat([z, condition], dim=1)

        return self.decoder(combined)
    
    def forward(self, x, condition):
        """
        Forward pass.
        """

        mu, logvar = self.encode(x, condition)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, condition)

        return recon_x, mu, logvar
    
def cvae_loss(recon_x, x, mu, logvar, beta=0.001):
    """
    Custom loss function for CVAE.
    """

    mse = F.mse_loss(recon_x, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2)- logvar.exp())

    return mse + beta * kld
    

def encode_generation_dataframe(df, condition_columns, output_columns):
    """
    Convert condition and output coulumns into numeric matrices

    Categorical columns are one-hot encoded.
    Numeric columns are kept numeric.

    """

    condition_df = df[condition_columns].copy()
    output_df = df[output_columns].copy()

    condition_encoded = pd.get_dummies(condition_df, drop_first=False)
    output_encoded = pd.get_dummies(output_df, drop_first=False)

    condition_encoded = condition_encoded.apply(pd.to_numeric, errors="coerce")
    output_encoded = output_encoded.apply(pd.to_numeric, errors="coerce")

    combined = pd.concat([condition_encoded, output_encoded], axis=1)
    combined = combined.dropna().reset_index(drop=True)

    condition_encoded = combined[condition_encoded.columns]
    output_encoded = combined[output_encoded.columns]

    return condition_encoded, output_encoded

def prepare_cvae_data(
        df,
        condition_columns,
        output_columns,
        test_size=0.1,
        val_size=0.1,
        random_state=42
):
    """
    Prepare train/val/test split for CVAE
    """

    condition_encoded, output_encoded = encode_generation_dataframe(df, condition_columns=condition_columns, output_columns=output_columns)

    cond_train_val, cond_test, out_train_val, out_test = train_test_split(
        condition_encoded,
        output_encoded,
        test_size=test_size,
        random_state=random_state 
    )

    relative_val_size = val_size / (1.0 - test_size)

    cond_train, cond_val, out_train, out_val = train_test_split(
    cond_train_val,
    out_train_val,
    test_size=relative_val_size,
    random_state=random_state
        )

    condition_scaler = StandardScaler()
    output_scaler = StandardScaler()

    cond_train_scaled = condition_scaler.fit_transform(cond_train).astype(np.float32)
    cond_val_scaled = condition_scaler.transform(cond_val).astype(np.float32)
    cond_test_scaled = condition_scaler.transform(cond_test).astype(np.float32)

    out_train_scaled = output_scaler.fit_transform(out_train).astype(np.float32)
    out_val_scaled = output_scaler.transform(out_val).astype(np.float32)
    out_test_scaled = output_scaler.transform(out_test).astype(np.float32)

    return {
        "condition_columns": condition_columns,
        "output_columns": output_columns,
        "encoded_condition_columns": condition_encoded.columns.tolist(),
        "encoded_output_columns": output_encoded.columns.tolist(),
        "condition_train_original": cond_train.reset_index(drop=True),
        "condition_val_original": cond_val.reset_index(drop=True),
        "condition_test_original": cond_test.reset_index(drop=True),
        "output_train_original": out_train.reset_index(drop=True),
        "output_val_original": out_val.reset_index(drop=True),
        "output_test_original": out_test.reset_index(drop=True),
        "condition_train_scaled": cond_train_scaled,
        "condition_val_scaled": cond_val_scaled,
        "condition_test_scaled": cond_test_scaled,
        "output_train_scaled": out_train_scaled,
        "output_val_scaled": out_val_scaled,
        "output_test_scaled": out_test_scaled,
        "condition_scaler": condition_scaler,
        "output_scaler": output_scaler
    }

def make_cvae_loaders(prepared, batch_size=32):
    """
    Create PyTorch dataloader for CVAE
    """    

    train_condition = torch.tensor(prepared["condition_train_scaled"], dtype=torch.float32)
    train_output = torch.tensor(prepared["output_train_scaled"], dtype=torch.float32)

    val_condition = torch.tensor(prepared["condition_val_scaled"], dtype=torch.float32)
    val_output = torch.tensor(prepared["output_val_scaled"], dtype=torch.float32)    

    train_dataset = TensorDataset(train_condition, train_output)
    val_dataset = TensorDataset(val_condition, val_output)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def train_cvae(
        model,
        train_loader,
        val_loader,
        epochs=200,
        patience=20,
        beta=0.001,
        learning_rate=1e-3,
        weight_decay=0.01,
        device=None
):
    """
    Train CVAE with early stopping.
    """

    if device is None:
        device = get_device()

    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.7)

    best_val_loss = float('inf')
    best_state = None
    best_epoch = -1
    no_improve_epoch = 0

    history = {
        "train_loss" : [],
        "val_loss" : []
    }

    for epoch in range(epochs):
        model.train()
        train_total = 0.0 

        for condition_batch, output_batch in train_loader:
            condition_batch = condition_batch.to(device)
            output_batch = output_batch.to(device)

            optimizer.zero_grad()

            recon_batch , mu, logvar = model(output_batch, condition_batch)
            loss = cvae_loss(
                recon_batch,
                output_batch,
                mu,
                logvar,
                beta=beta)
            
            loss.backward()
            optimizer.step()
            train_total += loss.item()

        avg_train_loss = train_total / len(train_loader.dataset)
        history["train_loss"].append(float(avg_train_loss))

        model.eval()
        val_total = 0.0

        with torch.no_grad():
            for condition_batch, output_batch in val_loader:
                condition_batch = condition_batch.to(device)
                output_batch = output_batch.to(device)

                recon_batch, mu, logvar = model(output_batch, condition_batch)
                val_loss = cvae_loss(
                    recon_batch,
                    output_batch,
                    mu,
                    logvar,
                    beta=beta
                )

                val_total += val_loss.item()   

        avg_val_loss = val_total / len(val_loader.dataset)
        history["val_loss"].append(float(avg_val_loss))

        print(
            f"Epoch: {epoch+1}, "
            f"Train loss: {avg_train_loss:.4f}, "
            f"Val loss: {avg_val_loss:.4f}"
        )        

        if avg_val_loss<best_val_loss:
            best_val_loss = avg_val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            no_improve_epoch = 0
        else:
            no_improve_epoch += 1

        if no_improve_epoch >= patience:
            print(f"Early stopping epoch {epoch + 1}")
            break

        scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Loaded best CVAE model with val loss {best_val_loss:.4f}")

    return model, {
        "best_val_loss" : float(best_val_loss),
        "best_epoch" : best_epoch,
        "history" : history 
    }                

def generate_with_cvae(
        model,
        condition_df,
        condition_scaler,
        output_scaler,
        encoded_output_columns,
        latent_dim=30,
        num_samples_per_condition=1,
        device=None
):
    """
    Generate output variables from condition dataframe.
    """

    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    condition_scaled = condition_scaler.transform(condition_df).astype(np.float32)
    condition_tensor = torch.tensor(condition_scaled, dtype=torch.float32, device=device)

    generated_rows = []
    condition_rows = []

    with torch.no_grad():
        for _ in range(num_samples_per_condition):
            z = torch.randn(condition_tensor.size(0), latent_dim, device=device)
            generated_scaled = model.decode(z, condition_tensor)
            generated_scaled = generated_scaled.cpu().numpy()
            generated_original = output_scaler.inverse_transform(generated_scaled)

            generated_rows.append(generated_original)
            condition_rows.append(condition_df.values) 

    generated_rows = np.vstack(generated_rows)
    condition_rows = np.vstack(condition_rows)

    generated_output_df = pd.DataFrame(
        generated_rows,
        columns=encoded_output_columns
    )           

    repeated_condition_df = pd.DataFrame(
        condition_rows,
        columns=condition_df.columns
    )

    return repeated_condition_df, generated_output_df

def postprocess_generated_outputs(
        generated_output_df,
        real_output_df,
        clip_to_training_range=True,
        round_encoded_columns=False
):
    """
    Postprocess generated output variables.
    """

    generated_output_df = generated_output_df.copy()

    if clip_to_training_range:
        for col in generated_output_df.columns:
            if col in real_output_df.columns:
                min_val = real_output_df[col].min()
                max_val = real_output_df[col].max()
                generated_output_df[col] = generated_output_df[col].clip(min_val, max_val)

    if round_encoded_columns:
        for col in generated_output_df.columns:
            col_lower = str(col).lower()

            if col_lower.endswith("_encoded"):
                generated_output_df[col] = generated_output_df[col].round()

                if col in real_output_df.columns:
                    min_val = real_output_df[col].min()
                    max_val = real_output_df[col].max()
                    generated_output_df[col] = generated_output_df[col].clip(min_val, max_val)

    return generated_output_df

def run_cvae_generation(
        df,
        condition_columns,
        output_columns,
        generation_condition_df=None,
        run_name="cvae_generation",
        latent_dim=30,
        hidden_dims=(512, 256, 128),
        dropout=0.2,
        beta=0.001,
        learning_rate=1e-3,
        epochs=200,
        patience=20,
        batch_size=32,
        num_samples_per_condition=1,
        round_encoded_columns=False,
        seed=42
):
    """
    Full CVAE workflow.
    """                

    set_seed(seed)
    device = get_device()
    print(f"Using device: {device}")

    prepared = prepare_cvae_data(
        df,
        condition_columns=condition_columns,
        output_columns=output_columns,
        random_state=seed
    )       

    if generation_condition_df is None:
        generation_condition_df = prepared["condition_test_original"]
    else:
        missing_columns = [
            col for col in condition_columns
            if col not in generation_condition_df.columns
            ]

        if missing_columns:
            raise ValueError(
                f"generation_condition_df is missing required condition columns: {missing_columns}"
            )

        generation_condition_df = generation_condition_df[condition_columns].copy()

        generation_condition_df = pd.get_dummies(generation_condition_df, drop_first=False)

        for col in prepared["encoded_condition_columns"]:
            if col not in generation_condition_df.columns:
                generation_condition_df[col] = 0

        generation_condition_df = generation_condition_df[prepared["encoded_condition_columns"]]

    train_loader, val_loader = make_cvae_loaders(prepared, batch_size=batch_size)

    x_dim = prepared["output_train_scaled"].shape[1]
    condition_dim = prepared["condition_train_scaled"].shape[1]

    model = ConditionalVAE(
        x_dim=x_dim,
        condition_dim=condition_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout
    )

    model, train_report = train_cvae(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        patience=patience,
        beta=beta,
        learning_rate=learning_rate,
        device=device
    )

    repeated_condition_df, generated_output_df = generate_with_cvae(
        model=model,
        condition_df=generation_condition_df,
        condition_scaler=prepared["condition_scaler"],
        output_scaler=prepared["output_scaler"],
        encoded_output_columns=prepared["encoded_output_columns"],
        latent_dim=latent_dim,
        num_samples_per_condition=num_samples_per_condition,
        device=device
    )

    generated_output_df = postprocess_generated_outputs(
        generated_output_df=generated_output_df,
        real_output_df=prepared["output_train_original"],
        clip_to_training_range=True,
        round_encoded_columns=round_encoded_columns
    )

    synthetic_df = pd.concat(
        [
            generated_output_df.reset_index(drop=True),
            repeated_condition_df.reset_index(drop=True)
        ],
        axis=1
    )

    model_path = MODELS_DIR / f"{run_name}_model.pt"
    synthetic_path = GENERATED_DIR / f"{run_name}_synthetic.csv"
    report_path = REPORTS_DIR / f"{run_name}_training_report.json"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "x_dim": x_dim,
            "condition_dim": condition_dim,
            "latent_dim": latent_dim,
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "condition_columns": condition_columns,
            "output_columns": output_columns,
            "encoded_condition_columns": prepared["encoded_condition_columns"],
            "encoded_output_columns": prepared["encoded_output_columns"],
            "condition_scaler": prepared["condition_scaler"],
            "output_scaler": prepared["output_scaler"],
            "train_report": train_report
        },
        model_path
    )

    synthetic_df.to_csv(synthetic_path, index=False)

    report = {
        "model_type": "CVAE",
        "run_name": run_name,
        "condition_columns": condition_columns,
        "output_columns": output_columns,
        "encoded_condition_columns": prepared["encoded_condition_columns"],
        "encoded_output_columns": prepared["encoded_output_columns"],
        "x_dim": x_dim,
        "condition_dim": condition_dim,
        "latent_dim": latent_dim,
        "hidden_dims": list(hidden_dims),
        "dropout": dropout,
        "beta": beta,
        "learning_rate": learning_rate,
        "epochs_requested": epochs,
        "patience": patience,
        "best_val_loss": train_report["best_val_loss"],
        "best_epoch": train_report["best_epoch"],
        "synthetic_shape": synthetic_df.shape,
        "model_path": str(model_path),
        "synthetic_path": str(synthetic_path)
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    return {
        "model": model,
        "prepared": prepared,
        "synthetic_df": synthetic_df,
        "model_path": str(model_path),
        "synthetic_path": str(synthetic_path),
        "report_path": str(report_path),
        "training_report": report
    }

if __name__ == "__main__":
    from utils.material_dataset_utils import preview_dataset
    from utils.material_schema_discovery import build_schema_report

    df = preview_dataset("paa_hydrogel")
    schema_report = build_schema_report(df)

    condition_columns = schema_report["suggested_generation_candidates"]["condition_columns"]
    output_columns = schema_report["suggested_generation_candidates"]["output_columns"]

    result = run_cvae_generation(
        df,
        condition_columns=condition_columns,
        output_columns=output_columns,
        run_name="paa_hydrogel_cvae_test",
        epochs=10,
        patience=3
    )

    print(f"Synthetic CSV: {result['synthetic_path']}")
    print(f"Report: {result['report_path']}")




