"""
generation_ddpm.py

Conditional DDPM utilities for hydrogel material generation.
"""

from pathlib import Path
import copy
import json
import math
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.generation_cvae import (
    prepare_cvae_data,
    postprocess_generated_outputs
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "outputs" / "models"
GENERATED_DIR = REPO_ROOT / "outputs" / "generated_data"
REPORTS_DIR = REPO_ROOT / "outputs" / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed=42):
    """
    Set random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Use GPU if available, otherwise CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_beta_scheduler(timesteps, beta_start=1e-4, beta_end=2e-2):
    """
    Create a liner beta scheduler for diffusion.
    """

    
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

def extract(values, timesteps, x_shape):
    """
    Extract timestep specific values and reshape for broadcasting
    """

    out = values.gather(0, timesteps)
    return out.view(-1, 1).expand(x_shape)

class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding for DDPM.
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        half_dim = self.embedding_dim // 2
        emb_scale = math.log(10000) / max(half_dim-1, 1)

        emb = torch.exp(torch.arange(half_dim, device=timesteps.device,)*-emb_scale)

        emb = timesteps.float().unsqueeze(1)*emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        if self.embedding_dim % 2 == 1:
            emb = F.pad(emb, (0,1))

        return emb

class ConditionalDenoiser(nn.Module):    
    """
    Conditional denoising network for tabular DDPM.
    """

    def __init__(self, x_dim, condition_dim, hidden_dim=256, time_dim=64, dropout=0.1):
        super().__init__()

        self.x_dim = x_dim
        self.condition_dim = condition_dim
        self.time_dim = time_dim

        self.time_embedding = SinusoidalTimeEmbedding(time_dim)

        input_dim = x_dim + condition_dim + time_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, x_dim)
        )    

    def forward(self, x_t, timesteps, condition):
        time_emb = self.time_embedding(timesteps)
        combined = torch.cat([x_t, condition, time_emb], dim=1)

        return self.net(combined)

class ConditionalTabularDDPM(nn.Module):
    """
    Conditional DDPM wrapper for tabular data.
    """

    def __init__(
        self,
        x_dim,
        condition_dim,
        timesteps=1000,
        beta_start=1e-4,
        beta_end=2e-2,
        hidden_dim=256,
        time_dim=64,
        dropout=0.1
    ):
        super().__init__()

        self.x_dim = x_dim
        self.condition_dim = condition_dim
        self.num_timesteps = timesteps

        self.denoiser = ConditionalDenoiser(
            x_dim=x_dim,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            dropout=dropout
        )

        betas = make_beta_scheduler(
            timesteps,
            beta_start=beta_start,
            beta_end=beta_end
        )

        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        alpha_bars_prev = torch.cat(
            [torch.tensor([1.0], dtype=torch.float32), alpha_bars[:-1]],
            dim=0
        )

        posterior_variance = (
            betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        )
        posterior_variance[0] = 1e-8

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer(
            "sqrt_one_minus_alpha_bars",
            torch.sqrt(1.0 - alpha_bars)
        )
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("posterior_variance", posterior_variance)

    def q_sample(self, x_0, timesteps, noise):
        """
        Add noise to clean data x_0 at selected timesteps.
        """

        sqrt_alpha_bar = extract(
            self.sqrt_alpha_bars,
            timesteps,
            x_0.shape
        )

        sqrt_one_minus_alpha_bar = extract(
            self.sqrt_one_minus_alpha_bars,
            timesteps,
            x_0.shape
        )

        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
    
    def compute_loss(self, x_0, condition):
        """
        DDPM noise prediction loss.
        """

        batch_size = x_0.size(0)

        timesteps = torch.randint(
            0,
            self.num_timesteps,
            (batch_size,),
            device=x_0.device
        ).long()

        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, timesteps, noise)

        predicted_noise = self.denoiser(x_t, timesteps, condition)

        return F.mse_loss(predicted_noise, noise)
    
    def sample(self, condition, num_steps=None):
        """
        Generate x from random noise conditioned on condition variables.
        """

        if num_steps is None:
            num_steps = self.num_timesteps

        num_steps = min(num_steps, self.num_timesteps)

        self.eval()

        n_samples = condition.size(0)
        x_t = torch.randn(n_samples, self.x_dim, device=condition.device)

        sampling_schedule = torch.linspace(
            self.num_timesteps - 1,
            0,
            num_steps,
            device=condition.device
        ).long()

        with torch.no_grad():
            for step in sampling_schedule:
                timesteps = torch.full(
                    (n_samples,),
                    int(step.item()),
                    device=condition.device,
                    dtype=torch.long
                )

                beta_t = extract(self.betas, timesteps, x_t.shape)

                sqrt_one_minus_alpha_bar_t = extract(
                    self.sqrt_one_minus_alpha_bars,
                    timesteps,
                    x_t.shape
                )

                sqrt_recip_alpha_t = extract(
                    self.sqrt_recip_alphas,
                    timesteps,
                    x_t.shape
                )

                predicted_noise = self.denoiser(x_t, timesteps, condition)

                model_mean = sqrt_recip_alpha_t * (
                    x_t - (beta_t / sqrt_one_minus_alpha_bar_t) * predicted_noise
                )

                if int(step.item()) > 0:
                    variance_t = extract(
                        self.posterior_variance,
                        timesteps,
                        x_t.shape
                    )
                    noise = torch.randn_like(x_t)
                    x_t = model_mean + torch.sqrt(variance_t) * noise
                else:
                    x_t = model_mean

        return x_t

def make_ddpm_loaders(prepared, batch_size=32):
    """
    Create PyTorch dataloaders for DDPM
    """            
    train_output = torch.tensor(prepared["output_train_scaled"], dtype=torch.float32)
    train_condition = torch.tensor(prepared["condition_train_scaled"], dtype=torch.float32)

    val_output = torch.tensor(prepared["output_val_scaled"], dtype=torch.float32)
    val_condition = torch.tensor(prepared["condition_val_scaled"], dtype=torch.float32)

    train_dataset = TensorDataset(train_output, train_condition)
    val_dataset = TensorDataset(val_output, val_condition)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def evaluate_ddpm_loss(model, data_loader, device):
    """
    Evaluate DDPM validation loss.
    """
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for output_batch, condition_batch in data_loader:
            output_batch = output_batch.to(device)
            condition_batch = condition_batch.to(device)

            loss = model.compute_loss(output_batch, condition_batch)

            batch_size = output_batch.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples

def train_ddpm(
    model,
    train_loader,
    val_loader,
    epochs=500,
    patience=20,
    learning_rate=1e-3,
    weight_decay=0.01,
    device=None
):
    """
    Train conditional DDPM with early stopping.
    """
    if device is None:
        device = get_device()

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=8,
        threshold=1e-4,
        min_lr=1e-6
    )

    best_state = None
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_no_improve = 0

    history = {
        "train_loss": [],
        "val_loss": []
    }

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        total_samples = 0

        for output_batch, condition_batch in train_loader:
            output_batch = output_batch.to(device)
            condition_batch = condition_batch.to(device)

            loss = model.compute_loss(output_batch, condition_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = output_batch.size(0)
            running_train_loss += loss.item() * batch_size
            total_samples += batch_size

        train_loss = running_train_loss / total_samples
        val_loss = evaluate_ddpm_loss(model, val_loader, device)

        scheduler.step(val_loss)

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"[DDPM] Epoch {epoch + 1:4d}/{epochs} | "
                f"train_loss = {train_loss:.6f} | "
                f"val_loss = {val_loss:.6f} | "
                f"best_val = {best_val_loss:.6f} @ epoch {best_epoch}"
            )

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {
        "best_val_loss": float(best_val_loss),
        "best_epoch": best_epoch,
        "history": history
    }


def generate_with_ddpm(
        model,
        condition_df,
        condition_scaler,
        output_scaler,
        encoded_output_columns,
        num_samples_per_condition=1,
        sampling_steps=None,
        device=None
):
    """
    Generate output variables from condition dataframe using DDPM.
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
            generated_scaled = model.sample(
                condition_tensor,
                num_steps=sampling_steps
            )    

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

def run_ddpm_generation(
    df,
    condition_columns,
    output_columns,
    generation_condition_df=None,
    run_name="ddpm_generation",
    timesteps=1000,
    sampling_steps=None,
    hidden_dim=256,
    time_dim=64,
    dropout=0.1,
    learning_rate=1e-3,
    weight_decay=0.01,
    epochs=500,
    patience=20,
    batch_size=32,
    num_samples_per_condition=1,
    round_encoded_columns=False,
    seed=42
):
    """
    Full DDPM workflow:
    - prepare data
    - train conditional DDPM
    - generate synthetic data for test conditions
    - clip generated outputs to training range
    - save model, synthetic CSV, and training report
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

    train_loader, val_loader = make_ddpm_loaders(
        prepared,
        batch_size=batch_size
    )

    x_dim = prepared["output_train_scaled"].shape[1]
    condition_dim = prepared["condition_train_scaled"].shape[1]

    model = ConditionalTabularDDPM(
        x_dim=x_dim,
        condition_dim=condition_dim,
        timesteps=timesteps,
        hidden_dim=hidden_dim,
        time_dim=time_dim,
        dropout=dropout
    )

    model, train_report = train_ddpm(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        patience=patience,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device
    )

    repeated_condition_df, generated_output_df = generate_with_ddpm(
        model=model,
        condition_df=generation_condition_df,
        condition_scaler=prepared["condition_scaler"],
        output_scaler=prepared["output_scaler"],
        encoded_output_columns=prepared["encoded_output_columns"],
        num_samples_per_condition=num_samples_per_condition,
        sampling_steps=sampling_steps,
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
            "timesteps": timesteps,
            "hidden_dim": hidden_dim,
            "time_dim": time_dim,
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
        "model_type": "DDPM",
        "run_name": run_name,
        "condition_columns": condition_columns,
        "output_columns": output_columns,
        "encoded_condition_columns": prepared["encoded_condition_columns"],
        "encoded_output_columns": prepared["encoded_output_columns"],
        "x_dim": x_dim,
        "condition_dim": condition_dim,
        "timesteps": timesteps,
        "sampling_steps": sampling_steps,
        "hidden_dim": hidden_dim,
        "time_dim": time_dim,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
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

    result = run_ddpm_generation(
        df,
        condition_columns=condition_columns,
        output_columns=output_columns,
        run_name="paa_hydrogel_ddpm_test",
        timesteps=100,
        sampling_steps=100,
        epochs=5,
        patience=2
    )

    print(f"Synthetic CSV, {result['synthetic_path']}")
    print(f"Report: {result['report_path']}")