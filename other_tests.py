"""
Evaluate baseline models on a test dataset with configurable noise and sampling settings.
Loads model checkpoints, runs test evaluation, computes RMSE, and saves LaTeX-formatted results.
"""

import argparse
import pickle
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baseline_models.neural_laplace import GeneralNeuralLaplace
from baseline_models.ode_models import GeneralLatentODE
from baseline_models.original_ode_models import GeneralNODE
from baseline_models.sig_ode_models import SigGeneralLatentODE
from baseline_models.sig_neural_laplace import GeneralSigNeuralLaplace
from datasets import generate_data_set
from config import *

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="lotka_volterra_system_with_delay")
parser.add_argument("--noise_std", type=float, default=0.0)
parser.add_argument("--time_points_to_sample", type=int, default=200)
args = parser.parse_args()

dataset = args.dataset
noise_std = args.noise_std
time_points_to_sample = args.time_points_to_sample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

(
    input_dim,
    output_dim,
    dltrain,
    dlval,
    dltest,
    *_,
) = generate_data_set(
    dataset,
    device_param,
    double=False,
    batch_size=batch_size,
    trajectories_to_sample=trajectories_to_sample,
    extrap=extrapolate,
    normalize=normalize_dataset,
    noise_std=noise_std,
    t_nsamples=time_points_to_sample,
    observe_step=observe_step,
    predict_step=predict_step,
    coupling_factor=coupling_factor,
    percent_missing_at_random=percent_missing_at_random,
)

df_list_baseline_results = []

for seed in range(seed, seed + run_number_of_seeds):
    models = [
    ("Neural Laplace", GeneralNeuralLaplace(
        input_dim=input_dim,
        output_dim=output_dim,
        latent_dim=latent_dim,
        hidden_units=42,
        s_recon_terms=s_recon_terms,
        use_sphere_projection=use_sphere_projection,
        ilt_algorithm=ilt_algorithm,
        encode_obs_time=encode_obs_time,
        encoder_type='gru'
    ).to(device)),
    ("Sig Neural Laplace", GeneralNeuralLaplace(
        input_dim=input_dim,
        output_dim=output_dim,
        latent_dim=latent_dim,
        hidden_units=42,
        s_recon_terms=s_recon_terms,
        use_sphere_projection=use_sphere_projection,
        ilt_algorithm=ilt_algorithm,
        encode_obs_time=encode_obs_time,
        encoder_type='signature',
        signature_kwargs={
            "n_features": 4,
            "kernel_size": 40,
            "depth": 3,
            "stride": 1,
            "use_augment": True
        }
    ).to(device)),
    ("Neural Flow ResNet", GeneralLatentODE(
        dim=input_dim,
        model="flow",
        flow_model="resnet",
        hidden_dim=26,
        hidden_layers=latent_dim,
        latents=latent_dim,
        n_classes=input_dim,
        z0_encoder='ode_rnn'
    ).to(device)),
    ("Sig Neural Flow ResNet", GeneralLatentODE(
        dim=input_dim,
        model="flow",
        flow_model="resnet",
        hidden_dim=26,
        hidden_layers=latent_dim,
        latents=latent_dim,
        n_classes=input_dim,
        z0_encoder="signature",
        encoder_kwargs={
            "n_features": 4,
            "kernel_size": 40,
            "depth": 3,
            "stride": 1,
            "use_augment": True
        }
    ).to(device))
    ]

    path = f"results/exp_all_baselines-{dataset}-{seed}.pkl"
    with open(path, "rb") as f:
        saved_dict = pickle.load(f)

    for model_name, system in models:
        system.model.load_state_dict(saved_dict[model_name]["model_state_dict"])
        system.model.eval()
        system.model.to(device_param)
        _, test_mse = system.test_step(dltest)
        test_rmse = np.sqrt(test_mse.item())

        df_list_baseline_results.append({
            "method": model_name,
            "test_rmse": test_rmse,
            "seed": seed,
        })

df_results = pd.DataFrame(df_list_baseline_results)
test_rmse_df_inner = df_results.groupby("method")[["test_rmse"]].agg(["mean", "std"])
latex_str = test_rmse_df_inner.style.to_latex()

filename = f"other_tests_{dataset}_{noise_std}_{time_points_to_sample}"
with open(filename, "w") as f:
    f.write(latex_str)
