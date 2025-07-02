# pytype: skip-file
"""
This module runs sensitivity and ablation studies comparing signature-based models 
to RNN-based and ODE-based baselines across multiple dynamical systems with delay.
"""

# Based on the original implementation by Samuel Holt, 
# available at: https://github.com/samholt/NeuralLaplace

import argparse
import logging
import pickle
from pathlib import Path
from time import strftime
import random
import inspect
import os
import numpy as np
import pandas as pd
import torch
from baseline_models.neural_laplace import GeneralNeuralLaplace
from baseline_models.ode_models import GeneralLatentODE
from baseline_models.original_ode_models import GeneralNODE
from datasets import generate_data_set
from utils import train_and_test
from config import *

datasets = [
    "spiral_dde",
    "lotka_volterra_system_with_delay",
    "fitzhugh_nagumo_with_delay",
    "rossler_system_with_delay"
]

np.random.seed(999)
torch.random.manual_seed(999)

file_name = Path(__file__).stem

def experiment_with_all_baselines(
    dataset,
    device_param,
    path_name
):
  observe_samples = (time_points_to_sample // 2) // observe_step
  logger.info("Experimentally observing %d samples", observe_samples)  # pylint: disable=possibly-used-before-assignment

  df_list_baseline_results = []

  for seed in range(start_seed, start_seed + run_number_of_seeds):
    torch.random.manual_seed(seed)

    Path("./results_temp").mkdir(parents=True, exist_ok=True)
    path = f"./results_temp/{path_name}-{seed}.pkl"

    (
        input_dim,
        output_dim,
        dltrain,
        dlval,
        dltest,
        _,
        _,
        _,
    ) = generate_data_set(
        dataset,
        device_param,
        double=True,
        batch_size=batch_size,
        trajectories_to_sample=trajectories_to_sample,
        extrap=extrapolate,
        normalize=normalize_dataset,
        noise_std=noise_std,
        t_nsamples=time_points_to_sample,
        observe_step=observe_step,
        predict_step=predict_step,
    )

    saved_dict = {}

    saved_dict["dataset"] = dataset
    saved_dict["trajectories_to_sample"] = trajectories_to_sample
    saved_dict["extrapolate"] = extrapolate
    saved_dict["normalize_dataset"] = normalize_dataset
    saved_dict["input_dim"] = input_dim
    saved_dict["output_dim"] = output_dim

    # Pre-save
    with open(path, "wb") as f:
      pickle.dump(saved_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    depths = [1, 2, 3]
    use_augments = [False, True]

    models = []

    for use_augment, depth in itertools.product(use_augments, depths):
        models += [
        (  
            f"Sig Neural Laplace (use_augment = {use_augment}, depth = {depth})",
            GeneralNeuralLaplace(
                input_dim=input_dim,
                output_dim=output_dim,
                latent_dim=latent_dim,
                hidden_units=42,
                s_recon_terms=s_recon_terms,
                use_sphere_projection=use_sphere_projection,
                ilt_algorithm=ilt_algorithm,
                encode_obs_time=encode_obs_time,
                encoder_type='signature',
                signature_kwargs = {
                    "n_features": 4,
                    "kernel_size": 40,
                    "depth": depth,
                    "stride": 1,
                    "use_augment": use_augment
                }
            ).to(device_param),
        ),
        (
            f"Sig Neural Flow ResNet (use_augment = {use_augment}, depth = {depth})",
            GeneralLatentODE(
                dim=input_dim,
                model="flow",
                flow_model="resnet",
                hidden_dim=26,
                hidden_layers=latent_dim,
                latents=latent_dim,
                n_classes=input_dim,
                z0_encoder="signature",
                encoder_kwargs = {
                    "n_features": 4,
                    "kernel_size": 40,
                    "depth": depth,
                    "stride": 1,
                    "use_augment": use_augment
                }
            ).to(device_param),
        ),
        ]
    for model_name, system in models:
      try:
        logger.info("Training & testing for : %s \t | seed: %d", model_name,
                    seed)
        system.double()
        encoder = getattr(system.model, 'encoder_z0', None)
        if encoder is None:
            encoder = getattr(system.model, 'encoder', None)

        # Log encoder parameters if found
        if encoder is not None:
            logger.info("encoder num_params=%d", sum(p.numel() for p in encoder.parameters()))
        else:
            logger.warning("No encoder or encoder_z0 found in model.")

        # Log total parameters
        logger.info("total num_params=%d", sum(p.numel() for p in system.model.parameters()))
        optimizer = torch.optim.Adam(system.model.parameters(),
                                     lr=learning_rate)
        scheduler = None
        test_rmse, train_loss, train_nfes, train_epochs, mean_duration = train_and_test(
            seed,
            dataset,
            model_name,
            system,
            dltrain,
            dlval,
            dltest,
            optimizer,
            device_param,
            scheduler,
            epochs=epochs,
            patience=patience,
        )
        logger.info("Result: %s - TEST RMSE: %s", model_name, test_rmse)
        df_list_baseline_results.append({
            "method": model_name,
            "test_rmse": test_rmse,
            "seed": seed,
            "mean_duration": mean_duration
        })
        saved_dict[model_name] = {
            "test rmse": test_rmse,
            "seed": seed,
            "model_state_dict": system.model.state_dict(),
            "train_loss": train_loss.detach().cpu().numpy(),
            "train_nfes": train_nfes.detach().cpu().numpy(),
            "train_epochs": train_epochs.detach().cpu().numpy(),
        }
        # Checkpoint
        with open(path, "wb") as f:
          pickle.dump(saved_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        logger.error(e)
        logger.error("Error for model: %s", model_name)
        raise e

  df_results = pd.DataFrame(df_list_baseline_results)
  test_rmse_df_inner = df_results.groupby("method")[["test_rmse", "mean_duration"]].agg(["mean", "std"])
  logger.info("Test RMSE of experiment")
  latex_str = test_rmse_df_inner.style.to_latex()
  logger.info(latex_str)
  save_key = dataset
  filename = f"ablation_sensitivity_{dataset}.tex"
  with open(filename, "w") as f:
    f.write(latex_str)
  return test_rmse_df_inner


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description=
      "Run all baselines for an experiment (including Neural Laplace)")
  parser.add_argument(
      "-d",
      "--dataset",
      type=str,
      default="lotka_volterra_system_with_delay",
      help=f"Available datasets: {datasets}",
  )
  parser.add_argument("--gpu", type=int, default=0)
  args = parser.parse_args()

  assert args.dataset in datasets
  device = torch.device("cuda:" +
                        str(args.gpu) if torch.cuda.is_available() else "cpu")

  Path("./logs").mkdir(parents=True, exist_ok=True)
  path_run_name = f"{file_name}-{args.dataset}"
  logging.basicConfig(
      format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
      handlers=[
          logging.FileHandler(f"logs/{path_run_name}_log.txt"),
          logging.StreamHandler(),
      ],
      datefmt="%H:%M:%S",
      level=logging.INFO,
  )
  logger = logging.getLogger()

  logger.info("Using %s device", device)
  test_rmse_df = experiment_with_all_baselines(
      args.dataset,
      device,
      path_run_name
  )
