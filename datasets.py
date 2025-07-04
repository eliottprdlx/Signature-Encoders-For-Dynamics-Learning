# pytype: skip-file

"""
This module provides functions to generate the datasets used for training and
testing dynamics learning models.
"""

# Based on the original implementation by Samuel Holt, 
# available at: https://github.com/samholt/NeuralLaplace

import shelve
from functools import partial
from pathlib import Path
import numpy as np
import scipy.io as sio
import torch
from ddeint import ddeint
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchlaplace.data_utils import basic_collate_fn

local_path = Path(__file__).parent

def spiral_dde(device, double=False, trajectories_to_sample=100, t_nsamples=200):
    def model(XY, t, d):
        x, y = XY(t)
        xd, yd = XY(t - d)
        return np.array([
            -np.tanh(x + xd) + np.tanh(y + yd),
            -np.tanh(x + xd) - np.tanh(y + yd),
        ])

    compute_points = 1000
    t_end = 20
    tt = np.linspace(t_end / compute_points, t_end, compute_points)
    sample_step = compute_points // t_nsamples
    evaluate_points = int(np.floor(np.sqrt(trajectories_to_sample)))
    x0s = np.linspace(-2, 2, evaluate_points)

    key = f"spiral_dde_trajectories_{evaluate_points}"
    try:
        with shelve.open("datasets") as db:
            trajectories = db[key]
    except KeyError:
        trajectories = []
        for x0 in tqdm(x0s):
            for y0 in x0s:
                sol = ddeint(model, lambda t, x0=x0, y0=y0: np.array([x0, y0]), tt, fargs=(2.5,))
                trajectories.append(sol)
        trajectories = np.stack(trajectories)
        with shelve.open("datasets") as db:
            db[key] = trajectories

    trajectories = trajectories[:, ::sample_step]
    tt = tt[::sample_step]

    if double:
        trajectories = torch.from_numpy(trajectories).to(device).double()
        t = torch.from_numpy(tt).to(device).double()
    else:
        trajectories = torch.from_numpy(trajectories).float().to(device)
        t = torch.from_numpy(tt).float().to(device)

    return trajectories, t


def lotka_volterra_system_with_delay(device, double=False, trajectories_to_sample=100, t_nsamples=200):
    def model(Y, t, d):
        x, y = Y(t)
        xd, yd = Y(t - d)
        return np.array([0.5 * x * (1 - yd), -0.5 * y * (1 - xd)])

    compute_points = 1000
    t_end = 30
    tt = np.linspace(2, t_end, compute_points)
    sample_step = compute_points // t_nsamples
    evaluate_points = int(np.floor(np.sqrt(trajectories_to_sample)))
    x0s = np.linspace(0.1, 2, evaluate_points)

    key = f"lotka_volterra_system_with_delay_trajectories_{evaluate_points}"
    try:
        with shelve.open("datasets") as db:
            trajectories = db[key]
    except KeyError:
        trajectories = []
        for x0 in tqdm(x0s):
            for y0 in x0s:
                sol = ddeint(model, lambda t, x0=x0, y0=y0: np.array([x0, y0]), tt, fargs=(0.1,))
                trajectories.append(sol)
        trajectories = np.stack(trajectories)
        with shelve.open("datasets") as db:
            db[key] = trajectories

    trajectories = trajectories[:, ::sample_step]
    tt = tt[::sample_step]

    if double:
        trajectories = torch.from_numpy(trajectories).to(device).double()
        t = torch.from_numpy(tt).to(device).double()
    else:
        trajectories = torch.from_numpy(trajectories).float().to(device)
        t = torch.from_numpy(tt).float().to(device)

    return trajectories, t


def fitzhugh_nagumo_with_delay(device, double=False, trajectories_to_sample=100, t_nsamples=200, coupling_factor=1.0):
    def model(Y, t, d):
        v, w = Y(t)
        vd, wd = Y(t - d)
        I = 0.5
        a, b, eps = 0.5, 0.8, 0.02
        return np.array([
            v - v**3 / 3 - wd + I,
            eps * (coupling_factor * vd + a - b * w)
        ])

    compute_points = 1000
    t_end = 30
    tt = np.linspace(2, t_end, compute_points)
    sample_step = compute_points // t_nsamples
    evaluate_points = int(np.floor(np.sqrt(trajectories_to_sample)))
    x0s = np.linspace(-5.0, 5.0, evaluate_points)

    key = f"fitzhugh_nagumo_delay_trajectories_{evaluate_points}_{coupling_factor}"
    try:
        with shelve.open("datasets") as db:
            trajectories = db[key]
    except KeyError:
        trajectories = []
        for x0 in tqdm(x0s):
            for y0 in x0s:
                sol = ddeint(model, lambda t, x0=x0, y0=y0: np.array([x0, y0]), tt, fargs=(1,))
                trajectories.append(sol)
        trajectories = np.stack(trajectories)
        with shelve.open("datasets") as db:
            db[key] = trajectories

    trajectories = trajectories[:, ::sample_step]
    tt = tt[::sample_step]

    if double:
        trajectories = torch.from_numpy(trajectories).to(device).double()
        t = torch.from_numpy(tt).to(device).double()
    else:
        trajectories = torch.from_numpy(trajectories).float().to(device)
        t = torch.from_numpy(tt).float().to(device)

    return trajectories, t


def rossler_system_with_delay(device, double=False, trajectories_to_sample=100, t_nsamples=200):
    def model(Y, t, d):
        x, y, z = Y(t)
        xd, _, zd = Y(t - d)
        a, b, c = 0.2, 0.2, 4.5
        dxdt = -y - zd
        dydt = x + a * y
        dzdt = b + z * (xd - c)
        return np.array([dxdt, dydt, dzdt])

    compute_points = 1000
    t_end = 20
    tt = np.linspace(2, t_end, compute_points)
    sample_step = compute_points // t_nsamples
    evaluate_points = int(np.floor(np.cbrt(trajectories_to_sample)))
    x0s = np.linspace(0.1, 1.5, evaluate_points)

    key = f"rossler_system_with_delay_trajectories_{evaluate_points}"
    try:
        with shelve.open("datasets") as db:
            trajectories = db[key]
    except KeyError:
        trajectories = []
        for x0 in tqdm(x0s):
            for y0 in x0s:
                for z0 in x0s:
                    hist_func = lambda t, x0=x0, y0=y0, z0=z0: np.array([x0, y0, z0])
                    sol = ddeint(model, hist_func, tt, fargs=(2.5,))
                    trajectories.append(sol)
        trajectories = np.stack(trajectories)
        with shelve.open("datasets") as db:
            db[key] = trajectories

    trajectories = trajectories[:, ::sample_step]
    tt = tt[::sample_step]

    if double:
        trajectories = torch.from_numpy(trajectories).to(device).double()
        t = torch.from_numpy(tt).to(device).double()
    else:
        trajectories = torch.from_numpy(trajectories).float().to(device)
        t = torch.from_numpy(tt).float().to(device)

    return trajectories, t


def generate_data_set(
    name,
    device,
    double=False,
    batch_size=128,
    extrap=0,
    trajectories_to_sample=100,
    percent_missing_at_random=0.0,
    normalize=True,
    test_set_out_of_distribution=False,
    noise_std=None,
    t_nsamples=200,
    observe_step=1,
    predict_step=1,
    coupling_factor=1.0
):
    if name == "spiral_dde":
        trajectories, t = spiral_dde(device, 
                                     double, 
                                     trajectories_to_sample, 
                                     t_nsamples)
    elif name == "lotka_volterra_system_with_delay":
        trajectories, t = lotka_volterra_system_with_delay(device, 
                                                           double, 
                                                           trajectories_to_sample, 
                                                           t_nsamples)
    elif name == "fitzhugh_nagumo_with_delay":
        trajectories, t = fitzhugh_nagumo_with_delay(device, 
                                                     double, 
                                                     trajectories_to_sample, 
                                                     t_nsamples, 
                                                     coupling_factor)
    elif name == "rossler_system_with_delay":
        trajectories, t = rossler_system_with_delay(device, 
                                                    double, 
                                                    trajectories_to_sample, 
                                                    t_nsamples)
    else:
        raise ValueError("Unknown dataset")

    if not extrap:
        mask = torch.rand_like(trajectories) > percent_missing_at_random
        mask = mask.double() if double else mask.float()
        trajectories = trajectories * mask.to(device)

    if normalize:
        samples, _, dim = trajectories.shape
        flat = trajectories.reshape(-1, dim)
        normed = (flat - flat.mean(0)) / flat.std(0)
        trajectories = normed.reshape(samples, -1, dim)

    if noise_std:
        trajectories += torch.randn_like(trajectories) * noise_std

    train_split = int(0.8 * trajectories.shape[0])
    test_split = int(0.9 * trajectories.shape[0])
    if test_set_out_of_distribution:
        train = trajectories[:train_split]
        val = trajectories[train_split:test_split]
        test = trajectories[test_split:]
    else:
        indices = torch.randperm(trajectories.shape[0])
        train = trajectories[indices[:train_split]]
        val = trajectories[indices[train_split:test_split]]
        test = trajectories[indices[test_split:]]

    test_plot_traj = test[0]

    input_dim = train.shape[2]
    output_dim = input_dim

    dltrain = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: basic_collate_fn(batch, 
                                                  t, 
                                                  "train", 
                                                  extrap, 
                                                  observe_step, 
                                                  predict_step),
    )
    dlval = DataLoader(
        val,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: basic_collate_fn(batch, 
                                                  t, 
                                                  "test", 
                                                  extrap, 
                                                  observe_step, 
                                                  predict_step),
    )
    dltest = DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: basic_collate_fn(batch, 
                                                  t, 
                                                  "test", 
                                                  extrap, 
                                                  observe_step, 
                                                  predict_step),
    )

    return (input_dim, 
            output_dim, 
            dltrain, dlval, dltest, 
            test_plot_traj, t, test)

