"""
Default configuration settings for training and evaluation of baseline and signature-based models.

These hyperparameters control dataset preprocessing, model architecture, training schedule,
and evaluation settings across all experiments. The configuration is shared between training,
evaluation, plotting, and ablation study scripts.
"""

batch_size = 128
extrapolate = True
epochs = 1
start_seed = 0
run_number_of_seeds = 5
learning_rate = 1e-3
ode_solver_method = "euler"
trajectories_to_sample = 1000
time_points_to_sample = 200
observe_step = 1
predict_step = 1
noise_std = 0
normalize_dataset = True
encode_obs_time = False
latent_dim = 2
s_recon_terms = 33
patience = 500
use_sphere_projection = True
ilt_algorithm = "fourier"

line_styles = {
    'True Trajectory':        {'color': 'black', 'dash': 'dash'},
    'Neural Laplace':         {'color': '#1f77b4'},
    'Sig Neural Laplace':     {'color': '#aec7e8'},
    'NODE (euler)':           {'color': '#ff7f0e'},
    'ANODE (euler)':          {'color': '#2ca02c'},
    'Neural Flow ResNet':         {'color': '#d62728'},
    'Sig Neural Flow ResNet':     {'color': '#ff9896'},
}