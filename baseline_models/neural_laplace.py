# pytype: skip-file
"""
This module contains neural network models for learning diverse classes of
  differential equations in the Laplace domain. It includes:

- SphereSurfaceModel: Maps input coordinates to output coordinates on the
  surface of a sphere using Riemann Sphere coordinates.
- SurfaceModel: Maps input coordinates to output coordinates in Riemann Sphere
  coordinates.
- NeuralLaplace: Combines the encoder (either GRU or signature-based)
  models to predict values at given times.
- GeneralNeuralLaplace: Provides methods for training, validation, testing,
  prediction, and encoding of data using the NeuralLaplace model.
"""

# Based on the original implementation by Samuel Holt, 
# available at: https://github.com/samholt/NeuralLaplace

import logging

import torch
from torch import nn
import math

from torchlaplace import laplace_reconstruct
from .encoders import Encoder

logger = logging.getLogger()

# Add torch.pi if missing (for compatibility with older PyTorch versions)
if not hasattr(torch, 'pi'):
    torch.pi = math.pi


class SphereSurfaceModel(nn.Module):
  """
  A neural network model that maps input coordinates to output coordinates
  on the surface of a sphere using Riemann Sphere coordinates.

  Args:
    s_dim (int): The dimension of the input space.
    output_dim (int): The dimension of the output space.
    latent_dim (int): The dimension of the latent encoding.
    hidden_units (int, optional): The number of hidden units in each layer of
      the network. Default is 64.

  Attributes:
    s_dim (int): The dimension of the input space.
    output_dim (int): The dimension of the output space.
    latent_dim (int): The dimension of the latent encoding.
    linear_tanh_stack (nn.Sequential): A sequential container of linear and
      Tanh layers.
    phi_scale (torch.Tensor): The scaling factor for the phi angle.
    nfe (int): The number of function evaluations.

  Methods:
    forward(i):
      Forward pass through the network. Takes input tensor `i` and returns
      the theta and phi coordinates on the sphere.
      Args:
        i (torch.Tensor): Input tensor of shape (batch_size, s_dim * 2 +
          latent_dim).
      Returns:
        tuple: A tuple containing:
          - theta (torch.Tensor): Tensor of shape (batch_size, output_dim,
            s_dim) representing the theta coordinates.
          - phi (torch.Tensor): Tensor of shape (batch_size, output_dim, s_dim)
            representing the phi coordinates.
  """
  # C^{b+k} -> C^{bxd} - In Riemann Sphere Co ords : b dim s reconstruction
  #   terms, k is latent encoding dimension, d is output dimension
  def __init__(self, s_dim, output_dim, latent_dim, hidden_units=64):
    super(SphereSurfaceModel, self).__init__()
    self.s_dim = s_dim
    self.output_dim = output_dim
    self.latent_dim = latent_dim
    self.linear_tanh_stack = nn.Sequential(
        nn.Linear(s_dim * 2 + latent_dim, hidden_units),
        nn.Tanh(),
        nn.Linear(hidden_units, hidden_units),
        nn.Tanh(),
        nn.Linear(hidden_units, (s_dim) * 2 * output_dim),
    )

    for m in self.linear_tanh_stack.modules():
      if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)

    phi_max = math.pi / 2.0
    self.phi_scale = phi_max - -math.pi / 2.0
    self.nfe = 0

  def forward(self, i):
    self.nfe += 1
    out = self.linear_tanh_stack(i.view(-1,
                                        self.s_dim * 2 + self.latent_dim)).view(
                                            -1, 2 * self.output_dim, self.s_dim)
    theta = nn.Tanh()(
        out[:, :self.output_dim, :]) * math.pi  # From - pi to + pi
    phi = (nn.Tanh()(out[:, self.output_dim:, :]) * self.phi_scale / 2.0 -
           math.pi / 2.0 + self.phi_scale / 2.0)  # Form -pi / 2 to + pi / 2
    return theta, phi


class SurfaceModel(nn.Module):
  """
  SurfaceModel for reconstructing terms in Riemann Sphere coordinates.
  Args:
    s_dim (int): Dimension of the input space.
    output_dim (int): Dimension of the output space.
    latent_dim (int): Dimension of the latent encoding.
    hidden_units (int, optional): Number of hidden units in each layer.
      Default is 64.
  Attributes:
    s_dim (int): Dimension of the input space.
    output_dim (int): Dimension of the output space.
    latent_dim (int): Dimension of the latent encoding.
    linear_tanh_stack (nn.Sequential): Sequential model with linear and Tanh
      layers.
    nfe (int): Number of function evaluations.
  Methods:
    forward(i):
      Forward pass through the model.
      Args:
        i (torch.Tensor): Input tensor.
      Returns:
        Tuple[torch.Tensor, torch.Tensor]: Real and imaginary parts of the
          output.
  """
  # C^{b+k} -> C^{bxd} - In Riemann Sphere Co ords : b dim s reconstruction
  #   terms, k is latent encoding dimension, d is output dimension
  def __init__(self, s_dim, output_dim, latent_dim, hidden_units=64):
    super(SurfaceModel, self).__init__()
    self.s_dim = s_dim
    self.output_dim = output_dim
    self.latent_dim = latent_dim
    self.linear_tanh_stack = nn.Sequential(
        nn.Linear(s_dim * 2 + latent_dim, hidden_units),
        nn.Tanh(),
        nn.Linear(hidden_units, hidden_units),
        nn.Tanh(),
        nn.Linear(hidden_units, (s_dim) * 2 * output_dim),
    )

    for m in self.linear_tanh_stack.modules():
      if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)

    self.nfe = 0

  def forward(self, i):
    self.nfe += 1
    out = self.linear_tanh_stack(i.view(-1,
                                        self.s_dim * 2 + self.latent_dim)).view(
                                            -1, 2 * self.output_dim, self.s_dim)
    real = out[:, :self.output_dim, :]
    imag = out[:, self.output_dim:, :]
    return real, imag


class NeuralLaplace(nn.Module):
    """
    NeuralLaplace is a neural network model designed to predict values at given
    times by learning diverse classes of differential equations in the Laplace domain.

    Args:
        input_dim (int): The dimension of the input data.
        output_dim (int): The dimension of the output data.
        latent_dim (int): The dimension of the latent representation.
        hidden_units (int): Number of hidden units for the encoder.
        s_recon_terms (int): Number of Laplace reconstruction terms.
        use_sphere_projection (bool): Whether to use sphere projection.
        encode_obs_time (bool): Whether to include time in the input.
        ilt_algorithm (str): Inverse Laplace transform algorithm.
        encoder_type (str): Type of encoder: "gru" or "signature".
        signature_kwargs (dict): Additional kwargs for the signature encoder.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        latent_dim=2,
        hidden_units=64,
        s_recon_terms=33,
        use_sphere_projection=True,
        encode_obs_time=True,
        ilt_algorithm="fourier",
        encoder_type="gru",
        signature_kwargs=None,
    ):
        super(NeuralLaplace, self).__init__()

        self.encoder = Encoder(
            encoder_type=encoder_type,
            dimension_in=input_dim,
            latent_dim=latent_dim,
            hidden_units=hidden_units,
            encode_obs_time=encode_obs_time,
            signature_kwargs=signature_kwargs,
        )

        self.use_sphere_projection = use_sphere_projection
        self.output_dim = output_dim
        self.ilt_algorithm = ilt_algorithm

        if use_sphere_projection:
            self.laplace_rep_func = SphereSurfaceModel(s_recon_terms, output_dim, latent_dim)
        else:
            self.laplace_rep_func = SurfaceModel(s_recon_terms, output_dim, latent_dim)

    def forward(self, observed_data, observed_tp, tp_to_predict):
        p = self.encoder(observed_data, observed_tp)
        return laplace_reconstruct(
            self.laplace_rep_func,
            p,
            tp_to_predict,
            recon_dim=self.output_dim,
            use_sphere_projection=self.use_sphere_projection,
            ilt_algorithm=self.ilt_algorithm,
        )


class GeneralNeuralLaplace(nn.Module):
  """
  GeneralNeuralLaplace is a PyTorch module that encapsulates a neural network
  model for learning Laplace transforms. It provides methods for training,
  validation, testing, prediction, and encoding of data.

  Attributes:
    model (NeuralLaplace): The neural network model for Laplace transforms.
    loss_fn (torch.nn.MSELoss): The mean squared error loss function.

  Methods:
    _get_loss(dl): Computes the mean squared error loss over a given dataloader.
    training_step(batch): Performs a training step and returns the loss.
    validation_step(dlval): Computes the validation loss over a given
      dataloader.
    test_step(dltest): Computes the test loss over a given dataloader.
    predict(dl): Generates predictions for a given dataloader.
    encode(dl): Encodes the data from a given dataloader.
    get_and_reset_nfes(): Returns and resets the number of function
      evaluations for the model.
  """

  def __init__(
        self,
        input_dim,
        output_dim,
        latent_dim=2,
        hidden_units=64,
        s_recon_terms=33,
        use_sphere_projection=True,
        encode_obs_time=True,
        ilt_algorithm="fourier",
        encoder_type="gru",
        signature_kwargs=None,
    ):
      super(GeneralNeuralLaplace, self).__init__()

      self.model = NeuralLaplace(
          input_dim=input_dim,
          output_dim=output_dim,
          latent_dim=latent_dim,
          hidden_units=hidden_units,
          s_recon_terms=s_recon_terms,
          use_sphere_projection=use_sphere_projection,
          encode_obs_time=encode_obs_time,
          ilt_algorithm=ilt_algorithm,
          encoder_type=encoder_type,
          signature_kwargs=signature_kwargs,
      )

      self.loss_fn = torch.nn.MSELoss()

  def _get_loss(self, dl):
    cum_loss = 0
    cum_batches = 0
    for batch in dl:
      preds = self.model(batch["observed_data"], batch["observed_tp"],
                         batch["tp_to_predict"])
      cum_loss += self.loss_fn(torch.flatten(preds),
                               torch.flatten(batch["data_to_predict"]))
      cum_batches += 1
    mse = cum_loss / cum_batches
    return mse

  def training_step(self, batch):
    preds = self.model(batch["observed_data"], batch["observed_tp"],
                       batch["tp_to_predict"])
    return self.loss_fn(torch.flatten(preds),
                        torch.flatten(batch["data_to_predict"]))

  def validation_step(self, dlval):
    mse = self._get_loss(dlval)
    return mse, mse

  def test_step(self, dltest):
    mse = self._get_loss(dltest)
    return mse, mse

  def predict(self, dl):
    predictions = []
    for batch in dl:
      predictions.append(
          self.model(batch["observed_data"], batch["observed_tp"],
                     batch["tp_to_predict"]))
    return torch.cat(predictions, 0)

  def encode(self, dl):
    encodings = []
    for batch in dl:
      encodings.append(
          self.model.encoder(batch["observed_data"], batch["observed_tp"]))
    return torch.cat(encodings, 0)

  def get_and_reset_nfes(self):
    """Returns and resets the number of function evaluations for model."""
    iteration_nfes = self.model.laplace_rep_func.nfe
    self.model.laplace_rep_func.nfe = 0
    return iteration_nfes
