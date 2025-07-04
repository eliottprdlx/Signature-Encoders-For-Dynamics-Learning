# pytype: skip-file
"""
This module implements two meta-types of encoders: deterministic and variational.
For each meta-type, it provides both a signature-based encoder (using path signatures)
and an RNN-based encoder (e.g., GRU or ODE-RNN).

Encoders are organized to allow flexible switching between architectures for experimentation
and benchmarking in latent dynamics modeling tasks.
"""

import torch
from torch import nn
import torch.nn.functional as F
import signatory
from .data_utils import (
    check_mask,
    compute_binary_CE_loss,
    compute_loss_all_batches,
    compute_mse,
    compute_multiclass_CE_loss,
    compute_poisson_proc_likelihood,
    init_network_weights,
    masked_gaussian_log_density,
    sample_standard_gaussian,
)
from .flow import CouplingFlow, ResNetFlow
from .gru import GRUFlow
from .ode import ODEModel


class Encoder(nn.Module):
    """
    A unified interface for trajectory encoding using either a GRU or signature-based encoder.
    This wrapper class simplifies the selection and configuration of different encoder types 
    for temporal or sequential data.
    
    Args:
        encoder_type (str): Type of encoder to use. Must be either "gru" or "signature".
        dimension_in (int): Dimensionality of the input features.
        latent_dim (int): Dimensionality of the output latent vector.
        hidden_units (int): Number of hidden units used in the GRU or augmentation network.
        encode_obs_time (bool, optional): Whether to include observation times in the input. Default is True.
        signature_kwargs (dict, optional): Additional keyword arguments to configure the SignatureEncoder.

    Attributes:
        encoder_type (str): Indicates the selected encoder type.
        encoder (nn.Module): The instantiated encoder (either ReverseGRUEncoder or SignatureEncoder).

    Raises:
        ValueError: If the specified encoder_type is not "gru" or "signature".

    Methods:
        forward(observed_data, observed_tp):
            Encodes a batch of input trajectories into latent vectors.

            Args:
                observed_data (torch.Tensor): Input tensor of shape (batch_size, time_steps, dimension_in).
                observed_tp (torch.Tensor): Timepoints of observations, of shape (time_steps,).

            Returns:
                torch.Tensor: Latent representation of shape (batch_size, latent_dim).
    """
    def __init__(self,
                 encoder_type,
                 dimension_in,
                 latent_dim,
                 hidden_units,
                 encode_obs_time=True,
                 signature_kwargs=None):
      super(Encoder, self).__init__()

      if encoder_type not in ["gru", "signature"]:
          raise ValueError(f"Unknown encoder type '{encoder_type}'. Choose 'gru' or 'signature'.")

      self.encoder_type = encoder_type

      if encoder_type == "gru":
          self.encoder = ReverseGRUEncoder(
              dimension_in=dimension_in,
              latent_dim=latent_dim,
              hidden_units=hidden_units // 2,
              encode_obs_time=encode_obs_time
          )

      elif encoder_type == "signature":
          signature_kwargs = signature_kwargs or {}
          self.encoder = SignatureEncoder(
              dimension_in=dimension_in,
              latent_dim=latent_dim,
              hidden_units=hidden_units,
              encode_obs_time=encode_obs_time,
              **signature_kwargs
          )

    def forward(self, observed_data, observed_tp):
        return self.encoder(observed_data, observed_tp)


import torch
import torch.nn as nn
import signatory


class SignatureEncoder(nn.Module):
    """
    An encoder that uses path signatures to transform observed trajectories into latent representations.

    This encoder optionally augments the input trajectory using convolutional layers before computing 
    its path signature via the Signatory library. The resulting signature is then projected into 
    a lower-dimensional latent space via a linear transformation.

    Args:
        dimension_in (int): Dimensionality of the input data.
        latent_dim (int): Dimension of the output latent vector.
        hidden_units (int): Number of hidden units in the augmentation network.
        encode_obs_time (bool, optional): Whether to append observation times to the input. Default is True.
        n_features (int, optional): Number of output channels from the augmentation network. Default is 4.
        kernel_size (int, optional): Convolutional kernel size for augmentation. Default is 40.
        depth (int, optional): Depth of the path signature. Default is 3.
        stride (int, optional): Stride in the convolutional augmentation. Default is 1.
        use_augment (bool, optional): Whether to apply augmentation before computing the signature. Default is True.

    Attributes:
        encode_obs_time (bool): Indicates whether observation time is included in the input.
        augment (signatory.Augment): The optional augmentation module.
        signature (signatory.Signature): Computes the path signature.
        linear_out (nn.Linear): Projects the signature into the latent space.
        use_augment (bool): Controls whether augmentation is applied.

    Methods:
        forward(observed_data, observed_tp):
            Encodes a batch of input trajectories into a latent representation.
    """

    def __init__(self,
                 dimension_in,
                 latent_dim,
                 hidden_units,
                 encode_obs_time=True,
                 n_features=4,
                 kernel_size=40,
                 depth=3,
                 stride=1,
                 use_augment=True):
        super(SignatureEncoder, self).__init__()
        self.encode_obs_time = encode_obs_time
        self.dimension_in = dimension_in
        self.use_augment = use_augment

        self.augment = signatory.Augment(
            in_channels=dimension_in,
            layer_sizes=(hidden_units // 2,
                         hidden_units // 2,
                         n_features),
            kernel_size=kernel_size,
            stride=stride,
            include_original=(stride == 1),
            include_time=(stride == 1)
        )

        self.signature = signatory.Signature(depth=depth)

        channels = n_features
        if stride == 1:
            channels += 1 + dimension_in  # time + original input
        if not use_augment:
            channels = dimension_in

        sig_channels = signatory.signature_channels(channels=channels, depth=depth)
        self.linear_out = torch.nn.Linear(sig_channels, latent_dim)
        nn.init.xavier_uniform_(self.linear_out.weight)

    def forward(self, observed_data, observed_tp):
        trajs_to_encode = observed_data
        if self.encode_obs_time:
            trajs_to_encode = torch.cat(
                (observed_data,
                 observed_tp.view(1, -1, 1).repeat(observed_data.shape[0], 1, 1)),
                dim=2
            )

        x = self.augment(trajs_to_encode) if self.use_augment else trajs_to_encode

        if x.size(1) <= 1:
            raise RuntimeError("Input length is too short to take the signature")

        x = self.signature(x, basepoint=True)
        return self.linear_out(x)


# Original code by Samuel Holt, from: https://github.com/samholt/NeuralLaplace

class ReverseGRUEncoder(nn.Module):
  """
  A GRU-based encoder that encodes observed trajectories into latent vectors.
  The encoder processes the input data in reverse order and optionally includes
  the observation times as part of the input.

  Args:
    dimension_in (int): The dimension of the input data.
    latent_dim (int): The dimension of the latent vector.
    hidden_units (int): The number of hidden units in the GRU.
    encode_obs_time (bool, optional): Whether to include observation times as
     part of the input. Default is True.

  Attributes:
    encode_obs_time (bool): Whether to include observation times as part of the
    input.
    gru (nn.GRU): A GRU layer for encoding the input data.
    linear_out (nn.Linear): A linear layer for producing the latent vector.

  Methods:
    forward(observed_data, observed_tp):
      Encodes the observed data and observation times into a latent vector.
      Args:
        observed_data (torch.Tensor): The observed data of shape (batch_size,
          t_observed_dim, observed_dim).
        observed_tp (torch.Tensor): The observation times of shape
          (t_observed_dim,).
      Returns:
        torch.Tensor: The encoded latent vector of shape (batch_size,
          latent_dim).
  """
  def __init__(self,
               dimension_in,
               latent_dim,
               hidden_units,
               encode_obs_time=True):
    super(ReverseGRUEncoder, self).__init__()
    self.encode_obs_time = encode_obs_time
    if self.encode_obs_time:
      dimension_in += 1
    self.gru = nn.GRU(dimension_in, hidden_units, 2, batch_first=True)
    self.linear_out = nn.Linear(hidden_units, latent_dim)
    nn.init.xavier_uniform_(self.linear_out.weight)

  def forward(self, observed_data, observed_tp):
    trajs_to_encode = observed_data
    if self.encode_obs_time:
      trajs_to_encode = torch.cat(
          (observed_data, observed_tp.view(1, -1, 1).repeat(
              observed_data.shape[0], 1, 1)),
          dim=2)
    reversed_trajs_to_encode = torch.flip(trajs_to_encode, (1,))
    out, _ = self.gru(reversed_trajs_to_encode)
    return self.linear_out(out[:, -1, :])


class Encoder_z0(nn.Module):
    """
    A unified interface for variational initial state encoders in latent ODE models.

    This wrapper class abstracts over multiple encoder architectures for estimating
    the initial latent state z₀ from observed time series data. Supported architectures
    include a signature-based encoder and an ODE-RNN-based encoder.

    Args:
        input_dim (int): Dimensionality of the input features.
        encoder_type (str): Type of encoder to use. Must be either 'signature' or 'ode_rnn'.
        latent_dim (int): Dimensionality of the latent space.
        hidden_units (int): Number of hidden units used in the encoder.
        z0_dim (int, optional): Dimensionality of the latent initial state z₀. Defaults to `latent_dim` if not specified.
        device (torch.device, optional): Device on which computations will be performed. Default is CPU.
        encoder_kwargs (dict, optional): Additional keyword arguments specific to the selected encoder type.

    Raises:
        ValueError: If an unsupported encoder type is provided.
        ValueError: If 'z0_diffeq_solver' is missing for the 'ode_rnn' encoder.

    Attributes:
        encoder_type (str): The name of the selected encoder architecture.
        encoder (nn.Module): The instantiated encoder module (either `Encoder_z0_signature` or `Encoder_z0_ODE_RNN`).

    Methods:
        forward(*args, **kwargs):
            Forwards the input through the selected encoder.

            Returns:
                Tuple[torch.Tensor, torch.Tensor]: Mean and standard deviation of the initial latent state z₀.
    """
    def __init__(self,
                 input_dim,
                 encoder_type,
                 latent_dim,
                 hidden_units,
                 z0_dim=None,
                 device=torch.device("cpu"),
                 encoder_kwargs=None):
        super().__init__()

        self.encoder_type = encoder_type.lower()
        self.device = device

        if encoder_kwargs is None:
            encoder_kwargs = {}

        if self.encoder_type == "signature":
            self.encoder = Encoder_z0_signature(
                dimension_in=input_dim,
                latent_dim=latent_dim,
                hidden_units=hidden_units,
                z0_dim=z0_dim or latent_dim,
                **encoder_kwargs
            )

        elif self.encoder_type == "ode_rnn":
            if "z0_diffeq_solver" not in encoder_kwargs:
                raise ValueError("z0_diffeq_solver must be provided for ODE-RNN encoder.")

            self.encoder = Encoder_z0_ODE_RNN(
                input_dim=input_dim,
                latent_dim=latent_dim,
                n_gru_units=hidden_units,
                z0_dim=z0_dim or latent_dim,
                device=device,
                **encoder_kwargs
            )

        else:
            raise ValueError(f"Unknown encoder_type '{encoder_type}'. Use 'signature' or 'ode_rnn'.")

    def forward(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)


class Encoder_z0_signature(nn.Module):
    """
    A signature-based encoder for initializing the latent state z₀ in latent ODE models.

    This encoder applies optional convolutional augmentation to the input trajectory, computes 
    the path signature using the Signatory library, and projects it to the initial mean and 
    standard deviation of z₀ via a linear layer.

    Args:
        latent_dim (int): Dimensionality of the latent space.
        dimension_in (int): Dimensionality of the input features.
        hidden_units (int, optional): Number of hidden units in the augmentation network. Default is 32.
        z0_dim (int, optional): Dimensionality of the latent initial state z₀. Required.
        encode_obs_time (bool, optional): Whether to include observation times as part of the input. Default is False.
        n_features (int, optional): Number of output features from the augmentation network. Default is 4.
        kernel_size (int, optional): Kernel size for the convolutional augmentation layers. Default is 40.
        depth (int, optional): Depth of the path signature computation. Default is 3.
        stride (int, optional): Stride used in augmentation. Default is 1.
        use_augment (bool, optional): Whether to apply augmentation before signature computation. Default is True.

    Attributes:
        signature (signatory.Signature): Signature layer to compute path features.
        linear_out (nn.Linear): Linear projection to the z₀ mean and log std.
        sig_channels (int): Number of channels in the computed signature.

    Methods:
        forward(observed_data, observed_tp):
            Computes the mean and standard deviation of the initial latent state z₀.

            Args:
                observed_data (torch.Tensor): Input tensor of shape (batch_size, time_steps, dimension_in).
                observed_tp (torch.Tensor): Time points of shape (time_steps,).

            Returns:
                Tuple[torch.Tensor, torch.Tensor]: Mean and standard deviation of z₀,
                each of shape (1, batch_size, z0_dim).
    """

    def __init__(self,
                 latent_dim,
                 dimension_in,
                 hidden_units=32,
                 z0_dim=None,
                 encode_obs_time=False,
                 n_features=4,
                 kernel_size=40,
                 depth=3,
                 stride=1,
                 use_augment=True):
        super(Encoder_z0_signature, self).__init__()
        self.encode_obs_time = encode_obs_time
        self.dimension_in = dimension_in
        self.latent_dim = latent_dim
        self.z0_dim = z0_dim
        self.use_augment = use_augment

        self.augment = signatory.Augment(
            in_channels=dimension_in,
            layer_sizes=(hidden_units // 2,
                         hidden_units // 2,
                         n_features),
            kernel_size=kernel_size,
            stride=stride,
            include_original=(stride == 1),
            include_time=(stride == 1)
        )

        self.signature = signatory.Signature(depth=depth)

        channels = n_features
        if stride == 1:
            channels += 1 + dimension_in
        if not use_augment:
            channels = dimension_in

        sig_channels = signatory.signature_channels(channels=channels, depth=depth)
        self.sig_channels = sig_channels

        self.linear_out = nn.Linear(sig_channels, 2 * z0_dim)
        nn.init.xavier_uniform_(self.linear_out.weight)

    def forward(self, observed_data, observed_tp, **kwargs):
        trajs_to_encode = observed_data
        if self.encode_obs_time:
            trajs_to_encode = torch.cat(
                (observed_data,
                 observed_tp.view(1, -1, 1).repeat(observed_data.shape[0], 1, 1)),
                dim=2
            )

        x = self.augment(trajs_to_encode) if self.use_augment else trajs_to_encode

        if x.size(1) <= 1:
            raise RuntimeError("Input length is too short to take the signature")

        x = self.signature(x, basepoint=True)
        z = self.linear_out(x)

        mean_z0, std_z0 = z[:, :self.z0_dim], z[:, self.z0_dim:]
        mean_z0 = mean_z0.unsqueeze(0)
        std_z0 = F.softplus(std_z0.unsqueeze(0))  # Ensures positivity

        return mean_z0, std_z0


# Original code by Samuel Holt, from: https://github.com/samholt/NeuralLaplace

def get_mask(x):
  x = x.unsqueeze(0)
  n_data_dims = x.size(-1) // 2
  mask = x[:, :, n_data_dims:]
  check_mask(x[:, :, :n_data_dims], mask)
  mask = (torch.sum(mask, -1, keepdim=True) > 0).float()
  assert not torch.isnan(mask).any()
  return mask.squeeze(0)

class Encoder_z0_ODE_RNN(nn.Module): # pylint: disable=invalid-name
  """
  Encoder_z0_ODE_RNN is a neural network module that encodes input sequences
  into latent space using an ODE-RNN approach.
  """

  def __init__(
      self,
      latent_dim,
      input_dim,
      z0_diffeq_solver=None,
      z0_dim=None,
      n_gru_units=100, # pylint: disable=unused-argument
      device=torch.device("cpu"),
  ):
    super().__init__()

    if z0_dim is None:
      self.z0_dim = latent_dim
    else:
      self.z0_dim = z0_dim

    self.lstm = nn.LSTMCell(input_dim, latent_dim)

    self.z0_diffeq_solver = z0_diffeq_solver
    self.latent_dim = latent_dim
    self.input_dim = input_dim
    self.device = device
    self.extra_info = None

    self.transform_z0 = nn.Sequential(
        nn.Linear(latent_dim, 100),
        nn.Tanh(),
        nn.Linear(100, self.z0_dim * 2),
    )
    init_network_weights(self.transform_z0)

  def forward(self, data, time_steps, run_backwards=True, save_info=False): # pylint: disable=unused-argument
    assert not torch.isnan(data).any()
    assert not torch.isnan(time_steps).any()

    n_traj, _, _ = data.size()
    latent = self.run_odernn(data, time_steps, run_backwards)

    latent = latent.reshape(1, n_traj, self.latent_dim)

    mean_z0, std_z0 = self.transform_z0(latent).chunk(2, dim=-1)
    std_z0 = F.softplus(std_z0) # pylint: disable=not-callable

    return mean_z0, std_z0

  def run_odernn(self, data, time_steps, run_backwards=True):
    batch_size, _, _ = data.size()
    prev_t, t_i = time_steps[:, -1] + 0.01, time_steps[:, -1]

    time_points_iter = range(0, time_steps.shape[1])
    if run_backwards:
      time_points_iter = reversed(time_points_iter)

    h = torch.zeros(batch_size, self.latent_dim).to(data)
    c = torch.zeros(batch_size, self.latent_dim).to(data)

    for i in time_points_iter:
      t = (t_i - prev_t).unsqueeze(1)
      h = self.z0_diffeq_solver(h.unsqueeze(1), t).squeeze(1)

      xi = data[:, i, :]
      h_, c_ = self.lstm(xi, (h, c))
      mask = get_mask(xi)

      h = mask * h_ + (1 - mask) * h
      c = mask * c_ + (1 - mask) * c

      prev_t, t_i = time_steps[:, i], time_steps[:, i - 1]

    return h
