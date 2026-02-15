import numpy as np
import torch
import torch.nn as nn
from siren_pytorch import Modulator, SirenNet

# Code adapted from https://github.com/vishwa91/wire/blob/main/modules/wire.py


class ComplexGaborLayer2D(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        is_first=False,
        omega0=10.0,
        sigma0=10.0,
        trainable=False,
    ):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first

        self.in_features = in_features

        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat

        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0 * torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0 * torch.ones(1), trainable)

        self.linear = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)

        # Second Gaussian window
        self.scale_orth = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)

    def forward(self, input):
        lin = self.linear(input)

        scale_x = lin
        scale_y = self.scale_orth(input)

        freq_term = torch.exp(1j * self.omega_0 * lin)

        arg = scale_x.abs().square() + scale_y.abs().square()
        gauss_term = torch.exp(-self.scale_0 * self.scale_0 * arg)

        return freq_term * gauss_term


class WIREWithModulation2D(nn.Module):
    def __init__(
        self,
        in_features,
        latent_dim,
        hidden_features,
        hidden_layers,
        out_features,
        outermost_linear=True,
        first_omega_0=10,
        hidden_omega_0=10.0,
        scale=10.0,
        pos_encode=False,
        sidelength=512,
        fn_samples=None,
        use_nyquist=True,
    ):
        super().__init__()

        self.nonlin = ComplexGaborLayer2D

        hidden_features = int(hidden_features / 2)
        dtype = torch.cfloat
        self.complex = True
        self.wavelet = "gabor"

        self.pos_encode = False

        self.layers = nn.ModuleList()
        self.modulators = nn.ModuleList()

        self.layers.append(
            self.nonlin(
                in_features,
                hidden_features,
                omega0=first_omega_0,
                sigma0=scale,
                is_first=True,
                trainable=False,
            )
        )
        self.modulators.append(nn.Linear(latent_dim, hidden_features))

        for i in range(hidden_layers):
            self.layers.append(
                self.nonlin(
                    hidden_features,
                    hidden_features,
                    omega0=hidden_omega_0,
                    sigma0=scale,
                )
            )
            self.modulators.append(nn.Linear(latent_dim, hidden_features))

        self.final_linear = nn.Linear(hidden_features, out_features, dtype=dtype)

    def forward(self, coords, z):
        x = coords

        for i, layer in enumerate(self.layers):
            x = layer(x)

            mod = torch.tanh(self.modulators[i](z))

            while mod.dim() < x.dim():
                mod = mod.unsqueeze(-2)

            x = x * (1.0 + mod)

        output = self.final_linear(x)

        if self.wavelet == "gabor":
            return output.real


class ComplexGaborLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        is_first=False,
        omega0=10.0,
        sigma0=10.0,
        trainable=True,
    ):
        super().__init__()
        self.is_first = is_first
        self.in_features = in_features
        self.out_features = out_features

        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat

        # Per neuron params
        self.omega_0 = nn.Parameter(
            torch.full((out_features,), float(omega0))
            + torch.randn(out_features) * 2.0,
            requires_grad=trainable,
        )
        self.scale_0 = nn.Parameter(
            torch.full((out_features,), float(sigma0))
            + torch.randn(out_features) * 2.0,
            requires_grad=trainable,
        )

        self.linear = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)

    def forward(self, input):
        lin = self.linear(input)

        # Apply per-neuron frequency and scale
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin

        return torch.exp(1j * omega - scale.abs().square())


class WIREWithModulation(nn.Module):
    def __init__(
        self,
        in_features,
        latent_dim,
        hidden_features,
        hidden_layers,
        out_features,
        first_omega_0=10.0,
        hidden_omega_0=10.0,
        scale=20.0,
    ):
        super().__init__()

        self.nonlin = ComplexGaborLayer
        self.complex = True
        self.wavelet = "gabor"

        self.layers = nn.ModuleList()
        self.modulators = nn.ModuleList()

        # First layer
        self.layers.append(
            self.nonlin(
                in_features,
                hidden_features,
                omega0=first_omega_0,
                sigma0=scale,
                is_first=True,
                trainable=True,
            )
        )
        self.modulators.append(nn.Linear(latent_dim, hidden_features))

        # Hidden layers
        for i in range(hidden_layers):
            self.layers.append(
                self.nonlin(
                    hidden_features,
                    hidden_features,
                    omega0=hidden_omega_0,
                    sigma0=scale,
                    trainable=True,
                )
            )
            self.modulators.append(nn.Linear(latent_dim, hidden_features))

        self.final_linear = nn.Linear(hidden_features, out_features, dtype=torch.cfloat)

    def forward(self, coords, z):
        x = coords

        for i, layer in enumerate(self.layers):
            x = layer(x)

            # Compute modulation from latent vector z
            mod = torch.tanh(self.modulators[i](z))

            # Broadcast mod to match x dimensions
            # by inserting a dimension at the second to last position.
            while mod.dim() < x.dim():
                mod = mod.unsqueeze(-2)

            # Apply modulation
            x = x * (1.0 + mod)

        output = self.final_linear(x)

        return output.real


class GaborWithModulation(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims, output_dim):
        super().__init__()

        self.hidden_dims = hidden_dims
        self.layers = nn.ModuleList()
        self.modulators = nn.ModuleList()
        self.gabor_scales = nn.ParameterList()
        self.gabor_frequencies = nn.ParameterList()
        # self.dropout = nn.Dropout(0.1)

        self.norms = nn.ModuleList([nn.LayerNorm(d) for d in hidden_dims])

        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                self.modulators.append(nn.Linear(latent_dim, dims[i + 1]))

                self.gabor_scales.append(
                    nn.Parameter(torch.empty(dims[i + 1]).uniform_(0.1, 2.0))
                )
                self.gabor_frequencies.append(
                    nn.Parameter(torch.empty(dims[i + 1]).uniform_(0.1, 2.0))
                )

    def forward(self, x, z):
        eps = 1e-6
        two_pi = 2.0 * torch.pi

        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.norms[i](x)

            # scale = torch.relu(self.gabor_scales[i]) + eps
            # frequency = torch.relu(self.gabor_frequencies[i]) + eps
            scale = torch.clamp(self.gabor_scales[i], 0.3, 1.5) + eps
            frequency = torch.clamp(self.gabor_frequencies[i], 0.3, 1.5) + eps

            x_norm = x / scale

            envelope = torch.exp(-0.5 * (x_norm**2))
            oscillation = torch.cos(two_pi * frequency * x_norm)

            mod = torch.tanh(self.modulators[i](z))
            x = envelope * oscillation * (1.0 + mod)
            # x = self.dropout(x)

        return self.layers[-1](x)


class SirenWithModulation(nn.Module):
    def __init__(
        self, input_dim, latent_dim, hidden_dims, output_dim, w0=1.0, w0_initial=30.0
    ):
        super().__init__()
        self.net = SirenNet(
            dim_in=input_dim,
            dim_hidden=hidden_dims,
            dim_out=output_dim,
            num_layers=len(hidden_dims),
            w0=w0,
            w0_initial=w0_initial,
            final_activation="id",
        )

        self.modulator = Modulator(
            dim_in=latent_dim,
            dim_hidden=self.net.dim_hidden,
            num_layers=self.net.num_layers,
        )

    def forward(self, x, z):
        mods = self.modulator(z.squeeze())
        return self.net(x, mods)
