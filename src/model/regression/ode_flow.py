import torch
import torch.nn as nn
import torchcde
from torchdiffeq import odeint

class LipSwish(nn.Module):
    def forward(self, x):
        return 0.909 * torch.nn.functional.silu(x)

class MLP(nn.Module):
    def __init__(self, in_size, out_size, hidden_dim, num_layers, tanh=False, activation='lipswish'):
        super().__init__()

        if activation == 'lipswish':
            activation_fn = LipSwish()
        else:
            activation_fn = nn.ReLU()

        model = [nn.Linear(in_size, hidden_dim), activation_fn]
        for _ in range(num_layers - 1):
            model.append(nn.Linear(hidden_dim, hidden_dim))
            model.append(activation_fn)
        model.append(nn.Linear(hidden_dim, out_size))
        if tanh:
            model.append(nn.Tanh())
        self._model = nn.Sequential(*model)

    def forward(self, x):
        return self._model(x)

class GeneratorFunc(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_hidden_dim, num_layers, activation='lipswish'):
        super(GeneratorFunc, self).__init__()
        self.linear_in = nn.Linear(hidden_dim + 1, hidden_dim)
        self.linear_X = nn.Linear(input_dim, hidden_dim)
        self.emb = nn.Linear(hidden_dim * 2, hidden_dim)
        self.f_net = MLP(hidden_dim, hidden_dim, hidden_hidden_dim, num_layers, activation=activation)
        self.g_net = MLP(hidden_dim, hidden_dim, hidden_hidden_dim, num_layers, activation=activation)
        self.linear_out = nn.Linear(hidden_dim, hidden_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def set_X(self, coeffs, times):
        self.coeffs = coeffs
        self.times = times
        self.X = torchcde.CubicSpline(self.coeffs, self.times)

    def f(self, t, y):
        Xt = self.X.evaluate(t)
        Xt = self.linear_X(Xt)

        if t.dim() == 0:
            t = torch.full_like(y[:, 0], fill_value=t).unsqueeze(-1)
        yy = self.linear_in(torch.cat((t, y), dim=-1))
        z = self.emb(torch.cat([yy,Xt], dim=-1))
        z = self.f_net(z) * y # (1 - torch.nan_to_num(y).sigmoid())
        return self.linear_out(z)

    def g(self, t, y):
        if t.dim() == 0:
            t = torch.full_like(y[:, 0], fill_value=t).unsqueeze(-1)
        tt = self.linear_in(torch.cat((t, y), dim=-1))
        return self.g_net(tt)

class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, vector_field=None):
        super(Generator, self).__init__()
        self.func = vector_field(input_dim, hidden_dim, hidden_dim, num_layers, activation='lipswish')
        # encoder
        self.initial = nn.Linear(input_dim, hidden_dim)
        # decoder
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def dynamics(self, t, y, inject_noise, noise_fn=None):
        deterministic = self.func.f(t, y)
        if inject_noise:
            if noise_fn is None:
                noise_fn = torch.randn_like  # Default to Gaussian noise
            noise = noise_fn(y).to(y.device)  # Apply the noise function
            stochastic = self.func.g(t, y) * noise
            return deterministic + stochastic
        return deterministic

    def forward(self, coeffs, times, inject_noise=True):
        self.func.set_X(coeffs, times)
        y0 = self.func.X.evaluate(times)  # Cubic spline interpolation of input coefficients
        z0 = self.initial(y0)[:, 0, :]  # Initial embedding of data (z0)

        z = odeint(
            lambda t, y: self.dynamics(t, y, inject_noise),
            z0,
            times,
            method='euler',  # Use Euler solver for integration
            options={"step_size": 0.05}
        )

        z = z.permute(1, 0, 2)  # Reshape for batch processing
        return self.decoder(z)

    def interpolate_and_predict_velocity(self, coeffs, t, batch):
        """
        Compute interpolation and velocity for flow matching loss.

        Args:
        - coeffs: Input coefficients (z0 embeddings)
        - true: Ground truth labels (z1 embeddings)
        - t: Random time steps for interpolation

        Returns:
        - zt: Interpolated embeddings at time t
        - vt: Ground truth velocities
        - predicted_vt: Predicted velocities at time t
        """
        # Ensure that the coefficients and times are properly set for cubic spline
        times = torch.linspace(0, 1, batch[0].shape[1], device=coeffs.device)
        self.func.set_X(coeffs, times)
        y0 = self.func.X.evaluate(times)

        # Compute embeddings for data (z0) and labels (z1)
        z0 = self.initial(y0)[:, 0, :]
        z1 = odeint(
            lambda u, y: self.dynamics(u, y, inject_noise=False),
            z0,
            times,
            method='euler',  # Use Euler solver for integration
            options={"step_size": 0.05}
        )  # Final embedding (encoded labels)

        # Reshape t to allow broadcasting
        t = t.view(-1, 1, 1)  # Shape: (batch_size, 1, 1)

        # Interpolate between z0 and z1
        z_t = (1 - t) * z0.unsqueeze(1) + t * z1.unsqueeze(1)  # Interpolated embeddings
        # ????
        vt = z1 - z0  # Ground truth velocity

        # todo: problem here: how do you compute the predicted velocity?

        # Predict velocity at interpolated embeddings
        t_scalar = t.squeeze(-1)  # Extract scalar times for passing to `f`
        predicted_vt = self.dynamics(t_scalar, z_t, inject_noise=True)  # Evaluate f at interpolated points

        return z_t.squeeze(1), vt, predicted_vt


