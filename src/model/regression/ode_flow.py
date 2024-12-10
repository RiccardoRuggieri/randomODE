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
        self.initial = nn.Linear(input_dim, hidden_dim)  # Encoder for inputs
        self.decoder = nn.Linear(hidden_dim, output_dim)  # Decoder for outputs

    def dynamics(self, t, y, inject_noise=True, noise_fn=None):
        deterministic = self.func.f(t, y)
        if inject_noise:
            if noise_fn is None:
                noise_fn = torch.randn_like
            noise = noise_fn(y).to(y.device)
            stochastic = self.func.g(t, y) * noise
            return deterministic + stochastic
        return deterministic

    def forward(self, coeffs, times, inject_noise=True):
        self.func.set_X(coeffs, times)
        y0 = self.func.X.evaluate(times)  # Interpolated data
        z0 = self.initial(y0)[:, 0, :]  # Encode initial embeddings

        z = odeint(
            lambda t, y: self.dynamics(t, y, inject_noise),
            z0,
            times,
            method='euler',
            options={"step_size": 0.05}
        )
        z = z.permute(1, 0, 2)  # Reshape for batch processing
        return self.decoder(z)

    def interpolate_and_predict_velocity(self, coeffs, t1, t2, true_labels):
        """
        Compute interpolations and velocity for flow matching.
        Args:
        - coeffs: Input coefficients
        - t: Random time steps
        Returns:
        - zt: Interpolated embeddings at time t
        - vt: Ground truth velocities
        - predicted_vt: Predicted velocities
        """

        # Ensure the coefficients are prepared for interpolation
        times = torch.linspace(0, 1, coeffs.shape[1], device=coeffs.device)
        self.func.set_X(coeffs, times)
        y0 = self.func.X.evaluate(times)

        # Encode initial embedding (z0)
        z0 = self.initial(y0)[:, 0, :]
        # sampling at time t through f
        z_t1 = self.dynamics(t1.squeeze(), z0, inject_noise=False)
        z_t2 = self.dynamics(t2.squeeze(), z0, inject_noise=False)

        # Predicted velocity
        predicted_vt_1 = self.dynamics(t1.squeeze(), z_t1, inject_noise=False)
        predicted_vt_2 = self.dynamics(t2.squeeze(), z_t2, inject_noise=False)

        if t1.squeeze() < t2.squeeze():
            v_t = (z_t2 - z_t1) / (t2 - t1)
            predicted_vt = (predicted_vt_2 - predicted_vt_1) / (t2 - t1)
            z_t = z_t1
        else:
            v_t = (z_t1 - z_t2) / (t1 - t2)
            predicted_vt = (predicted_vt_1 - predicted_vt_2) / (t1 - t2)
            z_t = z_t2

        return z_t, v_t, predicted_vt



