import numpy as np
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

class ControlledOUFunc(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_hidden_dim, num_layers, activation='lipswish'):
        super(ControlledOUFunc, self).__init__()

        self.linear_in = nn.Linear(hidden_dim + 1, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, hidden_dim)
        self.linear_X = nn.Linear(input_dim, hidden_dim)
        self.emb = nn.Linear(hidden_dim * 2, hidden_dim)

        # self.theta_net = MLP(hidden_dim, hidden_dim, hidden_hidden_dim, num_layers, activation=activation)
        # self.mu_net = MLP(hidden_dim, hidden_dim, hidden_hidden_dim, num_layers, activation=activation)
        # self.sigma_net = MLP(hidden_dim, hidden_dim, hidden_hidden_dim, num_layers, activation=activation)

        self.theta_net = nn.Linear(hidden_dim, hidden_dim)
        self.mu_net = nn.Linear(hidden_dim, hidden_dim)
        self.sigma_net = nn.Linear(hidden_dim, hidden_dim)

    def set_X(self, coeffs, times):
        self.coeffs = coeffs
        self.times = times
        self.X = torchcde.CubicSpline(self.coeffs, self.times)

    def forward(self, t, y):
        # Xt = self.X.evaluate(t)  # Evaluate input time series at time t
        # Xt = self.linear_X(Xt)

        if t.dim() == 0:
            t = torch.full_like(y[:, 0], fill_value=t).unsqueeze(-1)

        # yy = self.linear_in(torch.cat((t, y), dim=-1))
        # z = self.emb(torch.cat([yy, Xt], dim=-1))

        # Controlled OU parameters
        theta = self.theta_net(y)  # Mean-reversion rate
        mu = self.mu_net(y)        # Long-term mean
        sigma = self.sigma_net(y)  # Volatility

        # OU dynamics
        drift = theta * (mu - y)  # Drift term
        diffusion = sigma * torch.randn_like(y.squeeze(-1))  # Stochastic term

        return drift + diffusion


class ControlledOUGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, activation=None):
        super(ControlledOUGenerator, self).__init__()
        self.func = ControlledOUFunc(input_dim, hidden_dim, hidden_dim, num_layers, activation=activation)
        self.initial = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def sampler(self, coeffs, times, index):
        self.func.set_X(coeffs, times)
        y0 = self.func.X.evaluate(times)[:, index, :]
        y0 = self.initial(y0)
        ys = [y0]

        # Iterate over the time points to simulate the controlled OU process
        for i in range(1, len(times[index:])):

            t_prev = times[i - 1]
            t_curr = times[i]
            delta_t = t_curr - t_prev

            # Current state
            y_prev = ys[-1]

            # Evaluate the controlled OU function at the previous time point
            theta = self.func.theta_net(y_prev)
            mu = self.func.mu_net(y_prev)
            sigma = self.func.sigma_net(y_prev)

            dW = np.random.normal(0, np.sqrt(delta_t))

            # OU dynamics
            drift = theta * (mu - y_prev)
            diffusion = sigma * torch.sqrt(delta_t) * dW

            # Update the state
            y_curr = y_prev + drift * delta_t + diffusion
            ys.append(y_curr)

        # Stack the results to form the time series
        ys = torch.stack(ys, dim=1)  # Shape: (batch_size, num_time_points, hidden_dim)

        # Decode the hidden states to output space
        return self.decoder(ys).squeeze(-1)


    def forward(self, coeffs, times):
        self.func.set_X(coeffs, times)
        y0 = self.func.X.evaluate(times)[:, 0, :]
        y0 = self.initial(y0)

        z = odeint(self.func,
                   y0,
                   times,
                   method='euler',
                   options={"step_size": 0.01})

        z = z.permute(1, 0, 2)
        return self.decoder(z)




