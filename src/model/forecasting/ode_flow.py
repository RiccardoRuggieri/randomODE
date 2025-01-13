import torch
import torch.nn as nn
import torchcde
from torchdiffeq import odeint

class MLP(nn.Module):
    def __init__(self, in_size, out_size, hidden_dim, num_layers):
        super().__init__()
        activation_fn = nn.ReLU()
        model = [nn.Linear(in_size, hidden_dim), activation_fn]
        for _ in range(num_layers - 1):
            model.append(nn.Linear(hidden_dim, hidden_dim))
            model.append(activation_fn)
        model.append(nn.Linear(hidden_dim, out_size))
        # model.append(activation_fn)
        self._model = nn.Sequential(*model)

    def forward(self, x):
        return self._model(x)

# Generator function
class GeneratorFunc(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_hidden_dim, num_layers):
        super(GeneratorFunc, self).__init__()

        self.linear_in = nn.Linear(hidden_dim + 1, hidden_dim)
        self.linear_X = nn.Linear(input_dim, hidden_dim)
        self.emb = nn.Linear(hidden_dim * 2, hidden_dim)
        self.f_net = MLP(hidden_dim, hidden_dim, hidden_hidden_dim, num_layers)
        self.linear_out = nn.Linear(hidden_dim, hidden_dim)

    def set_X(self, coeffs, times):
        self.coeffs = coeffs
        self.times = times
        self.X = torchcde.CubicSpline(self.coeffs, self.times)

    def forward(self, t, y):
        Xt = self.X.evaluate(t)
        Xt = self.linear_X(Xt)
        if t.dim() == 0:
            t = torch.full_like(y[:, 0], fill_value=t).unsqueeze(-1)
        yy = self.linear_in(torch.cat((t, y), dim=-1))
        z = self.emb(torch.cat([yy, Xt], dim=-1))
        # z = z.relu()
        z = self.f_net(z)
        return self.linear_out(z)

# Generator
class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, forecast_horizon, num_layers, vector_field=None):
        super(Generator, self).__init__()
        self.func = vector_field(input_dim, hidden_dim, hidden_dim, num_layers)
        self.initial = nn.Linear(input_dim, hidden_dim)
        self.linear_in = nn.Linear(hidden_dim, forecast_horizon)
        self.embedd_time = nn.Linear(2 * forecast_horizon, hidden_dim)
        self.sample = nn.Linear(hidden_dim, forecast_horizon)
        self.forecast_horizon = forecast_horizon

    def forward(self, coeffs, times):
        self.func.set_X(coeffs, times)
        y0 = self.func.X.evaluate(times)
        y0 = self.initial(y0)[:, 0, :]  # Initial hidden state

        z = odeint(self.func, y0, times, method='euler', options={'step_size': 0.02})

        # forecasting in with time embedding
        z = z[-1]

        t = torch.linspace(0, 1, self.forecast_horizon).to(z.device)
        t = t.unsqueeze(0).repeat(z.shape[0], 1)

        z = self.linear_in(z)
        z = z.relu()
        z = self.embedd_time(torch.cat([z, t], dim=-1))
        z = z.relu()
        z = self.sample(z)

        return z

# class Generator(nn.Module):
#     def __init__(self, input_dim, hidden_dim, forecast_horizon, num_layers, vector_field=None):
#         super(Generator, self).__init__()
#         self.func = vector_field(input_dim, hidden_dim, hidden_dim, num_layers)
#         self.initial = nn.Linear(input_dim, hidden_dim)
#         self.classifier = nn.Linear(hidden_dim, hidden_dim)
#         self.sample = nn.Linear(hidden_dim, forecast_horizon)
#         self.forecast_horizon = forecast_horizon
#
#         # Learnable time embedding
#         self.time_embedding = nn.Parameter(torch.randn(forecast_horizon, hidden_dim))
#
#         # Projection layer after concatenation
#         self.concat_projector = nn.Linear(2 * hidden_dim, hidden_dim)
#
#     def forward(self, coeffs, times):
#         # Set up the vector field with the input data
#         self.func.set_X(coeffs, times)
#         y0 = self.func.X.evaluate(times)  # Evaluate input data at time points
#         y0 = self.initial(y0)[:, 0, :]  # Initial hidden state (batch, hidden_dim)
#
#         # Solve the ODE
#         z = odeint(self.func, y0, times, method='euler', options={'step_size': 0.02})
#         z = z[-1]  # Final hidden state after ODE integration (batch, hidden_dim)
#
#         # Forecasting step
#         z = self.classifier(z).relu()  # Project to hidden space
#         z = z.unsqueeze(1).repeat(1, self.forecast_horizon, 1)  # Expand for each time step
#         time_embeddings = self.time_embedding.unsqueeze(0).repeat(z.shape[0], 1, 1)  # Broadcast time embeddings
#
#         # Concatenate z and time embeddings
#         concatenated = torch.cat([z, time_embeddings], dim=-1)  # Shape: (batch, forecast_horizon, 2 * hidden_dim)
#
#         # Project concatenated embeddings
#         projected = self.concat_projector(concatenated)  # Shape: (batch, forecast_horizon, hidden_dim)
#         projected = projected.relu()
#
#         # Aggregate and make final predictions
#         z = self.sample(projected.mean(dim=1))  # Mean pooling over forecast horizon
#
#         return z