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

        # todo: weight initialization

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
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers, activation='lipswish', vector_field=None):
        super(Generator, self).__init__()
        self.func = vector_field(input_dim, hidden_dim, hidden_dim, num_layers, activation=activation)
        self.initial = nn.Linear(input_dim, hidden_dim)
        # we choose Sequential + BatchNorm1d for multi-variate time series
        if input_dim > 1:
            self.classifier = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim),
                                                  torch.nn.BatchNorm1d(hidden_dim), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                                                  torch.nn.Linear(hidden_dim, num_classes))
        else:
            # we choose an MLP for uni-variate time series
            self.classifier = MLP(hidden_dim, num_classes, hidden_dim, num_layers, tanh=False)

    def dynamics(self, t, y, inject_noise, noise_fn=None):
        deterministic = self.func.f(t, y)
        if inject_noise:
            if noise_fn is None:
                noise_fn = torch.randn_like  # Default to Gaussian noise
            noise = 0.01 * noise_fn(y).to(y.device) # Apply the noise function
            # todo: think about how to add the noise, if any. Otherwise, easy_flow is better
            stochastic = self.func.g(t, y) #* noise
            return deterministic + stochastic
        return deterministic

    def forward(self, coeffs, times, inject_noise=True):
        self.func.set_X(coeffs, times)
        y0 = self.func.X.evaluate(times)
        y0 = self.initial(y0)[:, 0, :]

        # z = odeint(
        #     lambda t, y: self.dynamics(t, y, inject_noise),
        #     y0,
        #     times,
        #     method='euler',
        #     options={"step_size": 0.05}
        # )
        
        # todo: see if it has some sense

        z_trajectory = []
        z = y0  # Initial state
        for t in times:
            z = z + self.dynamics(t, z, False)  # Flow dynamics: z(t+1) = z(t) + F(t, z(t))
            z_trajectory.append(z.unsqueeze(0))  # Add time dimension by unsqueezing

        z = torch.cat(z_trajectory, dim=0)  # Concatenate along the time axis

        # final_index is a tensor of shape (...)
        # z_t is a tensor of shape (times, ..., dim)
        final_index = torch.tensor([len(times) - 1], device=z.device)
        final_index_indices = final_index.unsqueeze(-1).expand(z.shape[1:]).unsqueeze(0)
        z = z.gather(dim=0, index=final_index_indices).squeeze(0)

        # Linear map and return
        pred_y = self.classifier(z)
        return pred_y