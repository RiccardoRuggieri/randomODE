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

# Generator function
class GeneratorFunc(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_hidden_dim, num_layers, activation='lipswish'):
        super(GeneratorFunc, self).__init__()

        self.linear_in = nn.Linear(hidden_dim + 1, hidden_dim)
        self.linear_X = nn.Linear(input_dim, hidden_dim)
        self.emb = nn.Linear(hidden_dim * 2, hidden_dim)
        self.f_net = MLP(hidden_dim, hidden_dim, hidden_hidden_dim, num_layers, activation=activation)
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
        z = self.f_net(z)
        return self.linear_out(z)

# Generator
class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation=None, vector_field=None):
        super(Generator, self).__init__()
        self.func = vector_field(input_dim, hidden_dim, hidden_dim, num_layers, activation=activation)
        self.initial = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, coeffs, times):
        self.func.set_X(coeffs, times)
        y0 = self.func.X.evaluate(times)
        y0 = self.initial(y0)[:, 0, :]

        z = odeint(self.func, y0, times, method='rk4', options={"step_size": 0.05})

        z = z.permute(1, 0, 2)
        return self.decoder(z)



