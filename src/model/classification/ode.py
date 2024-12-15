import torch
import torch.nn as nn
import torchcde
import torchsde
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

class NeuralODEFunc(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_hidden_dim, num_layers, activation='lipswish'):
        super(NeuralODEFunc, self).__init__()
        self.sde_type = "ito"
        self.noise_type = "scalar"

        self.linear_in = nn.Linear(hidden_dim + 1, hidden_dim)
        self.f_net = MLP(hidden_dim, hidden_dim, hidden_hidden_dim, num_layers, activation=activation)

    def set_X(self, coeffs, times):
        self.coeffs = coeffs
        self.times = times
        self.X = torchcde.CubicSpline(self.coeffs, self.times)

    def f(self, t, y):
        if t.dim() == 0:
            t = torch.full_like(y[:, 0], fill_value=t).unsqueeze(-1)
        # yy = self.linear_in(torch.cat((torch.sin(t), torch.cos(t), y), dim=-1))
        yy = self.linear_in(torch.cat((t, y), dim=-1))
        return self.f_net(yy)

    def g(self, t, y):
        return torch.zeros(y.size(0), y.size(1), 1).to(y.device)

class NDE_model(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers, activation='lipswish', vector_field=None):
        super(NDE_model, self).__init__()
        self.func = vector_field(input_dim, hidden_dim, hidden_dim, num_layers, activation=activation)
        self.initial = nn.Linear(input_dim, hidden_dim)
        self.decoder = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim),
                                           torch.nn.BatchNorm1d(hidden_dim), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                                           torch.nn.Linear(hidden_dim, num_classes))

    def forward(self, coeffs, times):
        # control module
        self.func.set_X(coeffs, times)

        y0 = self.func.X.evaluate(times[0])
        y0 = self.initial(y0)

        z = torchsde.sdeint(sde=self.func,
                            y0=y0,
                            ts=times,
                            dt=0.05,
                            method='euler')

        final_index = torch.tensor([len(times) - 1], device=z.device)
        final_index_indices = final_index.unsqueeze(-1).expand(z.shape[1:]).unsqueeze(0)
        z = z.gather(dim=0, index=final_index_indices).squeeze(0)

        return self.decoder(z)