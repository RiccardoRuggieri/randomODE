import torch
import torch.nn as nn
import torchcde
import torchsde

class MLP(nn.Module):
    def __init__(self, in_size, out_size, hidden_dim, num_layers, tanh=False):
        super().__init__()

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
    def __init__(self, input_dim, hidden_dim, hidden_hidden_dim, num_layers):
        super(GeneratorFunc, self).__init__()

        # this is just to use sdeint, which is faster than odeint.
        self.sde_type = "ito"
        self.noise_type = "scalar"

        self.linear_in = nn.Linear(hidden_dim + 1, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, hidden_dim)
        self.linear_X = nn.Linear(input_dim, hidden_dim)
        self.emb = nn.Linear(hidden_dim * 2, hidden_dim)
        self.f_net = MLP(hidden_dim, hidden_dim, hidden_hidden_dim, num_layers)

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
        z = self.emb(torch.cat([yy, Xt], dim=-1))
        z = z.relu()
        z = self.f_net(z)
        return self.linear_out(z)

    # setting the diffusion term to zero
    def g(self, t, y):
        return torch.zeros(y.size(0), y.size(1), 1).to(y.device)

class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers, vector_field=None):
        super(Generator, self).__init__()
        self.func = vector_field(input_dim, hidden_dim, hidden_dim, num_layers)
        self.initial = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim, num_classes))

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

        z = z[-1]

        return self.decoder(z)