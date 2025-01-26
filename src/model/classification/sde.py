import torch
import torch.nn as nn
import torchcde
import torchsde

# simple langevin sde for benchmarking

class LipSwish(nn.Module):
    def forward(self, x):
        return 0.909 * torch.nn.functional.silu(x)

# Generator function
class GeneratorFunc(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_hidden_dim, num_layers, activation='lipswish'):
        super(GeneratorFunc, self).__init__()

        self.sde_type = "ito"
        self.noise_type = "diagonal"
        self.linear_in = nn.Linear(hidden_dim + 1, hidden_dim)
        self.linear_X = nn.Linear(input_dim, hidden_dim)
        self.emb = nn.Linear(hidden_dim * 2, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, hidden_dim)

        # noise
        self.noise_t = torch.nn.Sequential(torch.nn.Linear(2, hidden_dim), torch.nn.ReLU(),
                                           torch.nn.Linear(hidden_dim, hidden_dim))

        # network
        self.linears = torch.nn.ModuleList(torch.nn.Linear(hidden_hidden_dim, hidden_hidden_dim)
                                           for _ in range(num_layers - 1))

        # parameter
        theta = 1.0
        self.theta = torch.nn.Parameter(torch.tensor([[theta]]), requires_grad=True)  # scaling factor

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
        #z = self.emb(torch.cat([yy, Xt], dim=-1))
        z = yy
        z = z.relu()
        for linear in self.linears:
            z = linear(z)
            z = z.relu()
        z = self.linear_out(z)
        return z

    def g(self, t, y):
        if t.dim() == 0:
            t = torch.full_like(y[:, 0], fill_value=t).unsqueeze(-1)

        tt = self.noise_t(torch.cat([torch.sin(t), torch.cos(t)], dim=-1)).relu()
        noise = tt

        noise = self.theta.sigmoid() * torch.nan_to_num(noise)  # bounding # ignore nan
        noise = noise.tanh()
        return noise

# synthetic class for a 'Langevin' neural SDE
class Generator(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers, activation='lipswish', vector_field=None):
        super(Generator, self).__init__()
        self.func = vector_field(input_dim, hidden_dim, hidden_dim, num_layers, activation=activation)
        self.initial = nn.Linear(input_dim, hidden_dim)
        self.classifier = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim),
                                              torch.nn.BatchNorm1d(hidden_dim), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                                              torch.nn.Linear(hidden_dim, num_classes))
        # self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, coeffs, times):
        self.func.set_X(coeffs, times)
        y0 = self.func.X.evaluate(times)
        y0 = self.initial(y0)[:, 0, :]  # Initial hidden state

        z_t = torchsde.sdeint(sde=self.func,
                              y0=y0,
                              ts=times,
                              dt=0.02,
                              method='euler')

        final_index = torch.tensor([len(times) - 1], device=z_t.device)
        final_index_indices = final_index.unsqueeze(-1).expand(z_t.shape[1:]).unsqueeze(0)
        z_t = z_t.gather(dim=0, index=final_index_indices).squeeze(0)

        # Linear map and return
        pred_y = self.classifier(z_t)
        return pred_y
