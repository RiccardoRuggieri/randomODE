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
            #t = t.item()
            t = torch.full_like(y[:, 0], fill_value=t).unsqueeze(-1)
        yy = self.linear_in(torch.cat((t, y), dim=-1))
        z = self.emb(torch.cat([yy, Xt], dim=-1))
        z = self.f_net(z)
        return self.linear_out(z)

# Generator
class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers, activation='lipswish', vector_field=None):
        super(Generator, self).__init__()
        self.func = vector_field(input_dim, hidden_dim, hidden_dim, num_layers, activation=activation)
        self.initial = nn.Linear(input_dim, hidden_dim)
        # we choose Sequential + BatchNorm1d for multi-variate time series
        # todo: understand why sometimes batch norm is better and sometimes it is not
        if input_dim <= 2:
            self.classifier = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim),
                                               torch.nn.BatchNorm1d(hidden_dim), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                                               torch.nn.Linear(hidden_dim, num_classes))
        else:
            # we choose an MLP for uni-variate time series
            #self.classifier = MLP(hidden_dim, num_classes, hidden_dim, num_layers, tanh=True)
            self.classifier = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim),
                                                  torch.nn.ReLU(),
                                                  torch.nn.Linear(hidden_dim, num_classes))


    def forward(self, coeffs, times):
        self.func.set_X(coeffs, times)
        y0 = self.func.X.evaluate(times)
        #print(y0.shape)
        y0 = self.initial(y0)[:, 0, :]  # Initial hidden state

        z = odeint(self.func, y0, times, method='euler', options={"step_size": 0.05})

        # final_index is a tensor of shape (...)
        # z_t is a tensor of shape (times, ..., dim)
        final_index = torch.tensor([len(times) - 1], device=z.device)
        final_index_indices = final_index.unsqueeze(-1).expand(z.shape[1:]).unsqueeze(0)
        z = z.gather(dim=0, index=final_index_indices).squeeze(0)

        # Linear map and return
        pred_y = self.classifier(z)
        return pred_y