import torch
import torchcde
import torchsde

# synthetic class for a 'Langevin' neural SDE
class NeuralSDE(torch.nn.Module):
    def __init__(self, func, input_channels, hidden_channels, output_channels, initial=True):
        super().__init__()
        self.func = func
        self.initial = initial
        self.initial_network = torch.nn.Linear(input_channels, hidden_channels)

        self.linear = torch.nn.Sequential(torch.nn.Linear(hidden_channels, hidden_channels),
                                          torch.nn.BatchNorm1d(hidden_channels), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                                          torch.nn.Linear(hidden_channels, output_channels))

    def forward(self, times, coeffs, final_index, z0=None, stream=False, **kwargs):
        # control module
        self.func.set_X(*coeffs, times)

        # Hardcoded, but could be made more flexible - see the original class
        if z0 is None:
            assert self.initial, "Was not expecting to be given no value of z0."
            z0 = self.initial_network(self.func.X.evaluate(times[0]))

        # Figure out what times we need to solve for
        if stream:
            t = times
        else:
            # faff around to make sure that we're outputting at all the times we need for final_index.
            sorted_final_index, inverse_final_index = final_index.unique(sorted=True, return_inverse=True)
            if 0 in sorted_final_index:
                sorted_final_index = sorted_final_index[1:]
                final_index = inverse_final_index
            else:
                final_index = inverse_final_index + 1
            if len(times) - 1 in sorted_final_index:
                sorted_final_index = sorted_final_index[:-1]
            t = torch.cat([times[0].unsqueeze(0), times[sorted_final_index], times[-1].unsqueeze(0)])

        # Switch default solver
        if 'method' not in kwargs:
            kwargs['method'] = 'euler'  # use 'srk' for more accurate solution for SDE
        if kwargs['method'] == 'euler':
            if 'options' not in kwargs:
                kwargs['options'] = {}
            options = kwargs['options']
            if 'dt' not in options:
                time_diffs = times[1:] - times[:-1]
                options['dt'] = max(time_diffs.min().item(), 1e-3)

        time_diffs = times[1:] - times[:-1]
        # used to be 1e-3
        dt = max(time_diffs.min().item(), 1e-3)

        z_t = torchsde.sdeint(sde=self.func,
                              y0=z0,
                              ts=t,
                              dt=dt,
                              **kwargs)

        # Organise the output
        if stream:
            # z_t is a tensor of shape (times, ..., channels), so change this to (..., times, channels)
            for i in range(len(z_t.shape) - 2, 0, -1):
                z_t = z_t.transpose(0, i)
        else:
            # final_index is a tensor of shape (...)
            # z_t is a tensor of shape (times, ..., channels)
            final_index = torch.tensor([len(times) - 1], device=z_t.device)
            final_index_indices = final_index.unsqueeze(-1).expand(z_t.shape[1:]).unsqueeze(0)
            z_t = z_t.gather(dim=0, index=final_index_indices).squeeze(0)

        # Linear map and return
        pred_y = self.linear(z_t)
        return pred_y


class DiffusionModel(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers, theta=1.0, sigma=1.0):
        """
        Here you specify drift and diffusion functions (f and g respectively).
        'Diffusion' is a broader term referring to the solution of the SDE. Once you have a diffusion,
        you can sample from it to get a trajectory.
        """
        super().__init__()
        self.sde_type = "ito"
        self.noise_type = "diagonal"  # or "scalar"

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        # network
        self.initial_network = torch.nn.Linear(input_channels, hidden_channels)
        self.linear_in = torch.nn.Linear(hidden_channels, hidden_hidden_channels)
        self.emb = torch.nn.Linear(hidden_channels * 2, hidden_channels)

        self.linears = torch.nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
                                           for _ in range(num_hidden_layers - 1))
        self.linear_out = torch.nn.Linear(hidden_hidden_channels, hidden_channels)

        # parameter
        self.theta = torch.nn.Parameter(torch.tensor([[theta]]), requires_grad=True)  # scaling factor

        # noise
        self.noise_t = torch.nn.Sequential(torch.nn.Linear(2, hidden_channels), torch.nn.ReLU(),
                                               torch.nn.Linear(hidden_channels, hidden_channels))

    def set_X(self, coeffs, times):
        self.coeffs = coeffs
        self.times = times
        self.X = torchcde.CubicSpline(self.coeffs, self.times)

    def f(self, t, y):
        Xt = self.X.evaluate(t)
        Xt = self.initial_network(Xt)

        # time embedding
        yy = self.linear_in(y)

        # input option
        z = self.emb(torch.cat([yy, Xt], dim=-1))

        # NN
        z = z.relu()
        for linear in self.linears:
            z = linear(z)
            z = z.relu()
        z = self.linear_out(z)

        z = z.tanh()
        return z

    def g(self, t, y):
        if t.dim() == 0:
            t = torch.full_like(y[:, 0], fill_value=t).unsqueeze(-1)

        tt = self.noise_t(torch.cat([torch.sin(t), torch.cos(t)], dim=-1)).relu()
        noise = tt

        noise = self.theta.sigmoid() * torch.nan_to_num(noise)  # bounding # ignore nan
        noise = noise.tanh()
        return noise  # diagonal noise # return noise.unsqueeze(-1) # scalar noise