import os
import torch
import argparse
import common.common_sde as common
import Dataset.classification.utils.sepsis as sepsis

# CUDA for PyTorch
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Recall that in neural sde the information about the data is directly
# passed to the model through the initial condition.
class InitialValueNetwork(torch.nn.Module):
    """
    This class defines the initial condition network for the SDE model.
    Recall that the initial value x_0 is interpolated from the observed data x
    and subsequently used to define the mapping for z(0)
    """
    def __init__(self, intensity, hidden_channels, model):
        super(InitialValueNetwork, self).__init__()
        self.linear1 = torch.nn.Linear(7 if intensity else 5, 256)
        self.linear2 = torch.nn.Linear(256, hidden_channels)

        self.model = model

    def forward(self, times, coeffs, final_index, **kwargs):
        *coeffs, static = coeffs
        # computation of the initial condition through non-linearities (NN)
        z0 = self.linear1(static)
        z0 = z0.relu()
        z0 = self.linear2(z0)
        return self.model(times, coeffs, final_index, z0=z0, **kwargs)


def train_model(intensity, device='cuda', max_epochs=200, pos_weight=10, *, model_name, hidden_channels,
                hidden_hidden_channels, num_hidden_layers, dry_run=False, **kwargs):
    batch_size = 1024
    lr = 1e-3

    static_intensity = intensity
    time_intensity = intensity or (model_name in ('odernn', 'dt', 'decay'))

    times, train_dataloader, val_dataloader, test_dataloader = sepsis.get_data(static_intensity,
                                                                                        time_intensity,
                                                                                        batch_size)

    input_channels = 1 + (1 + time_intensity) * 34
    make_model = common.make_model(model_name, input_channels, 1, hidden_channels,
                                   hidden_hidden_channels, num_hidden_layers, use_intensity=intensity, initial=False)

    def new_make_model():
        model, regularise = make_model()
        model.linear[-1].weight.register_hook(lambda grad: 100 * grad)
        model.linear[-1].bias.register_hook(lambda grad: 100 * grad)
        return InitialValueNetwork(intensity, hidden_channels, model), regularise

    if dry_run:
        name = None
    else:
        intensity_str = '_intensity' if intensity else '_nointensity'
        name = 'sepsis' + intensity_str
    num_classes = 2
    return common.main(name, model_name, times, train_dataloader, val_dataloader, test_dataloader, device,
                       new_make_model, num_classes, max_epochs, lr, kwargs, pos_weight=torch.tensor(pos_weight),
                       step_mode='valauc')


def run_all(intensity, device, model_names=['staticsde', 'naivesde', 'neurallsde', 'neurallnsde', 'neuralgsde']):

    hidden = 16
    num_layer = 1

    model_kwargs = dict(staticsde=dict(hidden_channels=hidden, hidden_hidden_channels=hidden, num_hidden_layers=num_layer),
                        naivesde=dict(hidden_channels=hidden, hidden_hidden_channels=hidden, num_hidden_layers=num_layer),
                        neurallsde=dict(hidden_channels=hidden, hidden_hidden_channels=hidden, num_hidden_layers=num_layer),
                        neurallnsde=dict(hidden_channels=hidden, hidden_hidden_channels=hidden, num_hidden_layers=num_layer),
                        neuralgsde=dict(hidden_channels=hidden, hidden_hidden_channels=hidden, num_hidden_layers=num_layer),)
    for model_name in model_names:
        train_model(intensity, device, model_name=model_name, **model_kwargs[model_name])


if __name__ == "__main__":
    # Define parameters directly in the code
    intensity = False  # Set to True or False
    device = 'cuda'  # Choose 'cuda' or 'cpu'
    model_names = ['neurallnsde']  # List of models to run
    num_runs = 1  # Number of repetitions

    for _ in range(num_runs):
        run_all(intensity=intensity, device=device, model_names=model_names)



