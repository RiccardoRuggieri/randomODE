import os
import torch

# CUDA for PyTorch
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

import common.regression.trainer_regression as common
import Dataset.regression.utils.diffusions_data_regression as diffusion_data

def train_model(device='cuda', max_epochs=50, *,                                        # training parameters
         model_name, hidden_channels, hidden_hidden_channels, num_hidden_layers,  # model parameters
         dry_run=False,
         **kwargs):                                                               # kwargs passed on to cdeint

    lr = 1e-3

    times, train_dataloader, val_dataloader, test_dataloader = diffusion_data.get_data(batch_size=32)

    # time series channels + time channel
    input_channels = 1 + 1
    num_classes = 1

    make_model = common.make_model(model_name, input_channels, num_classes, hidden_channels, hidden_hidden_channels,
                                   num_hidden_layers, use_intensity=False, initial=True)

    def new_make_model():
        model, regularise = make_model()
        model.linear[-1].weight.register_hook(lambda grad: 100 * grad)
        model.linear[-1].bias.register_hook(lambda grad: 100 * grad)
        return model, regularise

    name = None if dry_run else 'speech_commands'

    return common.main(name, model_name, times, train_dataloader, val_dataloader, test_dataloader, device,
                       new_make_model, max_epochs, lr, kwargs)


def run_all(device, model_names=['staticsde', 'naivesde', 'neurallsde', 'neurallnsde', 'neuralgsde']):

    hidden = 32
    num_layer = 1


    model_kwargs = dict(staticsde=dict(hidden_channels=hidden, hidden_hidden_channels=hidden, num_hidden_layers=num_layer),
                        naivesde=dict(hidden_channels=hidden, hidden_hidden_channels=hidden, num_hidden_layers=num_layer),
                        neurallsde=dict(hidden_channels=hidden, hidden_hidden_channels=hidden, num_hidden_layers=num_layer),
                        neurallnsde=dict(hidden_channels=hidden, hidden_hidden_channels=hidden, num_hidden_layers=num_layer),
                        neuralgsde=dict(hidden_channels=hidden, hidden_hidden_channels=hidden, num_hidden_layers=num_layer),)
    for model_name in model_names:
        train_model(device, model_name=model_name, **model_kwargs[model_name])

if __name__ == "__main__":
    # Define parameters directly in the code
    device = 'cuda'  # Choose 'cuda' or 'cpu'
    model_names = ['neurallsde']  # List of models to run
    num_runs = 1  # Number of repetitions

    for _ in range(num_runs):
        run_all(device=device, model_names=model_names)