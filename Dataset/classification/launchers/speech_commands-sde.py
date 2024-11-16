import os
import torch

# CUDA for PyTorch
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

import src.common.common_sde as common
import Dataset.classification.utils.speech_commands as speech_commands


def main(device='cuda', max_epochs=200, *,                                        # training parameters
         model_name, hidden_channels, hidden_hidden_channels, num_hidden_layers,  # model parameters
         dry_run=False,
         **kwargs):                                                               # kwargs passed on to cdeint

    batch_size = 1024
    lr = 1e-3

    intensity_data = True if model_name in ('odernn', 'dt', 'decay') else False
    times, train_dataloader, val_dataloader, test_dataloader = speech_commands.get_data(intensity_data,
                                                                                                 batch_size)
    input_channels = 1 + (1 + intensity_data) * 20

    make_model = common.make_model(model_name, input_channels, 10, hidden_channels, hidden_hidden_channels,
                                   num_hidden_layers, use_intensity=False, initial=True)

    def new_make_model():
        model, regularise = make_model()
        model.linear[-1].weight.register_hook(lambda grad: 100 * grad)
        model.linear[-1].bias.register_hook(lambda grad: 100 * grad)
        return model, regularise

    name = None if dry_run else 'speech_commands'
    num_classes = 10
    return common.main(name, model_name, times, train_dataloader, val_dataloader, test_dataloader, device, new_make_model,
                       num_classes, max_epochs, lr, kwargs, step_mode='valaccuracy')


def run_all(device, model_names=['staticsde', 'naivesde', 'neurallsde', 'neurallnsde', 'neuralgsde']):

    hidden = 4
    num_layer = 1

    model_kwargs = dict(staticsde=dict(hidden_channels=hidden, hidden_hidden_channels=hidden, num_hidden_layers=num_layer),
                        naivesde=dict(hidden_channels=hidden, hidden_hidden_channels=hidden, num_hidden_layers=num_layer),
                        neurallsde=dict(hidden_channels=hidden, hidden_hidden_channels=hidden, num_hidden_layers=num_layer),
                        neurallnsde=dict(hidden_channels=hidden, hidden_hidden_channels=hidden, num_hidden_layers=num_layer),
                        neuralgsde=dict(hidden_channels=hidden, hidden_hidden_channels=hidden, num_hidden_layers=num_layer),)
    for model_name in model_names:
        main(device, model_name=model_name, **model_kwargs[model_name])

if __name__ == "__main__":
    # Define parameters directly in the code
    device = 'cpu'  # Choose 'cuda' or 'cpu'
    model_names = ['neurallnsde']  # List of models to run
    num_runs = 1  # Number of repetitions

    for _ in range(num_runs):
        run_all(device=device, model_names=model_names)