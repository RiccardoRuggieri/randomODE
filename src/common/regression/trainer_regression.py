import json
import numpy as np
import pathlib
import torch
import tqdm

from model.classification.my_neural_lsde import NeuralSDE, DiffusionModel

here = pathlib.Path(__file__).resolve().parent


def _add_weight_regularisation(loss_fn, regularise_parameters, scaling=0.01):
    def new_loss_fn(pred_y, true_y):
        total_loss = loss_fn(pred_y, true_y)
        for parameter in regularise_parameters.parameters():
            if parameter.requires_grad:
                total_loss = total_loss + scaling * parameter.norm()
        return total_loss

    return new_loss_fn


class _SqueezeEnd(torch.nn.Module):
    def __init__(self, model):
        super(_SqueezeEnd, self).__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs).squeeze(-1)


def _count_parameters(model):
    """Counts the number of parameters in a model."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad_)


class _AttrDict(dict):
    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, item):
        return self[item]


class _SuppressAssertions:
    def __init__(self, tqdm_range):
        self.tqdm_range = tqdm_range

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is AssertionError:
            self.tqdm_range.write('Caught AssertionError: ' + str(exc_val))
            return True


def _train_loop(train_dataloader, model, times, optimizer, loss_fn, max_epochs, device, kwargs):
    model.train()

    tqdm_range = tqdm.tqdm(range(max_epochs))
    tqdm_range.write('Starting training for model:\n\n' + str(model) + '\n\n')
    for epoch in tqdm_range:
        epoch_loss = 0.0
        for batch in train_dataloader:
            batch = tuple(b.to(device) for b in batch)
            with _SuppressAssertions(tqdm_range):
                *train_coeffs, train_y, lengths = batch
                pred_y = model(times, train_coeffs, lengths, **kwargs)
                loss = loss_fn(pred_y, train_y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()

        # Calculate and print the average loss for the epoch
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        tqdm_range.write(f'Epoch {epoch + 1}/{max_epochs}, MSE Loss: {avg_epoch_loss:.4f}')


class _TensorEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (torch.Tensor, np.ndarray)):
            return o.tolist()
        else:
            super(_TensorEncoder, self).default(o)


def main(name, model_name, times, train_dataloader, val_dataloader, test_dataloader, device, make_model,
         max_epochs,
         lr, kwargs):
    times = times.to(device)
    if device != 'cpu':
        torch.cuda.reset_max_memory_allocated(device)
        baseline_memory = torch.cuda.memory_allocated(device)
    else:
        baseline_memory = None

    model, regularise_parameters = make_model()

    # Regression
    loss_fn = torch.nn.MSELoss()
    loss_fn = _add_weight_regularisation(loss_fn, regularise_parameters)
    model.to(device)

    # choose optimizer here
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lr * 0.01)

    # start the training loop
    _train_loop(train_dataloader, model, times, optimizer, loss_fn, max_epochs, device, kwargs)

    model.eval()

    if device != 'cpu':
        memory_usage = torch.cuda.max_memory_allocated(device) - baseline_memory
    else:
        memory_usage = None

    result = _AttrDict(name=name,
                       model_name=model_name,
                       times=times,
                       memory_usage=memory_usage,
                       baseline_memory=baseline_memory,
                       train_dataloader=train_dataloader,
                       val_dataloader=val_dataloader,
                       test_dataloader=test_dataloader,
                       model=model.to('cpu'),
                       parameters=_count_parameters(model))

    return result


def make_model(name, input_channels, output_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers,
               use_intensity, initial):
    if name == 'neurallsde':
        def make_model():
            vector_field = DiffusionModel(input_channels=input_channels, hidden_channels=hidden_channels,
                                          hidden_hidden_channels=hidden_hidden_channels,
                                          num_hidden_layers=num_hidden_layers)
            model = NeuralSDE(func=vector_field, input_channels=input_channels,
                              hidden_channels=hidden_channels, output_channels=output_channels, initial=initial)
            return model, vector_field
    else:
        raise ValueError("Unrecognised model name {}."
                         "".format(name))
    return make_model