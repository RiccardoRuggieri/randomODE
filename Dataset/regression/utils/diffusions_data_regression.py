import pathlib

from Dataset.classification.utils import common

import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# paths
here = pathlib.Path(__file__).resolve().parent
processed_data_loc = here / "processed_data" / "stochastic_processes"

# Ensure directories exist
processed_data_loc.mkdir(parents=True, exist_ok=True)

class StochasticProcessDataset(Dataset):
    """
    Custom dataset for stochastic process classification.
    """
    def __init__(self, file_path):
        """
        Initializes the dataset by loading data from a file.
        :param file_path: Path to the dataset file saved in .pt format.
        """
        dataset = torch.load(file_path)
        self.data = dataset["data"].to(torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def generate_stochastic_process_dataset(output_file, num_path_to_generate=1002, timesteps=100, dt=0.1, seed=42):
    torch.manual_seed(seed)
    import numpy as np
    np.random.seed(seed)

    def generate_ou_process(size, theta=0.5, mu=0.0, sigma=10):
        x = torch.zeros(size)
        for t in range(1, size[1]):
            x[:, t] = x[:, t - 1] + theta * (mu - x[:, t - 1]) * dt + sigma * torch.randn(size[0]) * torch.sqrt(torch.tensor(dt))
        return x

    # regression task
    num_classes = 1
    num_paths_per_class = num_path_to_generate // num_classes
    data = []

    paths = generate_ou_process(size=(num_paths_per_class, timesteps))
    data.append(paths)

    data = torch.cat(data, dim=0)

    indices = torch.randperm(num_path_to_generate)
    data = data[indices]

    dataset = {"data": data}
    torch.save(dataset, output_file)


def get_data_loader(batch_size=32, file_path=None):
    if file_path is None:
        file_path = processed_data_loc / "stochastic_processes.pt"
        processed_data_loc.mkdir(parents=True, exist_ok=True)
        generate_stochastic_process_dataset(file_path)
        # plot_sample_paths(file_path)

    dataset = StochasticProcessDataset(file_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def tensor_data(batch_size=32):
    dataset = get_data_loader(batch_size)
    # Extract X and y from the dataset
    X_list = []
    for batch in dataset:
        X_batch = batch
        X_list.append(X_batch)

    # univariate time series, you must unsqueeze the last dimension
    X = torch.cat(X_list, dim=0).unsqueeze(-1)

    return X

def get_data(batch_size=32):
    X = tensor_data(batch_size)

    # Generate times based on sampling rate and segment length
    times = torch.linspace(0, X.size(1) - 1, X.size(1))
    final_index = torch.tensor(X.size(1) - 1).repeat(X.size(0))

    (times, train_coeffs, val_coeffs, test_coeffs, train_final_index, val_final_index,
     test_final_index, _) = common.preprocess_data_regression(times, X, None, final_index,
                                                   append_times=True,
                                                   append_intensity=False)


    common.save_data(processed_data_loc,
                     times=times,
                     train_coeffs=train_coeffs, val_coeffs=val_coeffs, test_coeffs=test_coeffs,
                     train_final_index=train_final_index,
                     val_final_index=val_final_index, test_final_index=test_final_index)

    tensors = common.load_data(processed_data_loc)
    times = tensors['times']
    train_coeffs = tensors['train_coeffs']
    val_coeffs = tensors['val_coeffs']
    test_coeffs = tensors['test_coeffs']
    train_final_index = tensors['train_final_index']
    val_final_index = tensors['val_final_index']
    test_final_index = tensors['test_final_index']

    times, train_dataloader, val_dataloader, test_dataloader = common.wrap_data_regression(times, train_coeffs, val_coeffs,
                                                                                test_coeffs,
                                                                                train_final_index, val_final_index,
                                                                                test_final_index, 'cpu',
                                                                                batch_size=batch_size)

    return times, train_dataloader, val_dataloader, test_dataloader


def plot_sample_paths(file_path=None, num_samples=100):
    # Load dataset
    dataset = StochasticProcessDataset(file_path)

    # Get data
    data = dataset.data

    # Select a subset of the data to plot
    data_to_plot = data[:num_samples]

    # Create a plot
    plt.figure(figsize=(10, 6))

    for i in range(num_samples):
        plt.plot(data_to_plot[i].numpy(), label=f'Sample {i+1}')

    plt.title('Sample Paths')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend(loc='best')
    plt.show()