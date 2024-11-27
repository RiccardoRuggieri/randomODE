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
        self.labels = dataset["labels"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def generate_stochastic_process_dataset(output_file, num_path_to_generate=1002, timesteps=100, dt=0.1, seed=42):
    torch.manual_seed(seed)
    import numpy as np
    np.random.seed(seed)

    def generate_ou_process(size, theta=0.5, mu=0.0, sigma=10):
        x = torch.zeros(size)
        for t in range(1, size[1]):
            x[:, t] = x[:, t - 1] + theta * (mu - x[:, t - 1]) * dt + sigma * torch.randn(size[0]) * torch.sqrt(torch.tensor(dt))
        return x

    def generate_gbm(size, mu=0.0, sigma=0.1):
        x = torch.ones(size)
        for t in range(1, size[1]):
            x[:, t] = x[:, t - 1] * torch.exp((mu - 0.5 * sigma**2) * dt + sigma * torch.randn(size[0]) * torch.sqrt(torch.tensor(dt)))
        return x

    def generate_jump_diffusion(size, mu=0.0, sigma=5, jump_intensity=5, jump_mean=0.5, jump_std=2):
        x = torch.zeros(size)
        for t in range(1, size[1]):
            jump = (torch.rand(size[0]) < jump_intensity * dt).float() * (jump_mean + jump_std * torch.randn(size[0]))
            x[:, t] = x[:, t - 1] + mu * dt + sigma * torch.randn(size[0]) * torch.sqrt(torch.tensor(dt)) + jump
        return x

    def generate_logistic_process(size, r=0.5, K=1.0, sigma=0.1):
        x = torch.rand(size) * 0.1  # Start close to 0
        for t in range(1, size[1]):
            x[:, t] = x[:, t - 1] + r * x[:, t - 1] * (1 - x[:, t - 1] / K) * dt + sigma * torch.randn(size[0]) * torch.sqrt(torch.tensor(dt))
            x[:, t] = torch.clamp(x[:, t], min=0, max=K)  # Ensure within bounds
        return x

    def generate_brownian_bridge(size, sigma=0.1):
        x = torch.zeros(size)
        for t in range(1, size[1]):
            drift = -x[:, t - 1] / (size[1] - t)  # Constrains the end point
            x[:, t] = x[:, t - 1] + drift * dt + sigma * torch.randn(size[0]) * torch.sqrt(torch.tensor(dt))
        return x

    def generate_vasicek_process(size, theta=0.5, mu=0.0, sigma=0.1):
        x = torch.zeros(size)
        for t in range(1, size[1]):
            x[:, t] = x[:, t - 1] + theta * (mu - x[:, t - 1]) * dt + sigma * torch.randn(size[0]) * torch.sqrt(torch.tensor(dt))
        return x


    def generate_cir_process(size, theta=0.5, mu=0.5, sigma=2):
        x = torch.ones(size) * mu  # Start near the long-term mean
        for t in range(1, size[1]):
            sqrt_x = torch.sqrt(torch.clamp(x[:, t - 1], min=0))  # Ensure no negative values
            x[:, t] = x[:, t - 1] + theta * (mu - x[:, t - 1]) * dt + sigma * sqrt_x * torch.randn(size[0]) * torch.sqrt(torch.tensor(dt))
            x[:, t] = torch.clamp(x[:, t], min=0)  # Keep values non-negative
        return x

    num_classes = 3
    num_paths_per_class = num_path_to_generate // num_classes
    data, labels = [], []

    process_generators = [generate_ou_process,
                          generate_cir_process,
                          generate_jump_diffusion]

    for i, generator in enumerate(process_generators):
        paths = generator(size=(num_paths_per_class, timesteps))
        data.append(paths)
        labels.extend([i] * num_paths_per_class)

    data = torch.cat(data, dim=0)
    labels = torch.tensor(labels)

    indices = torch.randperm(num_path_to_generate)
    data = data[indices]
    labels = labels[indices]

    dataset = {"data": data, "labels": labels}
    torch.save(dataset, output_file)


def get_data_loader(batch_size=32, file_path=None):
    if file_path is None:
        file_path = processed_data_loc / "stochastic_processes.pt"
        processed_data_loc.mkdir(parents=True, exist_ok=True)
        generate_stochastic_process_dataset(file_path)
        plot_sample_paths(file_path)

    dataset = StochasticProcessDataset(file_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def tensor_data(batch_size=32):
    dataset = get_data_loader(batch_size)
    # Extract X and y from the dataset
    X_list = []
    y_list = []
    for batch in dataset:
        X_batch, y_batch = batch
        X_list.append(X_batch)
        y_list.append(y_batch)

    # univariate time series, you must unsqueeze the last dimension
    X = torch.cat(X_list, dim=0).unsqueeze(-1)
    y = torch.cat(y_list, dim=0)

    # print(X.shape, y.shape)

    return X, y

def get_data(batch_size=32):
    X, y = tensor_data(batch_size)

    # Generate times based on sampling rate and segment length
    times = torch.linspace(0, X.size(1) - 1, X.size(1))
    final_index = torch.tensor(X.size(1) - 1).repeat(X.size(0))

    (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
     test_final_index, _) = common.preprocess_data(times, X, y, final_index,
                                                   append_times=True,
                                                   append_intensity=False)

    common.save_data(processed_data_loc,
                     times=times,
                     train_coeffs=train_coeffs, val_coeffs=val_coeffs, test_coeffs=test_coeffs,
                     train_y=train_y, val_y=val_y, test_y=test_y, train_final_index=train_final_index,
                     val_final_index=val_final_index, test_final_index=test_final_index)

    tensors = common.load_data(processed_data_loc)
    times = tensors['times']
    train_coeffs = tensors['train_coeffs']
    val_coeffs = tensors['val_coeffs']
    test_coeffs = tensors['test_coeffs']
    train_y = tensors['train_y']
    val_y = tensors['val_y']
    test_y = tensors['test_y']
    train_final_index = tensors['train_final_index']
    val_final_index = tensors['val_final_index']
    test_final_index = tensors['test_final_index']

    times, train_dataloader, val_dataloader, test_dataloader = common.wrap_data(times, train_coeffs, val_coeffs,
                                                                                test_coeffs, train_y, val_y, test_y,
                                                                                train_final_index, val_final_index,
                                                                                test_final_index, 'cpu',
                                                                                batch_size=batch_size)

    return times, train_dataloader, val_dataloader, test_dataloader


def plot_sample_paths(file_path=None, num_samples_per_class=1):
    # Load dataset
    dataset = StochasticProcessDataset(file_path)

    # Get data and labels
    data, labels = dataset.data, dataset.labels

    # Define the classes
    num_classes = 3

    # Create a plot
    plt.figure(figsize=(10, 6))

    # Iterate through each class
    for class_id in range(num_classes):
        # Get the indices of the samples belonging to the current class
        class_indices = (labels == class_id).nonzero(as_tuple=True)[0]

        # Select the first two sample paths for this class
        selected_indices = class_indices[:num_samples_per_class]

        # Plot each sample path for the class
        for idx in selected_indices:
            path = data[idx].cpu().numpy()  # Convert to numpy for plotting
            plt.plot(path, label=f'Class {class_id} Sample {idx.item()}')

    plt.title('Sample Paths per Class')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend(loc='best')
    plt.show()