import pathlib

import torch
import torchcde
from torch.utils.data import Dataset, DataLoader, random_split
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
        self.data = dataset["data"]
        self.coeffs = dataset["coeffs"]
        self.labels = dataset["labels"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Unpack the data correctly
        path = self.data[idx]
        coeffs = self.coeffs[idx]
        label = self.labels[idx]
        return path, coeffs, label


def generate_stochastic_process_dataset(output_file, num_path_to_generate=10002, timesteps=100, dt=0.1, seed=42):
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
    data, list_coeffs, labels = [], [], []

    process_generators = [generate_ou_process,
                          generate_cir_process,
                          generate_jump_diffusion]

    for i, generator in enumerate(process_generators):
        paths = generator(size=(num_paths_per_class, timesteps))
        times = torch.linspace(0, 1, timesteps)
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(paths.unsqueeze(-1), times)
        list_coeffs.append(coeffs)
        data.append(paths)
        labels.extend([i] * num_paths_per_class)

    data = torch.cat(data, dim=0)  # Concatenate all paths
    coeffs = torch.cat(list_coeffs, dim=0)  # Concatenate all coeffs
    labels = torch.tensor(labels)

    indices = torch.randperm(num_path_to_generate)
    data = data[indices]
    coeffs = coeffs[indices]
    labels = labels[indices]

    dataset = {"data": data, "coeffs": coeffs, "labels": labels}
    torch.save(dataset, output_file)


def get_dataloaders(batch_size=32, train_ratio=0.8, file_path=None, seed=42):
    """
    Generates or loads a dataset, splits it into train and test, and returns dataloaders.
    :param batch_size: Batch size for the dataloaders.
    :param train_ratio: Ratio of training data.
    :param file_path: Path to the dataset file in .pt format. If None, generates a new dataset.
    :param seed: Seed for reproducibility.
    :return: train_loader, test_loader
    """
    if file_path is None:
        file_path = processed_data_loc / "stochastic_processes.pt"
        generate_stochastic_process_dataset(file_path, seed=seed)
        plot_sample_paths(file_path)

    # Initialize dataset and split
    dataset = StochasticProcessDataset(file_path)
    train_data, test_data = split_data(dataset, train_ratio)

    # Create dataloaders
    train_loader, test_loader = create_dataloaders(train_data, test_data, batch_size=batch_size)
    return train_loader, test_loader


def split_data(dataset, train_ratio=0.8):
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    test_size = total_size - train_size

    train_data, test_data = random_split(dataset, [train_size, test_size])
    return train_data, test_data


def create_dataloaders(train_data, test_data, batch_size=16):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def get_data():
    """
    Sets up the dataset and dataloaders for training and testing.
    :return: train_loader, test_loader, batch_size
    """
    config = {
        'seed': 42,
        'train_ratio': 0.8,
        'batch_size': 32,
    }

    def seed_everything(seed):
        import random
        import os
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    # Set the random seed
    seed_everything(config['seed'])

    # Get dataloaders
    train_loader, test_loader = get_dataloaders(
        batch_size=config['batch_size'],
        train_ratio=config['train_ratio'],
        seed=config['seed']
    )

    return train_loader, test_loader, config['batch_size']


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