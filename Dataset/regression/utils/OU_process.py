import numpy as np
import torch
import torchcde
from torch.utils.data import Dataset, DataLoader
import random
import os
import matplotlib.pyplot as plt

def ou_process(T, N, theta, mu, sigma, X0):
    """
    Simulate the Ornstein-Uhlenbeck process.

    Parameters:
    T (float): Total time.
    N (int): Number of time steps.
    theta (float): Rate of mean reversion.
    mu (float): Long-term mean.
    sigma (float): Volatility.
    X0 (float): Initial value.

    Returns:
    np.ndarray: Simulated values of the OU process.
    """
    dt = T / N
    t = np.linspace(0, T, N)
    X = np.zeros(N)
    X[0] = X0

    for i in range(1, N):
        dW = np.random.normal(0, np.sqrt(dt))
        X[i] = X[i-1] + theta * (mu - X[i-1]) * dt + sigma * dW

    # sinusoidal function
    # t = np.linspace(0, T, N)
    # X = np.zeros(N)
    # X[0] = X0
    # for i in range(1, N):
    #     # simple sinusoidal function
    #     X[i] = 0.05 * ( X[i-1] + np.sin(t[i] * 2) + np.random.normal(0, 0.1))
    #
    # # normalize
    # X = (X - X.mean()) / X.std() + 0.5

    return t, X

def generate_data(num_samples, T, N, theta, mu, sigma, X0):
    data_list = []
    for _ in range(num_samples):
        t, X = ou_process(T, N, theta, mu, sigma, X0)
        data_list.append([t, X])

    total_data = torch.Tensor(np.array(data_list))  # [Batch size, Dimension, Length]
    total_data = total_data.permute(0, 2, 1)  # [Batch size, Length, Dimension]

    # normalize the data
    total_data = (total_data - total_data.mean()) / total_data.std() + 1

    max_len = total_data.shape[1]
    times = torch.linspace(0, 1, max_len)
    coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(total_data, times)

    return total_data, coeffs, times

class OU_Dataset(Dataset):
    def __init__(self, data, coeffs):
        self.data = data
        self.coeffs = coeffs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            self.data[idx, ...],
            self.coeffs[idx, ...],
        )

def split_data(data, coeffs, train_ratio=0.8):
    total_size = len(data)
    train_size = int(total_size * train_ratio)

    train_idx = np.random.choice(range(total_size), train_size, replace=False)
    test_idx = np.array([i for i in range(total_size) if i not in train_idx])

    train_data = data[train_idx, ...]
    test_data = data[test_idx, ...]
    train_coeffs = coeffs[train_idx, ...]
    test_coeffs = coeffs[test_idx, ...]

    return train_data, train_coeffs, test_data, test_coeffs

def create_data_loaders(train_data, train_coeffs, test_data, test_coeffs, batch_size=16):
    train_dataset = OU_Dataset(train_data, train_coeffs)
    test_dataset = OU_Dataset(test_data, test_coeffs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def get_data():
    # Setup seed for reproducibility
    def seed_everything(seed):
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    # Parameters
    config = {
        'num_samples': 5000,
        'T': 10.0,
        'N': 10,
        'theta': 0.5,
        'mu': 0.7,
        'sigma': 0.5,
        'X0': 1.5,
        'train_ratio': 0.8,
        'batch_size': 16,
        'seed': 42,
    }

    # Ensure reproducibility
    seed_everything(config['seed'])

    # Generate data
    total_data, coeffs, times = generate_data(config['num_samples'], config['T'], config['N'], config['theta'], config['mu'], config['sigma'], config['X0'])

    # Split data
    train_data, train_coeffs, test_data, test_coeffs = split_data(total_data, coeffs, config['train_ratio'])

    # Create data loaders
    train_loader, test_loader = create_data_loaders(train_data, train_coeffs, test_data, test_coeffs, config['batch_size'])

    # Plot the first sample for verification
    # plt.plot(times.numpy(), total_data[0, :, 1].numpy())
    # plt.xlabel('Time')
    # plt.ylabel('Value')
    # plt.title('OU Process Sample Path')
    # plt.grid('on')
    # plt.show()

    return train_loader, test_loader, config['batch_size']