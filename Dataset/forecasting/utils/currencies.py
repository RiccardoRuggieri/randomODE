import numpy as np
import torch
import torchcde
from torch.utils.data import Dataset, DataLoader
import yfinance as yf
import os
import random
import pandas as pd

def fetch(tickers, start_date, end_date, base_ticker="USD"):
    """
    Fetches data for given tickers and computes log returns for a simplified forecasting task.
    The output retains relevant predictors.

    Parameters:
    tickers (list): List of tickers to fetch.
    start_date (str): Start date for data fetching.
    end_date (str): End date for data fetching.
    base_ticker (str): Base ticker for normalization.

    Returns:
    np.ndarray: Log returns of selected tickers.
    """
    # Initialize the dataset
    data = pd.DataFrame()

    # Fetch data for each ticker
    for ticker in tickers:
        ticker_data = yf.download(ticker, start=start_date, end=end_date)['Close'].dropna()
        data[ticker] = ticker_data

    # Compute log returns: log(current_price / previous_price)
    log_returns = np.log(data / data.shift(10)).dropna()

    # Select the tickers of interest
    selected_data = log_returns[['^OVX', 'EURUSD=X', '^VXN', '^VIX']].dropna()

    return selected_data.values


def create_windows(prices, window_size=21, num_samples=1):
    """
    Split prices into non-overlapping windows.

    Parameters:
    prices (np.ndarray): Exchange rate data.
    window_size (int): Size of each window.

    Returns:
    np.ndarray: Non-overlapping windows of exchange rate data.
    """
    windows = []
    for i in range(0, len(prices) - window_size + 1, num_samples):
        windows.append(prices[i:i + window_size])
    return np.array(windows)

class TimeSeriesDataset(Dataset):
    def __init__(self, data, targets, coeffs):
        self.data = data
        self.targets = targets
        self.coeffs = coeffs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx, ...], self.targets[idx, ...], self.coeffs[idx, ...]

def preprocess_windows(windows, predict_ahead=1):
    """
    Preprocess overlapping windows by separating input (20 days) and target (21st day),
    normalizing, and interpolating data with cubic splines.

    Parameters:
    windows (np.ndarray): Overlapping windows of exchange rate data.

    Returns:
    torch.Tensor: Inputs for training (cubic spline interpolated).
    torch.Tensor: Targets for prediction.
    torch.Tensor: Hermite cubic coefficients.
    tuple: Mean and standard deviation of targets (for denormalization).
    """
    inputs = windows[:, :-1 - predict_ahead + 1]
    targets = windows[:, -1 - predict_ahead + 1:]

    print(inputs.shape)

    # Normalize inputs and targets independently
    input_mean = inputs.mean(axis=(0, 1), keepdims=True)
    input_std = inputs.std(axis=(0, 1), keepdims=True)
    inputs = (inputs - input_mean) / input_std

    target_mean = targets.mean(axis=(0, 1))
    target_std = targets.std(axis=(0, 1))
    targets = (targets - target_mean) / target_std

    # Convert to torch tensors
    inputs = torch.Tensor(inputs)  # Shape: [Batch size, 20, num_features]
    targets = torch.Tensor(targets)  # Shape: [Batch size, 1, num_features]

    # Append time dimension
    times = torch.linspace(0, 1, inputs.shape[1])
    inputs = torch.cat((inputs, times.unsqueeze(0).repeat(inputs.shape[0], 1).unsqueeze(-1)), dim=-1)

    # Create Hermite cubic coefficients
    coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(inputs, times)

    return inputs, targets, coeffs, target_mean, target_std

def create_data_loaders(data, targets, coeffs, train_ratio=0.8, batch_size=16):
    """
    Split data into train and test sets and create DataLoaders.

    Parameters:
    data (torch.Tensor): Input data.
    targets (torch.Tensor): Target data.
    coeffs (torch.Tensor): Hermite cubic coefficients for input data.
    train_ratio (float): Proportion of data to use for training.
    batch_size (int): Batch size for DataLoader.

    Returns:
    DataLoader, DataLoader: Train and test DataLoaders.
    """
    train_size = int(len(data) * train_ratio)

    train_data = data[:train_size]
    test_data = data[train_size:]
    train_targets = targets[:train_size]
    test_targets = targets[train_size:]
    train_coeffs = coeffs[:train_size]
    test_coeffs = coeffs[train_size:]

    train_dataset = TimeSeriesDataset(train_data, train_targets, train_coeffs)
    test_dataset = TimeSeriesDataset(test_data, test_targets, test_coeffs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def get_data(num_samples=1):
    """
    Fetch data, create windows, preprocess, and generate DataLoaders.

    Returns:
    DataLoader, DataLoader: Train and test DataLoaders.
    tuple: Mean and standard deviation of targets (for denormalization).
    """
    config = {
        'tickers': ['EURUSD=X', 'GBPUSD=X', 'JPYUSD=X', 'EURCHF=X', 'EURGBP=X', 'EURJPY=X', 'JPY=X',
                    '^VIX', '^VVIX', '^VXN', '^GVZ', '^OVX', 'GC=F'],
        'start_date': '2005-01-01',
        'end_date': '2025-01-01',
        'window_size': 40,
        'train_ratio': 0.8,
        'batch_size': 16,
        'seed': 42,
        'num_samples': num_samples,
    }

    def seed_everything(seed):
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # seed_everything(config['seed'])

    prices = fetch(config['tickers'], config['start_date'], config['end_date'])
    windows = create_windows(prices, config['window_size'], config['num_samples'])
    inputs, targets, coeffs, mean, std = preprocess_windows(windows, predict_ahead=config['num_samples'])
    train_loader, test_loader = create_data_loaders(inputs, targets, coeffs, config['train_ratio'], config['batch_size'])

    return train_loader, test_loader, mean, std