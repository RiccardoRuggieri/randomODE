import numpy as np
import torch
import torchcde
from torch.utils.data import Dataset, DataLoader
import yfinance as yf
import os
import random
import pandas as pd

def fetch(tickers, start_date, end_date, sp500_ticker="^SPX"):
    """
    Fetches data for given tickers and computes selected features, including SP500 correlation and interest rates.
    The output retains 'Close' as the second column and includes relevant predictors.
    """

    # Fetch S&P 500 data
    print(f"Fetching data for {sp500_ticker}...")
    sp500_data = yf.download(sp500_ticker, start=start_date, end=end_date)
    sp500_data['Daily_Return'] = sp500_data['Close'].pct_change()
    # Calculate 30-day returns (target output)
    sp500_data['30d_Returns'] = np.log(sp500_data['Close']) - np.log(sp500_data['Close'].shift(30))

    # Calculate realized volatility (RVOL): 30-day rolling standard deviation of daily returns
    sp500_data['RVOL'] = sp500_data['Daily_Return'].rolling(window=30).std()

    for ticker in tickers:
        sp500_data[ticker] = yf.download(ticker, start=start_date, end=end_date)['Close'].dropna()

    return sp500_data[['RVOL', '30d_Returns', '^VIX', '^VVIX', '^VXN', '^GVZ', '^OVX']].dropna().values


def create_windows(prices, window_size=21, num_samples=1):
    """
    Split prices into non-overlapping windows.

    Parameters:
    prices (np.ndarray): Stock prices.
    window_size (int): Size of each window.

    Returns:
    np.ndarray: Non-overlapping windows of stock prices.
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
    windows (np.ndarray): Overlapping windows of prices.

    Returns:
    torch.Tensor: Inputs for training (cubic spline interpolated).
    torch.Tensor: Targets for prediction.
    torch.Tensor: Hermite cubic coefficients.
    tuple: Mean and standard deviation of targets (for denormalization).
    """
    inputs = windows[:, :-1 - predict_ahead + 1]
    targets = windows[:, -1 - predict_ahead + 1:]

    print(inputs.shape)

    # Normalize each feature in inputs and targets independently
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
    Split data into train and test sets based on a temporal split,
    ensuring no overlap in forecasting horizons, then create DataLoaders.

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
    Main function to fetch data, create windows, preprocess, and generate DataLoaders.

    Returns:
    DataLoader, DataLoader: Train and test DataLoaders.
    tuple: Mean and standard deviation of targets (for denormalization).
    """
    config = {
        'ticker': ['^VIX', '^VVIX', '^VXN', '^GVZ', '^OVX'],
        'start_date': '2008-01-01',
        'end_date': '2025-01-01',
        'window_size': 60,
        'train_ratio': 0.8,
        'batch_size': 32,
        'seed': 36,
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

    prices = fetch(config['ticker'], config['start_date'], config['end_date'])
    windows = create_windows(prices, config['window_size'], num_samples)
    inputs, targets, coeffs, mean, std = preprocess_windows(windows, predict_ahead=config['num_samples'])
    train_loader, test_loader = create_data_loaders(inputs, targets, coeffs, config['train_ratio'], config['batch_size'])

    return train_loader, test_loader, mean, std










