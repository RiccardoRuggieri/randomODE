import numpy as np
import torch
import torchcde
from torch.utils.data import Dataset, DataLoader
import yfinance as yf
import os
import random

def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch historical stock data from yfinance.

    Parameters:
    ticker (str): Stock ticker symbol.
    start_date (str): Start date (YYYY-MM-DD).
    end_date (str): End date (YYYY-MM-DD).

    Returns:
    np.ndarray: Adjusted close prices.
    """
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data['Close'].dropna().values

def create_windows(prices, window_size=21):
    """
    Split prices into overlapping windows.

    Parameters:
    prices (np.ndarray): Stock prices.
    window_size (int): Size of each window.

    Returns:
    np.ndarray: Windows of stock prices.
    """
    windows = []
    for i in range(len(prices) - window_size + 1):
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
    """
    inputs = windows[:, :-1 - predict_ahead + 1]  # First 20 days
    targets = windows[:, -1 - predict_ahead + 1:]  # 21st day

    # Normalize inputs and targets
    inputs = (inputs - inputs.mean()) / inputs.std()
    targets = (targets - targets.mean()) / targets.std()

    # introduce noise to the inputs, just for fun
    inputs += np.random.normal(0, 1, inputs.shape)

    # Convert to torch tensors
    inputs = torch.Tensor(inputs)  # Shape: [Batch size, 20, 1]
    targets = torch.Tensor(targets)  # Shape: [Batch size, 1, 1]

    # Create time steps for cubic spline interpolation
    times = torch.linspace(0, 1, inputs.shape[1])
    coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(inputs, times)

    return inputs, targets, coeffs

def create_data_loaders(data, targets, coeffs, train_ratio=0.8, batch_size=16):
    """
    Split data into train and test sets, then create DataLoaders.

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
    test_size = len(data) - train_size

    train_data, test_data = torch.utils.data.random_split(
        dataset=list(zip(data, targets, coeffs)),
        lengths=[train_size, test_size],
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def get_data():
    # Configuration
    config = {
        'ticker': 'IBM',
        'start_date': '2003-01-01',
        'end_date': '2023-01-01',
        'window_size': 21,
        'train_ratio': 0.8,
        'batch_size': 32,
        'seed': 42,
        'num_samples': 1,
    }

    # Ensure reproducibility
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

    seed_everything(config['seed'])

    # Fetch stock data
    prices = fetch_stock_data(config['ticker'], config['start_date'], config['end_date'])

    # Create overlapping windows
    windows = create_windows(prices, config['window_size'])

    # Preprocess windows
    inputs, targets, coeffs = preprocess_windows(windows, predict_ahead=config['num_samples'])

    # Create DataLoaders
    train_loader, test_loader = create_data_loaders(
        inputs, targets, coeffs, config['train_ratio'], config['batch_size']
    )

    return train_loader, test_loader

if __name__ == "__main__":
    train_loader, test_loader = get_data()
    print(f"Train data: {len(train_loader.dataset)} samples")

