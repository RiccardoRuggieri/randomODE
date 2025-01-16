import numpy as np
import torch
import torchcde
from torch.utils.data import Dataset, DataLoader
import yfinance as yf
import os
import random

def fetch_volatility(ticker, start_date, end_date):
    """
    Fetch raw volatility data from yfinance.

    Parameters:
    ticker (str): Stock ticker symbol.
    start_date (str): Start date (YYYY-MM-DD).
    end_date (str): End date (YYYY-MM-DD).

    Returns:
    np.ndarray: Volatility time series.
    """
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data = stock_data['Close'].dropna()

    # Calculate daily returns
    daily_returns = stock_data.pct_change().dropna()

    # Calculate rolling volatility (standard deviation of returns)
    volatility = daily_returns.rolling(window=5).std().dropna()

    return volatility.values

def fetch_price(ticker, start_date, end_date):
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

def super_fetch(ticker, start_date, end_date):

    data = yf.download(ticker, start=start_date, end=end_date)
    data['Daily_Return'] = data['Close'].pct_change()
    data['High_Low_Spread'] = (data['High'] - data['Low']) / data['Low']

    # Step 2: Add rolling features
    data['SMA_50'] = data['Close'].rolling(window=1).mean()
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['Volatility_5d'] = data['Daily_Return'].rolling(window=21).std()

    # Step 3: Compute RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # Step 4: Add MACD and Signal Line
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # Step 5: Compute Bollinger Bands
    data['Rolling_Mean'] = data['Close'].rolling(window=20).mean()
    data['Rolling_Std'] = data['Close'].rolling(window=20).std()
    data['Bollinger_Upper'] = data['Rolling_Mean'] + 2 * data['Rolling_Std']
    data['Bollinger_Lower'] = data['Rolling_Mean'] - 2 * data['Rolling_Std']

    # Step 6: Normalize volume
    data['Volume_ZScore'] = (data['Volume'] - data['Volume'].mean()) / data['Volume'].std()

    # Step 7: Drop NA values from rolling computations
    data.dropna(inplace=True)

    return data[['Volatility_5d', 'Close', 'Volume']].values

def fetch_bivariate_series(ticker, start_date, end_date):
    """
    Fetch bivariate time series of adjusted close prices and volatility from yfinance.

    Parameters:
    ticker (str): Stock ticker symbol.
    start_date (str): Start date (YYYY-MM-DD).
    end_date (str): End date (YYYY-MM-DD).

    Returns:
    pd.DataFrame: DataFrame with 'Close' and 'Volatility' columns.
    """
    # Fetch historical stock data
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data = stock_data[['Close', 'Volume']].dropna()

    # Calculate daily returns
    daily_returns = stock_data['Close'].pct_change().dropna()

    # Calculate rolling volatility (standard deviation of returns)
    stock_data['Volatility'] = daily_returns.rolling(window=5).std()

    # Drop rows with NaN values (from rolling calculation)
    stock_data = stock_data[['Volatility', 'Close', 'Volume']].dropna()

    print(stock_data)

    return stock_data


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
    inputs = windows[:, :-1 -predict_ahead + 1]  # First 20 days
    targets = windows[:, -1 -predict_ahead + 1:]  # 21st day

    # save mean and std for later
    mean = torch.zeros(targets.shape[2])
    std = torch.zeros(targets.shape[2])

    print(inputs.shape)
    # Normalize inputs and targets separately
    for i in range(inputs.shape[2]):
        inputs[:, :, i] = (inputs[:, :, i] - inputs[:, :, i].mean()) / inputs[:, :, i].std()

    for i in range(targets.shape[2]):
        mean[i] = targets[:, :, i].mean()
        std[i] = targets[:, :, i].std()
        targets[:, :, i] = (targets[:, :, i] - targets[:, :, i].mean()) / targets[:, :, i].std()

    # introduce noise to the inputs, just for fun
    # inputs += np.random.normal(0, 1, inputs.shape)

    # Convert to torch tensors
    inputs = torch.Tensor(inputs)  # Shape: [Batch size, 20, 1]
    targets = torch.Tensor(targets)  # Shape: [Batch size, 1, 1]

    # append times
    inputs = torch.cat((inputs, torch.linspace(0, 1, inputs.shape[1]).unsqueeze(0).repeat(inputs.shape[0], 1).unsqueeze(-1)), dim=-1)

    # Create time steps for cubic spline interpolation
    times = torch.linspace(0, 1, inputs.shape[1])
    coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(inputs, times)

    return inputs, targets, coeffs, mean, std

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

def get_data(num_samples=1):
    # Configuration
    config = {
        'ticker': 'IBM',
        'start_date': '2003-01-01',
        'end_date': '2023-01-01',
        'window_size': 50,
        'train_ratio': 0.8,
        'batch_size': 64,
        'seed': 36,
        'num_samples': num_samples,
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

    # prices prediction
    # prices = fetch_volatility(config['ticker'], config['start_date'], config['end_date'])
    # prices = fetch_price(config['ticker'], config['start_date'], config['end_date'])
    # prices = fetch_bivariate_series(config['ticker'], config['start_date'], config['end_date'])

    # volatility prediction
    prices = super_fetch(config['ticker'], config['start_date'], config['end_date'])

    # Create overlapping windows
    windows = create_windows(prices, config['window_size'])

    # Preprocess windows
    inputs, targets, coeffs, mean, std = preprocess_windows(windows, predict_ahead=config['num_samples'])

    # Create DataLoaders
    train_loader, test_loader = create_data_loaders(
        inputs, targets, coeffs, config['train_ratio'], config['batch_size']
    )

    return train_loader, test_loader, mean, std




