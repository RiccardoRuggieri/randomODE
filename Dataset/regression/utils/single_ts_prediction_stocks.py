import numpy as np
import torch
import torchcde
from torch.utils.data import Dataset, DataLoader
import yfinance as yf
import os
import random

def super_fetch(tickers, start_date, end_date, sp500_ticker="^GSPC"):
    """
    Fetches data for given tickers and computes selected features, including SP500 correlation and interest rates.
    The output retains 'Close' as the second column and includes relevant predictors.
    """
    concatenated_series = []

    # Fetch S&P 500 data
    print(f"Fetching data for {sp500_ticker}...")
    sp500_data = yf.download(sp500_ticker, start=start_date, end=end_date)
    sp500_data['Daily_Return'] = sp500_data['Close'].pct_change()

    # Fetch interest rate data (e.g., from a relevant ETF or data source)
    print("Fetching interest rates...")
    # Replace this ticker with the appropriate one for interest rates if available
    interest_rate_data = yf.download("^IRX", start=start_date, end=end_date)
    interest_rate_data['Interest_Rate'] = interest_rate_data['Close']

    # Ensure the interest rate data aligns with the same date range
    interest_rate_data = interest_rate_data[['Interest_Rate']].reindex(sp500_data.index).ffill()

    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        data = yf.download(ticker, start=start_date, end=end_date)

        # Calculate predictors
        data['Daily_Return'] = data['Close'].pct_change()
        data['Volatility'] = data['Daily_Return'].rolling(window=5).std()
        data['Short_MA'] = data['Close'].rolling(window=10).mean()
        data['Long_MA'] = data['Close'].rolling(window=50).mean()

        # Relative Strength Index (RSI)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        data['RSI'] = 100 - (100 / (1 + rs))

        # Exponential smoothing
        data['Smoothed_Close'] = data['Close'].ewm(span=10, adjust=False).mean()

        # Add S&P 500 correlation
        data['SP500_Corr'] = data['Daily_Return'].rolling(window=30).corr(sp500_data['Daily_Return'])

        # Add interest rates
        data = data.join(interest_rate_data, how='left')

        # Drop rows with NaN values from rolling computations
        data.dropna(inplace=True)

        # Arrange 'Close' as the second column
        relevant_data = data[['Daily_Return', 'RSI', 'Volatility', 'Close', 'SP500_Corr', 'Interest_Rate']]
        concatenated_series.append(relevant_data.values)

    return np.vstack(concatenated_series)



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

    # Normalize each feature in inputs and targets independently
    input_mean = inputs.mean(axis=(0, 1), keepdims=True)
    input_std = inputs.std(axis=(0, 1), keepdims=True)
    inputs = (inputs - input_mean) / (input_std + 1e-8)

    target_mean = targets.mean(axis=(0, 1))
    target_std = targets.std(axis=(0, 1))
    targets = (targets - target_mean) / (target_std + 1e-8)

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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
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
        'ticker': ['IBM'],
        'start_date': '1985-01-01',
        'end_date': '2025-01-01',
        'window_size': 42,
        'train_ratio': 0.8,
        'batch_size': 16,
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

    seed_everything(config['seed'])

    prices = super_fetch(config['ticker'], config['start_date'], config['end_date'])
    windows = create_windows(prices, config['window_size'], num_samples)
    inputs, targets, coeffs, mean, std = preprocess_windows(windows, predict_ahead=config['num_samples'])
    train_loader, test_loader = create_data_loaders(inputs, targets, coeffs, config['train_ratio'], config['batch_size'])

    return train_loader, test_loader, mean, std


# import yfinance as yf
# import numpy as np
#
# def super_fetch(tickers, start_date, end_date):
#     concatenated_series = []
#
#     # Fetch market-wide data
#     print("Fetching S&P 500 data...")
#     sp500_data = yf.download("^GSPC", start=start_date, end=end_date)
#     sp500_data['Daily_Return'] = sp500_data['Close'].pct_change()
#
#     print("Fetching NASDAQ 100 data...")
#     nasdaq_data = yf.download("^NDX", start=start_date, end=end_date)
#     nasdaq_data['Daily_Return'] = nasdaq_data['Close'].pct_change()
#
#     print("Fetching VIX (Volatility Index)...")
#     vix_data = yf.download("^VIX", start=start_date, end=end_date)
#     vix_data['Volatility_Index'] = vix_data['Close']
#
#     # Reset VIX index to align with data
#     vix_data = vix_data.reset_index()
#     vix_data.set_index('Date', inplace=True)
#
#     print("Fetching Treasury Yields...")
#     treasury_data = yf.download(["^IRX", "^TNX"], start=start_date, end=end_date)['Close']
#     treasury_data.columns = ["3M_Yield", "10Y_Yield"]
#
#     # Reset and align Treasury index
#     treasury_data = treasury_data.reset_index()
#     treasury_data.set_index('Date', inplace=True)
#
#     for ticker in tickers:
#         print(f"Fetching data for {ticker}...")
#         data = yf.download(ticker, start=start_date, end=end_date)
#
#         # Calculate predictors
#         data['Daily_Return'] = data['Close'].pct_change()
#         data['Volatility'] = data['Daily_Return'].rolling(window=5).std()
#         data['Short_MA'] = data['Close'].rolling(window=10).mean()
#         data['Long_MA'] = data['Close'].rolling(window=50).mean()
#
#         # Relative Strength Index (RSI)
#         delta = data['Close'].diff()
#         gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
#         loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
#         rs = gain / (loss + 1e-10)
#         data['RSI'] = 100 - (100 / (1 + rs))
#
#         # Exponential smoothing
#         data['Smoothed_Close'] = data['Close'].ewm(span=10, adjust=False).mean()
#
#         # Add S&P 500 and NASDAQ correlations
#         data['SP500_Corr'] = data['Daily_Return'].rolling(window=30).corr(sp500_data['Daily_Return'])
#         data['NASDAQ_Corr'] = data['Daily_Return'].rolling(window=30).corr(nasdaq_data['Daily_Return'])
#
#         # Reindex VIX and Treasury data to align with the ticker's data
#         aligned_vix = vix_data.reindex(data.index, method='pad')
#         aligned_treasury = treasury_data.reindex(data.index, method='pad')
#
#         # Add Volatility Index (VIX)
#         data['Volatility_Index'] = aligned_vix['Volatility_Index']
#
#         # Add Treasury Yields
#         data['3M_Yield'] = aligned_treasury['3M_Yield']
#         data['10Y_Yield'] = aligned_treasury['10Y_Yield']
#
#         # Drop rows with NaN values from rolling computations
#         data.dropna(inplace=True)
#
#         # Arrange 'Close' as the second column
#         relevant_data = data[['Daily_Return', 'Close', 'Volatility', 'Short_MA', 'Long_MA', 'RSI',
#                               'Smoothed_Close', 'SP500_Corr', 'NASDAQ_Corr', 'Volatility_Index',
#                               '3M_Yield', '10Y_Yield']]
#         concatenated_series.append(relevant_data.values)
#
#     return np.vstack(concatenated_series)







