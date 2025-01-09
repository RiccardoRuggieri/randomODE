import numpy as np
import torch
import torchcde
from torch.utils.data import Dataset, DataLoader
import random
import os
import yfinance as yf
import matplotlib.pyplot as plt

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


def calibrate_gbm_parameters(prices, dt):
    """
    Calibrate GBM parameters from historical stock prices.

    Parameters:
    prices (np.ndarray): Historical prices.
    dt (float): Time step, default assumes daily data with 252 trading days per year.

    Returns:
    float, float: Calibrated drift (mu) and volatility (sigma).
    """
    log_returns = np.log(prices[1:] / prices[:-1])
    mu = np.mean(log_returns) / dt
    sigma = np.std(log_returns) / np.sqrt(dt)
    return mu, sigma

def gbm_process(T, N, mu, sigma, S0):
    """
    Simulate the Geometric Brownian Motion process.

    Parameters:
    T (float): Total time.
    N (int): Number of time steps.
    mu (float): Drift.
    sigma (float): Volatility.
    S0 (float): Initial value.

    Returns:
    np.ndarray: Simulated values of the GBM process.
    """
    dt = T / N
    t = np.linspace(0, T, N)
    S = np.zeros(N)
    S[0] = S0

    for i in range(1, N):
        dW = np.random.normal(0, np.sqrt(dt))
        S[i] = S[i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)

    return t, S

def generate_gbm_data(num_samples, T, N, mu, sigma, S0, prices_forecast):
    data_list = []
    target_list = []
    for _ in range(num_samples):
        t, S = gbm_process(T, N, mu, sigma, S0)
        data_list.append([t, S])
        target_list.append([t, prices_forecast])


    total_data = torch.Tensor(np.array(data_list))  # [Batch size, Dimension, Length]
    total_targets = torch.Tensor(np.array(target_list))  # [Batch size, Dimension, Length]
    total_data = total_data.permute(0, 2, 1)  # [Batch size, Length, Dimension]
    total_targets = total_targets.permute(0, 2, 1)  # [Batch size, Length, Dimension]

    # normalize the data
    total_data = (total_data - total_data.mean()) / total_data.std()
    total_targets = (total_targets - total_targets.mean()) / total_targets.std()

    max_len = total_data.shape[1]
    times = torch.linspace(0, 1, max_len)
    coeffs_data = torchcde.hermite_cubic_coefficients_with_backward_differences(total_data, times)
    coeffs_targets = torchcde.hermite_cubic_coefficients_with_backward_differences(total_targets, times)

    return total_data, total_targets, coeffs_data, coeffs_targets, times

class GBM_Dataset(Dataset):
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

def split_data(data, target, coeffs_data, coeffs_target, train_ratio=0.8):
    train_data = data
    train_coeffs = coeffs_data
    test_data = target
    test_coeffs = coeffs_target

    return train_data, train_coeffs, test_data, test_coeffs

def create_data_loaders(train_data, train_coeffs, test_data, test_coeffs, batch_size=16):
    train_dataset = GBM_Dataset(train_data, train_coeffs)
    test_dataset = GBM_Dataset(test_data, test_coeffs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def get_gbm_data():
    # Parameters
    config = {
        'ticker': 'AAPL',
        'start_date_train': '2022-06-01',
        'end_date_train': '2022-12-31',
        # Here you can introduce a gap between training and forecasting data
        # Keep in mind that these two data should have same dimension
        'start_date_forecast': '2022-07-06',
        'end_date_forecast': '2023-02-06',
        'num_samples': 1000,
        'T': 1.0,
        'N': 148,
        'train_ratio': 0.8,
        'batch_size': 16,
        'seed': 32,
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
        torch.backends.cudnn.benchmark = True

    seed_everything(config['seed'])

    # Fetch stock data and calibrate GBM parameters
    prices_train = fetch_stock_data(config['ticker'], config['start_date_train'], config['end_date_train'])
    prices_forecast = fetch_stock_data(config['ticker'], config['start_date_forecast'], config['end_date_forecast'])

    print(prices_train.shape, prices_forecast.shape)

    # Here we pass only training data
    S0 = prices_train[-1]
    mu, sigma = calibrate_gbm_parameters(prices_train, 1/config['N'])

    # Simulate a GBM path
    T = 1.0
    N = len(prices_train)
    t_gbm, S_gbm = gbm_process(T, N, mu, sigma, S0)

    # Plot fetched prices and GBM
    plot_price_and_gbm(prices_train, t_gbm, S_gbm)

    # Generate GBM sample paths (passing forecasting data) --> + 1 month and 6 days
    total_data, total_targets, coeffs_data, coeffs_targets, times = (
        generate_gbm_data(config['num_samples'], config['T'], config['N'], mu, sigma, S0, prices_forecast.squeeze(-1)))

    # Split data
    train_data, train_coeffs, test_data, test_coeffs = split_data(total_data, total_targets, coeffs_data, coeffs_targets, config['train_ratio'])

    # Create data loaders
    train_loader, test_loader = create_data_loaders(train_data, train_coeffs, test_data, test_coeffs, config['batch_size'])

    return train_loader, test_loader, config['batch_size']


def plot_price_and_gbm(prices, t_gbm, S_gbm):
    """
    Plot fetched stock prices and a simulated GBM for comparison.

    Parameters:
    prices (np.ndarray): Fetched stock prices.
    t_gbm (np.ndarray): Time steps for GBM simulation.
    S_gbm (np.ndarray): Simulated GBM prices.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(prices, label="Fetched Stock Prices", linestyle="-", marker="o")
    plt.plot(t_gbm * (len(prices) - 1), S_gbm, label="Calibrated GBM Simulation", linestyle="--")
    plt.title("Comparison of Fetched Stock Prices and Calibrated GBM Simulation")
    plt.xlabel("Time Steps")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.show()







