import os
import pathlib
import torch
import pandas as pd
import requests
from Dataset.forecasting.utils import common

here = pathlib.Path(__file__).resolve().parent


import os
import requests

def download_m4_csv_files(destination_path):
    base_url = "https://github.com/Mcompetitions/M4-methods/raw/master/Dataset/Train/"
    m4_files = ["Daily-train.csv"]

    os.makedirs(destination_path, exist_ok=True)
    for file_name in m4_files:
        file_url = base_url + file_name
        file_path = os.path.join(destination_path, file_name)

        if not os.path.exists(file_path):
            print(f"Downloading {file_name}...")
            response = requests.get(file_url, stream=True)
            response.raise_for_status()  # Ensure any HTTP errors are caught
            with open(file_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)
            print(f"{file_name} downloaded.")
        else:
            print(f"{file_name} already exists, skipping download.")

    return destination_path


def _process_m4_data(append_time, time_seq, missing_rate, y_seq):
    PATH = pathlib.Path(__file__).resolve().parent
    dataset_path = PATH / "datasets"  # Folder to store datasets
    dataset_path.mkdir(parents=True, exist_ok=True)

    m4_folder = dataset_path / "M4"
    m4_folder.mkdir(parents=True, exist_ok=True)

    # Ensure the dataset is downloaded
    daily_file = "Daily-train.csv"
    file_path = m4_folder / daily_file
    if not file_path.exists():
        download_m4_csv_files(str(m4_folder))  # Assuming it downloads Daily-train.csv

    # Load the dataset
    print(f"Loading {daily_file}...")
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip')
    except Exception as e:
        raise RuntimeError(f"Error reading {daily_file}: {e}")

    # Assume the first column is the ID and the rest are time series data
    ids = df.iloc[:, 0]  # IDs for tracking
    data = df.iloc[:, 1:].values  # Time series data (2D: [samples, time])

    # Convert to tensors
    X_times = [torch.tensor(series[~pd.isnull(series)], dtype=torch.float32) for series in data]

    # Pad sequences to the maximum length with NaN
    maxlen = max(len(ts) for ts in X_times)
    for i in range(len(X_times)):
        padding_length = maxlen - len(X_times[i])
        X_times[i] = torch.cat([X_times[i], torch.full((padding_length,), float("nan"))])
    X_times = torch.stack(X_times)

    # Generate X_reg and y_reg for forecasting
    X_reg, y_reg = [], []
    for i in range(X_times.shape[0]):
        for j in range(X_times.shape[1] - time_seq - y_seq):
            X_reg.append(X_times[i, j:j + time_seq])
            y_reg.append(X_times[i, j + time_seq:j + time_seq + y_seq])

    X_reg = torch.stack(X_reg)
    y_reg = torch.stack(y_reg)

    # Randomly mask missing values
    generator = torch.Generator().manual_seed(56789)
    for Xi in X_reg:
        removed_points = torch.randperm(X_reg.size(1), generator=generator)[
                         :int(X_reg.size(1) * missing_rate)].sort().values
        Xi[removed_points] = float("nan")

    # Prepare indices
    final_indices_reg = torch.full((X_reg.shape[0],), time_seq - 1, dtype=torch.long)
    times = torch.linspace(1, X_reg.size(1), X_reg.size(1))

    # Use common preprocessing utility
    (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
     test_final_index, _) = common.preprocess_data_forecasting(
        times, X_reg, y_reg, final_indices_reg, append_times=append_time
    )

    return (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
            test_final_index)



def get_data(batch_size, missing_rate, append_time, time_seq, y_seq):
    base_base_loc = here / 'processed_data'

    if append_time:
        loc = base_base_loc / ('mujoco' + str(time_seq) + '_' + str(y_seq) + '_' + str(missing_rate) + '_time_aug')
    else:
        loc = base_base_loc / ('mujoco' + str(time_seq) + '_' + str(y_seq) + '_' + str(missing_rate))
    if os.path.exists(loc):
        tensors = common.load_data(loc)
        times = tensors['times']
        train_coeffs = tensors['train_a'], tensors['train_b'], tensors['train_c'], tensors['train_d']
        val_coeffs = tensors['val_a'], tensors['val_b'], tensors['val_c'], tensors['val_d']
        test_coeffs = tensors['test_a'], tensors['test_b'], tensors['test_c'], tensors['test_d']
        train_y = tensors['train_y']
        val_y = tensors['val_y']
        test_y = tensors['test_y']
        train_final_index = tensors['train_final_index']
        val_final_index = tensors['val_final_index']
        test_final_index = tensors['test_final_index']
    else:

        (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
         test_final_index) = _process_m4_data(append_time, time_seq, missing_rate, y_seq)
        if not os.path.exists(base_base_loc):
            os.mkdir(base_base_loc)
        if not os.path.exists(loc):
            os.mkdir(loc)
        common.save_data(loc, times=times,
                         train_a=train_coeffs[0], train_b=train_coeffs[1], train_c=train_coeffs[2],
                         train_d=train_coeffs[3],
                         val_a=val_coeffs[0], val_b=val_coeffs[1], val_c=val_coeffs[2], val_d=val_coeffs[3],

                         test_a=test_coeffs[0], test_b=test_coeffs[1], test_c=test_coeffs[2], test_d=test_coeffs[3],

                         train_y=train_y, val_y=val_y, test_y=test_y, train_final_index=train_final_index,
                         val_final_index=val_final_index, test_final_index=test_final_index)

    times, train_dataloader, val_dataloader, test_dataloader = common.wrap_data(times, train_coeffs, val_coeffs,
                                                                                test_coeffs, train_y, val_y, test_y,
                                                                                train_final_index, val_final_index,
                                                                                test_final_index, 'cpu',
                                                                                batch_size=batch_size)

    return times, train_dataloader, val_dataloader, test_dataloader

if __name__ == "__main__":
    append_time = True
    time_seq = 10  # Length of input time series
    y_seq = 1  # Length of output time series
    missing_rate = 0.1  # 10% missing values

    print("Processing M4 dataset...")
    result = _process_m4_data(append_time, time_seq, missing_rate, y_seq)
    print("M4 dataset processed successfully!")

    # Output shapes for verification
    (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
     test_final_index) = result

    print(f"times: {times.shape}")
    print(f"train_coeffs: {train_coeffs.shape}")
    print(f"val_coeffs: {val_coeffs.shape}")
    print(f"test_coeffs: {test_coeffs.shape}")
    print(f"train_y: {train_y.shape}")
    print(f"val_y: {val_y.shape}")
    print(f"test_y: {test_y.shape}")
    print(f"train_final_index: {train_final_index.shape}")
    print(f"val_final_index: {val_final_index.shape}")
    print(f"test_final_index: {test_final_index.shape}")