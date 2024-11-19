from scipy.signal import butter, filtfilt
import pathlib
import requests
import zipfile
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from wfdb import rdsamp, rdann
from Dataset.classification.utils import common

# paths
here = pathlib.Path(__file__).resolve().parent
base_loc = here / "data"
dataset_loc = base_loc / "mit-bih-arrhythmia-database-1.0.0"
dataset_zip = base_loc / "mit-bih-arrhythmia-database-1.0.0.zip"
processed_data_loc = here / "processed_data" / "mit_bih"

# Ensure directories exist
processed_data_loc.mkdir(parents=True, exist_ok=True)

# Download and extract the dataset
def download_data():
    url = "https://physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip"
    if not dataset_zip.exists():
        if not base_loc.exists():
            base_loc.mkdir(parents=True)
        print(f"Downloading dataset from {url}...")
        response = requests.get(url, stream=True)
        with open(dataset_zip, "wb") as f:
            f.write(response.content)
        print("Download complete.")
    if not dataset_loc.exists():
        print("Extracting dataset...")
        with zipfile.ZipFile(dataset_zip, "r") as zip_ref:
            zip_ref.extractall(base_loc)
        print("Extraction complete.")


class MITBIHDataset(Dataset):
    def __init__(self, data_dir, segment_length=1800, sampling_rate=360, augment=False):
        self.data_dir = pathlib.Path(data_dir)
        self.segment_length = segment_length
        self.sampling_rate = sampling_rate
        self.augment = augment
        self.symbol_to_class = {
            'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,  # Normal beats
            'A': 1, 'a': 1, 'J': 1,                 # Supra-ventricular beats
            'V': 2, 'E': 2,                         # Ventricular beats
            'F': 3,                                 # Fusion beats
            '/': 4, 'f': 4, 'Q': 4                  # Unclassifiable beats
        }
        self.processed_data_loc = self.data_dir / "processed"
        self.processed_data_loc.mkdir(exist_ok=True)
        self.data = self._process_data()

        if not self.data:
            raise ValueError("No data processed. Verify dataset path and processing logic.")

    def _bandpass_filter(self, signals, lowcut=0.5, highcut=50, fs=360, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, signals, axis=0)

    def _augment_signal(self, signal):
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.05, signal.shape)
            return signal + noise
        return signal

    def _process_data(self):
        records = [f.stem for f in self.data_dir.glob("*.hea")]
        if not records:
            print(f"No records found in {self.data_dir}. Check dataset path.")
            return []

        data = []
        for record in records:
            try:
                signals, _ = rdsamp(str(self.data_dir / record))
                annotations = rdann(str(self.data_dir / record), 'atr')
            except Exception as e:
                print(f"Error reading record {record}: {e}")
                continue

            signals = self._bandpass_filter(signals)
            signals = 2 * (signals - np.min(signals, axis=0)) / \
                      (np.max(signals, axis=0) - np.min(signals, axis=0) + 1e-8) - 1
            signals = signals[:, :2]

            for ann_sample, ann_symbol in zip(annotations.sample, annotations.symbol):
                if ann_symbol in self.symbol_to_class:
                    start = max(0, ann_sample - self.segment_length // 2)
                    end = start + self.segment_length
                    if end > len(signals):
                        segment = np.pad(signals[start:], ((0, end - len(signals)), (0, 0)), mode='constant')
                    else:
                        segment = signals[start:end]

                    if len(segment) == self.segment_length:
                        label = self.symbol_to_class[ann_symbol]
                        if self.augment:
                            segment = self._augment_signal(segment)
                        data.append((torch.tensor(segment, dtype=torch.float32),
                                     torch.tensor(label, dtype=torch.long)))

        if not data:
            print(f"No valid segments found for records in {self.data_dir}.")
            return []

        torch.save(data, self.processed_data_loc / "processed_data.pt")
        print(f"Processed data saved at {self.processed_data_loc / 'processed_data.pt'}")

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# DataLoader function
def get_data_loader(batch_size=32, segment_length=1800, sampling_rate=360):
    download_data()
    dataset = MITBIHDataset(data_dir=dataset_loc,
                            segment_length=segment_length,
                            sampling_rate=sampling_rate)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def tensor_data(batch_size=32, segment_length=1800, sampling_rate=360):
    dataset = get_data_loader(batch_size, segment_length, sampling_rate)
    # Extract X and y from the dataset
    X_list = []
    y_list = []
    for batch in dataset:
        X_batch, y_batch = batch
        X_list.append(X_batch)
        y_list.append(y_batch)

    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)

    return X, y

def get_data(batch_size=32, segment_length=1800, sampling_rate=360):
    X, y = tensor_data(batch_size, segment_length, sampling_rate)

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
