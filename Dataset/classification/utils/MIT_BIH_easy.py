import pathlib
import requests
import zipfile

import torchcde
from torch.utils.data import Dataset, DataLoader, random_split
import torch

from wfdb import rdsamp, rdann
from Dataset.classification.utils import common

from sklearn.preprocessing import scale

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

# 1. N - Normal
# 2. V - PVC (Premature ventricular contraction)
# 3. \ - PAB (Paced beat)
# 4. R - RBB (Right bundle branch)
# 5. L - LBB (Left bundle branch)
# 6. A - APB (Atrial premature beat)
# 7. ! - AFW (Ventricular flutter wave)
# 8. E - VEB (Ventricular escape beat)

class MITBIHDataset(Dataset):
    def __init__(self, data_dir, segment_length=360, sampling_rate=360):
        self.data_dir = data_dir
        self.segment_length = segment_length
        self.sampling_rate = sampling_rate
        self.max_count = 20000
        self.symbol_to_class = {
            'N': 0,
            'V': 1,
            '\\': 2,
            'R': 3,
            'L': 4,
            'A': 5,
            '!': 6,
            'E': 7,
        }
        self.times = torch.linspace(0, 1, segment_length)  # Uniformly spaced time values
        self.data = self._process_data()
        if not self.data:
            raise ValueError("No data processed. Verify dataset path and processing logic.")

    def _process_data(self):
        records = [f.stem for f in self.data_dir.glob("*.dat")]
        if not records:
            print(f"No records found in {self.data_dir}. Check dataset path.")
            return []

        class_max_count = self.max_count // len(self.symbol_to_class)
        class_counts = {class_id: 0 for class_id in set(self.symbol_to_class.values())}
        data = []

        for record in records:
            record_path = str(self.data_dir / record)
            try:
                signals, _ = rdsamp(record_path)
                annotations = rdann(record_path, 'atr')
            except Exception as e:
                print(f"Error reading record {record}: {e}")
                continue

            signals = scale(signals).astype("float32")

            for i, annotation_sample in enumerate(annotations.sample):
                label = self.symbol_to_class.get(annotations.symbol[i], None)
                if label is None or class_counts[label] >= class_max_count:
                    continue

                start = annotation_sample - self.segment_length // 2
                end = annotation_sample + self.segment_length // 2

                if start < 0 or end > len(signals):
                    continue

                segment = torch.tensor(signals[start:end], dtype=torch.float32)
                coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(segment.unsqueeze(0), self.times)
                data.append((segment, coeffs.squeeze(0), torch.tensor(label, dtype=torch.long)))
                class_counts[label] += 1

                if sum(class_counts.values()) >= self.max_count:
                    break
            if sum(class_counts.values()) >= self.max_count:
                break

        if not data:
            print(f"No valid segments found for records in {self.data_dir}.")
            return []

        save_path = processed_data_loc / "processed_data.pt"
        torch.save(data, save_path)
        print(f"Processed data saved at {save_path}")
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        segment, coeffs, label = self.data[idx]
        return segment, coeffs, label

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
    config = {
        'batch_size': 32,
        'segment_length': 360,
        'sampling_rate': 360,
        'train_ratio': 0.8,
        'seed': 42,
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

    seed_everything(config['seed'])

    download_data()
    dataset = MITBIHDataset(data_dir=dataset_loc,
                            segment_length=config['segment_length'],
                            sampling_rate=config['sampling_rate'])

    train_data, test_data = split_data(dataset.data, config['train_ratio'])

    train_loader, test_loader = create_dataloaders(train_data, test_data, config['batch_size'])

    return train_loader, test_loader, config['batch_size']
