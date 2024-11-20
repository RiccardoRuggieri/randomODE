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

# Dataset class
class MITBIHDataset(Dataset):
    def __init__(self, data_dir, segment_length=360, sampling_rate=360):
        self.data_dir = data_dir
        self.segment_length = segment_length
        self.sampling_rate = sampling_rate
        self.max_count = 1000
        self.symbol_to_class = {
            'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,  # Normal beats
            'A': 1, 'a': 1, 'J': 1,                 # Supra-ventricular beats
            'V': 2, 'E': 2,                         # Ventricular beats
            'F': 3,                                 # Fusion beats
            '/': 4, 'f': 4, 'Q': 4                  # Unclassifiable beats
        }
        self.data = self._process_data()
        if not self.data:
            raise ValueError("No data processed. Verify dataset path and processing logic.")

    def _process_data(self):
        records = [f.stem for f in self.data_dir.glob("*.dat")]
        if not records:
            print(f"No records found in {self.data_dir}. Check dataset path.")
            return []

        # Define max count per class (20% of max_count for 5 classes)
        class_max_count = self.max_count // len(self.symbol_to_class)

        # Dictionary to track the count for each class
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

            # Normalize signals
            # signals = (signals - np.mean(signals, axis=0)) / np.std(signals, axis=0)

            # Process each annotation
            for i, annotation_sample in enumerate(annotations.sample):
                # Determine the label for the current annotation
                label = self.symbol_to_class.get(annotations.symbol[i], None)
                if label is None or class_counts[label] >= class_max_count:
                    continue

                start = annotation_sample - self.segment_length // 2
                end = annotation_sample + self.segment_length // 2

                # Ensure the segment stays within bounds
                if start < 0 or end > len(signals):
                    continue

                segment = signals[start:end]
                data.append((torch.tensor(segment, dtype=torch.float32),
                             torch.tensor(label, dtype=torch.long)))
                class_counts[label] += 1

                # Stop processing if we have reached the overall max count
                if sum(class_counts.values()) >= self.max_count:
                    print(f"Reached max count: {self.max_count}")
                    break
            if sum(class_counts.values()) >= self.max_count:
                break

        if not data:
            print(f"No valid segments found for records in {self.data_dir}.")
            return []

        # Save the processed data
        save_path = processed_data_loc / "processed_data.pt"
        torch.save(data, save_path)
        print(f"Processed data saved at {save_path}")

        return data


    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a single data sample at the given index.
        """
        return self.data[idx]

# DataLoader function
def get_data_loader(batch_size=32, segment_length=360, sampling_rate=360):
    download_data()
    dataset = MITBIHDataset(data_dir=dataset_loc,
                            segment_length=segment_length,
                            sampling_rate=sampling_rate)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def tensor_data(batch_size=32, segment_length=360, sampling_rate=360):
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

def get_data(batch_size=32, segment_length=360, sampling_rate=360):
    X, y = tensor_data(batch_size, segment_length, sampling_rate)

    # Generate times based on sampling rate and segment length
    time_step = 1 / sampling_rate
    times = torch.linspace(0, (X.size(1) - 1) * time_step, X.size(1))
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

if __name__ == "__main__":
    try:
        _, train_dataloader, _, _ = get_data(batch_size=32, segment_length=360, sampling_rate=360)
        first_batch = next(iter(train_dataloader))
        X, y = first_batch[0][0], first_batch[1][0]
        print(f"First feature shape: {X.shape}") # 1799 x 12
        print(f"First label: {y}") # a scalar value
    except ValueError as e:
        print(f"Error: {e}")

# # MIT-BIH Classification Task
#
# The MIT-BIH dataset consists of ECG data with various heartbeat types. This task involves classifying ECG segments into 5 different classes based on their annotations.
#
# ## Classes and Their Meanings
#
# 1. **Class 0: Normal Beats**
#    - Includes:
#      - `N` (Normal)
#      - `L` (Left bundle branch block)
#      - `R` (Right bundle branch block)
#      - `e` (Atrial escape)
#      - `j` (Nodal (junctional) escape)
#
# 2. **Class 1: Supraventricular Ectopic Beats**
#    - Includes:
#      - `A` (Atrial premature)
#      - `a` (Aberrated atrial premature)
#      - `J` (Nodal (junctional) premature)
#
# 3. **Class 2: Ventricular Ectopic Beats**
#    - Includes:
#      - `V` (Premature ventricular contraction)
#      - `E` (Ventricular escape)
#
# 4. **Class 3: Fusion Beats**
#    - Includes:
#      - `F` (Fusion of ventricular and normal)
#
# 5. **Class 4: Unknown or Unclassifiable Beats**
#    - Includes:
#      - `/` (Paced)
#      - `f` (Fusion of paced and normal)
#      - `Q` (Unclassifiable)
#
# ## Dataset Details
#
# - **Input (`X`)**:
#   - Shape: `[batch_size, 1800, 2]`
#   - `1800`: Number of time steps in each segment (5 seconds of data at a 360 Hz sampling rate).
#   - `2`: Number of channels (e.g., ECG leads).
#
# - **Target (`y`)**:
#   - Shape: `[batch_size]`
#   - Contains the class label for each segment.
#
# ## Task
#
# The objective is to classify each ECG segment into one of the 5 heartbeat classes based on the input data. This is a **multi-class classification task**.
