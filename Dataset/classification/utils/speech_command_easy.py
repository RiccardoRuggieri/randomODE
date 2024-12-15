import os
import tarfile
import urllib.request
from pathlib import Path
import torch
import torchcde
from torch.utils.data import DataLoader, random_split, Dataset
import torchaudio

here = Path(__file__).parent

class SpeechCommandsDataset(Dataset):
    """
    Custom dataset for the Speech Commands classification task.
    """
    def __init__(self, times, data, labels):
        self.times = times
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.times, self.data[idx], self.labels[idx]

class SpeechCommandsData:
    def __init__(self, train_ratio=0.8, batch_size=32, seed=42):
        self.train_ratio = train_ratio
        self.batch_size = batch_size
        self.seed = seed
        self.base_loc = here / 'data' / 'SpeechCommands'
        self.dataset = None
        self.train_loader = None
        self.test_loader = None

    def download(self):
        """
        Downloads and extracts the Speech Commands dataset if not already downloaded.
        """
        base_base_loc = here / 'data'
        loc = self.base_loc / 'speech_commands.tar.gz'

        if os.path.exists(loc):
            print("Dataset already downloaded.")
            return
        if not os.path.exists(base_base_loc):
            os.mkdir(base_base_loc)
        if not os.path.exists(self.base_loc):
            os.mkdir(self.base_loc)

        print("Downloading Speech Commands dataset...")
        urllib.request.urlretrieve('http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz', loc)
        with tarfile.open(loc, 'r') as f:
            print("Extracting files...")
            f.extractall(self.base_loc)
            print("Extraction completed.")

    def _process_data(self):
        """
        Processes the audio data into tensors for training and testing.
        """
        print("Processing data...")
        X = torch.empty(34975, 16000, 1)
        y = torch.empty(34975, dtype=torch.long)

        batch_index = 0
        y_index = 0
        for foldername in ('yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'):
            loc = self.base_loc / foldername
            for filename in os.listdir(loc):
                audio, _ = torchaudio.load(loc / filename, channels_first=False, normalize=True)
                audio = audio / 2**10  # Manual normalization.

                # Discard samples shorter than the expected length.
                if len(audio) != 16000:
                    continue

                X[batch_index] = audio
                _times = torch.linspace(0, 1, 16000)
                y[batch_index] = y_index
                batch_index += 1
            y_index += 1

        assert batch_index == 34975, f"Expected 34975 samples, but got {batch_index}."

        X = torchaudio.transforms.MFCC(log_mels=True, n_mfcc=20,
                                       melkwargs=dict(n_fft=200, hop_length=100, n_mels=128))(
            X.squeeze(-1)).transpose(1, 2).detach()

        times = torch.linspace(0, X.size(1) - 1, X.size(1))

        X = torchcde.hermite_cubic_coefficients_with_backward_differences(X, times)

        print(X.shape)

        print("Data processing completed.")
        # create processed_data folder
        # Ensure the 'processed_data/speech_commands' directory exists
        processed_data_dir = here / 'processed_data' / 'speech_commands'
        os.makedirs(processed_data_dir, exist_ok=True)

        # Save the processed data
        torch.save({'times': times, 'X': X, 'y': y}, processed_data_dir / 'data.pt')

        return times, X, y

    def _split_data(self, dataset):
        """
        Splits the dataset into training and testing subsets.
        """
        total_size = len(dataset)
        train_size = int(total_size * self.train_ratio)
        test_size = total_size - train_size

        train_data, test_data = random_split(dataset, [train_size, test_size])
        return train_data, test_data

    def _create_dataloaders(self, train_data, test_data):
        """
        Creates dataloaders for training and testing datasets.
        """
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader

    def get_data(self):
        """
        Downloads, processes, and splits the data into dataloaders.
        """
        base_base_loc = here / 'processed_data'
        loc = base_base_loc / 'speech_commands'
        if os.path.exists(loc):
            tensors = torch.load(loc / 'data.pt')
            times = tensors['times']
            X = tensors['X']
            y = tensors['y']
        else:
            self.download()
            times, X, y = self._process_data()

        # Initialize dataset
        self.dataset = SpeechCommandsDataset(times, X, y)

        # Split dataset
        train_data, test_data = self._split_data(self.dataset)

        # Create dataloaders
        self.train_loader, self.test_loader = self._create_dataloaders(train_data, test_data)

        return self.train_loader, self.test_loader

