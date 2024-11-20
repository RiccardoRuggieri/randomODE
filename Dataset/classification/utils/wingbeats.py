import kagglehub
import os
import zipfile
import pathlib
import torch
import torchaudio
from Dataset.classification.utils import common

here = pathlib.Path(__file__).resolve().parent


def download_wingbeats():
    base_base_loc = here / 'data'
    base_loc = base_base_loc / 'Wingbeats'

    if base_loc.exists():
        print("Wingbeats dataset already downloaded.")
        return base_loc

    if not base_base_loc.exists():
        base_base_loc.mkdir()

    print("Downloading Wingbeats dataset using kagglehub...")
    path = kagglehub.dataset_download("potamitis/wingbeats")

    print("Extracting Wingbeats dataset...")
    zip_file = pathlib.Path(path) / "wingbeats.zip"
    with zipfile.ZipFile(zip_file, 'r') as z:
        z.extractall(base_loc)
    print("Extraction completed.")

    return base_loc


def _process_wingbeats_data():
    base_loc = here / 'data' / 'Wingbeats'
    species_dirs = [d for d in base_loc.iterdir() if d.is_dir()]
    num_classes = len(species_dirs)

    # Count the total number of samples
    num_samples = sum(len(list(species_dir.glob("*.wav"))) for species_dir in species_dirs)

    X = torch.empty(num_samples, 16000, 1)
    y = torch.empty(num_samples, dtype=torch.long)

    batch_index = 0
    for y_index, species_dir in enumerate(species_dirs):
        for filename in species_dir.glob("*.wav"):
            audio, _ = torchaudio.load(filename, normalize=True)

            # Normalize audio signal
            audio = audio / (2 ** 10)

            # Discard audio samples shorter or longer than 16000 samples
            if audio.size(1) != 16000:
                continue

            X[batch_index] = audio.t().unsqueeze(-1)
            y[batch_index] = y_index
            batch_index += 1

    assert batch_index == num_samples, f"Expected {num_samples} samples, got {batch_index}."

    X = torchaudio.transforms.MFCC(log_mels=True, n_mfcc=20,
                                   melkwargs=dict(n_fft=200, hop_length=100, n_mels=128))(X.squeeze(-1)).transpose(1, 2).detach()
    print(f"Processed MFCC features shape: {X.shape}")

    times = torch.linspace(0, X.size(1) - 1, X.size(1))
    final_index = torch.tensor(X.size(1) - 1).repeat(X.size(0))

    (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
     test_final_index, _) = common.preprocess_data(times, X, y, final_index, append_times=True, append_intensity=False)

    return times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index, test_final_index


def get_wingbeats_data(batch_size):
    base_base_loc = here / 'processed_data'
    loc = base_base_loc / 'wingbeats_with_mels'

    if os.path.exists(loc):
        tensors = common.load_data(loc)
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
    else:
        download_wingbeats()
        (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
         test_final_index) = _process_wingbeats_data()
        if not os.path.exists(base_base_loc):
            os.mkdir(base_base_loc)
        if not os.path.exists(loc):
            os.mkdir(loc)
        common.save_data(loc, times=times,
                         train_coeffs=train_coeffs, val_coeffs=val_coeffs, test_coeffs=test_coeffs,
                         train_y=train_y, val_y=val_y, test_y=test_y, train_final_index=train_final_index,
                         val_final_index=val_final_index, test_final_index=test_final_index)

    times, train_dataloader, val_dataloader, test_dataloader = common.wrap_data(
        times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y,
        train_final_index, val_final_index, test_final_index, 'cpu', batch_size=batch_size)

    return times, train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":
    try:
        _, train_dataloader, _, _ = get_wingbeats_data(batch_size=32)
        first_batch = next(iter(train_dataloader))
        X, y = first_batch[0][0], first_batch[1][0]
        print(f"First feature shape: {X.shape}")  # MFCC feature dimensions
        print(f"First label: {y}")
    except ValueError as e:
        print(f"Error: {e}")
