import common.forecasting.trainer_forecasting as common
from random import SystemRandom
import Dataset.forecasting.M4 as M4
import os

from tensorboardX import SummaryWriter

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def main(
        device="cuda",
        max_epochs=100,
        missing_rate=0.0,
        *,
        model_name="neuralgsde",
        hidden_channels=16,
        hidden_hidden_channels=16,
        num_hidden_layers=1,
        dry_run=False,
        method="euler",
        step_mode="valloss",
        lr=0.001,
        weight_decay=0.0,
        loss="mse",
        reg=0.0,
        scale=1.0,
        time_seq=50,
        y_seq=10,
        **kwargs
):

    batch_size = 1024
    PATH = os.path.dirname(os.path.abspath(__file__))

    # Data loader
    times, train_dataloader, val_dataloader, test_dataloader = M4.get_data(
        batch_size, missing_rate, append_time=True, time_seq=time_seq, y_seq=y_seq
    )

    output_time = y_seq
    experiment_id = int(SystemRandom().random() * 100000)

    # Feature and time augmentation.
    input_channels = 1 + 1  # Fixed issue
    folder_name = f'M4_{missing_rate}'
    test_name = f"step_{method}_{model_name}_{experiment_id}"
    result_folder = f"{PATH}/tensorboard_M4"
    writer = SummaryWriter(f"{result_folder}/runs/{folder_name}/{test_name}")

    # Model initialize
    make_model = common.make_model(
        model_name, input_channels, 14, hidden_channels,
        hidden_hidden_channels, num_hidden_layers,
        initial=True, output_time=output_time
    )

    if dry_run:
        name = None
    else:
        name = f'M4_{missing_rate}'

    # Main for forecasting
    return common.main_forecasting(
        name, model_name, times, train_dataloader, val_dataloader, test_dataloader,
        device, make_model, max_epochs, lr, weight_decay, loss, reg, scale, writer, kwargs,
        step_mode=step_mode
    )

# remember to choose the device on which to run the code
if __name__ == "__main__":
    # Fixed parameter values
    main(
        device="cuda",
        hidden_channels=16,
        hidden_hidden_channels=16,
        num_hidden_layers=1,
        lr=0.001,
        method="euler",
        missing_rate=0.0,
        time_seq=50,
        y_seq=10,
        max_epochs=50,
        step_mode="valloss",
        model_name="neuralgsde",
    )
