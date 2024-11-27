import common.forecasting.common_sde as common
import torch
from random import SystemRandom
import Dataset.forecasting.mujoco as mujoco
import os

from tensorboardX import SummaryWriter
import argparse  # For command-line argument parsing

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def main(
        manual_seed=None,
        intensity='',
        device="cuda",
        max_epochs=500,
        missing_rate=0.0,
        pos_weight=10,
        *,
        model_name="neuralgsde",
        hidden_channels=16,
        hidden_hidden_channels=16,
        num_hidden_layers=1,
        ode_hidden_hidden_channels=16,
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

    time_augment = intensity
    # Data loader
    times, train_dataloader, val_dataloader, test_dataloader = mujoco.get_data(
        batch_size, missing_rate, time_augment, time_seq, y_seq
    )

    output_time = y_seq
    experiment_id = int(SystemRandom().random() * 100000)

    # Feature and time augmentation.
    input_channels = time_augment + 14
    folder_name = f'MuJoCo_{missing_rate}'
    test_name = f"step_{method}_{model_name}_{experiment_id}"
    result_folder = f"{PATH}/tensorboard_mujoco"
    writer = SummaryWriter(f"{result_folder}/runs/{folder_name}/{test_name}")

    # Model initialize
    make_model = common.make_model(
        model_name, input_channels, 14, hidden_channels,
        hidden_hidden_channels, ode_hidden_hidden_channels, num_hidden_layers,
        use_intensity=intensity, initial=True, output_time=output_time
    )

    def new_make_model():
        model, regularise = make_model()
        model.linear[-1].weight.register_hook(lambda grad: 100 * grad)
        model.linear[-1].bias.register_hook(lambda grad: 100 * grad)
        return model, regularise

    if dry_run:
        name = None
    else:
        name = f'MuJoCo_{missing_rate}'

    # Main for forecasting
    return common.main_forecasting(
        name, model_name, times, train_dataloader, val_dataloader, test_dataloader,
        device, make_model, max_epochs, lr, weight_decay, loss, reg, scale, writer, kwargs,
        pos_weight=torch.tensor(pos_weight), step_mode=step_mode
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MuJoCo forecasting with SDE models.")
    parser.add_argument("--h_channels", type=int, default=16, help="Number of hidden channels")
    parser.add_argument("--hh_channels", type=int, default=16, help="Number of hidden-hidden channels")
    parser.add_argument("--layers", type=int, default=1, help="Number of layers")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--method", type=str, default="euler", help="SDE solver method")
    parser.add_argument("--missing_rate", type=float, default=0.0, help="Missing rate in data")
    parser.add_argument("--time_seq", type=int, default=50, help="Time sequence length")
    parser.add_argument("--y_seq", type=int, default=10, help="Output sequence length")
    parser.add_argument("--intensity", type=str, default='', help="Time intensity augmentation")
    parser.add_argument("--epoch", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--step_mode", type=str, default="valloss", help="Step mode")
    parser.add_argument("--model", type=str, default="neuralgsde", help="Model name")

    args = parser.parse_args()

    main(
        hidden_channels=args.h_channels,
        hidden_hidden_channels=args.hh_channels,
        num_hidden_layers=args.layers,
        lr=args.lr,
        method=args.method,
        missing_rate=args.missing_rate,
        time_seq=args.time_seq,
        y_seq=args.y_seq,
        intensity=args.intensity,
        max_epochs=args.epoch,
        step_mode=args.step_mode,
        model_name=args.model
    )

    
