import model.forecasting.ode_flow as ode_flow
import model.forecasting.sde as sde
import torch
import torch.optim as optim
from common.forecasting.trainer_single_ts import _train_loop as train
import Dataset.regression.utils.single_ts_prediction_stocks as single_ts_prediction


def main_classical_training(type='ode', hidden_dim=16, num_layers=1):
    """
    Generate the OU process data and train the model for a regression task using a neural sde langevin model.
    :return:
    """
    input_dim = 3 + 1
    forecast_horizon = 10

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if type == 'ode':
        model = ode_flow.Generator(input_dim=input_dim,
                                   hidden_dim=hidden_dim,
                                   forecast_horizon=forecast_horizon,
                                   num_layers=num_layers,
                                   vector_field=ode_flow.GeneratorFunc).to(device)
    else:
        model = sde.Generator(input_dim=input_dim,
                              hidden_dim=hidden_dim,
                              forecast_horizon=forecast_horizon,
                              num_layers=num_layers,
                              vector_field=sde.GeneratorFunc).to(device)

    # 300 epochs
    num_epochs = 300
    lr = 1e-3

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    # Here we get the data
    train_loader, test_loader, mean, std = single_ts_prediction.get_data(num_samples=forecast_horizon)

    # Here we train the model for forecasting
    train(model, optimizer, num_epochs,
          train_loader, test_loader,
          device, criterion,
          forecast_horizon=forecast_horizon,
          mean=mean, std=std)

if __name__ == '__main__':
    main_classical_training('ode', 64, 1)