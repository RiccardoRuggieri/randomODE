import model.forecasting.randomODE as randomODE
import model.forecasting.sde as sde
import torch
import torch.optim as optim
from common.forecasting.trainer import _train_loop as train
import Dataset.forecasting.utils.currencies as currencies


def main_classical_training(type='ode', hidden_dim=16, num_layers=1):

    input_dim = 4 + 1
    forecast_horizon = 10

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if type == 'ode':
        model = randomODE.Generator(input_dim=input_dim,
                                    hidden_dim=hidden_dim,
                                    forecast_horizon=forecast_horizon,
                                    num_layers=num_layers,
                                    vector_field=randomODE.GeneratorFunc).to(device)
    else:
        model = sde.Generator(input_dim=input_dim,
                              hidden_dim=hidden_dim,
                              forecast_horizon=forecast_horizon,
                              num_layers=num_layers,
                              vector_field=sde.GeneratorFunc).to(device)

    # 300 epochs
    num_epochs = 100
    lr = 1e-3

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.L1Loss()

    # Here we get the data
    train_loader, test_loader, mean, std = currencies.get_data(num_samples=forecast_horizon)

    # Here we train the model for forecasting
    results = train(model, optimizer, num_epochs,
                    train_loader, test_loader,
                    device, criterion,
                    forecast_horizon=forecast_horizon,
                    mean=mean, std=std)

    return results

if __name__ == '__main__':
    main_classical_training('ode', 128, 2)