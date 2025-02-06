import model.forecasting.randomODE as randomODE
import model.forecasting.sde as sde
import torch
import torch.optim as optim
from common.forecasting.trainer import _train_loop as train
import Dataset.forecasting.utils.stocks as stocks

class MAPELoss(torch.nn.Module):
    def __init__(self, epsilon=1e-8):
        super(MAPELoss, self).__init__()
        self.epsilon = epsilon  # Avoid division by zero

    def forward(self, y_pred, y_true):
        return torch.mean(torch.abs((y_true - y_pred) / (y_true + self.epsilon)))


def main_classical_training(type='ode', hidden_dim=16, num_layers=1, criterion='L1', with_stdev=False):

    input_dim = 7 + 1
    forecast_horizon = 20

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
    if criterion == 'L1':
        criterion = torch.nn.L1Loss()
    elif criterion == 'MSE':
        criterion = torch.nn.MSELoss()
    elif criterion == 'MAPE':
        criterion = MAPELoss()

    # Here we get the data
    train_loader, test_loader, mean, std = stocks.get_data(num_samples=forecast_horizon)

    # Here we train the model for forecasting
    results = train(model, optimizer, num_epochs,
                    train_loader, test_loader,
                    device, criterion,
                    forecast_horizon=forecast_horizon,
                    mean=mean, std=std,
                    with_stdev=with_stdev)

    return results

if __name__ == '__main__':
    # available criterion: 'L1', 'MSE', 'MAPE'
    # available type: 'ode', 'sde'
    main_classical_training('ode', 64, 1, 'L1')