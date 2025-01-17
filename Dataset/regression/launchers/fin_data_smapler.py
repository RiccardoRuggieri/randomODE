import model.classification.ode_flow as ode_flow
import model.classification.sde as sde
import torch
import torch.optim as optim
from common.forecasting.trainer_ode_sampler import _train_loop as train
import Dataset.regression.utils.financial_data_prices as financial_data
import Dataset.regression.utils.OU_process as OU_process
import Dataset.regression.utils.single_ts_prediction_stocks as single_ts_prediction


def main_classical_training():
    """
    Generate the OU process data and train the model for a regression task using a neural sde langevin model.
    :return:
    """
    input_dim = 1 + 1
    hidden_dim = 32
    forecast_horizon = 2
    num_layers = 1

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model1 = ode_flow.Generator(input_dim=input_dim,
                                hidden_dim=hidden_dim,
                                num_layers=num_layers,
                                num_classes=forecast_horizon,
                                vector_field=ode_flow.GeneratorFunc).to(device)

    model2 = sde.Generator(input_dim=input_dim,
                            hidden_dim=hidden_dim,
                            num_layers=num_layers,
                            num_classes=forecast_horizon,
                            vector_field=sde.GeneratorFunc).to(device)

    num_epochs = 400
    lr = 1e-3

    optimizer = optim.Adam(model1.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    # Here we get the data
    train_loader, test_loader, _ = OU_process.get_data()
    # train_loader, test_loader, _ = financial_data.get_gbm_data()

    # Here we train the model for forecasting
    train(model1, optimizer, num_epochs, train_loader, test_loader, device, criterion, forecast_horizon=forecast_horizon)

if __name__ == '__main__':
    main_classical_training()