import model.regression.neuralsde_regression as neuralsde_regression
import model.forecasting.ode_flow as ode_flow_forecast
import model.regression.ode_flow as ode_flow
import model.regression.ode as ode
import torch
import torch.optim as optim
from common.regression.trainer_regression_forecasting import _train_loop as train
import Dataset.regression.utils.financial_data_prices as financial_data
import Dataset.regression.utils.OU_process as OU_process


def main_classical_training():
    """
    Generate the OU process data and train the model for a regression task using a neural sde langevin model.
    :return:
    """
    input_dim = 1 + 1
    output_dim = 1
    hidden_dim = 32
    num_layers = 1

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model0 = ode.Generator(input_dim=input_dim,
                            hidden_dim=hidden_dim,
                            output_dim=output_dim,
                            num_layers=num_layers,
                            vector_field=ode.GeneratorFunc).to(device)


    model1 = neuralsde_regression.Generator(input_dim=input_dim,
                                            hidden_dim=hidden_dim,
                                            output_dim=output_dim,
                                            num_layers=num_layers,
                                            vector_field=neuralsde_regression.GeneratorFunc).to(device)


    model2 = ode_flow.Generator(input_dim=input_dim,
                                hidden_dim=hidden_dim,
                                output_dim=output_dim,
                                num_layers=num_layers,
                                vector_field=ode_flow.GeneratorFunc).to(device)

    model3 = ode_flow_forecast.ControlledOUGenerator(input_dim=input_dim,
                                hidden_dim=hidden_dim,
                                num_layers=num_layers,
                                output_dim=output_dim).to(device)


    num_epochs = 100
    lr = 1e-3

    optimizer = optim.Adam(model2.parameters(), lr=lr)
    # absolute error
    criterion = torch.nn.MSELoss()

    # Here we get the data
    train_loader, test_loader, _ = OU_process.get_data()
    # train_loader, test_loader, _ = financial_data.get_gbm_data()

    # Here we train the model for forecasting
    train(model2, optimizer, num_epochs, train_loader, test_loader, device, criterion, forecast_horizon=0.2)

if __name__ == '__main__':
    main_classical_training()