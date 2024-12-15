import model.regression.neuralsde_regression as neuralsde_regression
import model.regression.ode_flow as ode_flow
import torch
import torch.optim as optim
from common.regression.trainer_regression import _train_loop
import Dataset.regression.utils.OU_process as OU_process


def main_classical_training():
    """
    Generate the OU process data and train the model for a regression task using a neural sde langevin model.
    :return:
    """
    input_dim = 2
    output_dim = 1
    hidden_dim = 32
    num_layers = 1

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")


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




    num_epochs = 50
    lr = 1e-3

    optimizer = optim.Adam(model1.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    # Here we get the data
    train_loader, test_loader, _ = OU_process.get_data()

    # Here we train the model
    all_preds, all_trues = _train_loop(model1, optimizer, num_epochs, train_loader, test_loader, device, criterion)

    # train_flow_matching(model4, optimizer, num_epochs, train_loader, test_loader, device, criterion)

    # Show some stats at the end of the training
    # show_distribution_comparison(all_preds, all_trues)

if __name__ == '__main__':
    main_classical_training()

