import model.regression.neuralsde_regression as neuralsde_regression
import model.regression.easy_flow_matching_regression as easy_flow_matching_regression
import model.regression.flow_matching_regression as flow_matching_regression
import model.regression.ode_flow as ode_flow
import torch
import torch.optim as optim
from common.regression.trainer_regression import _train_loop, train_flow_matching, _train_loop_asGAN
import Dataset.regression.utils.OU_process as OU_process
from common.regression.utils import show_distribution_comparison

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

    model2 = easy_flow_matching_regression.Generator(input_dim=input_dim,
                                              hidden_dim=hidden_dim,
                                              output_dim=output_dim,
                                              num_layers=num_layers,
                                              vector_field=easy_flow_matching_regression.GeneratorFunc).to(device)

    model3 = flow_matching_regression.Generator(input_dim=input_dim,
                                                    hidden_dim=hidden_dim,
                                                    output_dim=output_dim,
                                                    num_layers=num_layers,
                                                    vector_field=flow_matching_regression.GeneratorFunc).to(device)

    model4 = ode_flow.Generator(input_dim=input_dim,
                                            hidden_dim=hidden_dim,
                                            output_dim=output_dim,
                                            num_layers=num_layers,
                                            vector_field=ode_flow.GeneratorFunc).to(device)


    num_epochs = 200
    lr = 1e-3

    optimizer = optim.Adam(model4.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    # Here we get the data
    train_loader, test_loader, _ = OU_process.get_data()

    # Here we train the model
    # all_preds, all_trues = _train_loop(model3, optimizer, num_epochs, train_loader, test_loader, device, criterion)

    all_preds, all_trues = train_flow_matching(model4, optimizer, num_epochs, train_loader, device, criterion)

    # Show some stats at the end of the training
    # show_distribution_comparison(all_preds, all_trues)

def main_GAN_training():
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


    generator = neuralsde_regression.Generator(input_dim=input_dim,
                                           hidden_dim=hidden_dim,
                                           output_dim=output_dim,
                                           num_layers=num_layers,
                                           vector_field=neuralsde_regression.GeneratorFunc).to(device)

    # for the discriminator you can employ a simpler architecture
    # remark: generator outputs are discriminator inputs, namely
    # generator ------> discriminator
    # Important remark: the discriminator is a neuralCDE, so we need to append
    # times as a separate channel to the inputs
    discriminator = neuralsde_regression.Discriminator(input_dim=1,
                                                        hidden_dim=hidden_dim,
                                                        mlp_size=16,
                                                        num_layers=1)

    num_epochs = 100

    # Here we get the data
    train_loader, test_loader, batch_size = OU_process.get_data()

    # Here we define the loss function for the comparison with classical training in regression task
    criterion = torch.nn.MSELoss()

    # Here we train the model
    all_preds, all_trues = _train_loop_asGAN(generator=generator,
                                             discriminator=discriminator,
                                             num_epochs=num_epochs,
                                             train_loader=train_loader,
                                             test_loader=test_loader,
                                             device=device,
                                             criterion=criterion,
                                             batch_size=batch_size)

    # Show some stats at the end of the training
    # show_distribution_comparison(all_preds, all_trues)

if __name__ == '__main__':
    # choose between "GAN" and "classical" for the training method
    your_choice = "classical"
    if your_choice == "GAN":
        main_GAN_training()
    elif your_choice == "classical":
        main_classical_training()

