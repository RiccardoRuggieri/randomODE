import torch
import torch.optim as optim
from Dataset.classification.utils import MIT_BIH
import model.classification.randomODE as ode_flow
import model.classification.sde as sde
from common.classification.trainer import _train_loop


def main_classical_training(type='ode', hidden_dim=16, num_layers=1):

    input_dim = 2
    num_classes = 8

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if type == 'ode':
        model = ode_flow.Generator(input_dim=input_dim,
                                   hidden_dim=hidden_dim,
                                   num_classes=num_classes,
                                   num_layers=num_layers,
                                   vector_field=ode_flow.GeneratorFunc).to(device)
    elif type == 'sde':
        model = sde.Generator(input_dim=input_dim,
                              hidden_dim=hidden_dim,
                              num_classes=num_classes,
                              num_layers=num_layers,
                              vector_field=sde.GeneratorFunc).to(device)
    else:
        model = ode_flow.Generator(input_dim=input_dim,
                                   hidden_dim=hidden_dim,
                                   num_classes=num_classes,
                                   num_layers=num_layers,
                                   vector_field=ode_flow.GeneratorFunc,
                                   final_nonlinearity=True).to(device)

    # num_epochs = 100
    num_epochs = 30
    lr = 1e-3

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Here we get the data
    train_loader, test_loader, _ = MIT_BIH.get_data()

    # Here we train the model
    results = _train_loop(model, optimizer, num_epochs, train_loader, test_loader, device, criterion)

    # Show some stats at the end of the training
    # show_distribution_comparison(all_preds, all_trues)

    return results

if __name__ == '__main__':
    main_classical_training()