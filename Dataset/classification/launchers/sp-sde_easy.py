import torch
import torch.optim as optim
from Dataset.classification.utils import diffusion_data_easy
import model.classification.ode_flow as ode_flow
import model.classification.sde as sde
from common.classification.trainer_classification_easy import _train_loop

def main_classical_training():

    input_dim = 1
    num_classes = 4
    hidden_dim = 16
    num_layers = 1

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model0 = sde.Generator(input_dim=input_dim,
                           hidden_dim=hidden_dim,
                           num_classes=num_classes,
                           num_layers=num_layers,
                           vector_field=sde.GeneratorFunc).to(device)

    model1 = ode_flow.Generator(input_dim=input_dim,
                                hidden_dim=hidden_dim,
                                num_classes=num_classes,
                                num_layers=num_layers,
                                vector_field=ode_flow.GeneratorFunc).to(device)

    num_epochs = 200
    lr = 1e-3

    optimizer = optim.Adam(model0.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Here we get the data
    train_loader, test_loader, _ = diffusion_data_easy.get_data()

    # Here we train the model
    all_preds, all_trues = _train_loop(model0, optimizer, num_epochs, train_loader, test_loader, device, criterion)

    # Show some stats at the end of the training
    # show_distribution_comparison(all_preds, all_trues)

if __name__ == '__main__':
    main_classical_training()