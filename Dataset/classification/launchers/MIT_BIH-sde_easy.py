import torch
import torch.optim as optim
from Dataset.classification.utils import MIT_BIH_easy
import model.regression.easy_flow_matching_regression as easy_flow_matching_regression
from common.classification.trainer_classification_easy import _train_loop

def main_classical_training():
    """
    Generate the OU process data and train the model for a regression task using a neural sde langevin model.
    :return:
    """
    input_dim = 2
    num_classes = 8
    hidden_dim = 32
    num_layers = 1

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # todo: try to understand where num_classes comes into play and the relationship with labels from your dataset

    model = easy_flow_matching_regression.Generator(input_dim=input_dim,
                                                     hidden_dim=hidden_dim,
                                                     output_dim=num_classes,
                                                     num_layers=num_layers,
                                                     vector_field=easy_flow_matching_regression.GeneratorFunc).to(device)

    num_epochs = 50
    lr = 1e-3

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Here we get the data
    train_loader, test_loader, _ = MIT_BIH_easy.get_data()

    # Here we train the model
    all_preds, all_trues = _train_loop(model, optimizer, num_epochs, train_loader, test_loader, device, criterion)

    # Show some stats at the end of the training
    # show_distribution_comparison(all_preds, all_trues)

if __name__ == '__main__':
    main_classical_training()