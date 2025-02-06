import torch
import torch.optim as optim
from Dataset.classification.utils import speech_commands
import model.classification.sde as sde
import model.classification.ode as ode
from common.classification.trainer import _train_loop

def main_classical_training(type='ode', hidden_dim=16, num_layers=1):

    input_dim = 20
    num_classes = 10

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if type == 'ode':
        model = ode.Generator(input_dim=input_dim,
                              hidden_dim=hidden_dim,
                              num_classes=num_classes,
                              num_layers=num_layers,
                              vector_field=ode.GeneratorFunc).to(device)
    else:
        model = sde.Generator(input_dim=input_dim,
                              hidden_dim=hidden_dim,
                              num_classes=num_classes,
                              num_layers=num_layers,
                              vector_field=sde.GeneratorFunc).to(device)


    # 200 epochs
    num_epochs = 200
    lr = 1e-3

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Here we get the data
    data_manager = speech_commands.SpeechCommandsData(train_ratio=0.8, batch_size=256, seed=42)
    train_loader, test_loader = data_manager.get_data()

    # Here we train the model
    results = _train_loop(model, optimizer, num_epochs, train_loader, test_loader, device, criterion)

    return results

if __name__ == '__main__':
    # available types: 'ode', 'sde'
    main_classical_training(type='ode', hidden_dim=128, num_layers=1)