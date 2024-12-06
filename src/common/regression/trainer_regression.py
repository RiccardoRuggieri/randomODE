import torch
import matplotlib.pyplot as plt
import torch.optim.swa_utils as swa_utils
import torchcde


def _train_loop(model, optimizer, num_epochs, train_loader, test_loader, device, criterion):
    global all_preds, all_trues
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            coeffs = batch[1].to(device)
            times = torch.linspace(0, 1, batch[0].shape[1]).to(device)

            optimizer.zero_grad()
            true = batch[0][:, :, 1].to(device)
            pred = model(coeffs, times).squeeze(-1)
            loss = criterion(pred, true)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch}, Loss: {avg_loss}')

            ##
            model.eval()
            total_loss = 0
            all_preds = []
            all_trues = []
            with torch.no_grad():
                for batch in test_loader:
                    coeffs = batch[1].to(device)
                    times = torch.linspace(0, 1, batch[0].shape[1]).to(device)

                    true = batch[0][:, :, 1].to(device)
                    pred = model(coeffs, times).squeeze(-1)
                    loss = criterion(pred, true)
                    total_loss += loss.item()

                    all_preds.append(pred.cpu())
                    all_trues.append(true.cpu())

            avg_loss = total_loss / len(test_loader)
            print(f'Test Loss: {avg_loss}')

            all_preds = torch.cat(all_preds, dim=0)
            all_trues = torch.cat(all_trues, dim=0)

            ## plotting
            # num_samples = 5
            #
            # plt.figure(figsize=(8, 4))
            # for i in range(num_samples):
            #     plt.plot(all_trues[i].numpy(), color='r')
            #     plt.plot(all_preds[i].numpy(), color='b')
            # plt.xlabel('Time')
            # plt.ylabel('Value')
            # plt.ylim(-0.75,1.25)
            # plt.title('Model Predictions vs True Values')
            # plt.show()

    return all_preds, all_trues

def _train_loop_asGAN(generator, discriminator, num_epochs, train_loader, test_loader, device, criterion, batch_size):
    global all_preds, all_trues

    # we define a particular train loader for the discriminator
    infinite_train_dataloader = (elem for it in iter(lambda: train_loader, None) for elem in it)

    # Weight averaging improve GAN training.
    averaged_generator = swa_utils.AveragedModel(generator)
    averaged_discriminator = swa_utils.AveragedModel(discriminator)

    ########## Initialization of parameters ##########

    # hyperparameters
    init_mult1 = 3
    init_mult2 = 0.5

    with torch.no_grad():
        for param in generator.initial.parameters():
            param *= init_mult1
        for param in generator.func.parameters():
            param *= init_mult2

    # the choice of the optimizer is restrained by the fact that GAN architecture
    # has its own specificities

    # some other hyperparameters
    generator_lr = 2e-4
    discriminator_lr = 1e-3
    weight_decay = 0.01

    generator_optimiser = torch.optim.Adadelta(generator.parameters(), lr=generator_lr, weight_decay=weight_decay)
    discriminator_optimiser = torch.optim.Adadelta(discriminator.parameters(), lr=discriminator_lr, weight_decay=weight_decay)

    # Training involves both generator and discriminator
    for epoch in range(1, num_epochs + 1):

        generator.train()
        discriminator.train()

        total_loss = 0

        for batch in train_loader:
            coeffs = batch[1].to(device)
            # times are directly available as a separate channel here!!
            times = torch.linspace(0, 1, batch[0].shape[1]).to(device)

            generated_samples = generator(coeffs, times)

            # now a little bit of data formatting to get the discriminator work
            times = times.unsqueeze(0).unsqueeze(-1).expand(batch_size, times.size(0), 1)
            generated_samples = torchcde.linear_interpolation_coeffs(torch.cat([times, generated_samples], dim=2))
            print(generated_samples.shape)

            generated_score = discriminator(generated_samples)

            # todo: ############### resolve this issue ################
            real_samples = next(infinite_train_dataloader)
            # again some data formatting
            print(real_samples.shape)
            real_samples = torchcde.linear_interpolation_coeffs(torch.cat([times, real_samples], dim=2))

            real_score = discriminator(real_samples)

            loss = generated_score - real_score
            loss.backward()

            for param in generator.parameters():
                param.grad *= -1

            generator_optimiser.step()
            discriminator_optimiser.step()

            generator_optimiser.zero_grad()
            discriminator_optimiser.zero_grad()

            # constraining the Lipschitz constant of the discriminator with clipping
            with torch.no_grad():
                for module in discriminator.modules():
                    if isinstance(module, torch.nn.Linear):
                        lim = 1 / module.out_features
                        module.weight.clamp_(-lim, lim)

            # stochastic weight averaging
            # hyperparameter
            swa_step_start = 5000

            if epoch > swa_step_start:
                averaged_generator.update_parameters(generator)
                averaged_discriminator.update_parameters(discriminator)


        if epoch % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch}, Loss: {avg_loss}')

            # evaluation for visualization purposes
            # here we can use the same function as for the classical training to have a fair comparison
            generator.eval()
            total_loss = 0
            all_preds = []
            all_trues = []
            with torch.no_grad():
                for batch in test_loader:
                    coeffs = batch[1].to(device)
                    times = torch.linspace(0, 1, batch[0].shape[1]).to(device)

                    true = batch[0][:, :, 1].to(device)
                    pred = generator(coeffs, times).squeeze(-1)
                    loss = criterion(pred, true)
                    total_loss += loss.item()

                    all_preds.append(pred.cpu())
                    all_trues.append(true.cpu())

            avg_loss = total_loss / len(test_loader)
            print(f'Test Loss: {avg_loss}')

            all_preds = torch.cat(all_preds, dim=0)
            all_trues = torch.cat(all_trues, dim=0)

            ## plotting
            # num_samples = 5
            #
            # plt.figure(figsize=(8, 4))
            # for i in range(num_samples):
            #     plt.plot(all_trues[i].numpy(), color='r')
            #     plt.plot(all_preds[i].numpy(), color='b')
            # plt.xlabel('Time')
            # plt.ylabel('Value')
            # plt.ylim(-0.75,1.25)
            # plt.title('Model Predictions vs True Values')
            # plt.show()

    # end of training
    generator.load_state_dict(averaged_generator.module.state_dict())
    discriminator.load_state_dict(averaged_discriminator.module.state_dict())

    return all_preds, all_trues

def evaluate_loss(ts, batch_size, dataloader, generator, discriminator):
    with torch.no_grad():
        total_samples = 0
        total_loss = 0
        for real_samples, in dataloader:
            generated_samples = generator(ts, batch_size)
            generated_score = discriminator(generated_samples)
            real_score = discriminator(real_samples)
            loss = generated_score - real_score
            total_samples += batch_size
            total_loss += loss.item() * batch_size
    return total_loss / total_samples