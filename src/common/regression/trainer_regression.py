import torch
import matplotlib.pyplot as plt
import tqdm
import model.regression.neuralsde_regression as neuralsde_regression
import torch.optim.swa_utils as swa_utils

def _train_loop(model, optimizer, num_epochs, train_loader, test_loader, device, criterion):
    global all_preds, all_trues
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            coeffs = batch[1].to(device)
            times = torch.linspace(0, 1, batch[0].shape[1]).to(device)

            optimizer.zero_grad()
            true = batch[0][:,:,1].to(device)
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

                    true = batch[0][:,:,1].to(device)
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

def _train_loop_asGAN(model, optimizer, num_epochs, train_loader, test_loader, device, criterion):
    global all_preds, all_trues
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            coeffs = batch[1].to(device)
            times = torch.linspace(0, 1, batch[0].shape[1]).to(device)

            optimizer.zero_grad()
            true = batch[0][:,:,1].to(device)
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

                    true = batch[0][:,:,1].to(device)
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


def main(
        # Architectural hyperparameters. These are quite small for illustrative purposes.
        initial_noise_size=5,  # How many noise dimensions to sample at the start of the SDE.
        noise_size=3,          # How many dimensions the Brownian motion has.
        hidden_size=16,        # How big the hidden size of the generator SDE and the discriminator CDE are.
        mlp_size=16,           # How big the layers in the various MLPs are.
        num_layers=1,          # How many hidden layers to have in the various MLPs.

        # Training hyperparameters. Be prepared to tune these very carefully, as with any GAN.
        generator_lr=2e-4,      # Learning rate often needs careful tuning to the problem.
        discriminator_lr=1e-3,  # Learning rate often needs careful tuning to the problem.
        batch_size=512,        # Batch size.
        steps=1000,            # How many steps to train both generator and discriminator for.
        init_mult1=3,           # Changing the initial parameter size can help.
        init_mult2=0.5,         #
        weight_decay=0.01,      # Weight decay.
        swa_step_start=5000,    # When to start using stochastic weight averaging.

        # Evaluation and plotting hyperparameters
        steps_per_print=10,                   # How often to print the loss.
        num_plot_samples=50,                  # How many samples to use on the plots at the end.
        plot_locs=(0.1, 0.3, 0.5, 0.7, 0.9),  # Plot some marginal distributions at this proportion of the way along.
):
    is_cuda = torch.cuda.is_available()
    device = 'cuda' if is_cuda else 'cpu'
    if not is_cuda:
        print("Warning: CUDA not available; falling back to CPU but this is likely to be very slow.")

    # Data
    ts, data_size, train_dataloader = OU_process.get_data(batch_size=batch_size, device=device)
    infinite_train_dataloader = (elem for it in iter(lambda: train_dataloader, None) for elem in it)

    # Models
    generator = neuralsde_regression.Generator(data_size, initial_noise_size, noise_size, hidden_size, mlp_size, num_layers).to(device)
    discriminator = neuralsde_regression.Discriminator(data_size, hidden_size, mlp_size, num_layers).to(device)
    # Weight averaging really helps with GAN training.
    averaged_generator = swa_utils.AveragedModel(generator)
    averaged_discriminator = swa_utils.AveragedModel(discriminator)

    # Picking a good initialisation is important!
    # In this case these were picked by making the parameters for the t=0 part of the generator be roughly the right
    # size that the untrained t=0 distribution has a similar variance to the t=0 data distribution.
    # Then the func parameters were adjusted so that the t>0 distribution looked like it had about the right variance.
    # What we're doing here is very crude -- one can definitely imagine smarter ways of doing things.
    # (e.g. pretraining the t=0 distribution)
    with torch.no_grad():
        for param in generator._initial.parameters():
            param *= init_mult1
        for param in generator._func.parameters():
            param *= init_mult2

    # Optimisers. Adadelta turns out to be a much better choice than SGD or Adam, interestingly.
    generator_optimiser = torch.optim.Adadelta(generator.parameters(), lr=generator_lr, weight_decay=weight_decay)
    discriminator_optimiser = torch.optim.Adadelta(discriminator.parameters(), lr=discriminator_lr,
                                                   weight_decay=weight_decay)

    # Train both generator and discriminator.
    trange = tqdm.tqdm(range(steps))
    for step in trange:
        real_samples, = next(infinite_train_dataloader)

        generated_samples = generator(ts, batch_size)
        generated_score = discriminator(generated_samples)
        real_score = discriminator(real_samples)
        loss = generated_score - real_score
        loss.backward()

        for param in generator.parameters():
            param.grad *= -1
        generator_optimiser.step()
        discriminator_optimiser.step()
        generator_optimiser.zero_grad()
        discriminator_optimiser.zero_grad()

        ###################
        # We constrain the Lipschitz constant of the discriminator using carefully-chosen clipping (and the use of
        # LipSwish activation functions).
        ###################
        with torch.no_grad():
            for module in discriminator.modules():
                if isinstance(module, torch.nn.Linear):
                    lim = 1 / module.out_features
                    module.weight.clamp_(-lim, lim)

        # Stochastic weight averaging typically improves performance.
        if step > swa_step_start:
            averaged_generator.update_parameters(generator)
            averaged_discriminator.update_parameters(discriminator)

        if (step % steps_per_print) == 0 or step == steps - 1:
            total_unaveraged_loss = evaluate_loss(ts, batch_size, train_dataloader, generator, discriminator)
            if step > swa_step_start:
                total_averaged_loss = evaluate_loss(ts, batch_size, train_dataloader, averaged_generator.module,
                                                    averaged_discriminator.module)
                trange.write(f"Step: {step:3} Loss (unaveraged): {total_unaveraged_loss:.4f} "
                             f"Loss (averaged): {total_averaged_loss:.4f}")
            else:
                trange.write(f"Step: {step:3} Loss (unaveraged): {total_unaveraged_loss:.4f}")
    generator.load_state_dict(averaged_generator.module.state_dict())
    discriminator.load_state_dict(averaged_discriminator.module.state_dict())

    _, _, test_dataloader = OU_process.get_data(batch_size=batch_size, device=device)

    # plot(ts, generator, test_dataloader, num_plot_samples, plot_locs)