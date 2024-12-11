import torch

def _train_loop(model, optimizer, num_epochs, train_loader, test_loader, device, criterion):
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            coeffs, labels = batch[1].to(device), batch[2].to(device)
            times = torch.linspace(0, 1, batch[0].shape[1]).to(device)

            optimizer.zero_grad()

            # Forward pass
            logits = model(coeffs, times)  # Include the 'times' argument
            loss = criterion(logits, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 1 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch}, Train Loss: {avg_loss}')

            # Evaluation on test set
            model.eval()
            total_loss = 0
            all_preds = []
            all_trues = []
            with torch.no_grad():
                for batch in test_loader:
                    inputs, labels = batch[1].to(device), batch[2].to(device)
                    times = torch.linspace(0, 1, batch[0].shape[1]).to(device)

                    logits = model(inputs, times)  # Include the 'times' argument
                    loss = criterion(logits, labels)
                    total_loss += loss.item()

                    preds = torch.argmax(logits, dim=1)
                    all_preds.append(preds.cpu())
                    all_trues.append(labels.cpu())

            avg_loss = total_loss / len(test_loader)
            print(f'Epoch {epoch}, Test Loss: {avg_loss}')

            # Concatenate all predictions and true labels
            all_preds = torch.cat(all_preds, dim=0)
            all_trues = torch.cat(all_trues, dim=0)

            # Accuracy Calculation
            accuracy = (all_preds == all_trues).float().mean().item()
            print(f'Epoch {epoch}, Test Accuracy: {accuracy:.4f}')

            # Plot sample predictions
            num_samples = 5
            print("Sample Random Predictions:")
            # todo: fix a random number between 0 and len(all_preds) - 1
            for i  in range(num_samples):
                print(f"True: {all_trues[i].item()}, Pred: {all_preds[i].item()}")

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