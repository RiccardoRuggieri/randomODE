import time
import torch

def _train_loop(model, optimizer, num_epochs, train_loader, test_loader, device, criterion):
    results = {
        "accuracy": None,
        "stdev": None,
    }

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            coeffs, labels = batch[1].to(device), batch[2].to(device)
            times = torch.linspace(0, 1, batch[0].shape[1]).to(device)
            optimizer.zero_grad()

            # Forward pass
            logits = model(coeffs, times)  # Include the 'times' argument
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        epoch_time = time.time() - start_time

        avg_train_loss = total_train_loss / len(train_loader)
        if epoch % 1 == 0:
            print(f'Epoch {epoch}, Train Loss: {avg_train_loss}, Time: {epoch_time:.2f} seconds')

            # Evaluation on test set
            model.eval()
            total_test_loss = 0
            all_preds = []
            all_trues = []
            with torch.no_grad():
                for batch in test_loader:
                    inputs, labels = batch[1].to(device), batch[2].to(device)
                    times = torch.linspace(0, 1, batch[0].shape[1]).to(device)

                    logits = model(inputs, times)  # Include the 'times' argument
                    loss = criterion(logits, labels)
                    total_test_loss += loss.item()

                    preds = torch.argmax(logits, dim=1)
                    all_preds.append(preds.cpu())
                    all_trues.append(labels.cpu())

            avg_test_loss = total_test_loss / len(test_loader)
            all_preds = torch.cat(all_preds, dim=0)
            all_trues = torch.cat(all_trues, dim=0)
            accuracy = (all_preds == all_trues).float().mean().item()

            # Plot sample predictions
            # num_samples = 5
            # print("Sample Random Predictions:")
            # for i  in range(num_samples):
            #     print(f"True: {all_trues[i].item()}, Pred: {all_preds[i].item()}")

            print(f'Epoch {epoch}, Test Loss: {avg_test_loss}, Test Accuracy: {accuracy:.4f}')

    results["accuracy"] = accuracy
    results["stdev"] = torch.std(all_preds.float()).item()

    return results


